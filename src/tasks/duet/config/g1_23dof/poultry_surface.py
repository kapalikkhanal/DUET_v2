"""Independent poultry-surface task; existing DUET variants keep their configs."""

from dataclasses import replace

from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

import src.tasks.duet.mdp as mdp
from src.tasks.duet.mdp.poultry_surface import (
  PoultrySurfaceActionCfg,
  action_excess_l2,
  backward_speed_deficit,
  joint_pos_limits_widened,
  poultry_clearance,
  poultry_foot_level,
  poultry_metric,
  poultry_plow,
  poultry_swing_height,
  stop_backward_walkers,
)
from .env_cfgs import (
  HEIGHT_RANGE,
  SQUAT_REACH_POSES,
  _arm_pose,
  unitree_g1_23dof_duet_rough_env_cfg,
)
SQUAT_FLOOR = 0.22  # m; was HEIGHT_RANGE[0] = 0.24

from .rl_cfg import ROUGH_REWARD_FLOOR, unitree_g1_23dof_duet_ppo_runner_cfg


def unitree_g1_23dof_duet_poultry_surface_env_cfg(
  play=False, litter_enabled=True, *, litter_depth_mode=None,
  litter_depth_range=(0.0, 0.06),
):
  """Random episode depths in training; legacy row depths in play/evaluation.

  Pin a depth in either mode with litter_depth_mode="random" and
  litter_depth_range=(d, d), or set those fields on actions["poultry_surface"].
  """
  cfg = unitree_g1_23dof_duet_rough_env_cfg(play=play)
  # Cap every training stage as well as the unstaged play command range.
  if "command_vel" in cfg.curriculum:
    for stage in cfg.curriculum["command_vel"].params["velocity_stages"]:
      lo, hi = stage["lin_vel_x"]
      stage["lin_vel_x"] = (max(lo, -0.6), hi)
  twist = cfg.commands["twist"]
  lo, hi = twist.ranges.lin_vel_x
  twist.ranges.lin_vel_x = (max(lo, -0.6), hi)
  cfg.rewards["backward_speed_deficit"] = RewardTermCfg(
    func=backward_speed_deficit, weight=-1.5,
  )
  cfg.events["stop_backward_walkers"] = EventTermCfg(
    func=stop_backward_walkers, mode="interval", interval_range_s=(1.0, 3.0),
    params={"prob": 0.3},
  )

  # Restore squat balance in this Rough-derived task, including medium-height
  # squats and a fading centering penalty at the nominal standing height.
  cfg.rewards["squat_com_centering"] = RewardTermCfg(
    func=mdp.squat_com_centering, weight=-2.0,
    params={
      "sensor_name": "feet_ground_contact",
      "command_name": "twist",
      "height_command_name": "base_height",
      "squat_below": 0.65,
      "full_above": 0.85,
      "foot_half_length": 0.09,
      "command_threshold": 0.1,
      "asset_cfg": SceneEntityCfg("robot", site_names=("left_foot", "right_foot")),
    },
  )
  cfg.rewards["body_orientation_l2"].params.update(
    relax_below=0.35, full_above=0.60, relax_factor=0.15,
  )
  arm = cfg.actions["upper_body_pose"]
  arm.squat_reach_poses = SQUAT_REACH_POSES + (
    _arm_pose(pitch=0.0, roll=1.2, yaw=0.0, elbow=0.30),
    _arm_pose(pitch=-0.6, roll=0.9, yaw=0.0, elbow=0.50),
  )
  arm.squat_reach_prob = 0.75
  arm.squat_reach_below = 0.65
  cfg.commands["base_height"].deep_squat_fraction = 0.6
  # Deeper squat (2026-10-12): the floor ramps 0.24 -> 0.22 m over iterations
  # 90k-92k (knots are policy steps, 24 per iteration), so a resume from 90k
  # is not asked for the new depth at once. Paired with the 97% squat-joint
  # limit band below; at 90% the kinematic floor was 0.29 m.
  base_height = cfg.commands["base_height"]
  base_height.height_range = (SQUAT_FLOOR, base_height.height_range[1])
  if base_height.floor_stages:
    base_height.floor_stages = tuple(base_height.floor_stages) + (
      (90_000 * 24, HEIGHT_RANGE[0]), (92_000 * 24, SQUAT_FLOOR),
    )
  elif base_height.floor_curriculum_start <= HEIGHT_RANGE[0]:
    base_height.floor_curriculum_start = SQUAT_FLOOR  # pinned (play/eval): full depth
  base_height.deep_squat_range = (SQUAT_FLOOR, 0.65)
  # Same penalty as joint_pos_limits (-10), but hip pitch, knee and ankle pitch
  # may use 97% of their range; every other joint keeps the 90% soft limit.
  cfg.rewards["joint_pos_limits"] = RewardTermCfg(
    func=joint_pos_limits_widened, weight=cfg.rewards["joint_pos_limits"].weight,
    params={"factor": 0.97},
  )

  # The new task uses identifiable terrain columns even in play. Membership is
  # checked at each foot point's actual xy, including crossings between tiles.
  proportions = {
    "flat": 0.15, "soft_floor": 0.60,
    "hf_pyramid_slope": 0.05, "hf_pyramid_slope_inv": 0.05,
    "random_rough": 0.05, "wave_terrain": 0.10,
  }
  generator = cfg.scene.terrain.terrain_generator
  cfg.scene.terrain.terrain_generator = replace(
    generator, curriculum=True, num_cols=20,
    sub_terrains={
      name: replace(sub, proportion=proportions[name])
      for name, sub in generator.sub_terrains.items()
    },
  )
  cfg.actions["poultry_surface"] = PoultrySurfaceActionCfg(
    entity_name="robot", litter_enabled=litter_enabled,
    litter_depth_mode=("row" if play else "random") if litter_depth_mode is None else litter_depth_mode,
    litter_depth_range=litter_depth_range,
  )
  # V1 saved rollouts: sole apex ~1 cm, ankle stays near default, and a level
  # 10 cm swing competes with ~0.23/s posture reward. These tolerances reduce
  # that estimated cost to ~0.04/s while retaining the balance-joint settings.
  cfg.rewards["pose"].params["std_walking"].update({
    r".*hip_pitch.*": 0.8, r".*knee.*": 0.8, r".*ankle_pitch.*": 0.4,
  })
  # Lowest sole above the stance floor; contact-independent travel cost. The
  # old world-z/midfoot foot_clearance is removed here (a separate key keeps the
  # curves from reading as the old term) and untouched for every other task.
  cfg.rewards.pop("foot_clearance")
  # Boost 1 (2026-10-10, at 60k): -2.0 -> -3.0. Boost 2 (at 84k): lift flat at
  # ~7.3 cm from 70k to 85k against the 9 cm target; -3.0 -> -4.0.
  lift_target = {"above_litter": 0.05, "min_target": 0.09, "max_target": 0.11}
  cfg.rewards["poultry_clearance"] = RewardTermCfg(
    func=poultry_clearance, weight=-4.0, params=lift_target.copy(),
  )
  # Offline V1 rates at d=4 cm: ~0.012-0.015 raw plow on bare flat/soft floor.
  # Weight -6 makes that ~0.075-0.09/s, above the relaxed posture cost and well
  # below individual tracking rewards (~1.4-1.95/s). Layer-relative cost is
  # larger for buried toes. Measured on v1 at 0.35-0.5 m/s: -0.08 to -0.12/s
  # on bare flat/soft floor, -0.18 to -0.25/s on the 4 cm litter row.
  cfg.rewards["poultry_plow"] = RewardTermCfg(func=poultry_plow, weight=-6.0)
  # At 3.33 landings/s, a fully missed target costs about 0.2/s. Unlike just
  # raising an L1 target, this supplies height pressure to an already-low gait.
  # Boost 1 (at 60k): -0.06 -> -0.12. Boost 2 (at 84k): -0.12 -> -0.20.
  cfg.rewards["poultry_swing_height"] = RewardTermCfg(
    func=poultry_swing_height, weight=-0.20, params=lift_target.copy(),
  )
  cfg.rewards["poultry_foot_level"] = RewardTermCfg(func=poultry_foot_level, weight=-1.0)
  # Raw outputs drifted past the deploy guard in all three 100k lines (66k,
  # 66k, 75k; worst joint ankle pitch). At raw 12 this costs ~20% of a step's
  # reward, at the 15 clip ~60%; below 8 it is zero. -0.05 was too weak: 85k
  # still reached 17.8 (guard trip, ankle pitch), because the 15 clip hid
  # everything above it from the penalty. Now -0.2, with the clip at 30.
  cfg.rewards["action_excess_l2"] = RewardTermCfg(
    func=action_excess_l2, weight=-0.2, params={"threshold": 8.0},
  )
  # Actions are clipped at 30 (not 15) so the penalty above sees the real
  # magnitude; the last-action input stays within the deploy guard's +-15 so a
  # rare outlier cannot swamp the frozen-normalizer inputs.
  for group in ("actor", "critic"):
    terms = cfg.observations[group].terms
    terms["actions"] = replace(terms["actions"], clip=(-15.0, 15.0))
  for key, name in {
    "poultry_swing_apex_m": "apex",  # held last completed apex, episode mean
    "poultry_swing_clearance_m": "clearance",
    "poultry_plow_cost": "plow",
    "poultry_toe_low_travel_mps": "low_travel",
    "poultry_toe_contact_slide_mps": "contact_slide",
    "poultry_liftoff_toe_down_deg": "liftoff_toe_down",
    "poultry_drag_force_n": "drag_force",
    "poultry_layer_height_m": "layer",
    "poultry_falls": "falls",
  }.items():
    cfg.metrics[key] = MetricsTermCfg(func=poultry_metric, params={"name": name})
  return cfg


def poultry_surface_ppo_runner_cfg():
  cfg = unitree_g1_23dof_duet_ppo_runner_cfg(
    experiment_name="DUET_G1_23dof_poultry",
    reward_floor=ROUGH_REWARD_FLOOR,
    max_iterations=100_001,
  )
  # Fine-tunes resume from a mature checkpoint; see rl_cfg.DuetPpoAlgorithmCfg.
  cfg.algorithm.freeze_obs_normalizer_on_load = True
  # Healthy checkpoints never exceed |a| ~13 (60k-65.5k); from 66k rare states
  # produced |a| up to 1000, blowing up the physics and the value loss. 30
  # bounds that while leaving 15-30 visible to action_excess_l2 (the deploy
  # runaway guard trips above 15); the last-action input is clipped at 15.
  cfg.clip_actions = 30.0
  return cfg
