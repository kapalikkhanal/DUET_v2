"""Litter depth and soft-floor sinkage sweeps, beyond the trained range.

PoultrySurface play environment (litter drag on, no pushes), arms at rest,
0.6 kg per hand, walking at +0.4 / -0.4 / sideways 0.3 m/s for 8 s at 0.78 m.
One soft tile column with R rows; envs spread over rows.

  --sweep litter : softness fixed at the training maximum, friction 0.5,
                   litter depth per row 0, 2, 4 (= training max), 6, 8, 10, 12 cm.
  --sweep sink   : no litter, friction 0.5, softness per row 0, 0.5, 1.0
                   (= training max), 1.5, 2.0 x the trained softness range
                   (solref/solimp extrapolated linearly). Reports measured
                   stance sinkage in cm.

Per row and command: falls, achieved/commanded speed, swing apex of the
lowest foot point above the litter surface, and the share of swing time the
toe spends inside the litter (scoop).
"""
import argparse, os, sys
from dataclasses import asdict, replace
os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np, torch
import mjlab.tasks, src.tasks  # noqa
import duet_demo_video as demo
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from src.tasks.common.terrains import SoftFlatTerrainCfg

p = argparse.ArgumentParser()
p.add_argument("--sweep", choices=("litter", "sink"), required=True)
p.add_argument("policies", nargs="+")
a = p.parse_args()
TASK, DEV = "Unitree-G1-23Dof-Duet-PoultrySurface", "cuda:0"
CMDS = [(0.4, 0.0), (-0.4, 0.0), (0.0, 0.3)]
if a.sweep == "litter":
  ROWS = 7; MAX_LAYER = 0.12
  soft = SoftFlatTerrainCfg(proportion=1.0, friction_range=(0.5, 0.5))
  diff = (1.0, 1.0)
  labels = [f"litter {100 * MAX_LAYER * r / (ROWS - 1):.0f} cm" for r in range(ROWS)]
else:
  ROWS = 5; MAX_LAYER = 0.0
  # Training: t in [0, 1] over (hard, soft). Here t' = 2t, extrapolated.
  ext = lambda lo, hi: (lo, lo + 2 * (hi - lo))
  soft = SoftFlatTerrainCfg(proportion=1.0, friction_range=(0.5, 0.5),
                            timeconst_range=ext(0.02, 0.05), dampratio_range=ext(1.0, 2.0),
                            dmin_range=ext(0.9, 0.6), dmax_range=ext(0.95, 0.9), width_range=ext(0.001, 0.02))
  diff = (0.0, 1.0)
  labels = [f"soft x{2 * r / (ROWS - 1):.1f}" for r in range(ROWS)]
N = ROWS * len(CMDS) * 48

cfg = load_env_cfg(TASK, play=True)
cfg.scene.num_envs = N
cfg.episode_length_s = 1e6
for k in ("push_robot", "stop_backward_walkers", "randomize_terrain"):
  cfg.events.pop(k, None)
cfg.events["hand_payload"].params["ranges"] = (0.6, 0.6)
cfg.events["reset_base"].params["pose_range"] = {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (0, 0), "yaw": (0, 0)}
tw_cfg = cfg.commands["twist"]; tw_cfg.heading_command = False; tw_cfg.ranges.heading = None
tw_cfg.resampling_time_range = (1e9, 1e9); cfg.commands["base_height"].resampling_time_range = (1e9, 1e9)
cfg.curriculum = {}
gen = cfg.scene.terrain.terrain_generator
cfg.scene.terrain.terrain_generator = replace(gen, num_rows=ROWS, num_cols=1, curriculum=True,
                                              difficulty_range=diff, sub_terrains={"soft_floor": soft})
cfg.scene.terrain.max_init_terrain_level = ROWS - 1
cfg.actions["poultry_surface"].max_layer_height = MAX_LAYER
cfg.actions["poultry_surface"].litter_enabled = MAX_LAYER > 0

env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=cfg, device=DEV), clip_actions=None)
u = env.unwrapped; robot = u.scene["robot"]
runner = (load_runner_cls(TASK) or MjlabOnPolicyRunner)(env, asdict(load_rl_cfg(TASK)), device=DEV)
terrain = u.scene.terrain
# Even spread: row = env % ROWS, command group = (env // ROWS) % 3.
ids = torch.arange(N, device=DEV)
rows = ids % ROWS; grp = (ids // ROWS) % len(CMDS)
terrain.terrain_levels[:] = rows
terrain.env_origins[:] = terrain.terrain_origins[rows, terrain.terrain_types]
surf = u.action_manager.get_term("poultry_surface")
arm = u.action_manager.get_term("upper_body_pose"); tw = u.command_manager.get_term("twist"); bh = u.command_manager.get_term("base_height")
rest = demo.pose_vec(arm.joint_names, demo.REST, demo.REST, DEV)
vx = torch.tensor([CMDS[g][0] for g in grp.tolist()], device=DEV); vy = torch.tensor([CMDS[g][1] for g in grp.tolist()], device=DEV)
toe = surf.toe_index
dt = u.step_dt; T0, T1 = 2.0, 10.0

for pol_path in a.policies:
  runner.load(pol_path, load_cfg={"actor": True}, strict=True, map_location=DEV)
  policy = runner.get_inference_policy(device=DEV)
  torch.manual_seed(0)
  falls = torch.zeros(N, device=DEV); spd = torch.zeros(N, device=DEV); nsp = torch.zeros(N, device=DEV)
  air_t = torch.zeros(N, device=DEV); scoop_t = torch.zeros(N, device=DEV)
  sink = torch.zeros(N, device=DEV); nsink = torch.zeros(N, device=DEV)
  apex_sum = torch.zeros(N, device=DEV); apex_n = torch.zeros(N, device=DEV)
  peak = torch.zeros(N, 2, device=DEV); slen = torch.zeros(N, 2, device=DEV)
  with torch.inference_mode():
    obs, _ = env.reset()
    for k in range(int(T1 / dt)):
      t = k * dt; walking = t >= T0
      tw.vel_command_b[:, 0] = vx if walking else 0.0; tw.vel_command_b[:, 1] = vy if walking else 0.0; tw.vel_command_b[:, 2] = 0.0
      tw.is_standing_env[:] = not walking
      bh._squat_target[:, 0] = 0.78; bh._walk_target[:, 0] = 0.78; bh._height[:, 0] = 0.78
      arm.set_external_target(rest.unsqueeze(0).expand(N, -1).contiguous())
      obs, _, _, _ = env.step(policy(obs))
      fell = u.termination_manager.get_term("fell_over").float()
      if t < T0 + 1.0:
        continue
      falls += fell
      s = surf.sample()
      v = robot.data.root_link_lin_vel_b
      along = torch.where(grp == 2, v[:, 1] / 0.3, v[:, 0] / vx.clamp(min=-1, max=1).where(grp != 2, torch.ones_like(vx)))
      spd += along * (1 - fell); nsp += (1 - fell)
      contact, air = s["contact"], s["air"]
      low_lit = s["lowest_litter"]                                   # lowest point above litter surface
      toe_clear = s["clearance"].gather(-1, toe[None, :, None].expand(N, -1, -1)).squeeze(-1)
      air_t += air.float().sum(1); scoop_t += (air & (toe_clear < 0.0)).float().sum(1)
      sink += (contact.float() * (-s["lowest"]).clamp(min=0)).sum(1); nsink += contact.float().sum(1)
      peak = torch.where(air, torch.maximum(peak, low_lit), peak); slen = torch.where(air, slen + dt, slen)
      ended = contact & (slen > 0.12)
      apex_sum += (peak * ended).sum(1); apex_n += ended.float().sum(1)
      peak = torch.where(contact, torch.full_like(peak, -1.0), peak); slen = torch.where(contact, torch.zeros_like(slen), slen)
  mins = (T1 - T0 - 1.0) / 60.0
  print(f"\n== {os.path.basename(os.path.dirname(pol_path))[-24:]}/{os.path.basename(pol_path)}  sweep={a.sweep}")
  print(f"{'condition':>14} | " + " | ".join(f"{lab:^38}" for lab in ("fwd +0.4", "back -0.4", "side 0.3")) + f" | {'sinkage':>7}")
  print(f"{'':>14} | " + " | ".join(f"{'falls/min':>9} {'speed':>6} {'apex':>7} {'scoop':>6}" + " " * 6 for _ in CMDS) + " |")
  for r in range(ROWS):
    cells = []
    for g in range(len(CMDS)):
      m = (rows == r) & (grp == g)
      fpm = float(falls[m].sum()) / (int(m.sum()) * mins)
      sp = float((spd[m] / nsp[m].clamp(min=1)).mean())
      ap = float((apex_sum[m].sum() / apex_n[m].sum().clamp(min=1)))
      sc = float(scoop_t[m].sum() / air_t[m].sum().clamp(min=1))
      cells.append(f"{fpm:9.3f} {100 * sp:5.0f}% {100 * ap:5.1f}cm {100 * sc:5.0f}%      ")
    m = rows == r
    sk = float(sink[m].sum() / nsink[m].sum().clamp(min=1))
    print(f"{labels[r]:>14} | " + " | ".join(cells) + f" | {100 * sk:5.2f}cm")
print("\n(speed = achieved / commanded along the command axis; apex = mean swing peak of the lowest foot point "
      "above the litter surface; scoop = share of airborne time with the toe below the litter surface; "
      "sinkage = mean depth of the lowest foot point below the floor while in contact)")
