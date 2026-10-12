"""Record a scripted one-minute demo video of a DUET checkpoint.

One robot, no pushes, no observation noise. Commands and arm targets are set
directly every control step (the same hooks ``duet_arm_scenarios.py`` uses),
so the video shows exactly the commanded behaviour:

   0- 3 s  stand, arms at rest            35-41 s  height 0.78 -> 0.55 -> 0.78 m
   3- 9 s  walk forward 0.5 m/s           41-48 s  deep squat to 0.24 m, arms down
   9-15 s  walk backward 0.5 m/s          48-54 s  arm poses: forward / high /
  15-25 s  turn left, then right 0.8 rad/s          asymmetric / wide
  25-35 s  strafe left, then right 0.3 m/s 54-60 s  random fast arm motion (the
                                                    training generator, full
                                                    workspace), walking then still

Each frame carries the segment label, the command and the measured response.
A per-segment summary (velocity / yaw / height error, max torso tilt, falls)
is printed at the end.

Usage:
  PYTHONPATH=. python scripts/duet_demo_video.py \
      --checkpoint logs/rsl_rl/DUET_G1_23dof/<run>/model_XXXX.pt \
      --out logs/videos/demo.mp4 [--terrain plane|soft] [--soft-difficulty 0.5] [--soft-friction 0.6]
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import asdict, replace

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

import mjlab.tasks  # noqa: F401
import src.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.terrains.terrain_generator import TerrainGeneratorCfg
from mjlab.utils.lab_api.math import quat_apply_inverse

from src.tasks.common.terrains import SoftFlatTerrainCfg

STAND = 0.78

# Arm poses (left side; the right side mirrors roll / yaw / wrist).
REST = {"pitch": 0.35, "roll": 0.18, "yaw": 0.0, "elbow": 0.87, "wrist": 0.0}
FORWARD = {"pitch": -1.2, "roll": 0.10, "yaw": 0.0, "elbow": 0.30, "wrist": 0.0}
HIGH = {"pitch": -1.6, "roll": 0.10, "yaw": 0.0, "elbow": 0.40, "wrist": 0.0}
DOWN_BACK = {"pitch": 0.8, "roll": 0.15, "yaw": 0.0, "elbow": 0.20, "wrist": 0.0}
LOW_PICK = {"pitch": 0.60, "roll": 0.15, "yaw": 0.0, "elbow": 0.15, "wrist": 0.0}
WIDE = {"pitch": 0.0, "roll": 1.2, "yaw": 0.0, "elbow": 0.30, "wrist": 0.0}

# (start s, end s, label, vx, vy, yaw rate)
SEGMENTS = [
  (0, 3, "Stand still, arms at rest", 0.0, 0.0, 0.0),
  (3, 9, "Walk forward 0.5 m/s", 0.5, 0.0, 0.0),
  (9, 15, "Walk backward 0.5 m/s", -0.5, 0.0, 0.0),
  (15, 20, "Turn left 0.8 rad/s", 0.0, 0.0, 0.8),
  (20, 25, "Turn right 0.8 rad/s", 0.0, 0.0, -0.8),
  (25, 30, "Strafe left 0.3 m/s", 0.0, 0.3, 0.0),
  (30, 35, "Strafe right 0.3 m/s", 0.0, -0.3, 0.0),
  (35, 41, "Height change 0.78 -> 0.55 -> 0.78 m", 0.0, 0.0, 0.0),
  (41, 48, "Deep squat to 0.24 m, arms down (low pick)", 0.0, 0.0, 0.0),
  (48, 54, "Arm poses: forward / high / asymmetric / wide", 0.0, 0.0, 0.0),
  (54, 58, "Random fast arm motion, walking 0.3 m/s", 0.3, 0.0, 0.0),
  (58, 60, "Random fast arm motion, standing", 0.0, 0.0, 0.0),
]
TOTAL_S = SEGMENTS[-1][1]

# Pelvis-height keyframes (s, m), cosine-blended between knots.
HEIGHT_KEYS = [
  (0, STAND), (35, STAND), (38, 0.55), (41, STAND),
  (43.5, 0.24), (45.5, 0.24), (48, STAND), (TOTAL_S, STAND),
]
# Arm keyframes (s, (left pose, right pose)), cosine-blended; after
# GENERATOR_FROM the training-time random generator drives the arms.
ARM_KEYS = [
  (0, (REST, REST)), (41, (REST, REST)), (43.5, (LOW_PICK, LOW_PICK)),
  (45.5, (LOW_PICK, LOW_PICK)), (48, (REST, REST)),
  (48.7, (FORWARD, FORWARD)), (49.2, (FORWARD, FORWARD)),
  (49.9, (HIGH, HIGH)), (50.4, (HIGH, HIGH)),
  (51.1, (FORWARD, DOWN_BACK)), (51.6, (FORWARD, DOWN_BACK)),
  (52.3, (WIDE, WIDE)), (52.8, (WIDE, WIDE)),
  (53.5, (REST, REST)), (54, (REST, REST)),
]
GENERATOR_FROM = 54.0


def pose_vec(names, left, right, device) -> torch.Tensor:
  vals = []
  for n in names:
    side, sign = (left, 1.0) if n.startswith("left_") else (right, -1.0)
    if "shoulder_pitch" in n:
      v = side["pitch"]
    elif "shoulder_roll" in n:
      v = sign * side["roll"]
    elif "shoulder_yaw" in n:
      v = sign * side["yaw"]
    elif "elbow" in n:
      v = side["elbow"]
    elif "wrist" in n:
      v = sign * side["wrist"]
    else:
      raise ValueError(n)
    vals.append(v)
  return torch.tensor(vals, device=device, dtype=torch.float32)


def blend(keys, t):
  """Cosine interpolation through (time, value) keyframes."""
  if t <= keys[0][0]:
    return keys[0][1], keys[0][1], 0.0
  for (t0, v0), (t1, v1) in zip(keys[:-1], keys[1:]):
    if t <= t1:
      a = 0.5 - 0.5 * math.cos(math.pi * (t - t0) / max(t1 - t0, 1e-9))
      return v0, v1, a
  return keys[-1][1], keys[-1][1], 0.0


def segment_at(t):
  for seg in SEGMENTS:
    if seg[0] <= t < seg[1]:
      return seg
  return SEGMENTS[-1]


def build_cfg(task: str, terrain: str, soft_difficulty: float, soft_friction: float, width: int, height: int):
  cfg = load_env_cfg(task, play=True)
  cfg.scene.num_envs = 1
  cfg.episode_length_s = int(1e9)
  cfg.events.pop("randomize_terrain", None)
  cfg.events["reset_base"].params["pose_range"] = {"x": (0, 0), "y": (0, 0), "z": (0, 0), "yaw": (0, 0)}
  tw = cfg.commands["twist"]
  tw.heading_command = False  # yaw rate is commanded directly
  tw.ranges.heading = None
  tw.resampling_time_range = (1e9, 1e9)
  cfg.commands["base_height"].resampling_time_range = (1e9, 1e9)
  arm = cfg.actions["upper_body_pose"]
  arm.init_ratio = 1.0  # full trained arm distribution for the random segment
  arm.clean_fraction_stages = ((0, 0.0),)
  for s in cfg.scene.sensors or ():
    if s.name == "terrain_scan":
      s.debug_vis = False
  assert cfg.scene.terrain is not None
  if terrain == "plane":
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None
  else:  # one large soft-litter tile at a fixed softness
    gen = cfg.scene.terrain.terrain_generator
    assert isinstance(gen, TerrainGeneratorCfg)
    cfg.scene.terrain.terrain_generator = replace(
      gen, size=(30.0, 30.0), num_rows=1, num_cols=1, curriculum=True,
      difficulty_range=(soft_difficulty, soft_difficulty),
      sub_terrains={"soft_floor": SoftFlatTerrainCfg(
        proportion=1.0, friction_range=(soft_friction, soft_friction))},
    )
    cfg.scene.terrain.max_init_terrain_level = 0
  cfg.viewer.width, cfg.viewer.height = width, height
  cfg.viewer.distance = 3.2
  cfg.viewer.elevation = -12.0
  cfg.viewer.azimuth = 115.0
  return cfg


def main() -> None:
  p = argparse.ArgumentParser()
  p.add_argument("--checkpoint", required=True)
  p.add_argument("--out", required=True)
  p.add_argument("--task", default="Unitree-G1-23Dof-Duet-Rough")
  p.add_argument("--terrain", choices=("plane", "soft"), default="plane")
  p.add_argument("--soft-difficulty", type=float, default=0.5)
  p.add_argument("--soft-friction", type=float, default=0.6)
  p.add_argument("--width", type=int, default=1280)
  p.add_argument("--height", type=int, default=720)
  p.add_argument("--device", default="cuda:0")
  args = p.parse_args()

  cfg = build_cfg(args.task, args.terrain, args.soft_difficulty, args.soft_friction, args.width, args.height)
  agent_cfg = load_rl_cfg(args.task)
  env_raw = ManagerBasedRlEnv(cfg=cfg, device=args.device, render_mode="rgb_array")
  env = RslRlVecEnvWrapper(env_raw, clip_actions=agent_cfg.clip_actions)
  runner_cls = load_runner_cls(args.task) or MjlabOnPolicyRunner
  runner = runner_cls(env, asdict(agent_cfg), device=args.device)
  runner.load(args.checkpoint, load_cfg={"actor": True}, strict=True, map_location=args.device)
  policy = runner.get_inference_policy(device=args.device)

  u = env.unwrapped
  robot = u.scene["robot"]
  arm = u.action_manager.get_term("upper_body_pose")
  tw = u.command_manager.get_term("twist")
  bh = u.command_manager.get_term("base_height")
  torso_id = u.reward_manager.get_term_cfg("body_orientation_l2").params["asset_cfg"].body_ids
  foot_ids = u.reward_manager.get_term_cfg("track_base_height").params["asset_cfg"].site_ids
  names = arm.joint_names
  dt = u.step_dt
  fps = int(round(1.0 / dt))

  font = ImageFont.load_default(size=max(18, args.height // 28))
  small = ImageFont.load_default(size=max(14, args.height // 38))
  os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
  writer = imageio.get_writer(args.out, fps=fps, codec="libx264", quality=8, macro_block_size=1)

  stats: dict[str, dict[str, list[float]]] = {s[2]: {"v": [], "yaw": [], "h": [], "tilt": []} for s in SEGMENTS}
  falls: dict[str, int] = {s[2]: 0 for s in SEGMENTS}
  total_falls = 0
  tag = (f"soft floor  difficulty {args.soft_difficulty}  friction {args.soft_friction}"
         if args.terrain == "soft" else "flat hard ground")
  obs = env.get_observations()
  with torch.inference_mode():
    for step in range(int(TOTAL_S / dt)):
      t = step * dt
      _, _, label, vx, vy, wz = segment_at(t)
      tw.vel_command_b[:, 0] = vx
      tw.vel_command_b[:, 1] = vy
      tw.vel_command_b[:, 2] = wz
      if hasattr(tw, "is_standing_env"):
        tw.is_standing_env[:] = vx == 0 and vy == 0 and wz == 0
      h0, h1, a = blend(HEIGHT_KEYS, t)
      h = h0 + a * (h1 - h0)
      bh._squat_target[:] = h
      bh._walk_target[:] = h
      bh._height[:] = h
      if t < GENERATOR_FROM:
        (l0, r0), (l1, r1), a = blend(ARM_KEYS, t)
        q0 = pose_vec(names, l0, r0, args.device)
        q1 = pose_vec(names, l1, r1, args.device)
        arm.set_external_target((q0 + a * (q1 - q0)).unsqueeze(0))
      else:
        arm.set_external_target(None)

      obs, _, _, _ = env.step(policy(obs))

      fell = bool(u.termination_manager.get_term("fell_over").any())
      if fell:
        falls[label] += 1
        total_falls += 1
      lin = robot.data.root_link_lin_vel_b[0]
      ang = robot.data.root_link_ang_vel_b[0]
      root_z = robot.data.root_link_pos_w[0, 2]
      foot_z = robot.data.site_pos_w[0, foot_ids, 2]
      h_meas = float(root_z - foot_z.min() + 0.02)
      quat = robot.data.body_link_quat_w[:, torso_id, :].squeeze(1)
      g = quat_apply_inverse(quat, robot.data.gravity_vec_w)[0]
      tilt = math.degrees(math.acos(max(-1.0, min(1.0, float(-g[2])))))
      st = stats[label]
      st["v"].append(float(torch.norm(lin[:2] - torch.tensor([vx, vy], device=lin.device))))
      st["yaw"].append(abs(float(ang[2]) - wz))
      st["h"].append(abs(h_meas - h))
      st["tilt"].append(tilt)

      frame = Image.fromarray(env.unwrapped.render())
      d = ImageDraw.Draw(frame)
      d.rectangle((0, 0, args.width, font.size + small.size + 30), fill=(0, 0, 0))
      d.text((16, 10), f"{t:5.1f} s   {label}", font=font, fill=(255, 255, 255))
      d.text((args.width - 16, 10), tag, font=small, fill=(255, 210, 120), anchor="ra")
      d.text(
        (16, 10 + font.size + 8),
        f"cmd  vx {vx:+.2f}  vy {vy:+.2f}  yaw {wz:+.2f}  h {h:.2f} m      "
        f"meas  vx {float(lin[0]):+.2f}  vy {float(lin[1]):+.2f}  yaw {float(ang[2]):+.2f}  "
        f"h {h_meas:.2f} m   tilt {tilt:4.1f} deg   falls {total_falls}",
        font=small, fill=(200, 230, 255),
      )
      if fell:
        d.text((args.width - 140, 10), "FELL", font=font, fill=(255, 60, 60))
      writer.append_data(np.asarray(frame))
  writer.close()

  surface = (f"soft, difficulty {args.soft_difficulty}, friction {args.soft_friction}"
             if args.terrain == "soft" else "plane")
  print(f"\nSaved {args.out}  ({TOTAL_S} s, {fps} fps, {args.width}x{args.height}, terrain={surface})")
  print(f"{'segment':48s} {'v err':>7s} {'yaw err':>8s} {'h err':>7s} {'max tilt':>9s} {'falls':>6s}")
  for _, _, label, *_ in SEGMENTS:
    st = stats[label]
    mean = lambda x: sum(x) / max(len(x), 1)
    print(f"{label:48s} {mean(st['v']):7.3f} {mean(st['yaw']):8.3f} {mean(st['h']):7.3f} "
          f"{max(st['tilt'] or [0]):9.1f} {falls[label]:6d}")
  env.close()


if __name__ == "__main__":
  main()
