"""Extreme raw actions and deploy-guard trips (derived from falls_breakdown.py).

Where do falls happen? Training distribution, split by terrain type and activity.

PoultrySurface task as trained (pushes, arm motion, payloads, litter, obs noise),
curriculum at its final stage (step counter set to the trained value), terrain
rows frozen at their random initial level (0-5). Each fall is attributed to the
tile under the robot and to the command it had on the step it fell. Rates are
falls per robot-minute of exposure in that class. NOISE=1 adds the policy's own
action noise (std from the checkpoint), as during training rollouts.

Usage: falls_breakdown.py CKPT [CKPT ...]
"""
import os, sys
from dataclasses import asdict
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np, torch
import mjlab.tasks, src.tasks  # noqa
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.lab_api.math import quat_apply_inverse

TASK, DEV = "Unitree-G1-23Dof-Duet-PoultrySurface", "cuda:0"
N = int(os.environ.get("N_ENV", 2048)); STEPS = int(os.environ.get("STEPS", 2000))
NOISE = os.environ.get("NOISE", "1") == "1"; SEED = int(os.environ.get("SEED", 0))

cfg = load_env_cfg(TASK, play=False)
cfg.scene.num_envs = N
cfg.curriculum.pop("terrain_levels", None)
ag = load_rl_cfg(TASK)
env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=cfg, device=DEV), clip_actions=ag.clip_actions)
runner = (load_runner_cls(TASK) or MjlabOnPolicyRunner)(env, asdict(ag), device=DEV)
u = env.unwrapped; robot = u.scene["robot"]
tg = cfg.scene.terrain.terrain_generator
names = list(tg.sub_terrains.keys())
props = np.array([s.proportion for s in tg.sub_terrains.values()], dtype=float)
colmap = torch.tensor(np.searchsorted(np.cumsum(props / props.sum()),
                                      np.arange(tg.num_cols) / tg.num_cols + 0.001, side="right"), device=DEV)
T_NAMES = names + ["border"]
A_NAMES = ["stand_deep(<0.6)", "stand_tall", "walk_fwd", "walk_back", "side/turn"]
tw = u.command_manager.get_term("twist")


def classify(pos):
  row = torch.floor(pos[:, 0] / tg.size[0] + tg.num_rows / 2).long()
  col = torch.floor(pos[:, 1] / tg.size[1] + tg.num_cols / 2).long()
  inside = (row >= 0) & (row < tg.num_rows) & (col >= 0) & (col < tg.num_cols)
  t = torch.where(inside, colmap[col.clamp(0, tg.num_cols - 1)], torch.full_like(col, len(names)))
  cmd = u.command_manager.get_command("twist"); h = u.command_manager.get_command("base_height")[:, 0]
  speed = cmd[:, :2].norm(dim=1) + cmd[:, 2].abs()
  standing = speed <= 0.1
  a = torch.full_like(col, 4)
  a = torch.where(~standing & (cmd[:, 0] > 0.1) & (cmd[:, 0].abs() >= cmd[:, 1].abs()), 0 * a + 2, a)
  a = torch.where(~standing & (cmd[:, 0] < -0.1) & (cmd[:, 0].abs() >= cmd[:, 1].abs()), 0 * a + 3, a)
  a = torch.where(standing & (h < 0.6), 0 * a, a)
  a = torch.where(standing & (h >= 0.6), 0 * a + 1, a)
  return t, a



for ckpt in sys.argv[1:]:
  runner.load(ckpt, load_cfg={"actor": True}, strict=True, map_location=DEV)
  pol = runner.get_inference_policy(device=DEV)
  std = torch.load(ckpt, map_location=DEV, weights_only=False)["actor_state_dict"]["distribution.std_param"]
  torch.manual_seed(SEED)
  u.common_step_counter = int(os.path.basename(ckpt).split("_")[1].split(".")[0]) * 24
  hist = {10: 0, 15: 0, 30: 0, 100: 0}; amax = 0.0; vmax = 0.0; n = 0; falls = 0
  run = torch.zeros(N, device=DEV); trips = 0; joint_hits = torch.zeros(13, device=DEV)
  timer = torch.zeros(N, device=DEV); ext_events = 0; ext_then_fall = 0; ext_tilted = 0
  tid = u.reward_manager.get_term_cfg('body_orientation_l2').params['asset_cfg'].body_ids
  with torch.inference_mode():
    obs, _ = env.reset()
    for k in range(STEPS):
      act = pol(obs)
      if NOISE:
        act = act + std * torch.randn_like(act)
      m = act.abs().max(dim=1).values
      if k >= 50:
        for th in hist: hist[th] += int((m > th).sum())
        amax = max(amax, float(m.max())); n += N
        vmax = max(vmax, float(robot.data.joint_vel.abs().max()))
        over = m > 15
        joint_hits += (act.abs() > 15).float().sum(0)
        run = torch.where(over, run + 1, torch.zeros_like(run))
        trips += int((run == 3).sum())  # deploy guard: |a| > 15 for 60 ms = 3 policy steps
        g = quat_apply_inverse(robot.data.body_link_quat_w[:, tid, :].squeeze(1), robot.data.gravity_vec_w)
        tilt = torch.rad2deg(torch.acos((-g[:, 2]).clamp(-1, 1)))
        new = over & (timer <= 0)
        ext_events += int(new.sum()); ext_tilted += int((new & (tilt > 30)).sum())
        timer = torch.where(new, torch.full_like(timer, 25), timer - 1)
      obs, _, _, _ = env.step(act)
      if k >= 50:
        fell = u.termination_manager.get_term("fell_over")
        falls += int(fell.sum()); ext_then_fall += int((fell & (timer > 0)).sum()); timer = torch.where(fell, torch.zeros_like(timer), timer)
  tag = os.path.basename(os.path.dirname(ckpt))[-14:] + "/" + os.path.basename(ckpt)
  print(f"{tag}: env-steps {n}, max|a| {amax:.1f}, " + ", ".join(f">{th}: {hist[th]} ({1e6*hist[th]/n:.1f} per M)" for th in hist) + f", max|qvel| {vmax:.0f}, falls {falls}, GUARD TRIPS (>15 for 3 steps) {trips}, joints>15 {[int(x) for x in joint_hits.tolist()]}, extreme events {ext_events}: torso tilt >30deg at onset {ext_tilted}, fell within 0.5 s {ext_then_fall}")
