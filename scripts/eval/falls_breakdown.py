"""Where do falls happen? Training distribution, split by terrain type and activity.

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


results = {}
for ckpt in sys.argv[1:]:
  runner.load(ckpt, load_cfg={"actor": True}, strict=True, map_location=DEV)
  pol = runner.get_inference_policy(device=DEV)
  std = torch.load(ckpt, map_location=DEV, weights_only=False)["actor_state_dict"]["distribution.std_param"]
  torch.manual_seed(SEED)
  u.common_step_counter = int(os.path.basename(ckpt).split("_")[1].split(".")[0]) * 24
  with torch.inference_mode():
    obs, _ = env.reset()
    ft = torch.zeros(len(T_NAMES), device=DEV); et = torch.zeros_like(ft)
    fa = torch.zeros(len(A_NAMES), device=DEV); ea = torch.zeros_like(fa)
    fta = torch.zeros(len(T_NAMES), len(A_NAMES), device=DEV)
    for k in range(STEPS):
      t, a = classify(robot.data.root_link_pos_w)
      act = pol(obs)
      if NOISE:
        act = act + std * torch.randn_like(act)
      obs, _, _, _ = env.step(act)
      if k < 50:  # skip the spawn transient
        continue
      fell = u.termination_manager.get_term("fell_over").float()
      et += torch.bincount(t, minlength=len(T_NAMES)).float(); ft += torch.bincount(t, weights=fell, minlength=len(T_NAMES))
      ea += torch.bincount(a, minlength=len(A_NAMES)).float(); fa += torch.bincount(a, weights=fell, minlength=len(A_NAMES))
      fta += torch.bincount(t * len(A_NAMES) + a, weights=fell, minlength=len(T_NAMES) * len(A_NAMES)).view(len(T_NAMES), len(A_NAMES))
  mins = u.step_dt / 60.0
  tag = os.path.basename(os.path.dirname(ckpt))[-12:] + "/" + os.path.basename(ckpt)
  print(f"\n== {tag}  noise={NOISE}  envs={N} steps={STEPS}: {int(ft.sum())} falls, "
        f"{float(ft.sum() / (et.sum() * mins)):.3f} per robot-minute")
  print(f"{'terrain':>16} {'robot-min':>9} {'falls':>6} {'per min':>8}")
  for i, n in enumerate(T_NAMES):
    if et[i] > 0:
      print(f"{n:>16} {float(et[i] * mins):9.0f} {int(ft[i]):6d} {float(ft[i] / (et[i] * mins)):8.3f}")
  print(f"{'activity':>16} {'robot-min':>9} {'falls':>6} {'per min':>8}")
  for i, n in enumerate(A_NAMES):
    if ea[i] > 0:
      print(f"{n:>16} {float(ea[i] * mins):9.0f} {int(fa[i]):6d} {float(fa[i] / (ea[i] * mins)):8.3f}")
  print("falls by terrain x activity:", {T_NAMES[i]: [int(x) for x in fta[i].tolist()] for i in range(len(T_NAMES)) if fta[i].sum() > 0})
