"""Foot clearance while walking: does the foot lift, or skim and scoop?

Flat ground, +0.6 kg/hand, no pushes, arms at rest. Env groups walk at
+0.4, -0.4 m/s (forward/backward) and 0.3 m/s sideways for 8 s at 0.78 m.
For every foot, from MuJoCo geometry (the foot's capsule collision geoms):
  * swing apex  : highest point of the sole's lowest point during each swing
  * toe/heel min: lowest point of the foot geometry in mid-swing (scoop risk)
  * skim dist   : horizontal distance the foot travels while airborne but
                  with its lowest point < 2 cm off the ground, per metre walked
  * stance slip : horizontal foot speed while in contact
"""
import argparse, math, os, sys
from dataclasses import asdict
os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np, torch
import duet_demo_video as demo
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_rl_cfg, load_runner_cls

p = argparse.ArgumentParser(); p.add_argument("--policy", required=True); a = p.parse_args()
TASK, DEV, N = "Unitree-G1-23Dof-Duet-Rough", "cuda:0", 96
CMDS = [(0.4, 0.0), (-0.4, 0.0), (0.0, 0.3)]
cfg = demo.build_cfg(TASK, "plane", 0.5, 0.6, 320, 240); cfg.scene.num_envs = N
cfg.events["hand_payload"].params["ranges"] = (0.6, 0.6)
env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=cfg, device=DEV), clip_actions=None)
u = env.unwrapped; robot = u.scene["robot"]
if a.policy.endswith(".onnx"):
  import onnxruntime as ort
  sess = ort.InferenceSession(a.policy, providers=["CPUExecutionProvider"])
  def policy(obs):
    x = obs["actor"].cpu().numpy().astype(np.float32)
    return torch.tensor(np.concatenate([sess.run(None, {"obs": x[i:i+1]})[0] for i in range(len(x))]), device=DEV)
else:
  runner = (load_runner_cls(TASK) or MjlabOnPolicyRunner)(env, asdict(load_rl_cfg(TASK)), device=DEV)
  runner.load(a.policy, load_cfg={"actor": True}, strict=True, map_location=DEV)
  policy = runner.get_inference_policy(device=DEV)

mm = u.sim.mj_model
feet = {}
for g in range(mm.ngeom):
  n = mm.geom(g).name or ""
  if "_foot" in n and "collision" in n and ("left_foot" in n or "right_foot" in n):
    feet.setdefault("left" if "left_foot" in n else "right", []).append(g)
print("foot geoms:", {k: len(v) for k, v in feet.items()})
gid = {s: torch.tensor(v, device=DEV) for s, v in feet.items()}
size = torch.tensor(mm.geom_size, device=DEV, dtype=torch.float32)
gtype = mm.geom_type
is_cap = {sd: torch.tensor([bool(int(gtype[int(g)]) == 3) for g in v], device=DEV) for sd, v in feet.items()}
sensor = u.scene["feet_ground_contact"]
arm = u.action_manager.get_term("upper_body_pose"); tw = u.command_manager.get_term("twist"); bh = u.command_manager.get_term("base_height")
rest = demo.pose_vec(arm.joint_names, demo.REST, demo.REST, DEV)
grp = torch.tensor([i * len(CMDS) // N for i in range(N)], device=DEV)
vx = torch.tensor([CMDS[g][0] for g in grp.tolist()], device=DEV); vy = torch.tensor([CMDS[g][1] for g in grp.tolist()], device=DEV)

def lowest(side):
  """Lowest point (z) of a foot's collision geometry, per env."""
  ids = gid[side]
  pos = u.sim.data.geom_xpos[:, ids, :]                       # [N, G, 3]
  mat = u.sim.data.geom_xmat[:, ids].reshape(N, len(ids), 3, 3)
  r = size[ids, 0]
  low = pos[..., 2] - r
  caps = is_cap[side]   # mjGEOM_CAPSULE
  half = size[ids, 1]
  axis_z = mat[..., 2, 2]                                     # z of the capsule's local z-axis
  cap_low = pos[..., 2] - (half * axis_z).abs() - r
  return torch.where(caps, cap_low, low).min(dim=1).values

dt = u.step_dt; T0, T1 = 2.0, 10.0
apex = {g: [] for g in range(len(CMDS))}; midmin = {g: [] for g in range(len(CMDS))}
skim = torch.zeros(N, device=DEV); slip = torch.zeros(N, device=DEV); nslip = torch.zeros(N, device=DEV)
swing_max = torch.zeros(N, 2, device=DEV); swing_len = torch.zeros(N, 2, device=DEV)
swing_lows = [[[] for _ in range(2)] for _ in range(N)]
start_xy = None
with torch.inference_mode():
  obs, _ = env.reset()
  for k in range(int(T1 / dt)):
    t = k * dt; walking = t >= T0
    tw.vel_command_b[:, 0] = vx if walking else 0.0; tw.vel_command_b[:, 1] = vy if walking else 0.0; tw.vel_command_b[:, 2] = 0.0
    tw.is_standing_env[:] = not walking
    bh._squat_target[:, 0] = 0.78; bh._walk_target[:, 0] = 0.78; bh._height[:, 0] = 0.78
    arm.set_external_target(rest.unsqueeze(0).expand(N, -1).contiguous())
    obs, _, _, _ = env.step(policy(obs))
    if t < T0 + 1.0:
      start_xy = robot.data.root_link_pos_w[:, :2].clone(); continue
    contact = sensor.data.found > 0                            # [N, 2]
    lows = torch.stack([lowest("left"), lowest("right")], dim=1)
    vel = torch.norm(robot.data.site_lin_vel_w[:, u.reward_manager.get_term_cfg("track_base_height").params["asset_cfg"].site_ids, :2], dim=-1)
    air = ~contact
    skim += ((air & (lows < 0.02)).float() * vel * dt).sum(1)
    slip += (contact.float() * vel).sum(1); nslip += contact.float().sum(1)
    # swing tracking: apex = max lowest-point height over each airborne interval
    swing_max = torch.where(air, torch.maximum(swing_max, lows), swing_max)
    swing_len = torch.where(air, swing_len + dt, swing_len)
    ended = contact & (swing_len > 0.12)                       # a real step (>120 ms airborne)
    for e, f in ended.nonzero().tolist():
      apex[int(grp[e])].append(float(swing_max[e, f]))
    swing_max = torch.where(contact, torch.zeros_like(swing_max), swing_max)
    swing_len = torch.where(contact, torch.zeros_like(swing_len), swing_len)
    # mid-swing clearance sample: airborne for 0.12-0.30 s
    mid = air & (swing_len > 0.12) & (swing_len < 0.30)
    for e, f in mid.nonzero().tolist()[:4000]:
      midmin[int(grp[e])].append(float(lows[e, f]))
  dist = torch.norm(robot.data.root_link_pos_w[:, :2] - start_xy, dim=1)
print(f"\npolicy {os.path.basename(os.path.dirname(a.policy))}/{os.path.basename(a.policy)}")
print(f"{'command':>14} {'swing apex':>10} {'apex p10':>8} {'mid-swing low p10':>17} {'skim / m walked':>15} {'stance slip':>11} {'steps':>6}")
for g, (cx, cy) in enumerate(CMDS):
  sel = (grp == g)
  ap = np.array(apex[g]) if apex[g] else np.array([np.nan]); mm_ = np.array(midmin[g]) if midmin[g] else np.array([np.nan])
  sk = float((skim[sel] / dist[sel].clamp(min=0.1)).mean())
  sl = float((slip[sel] / nslip[sel].clamp(min=1)).mean())
  lab = f"vx {cx:+.1f} vy {cy:+.1f}"
  print(f"{lab:>14} {100*np.nanmean(ap):8.1f}cm {100*np.nanpercentile(ap,10):6.1f}cm {100*np.nanpercentile(mm_,10):15.1f}cm "
        f"{100*sk:12.1f}cm {100*sl:9.1f}cm/s {len(apex[g]):6d}")
print("(heights = lowest point of the foot geometry above the floor; skim = horizontal travel while airborne "
      "but <2 cm off the floor, per metre walked)")
