"""What limits squat depth? Policy behaviour at deep commands + kinematic reach.

Part A (--policy): PoultrySurface play env on flat ground, 0.6 kg per hand, arms
at rest, standing still. Height command ramps from 0.78 to H over 2-5 s, held to
9 s; measured over 7-9 s. Reports achieved height (the training metric), leg
pitch joints vs soft (training penalty) and hard limits, torso pitch, CoM
offset, foot pitch (heel lift), self-collision force, and the weighted reward of
the terms that act on a deep squat.

Part B (always): kinematic search on the robot model. Symmetric legs, feet flat
(ankle pitch solved so the sole is level), arms at rest, no payload. For each
pelvis pitch / hip pitch / knee, records pelvis height above the sole (training
metric) and whole-body CoM fore-aft offset from the foot site. Reports the
lowest height with |CoM offset| <= 3 cm, within soft limits and within hard
limits.
"""
import argparse, math, os, sys
from dataclasses import asdict
os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np, torch, mujoco
import mjlab.tasks, src.tasks  # noqa
import duet_demo_video as demo
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_rl_cfg, load_runner_cls
from mjlab.utils.lab_api.math import quat_apply_inverse

p = argparse.ArgumentParser()
p.add_argument("--policy", default=None)
p.add_argument("--heights", type=float, nargs="+", default=[0.30, 0.26, 0.24, 0.22, 0.20, 0.18, 0.16])
a = p.parse_args()
TASK, DEV = "Unitree-G1-23Dof-Duet-PoultrySurface", "cuda:0"
per = 24
N = per * len(a.heights)
cfg = demo.build_cfg(TASK, "plane", 0.5, 0.6, 320, 240)
cfg.scene.num_envs = N
cfg.events["hand_payload"].params["ranges"] = (0.6, 0.6)
env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=cfg, device=DEV), clip_actions=None)
u = env.unwrapped; robot = u.scene["robot"]
jn = robot.joint_names
PITCH = [f"{s}_{j}_joint" for s in ("left", "right") for j in ("hip_pitch", "knee", "ankle_pitch")]
jidx = [jn.index(n) for n in PITCH]
hard = robot.data.joint_pos_limits[0, jidx].cpu().numpy()
soft = robot.data.soft_joint_pos_limits[0, jidx].cpu().numpy()
print("limits (hard | soft):")
for i, n in enumerate(PITCH[:3]):
  print(f"  {n:>22}: hard [{hard[i,0]:+.2f}, {hard[i,1]:+.2f}]  soft [{soft[i,0]:+.2f}, {soft[i,1]:+.2f}]  default {float(robot.data.default_joint_pos[0, jidx[i]]):+.2f}")

if a.policy:
  runner = (load_runner_cls(TASK) or MjlabOnPolicyRunner)(env, asdict(load_rl_cfg(TASK)), device=DEV)
  runner.load(a.policy, load_cfg={"actor": True}, strict=True, map_location=DEV)
  policy = runner.get_inference_policy(device=DEV)
  arm = u.action_manager.get_term("upper_body_pose"); tw = u.command_manager.get_term("twist")
  bh = u.command_manager.get_term("base_height")
  rm = u.reward_manager
  tid = rm.get_term_cfg("body_orientation_l2").params["asset_cfg"].body_ids
  fcfg = rm.get_term_cfg("track_base_height").params["asset_cfg"]
  rest = demo.pose_vec(arm.joint_names, demo.REST, demo.REST, DEV)
  H = torch.tensor([a.heights[i // per] for i in range(N)], device=DEV)
  TERMS = ["track_base_height", "joint_pos_limits", "self_collisions", "squat_com_centering",
           "body_orientation_l2", "pose", "com_support", "action_excess_l2"]
  TERMS = [t for t in TERMS if t in rm.active_terms]
  acc = {k: torch.zeros(N, device=DEV) for k in ["h", "torso", "cfwd", "footpitch", "coll"] + TERMS}
  qacc = torch.zeros(N, len(jidx), device=DEV); cnt = 0
  sc = u.scene.sensors.get("self_collision") if hasattr(u.scene, "sensors") else None
  fell = torch.zeros(N, dtype=torch.bool, device=DEV)
  obs = env.get_observations(); dt = u.step_dt
  with torch.inference_mode():
    for k in range(int(9.0 / dt)):
      t = k * dt
      s = min(1.0, max(0.0, (t - 2.0) / 3.0)); s = 0.5 - 0.5 * math.cos(math.pi * s)
      h = 0.78 + (H - 0.78) * s
      tw.vel_command_b[:] = 0.0; tw.is_standing_env[:] = True
      bh._squat_target[:, 0] = h; bh._walk_target[:, 0] = h; bh._height[:, 0] = h
      arm.set_external_target(rest.unsqueeze(0).expand(N, -1).contiguous())
      obs, _, _, _ = env.step(policy(obs))
      fell |= u.termination_manager.get_term("fell_over")
      if t < 7.0:
        continue
      cnt += 1
      feet = robot.data.site_pos_w[:, fcfg.site_ids]
      acc["h"] += robot.data.root_link_pos_w[:, 2] - feet[..., 2].min(1).values + 0.02
      g = quat_apply_inverse(robot.data.body_link_quat_w[:, tid, :].squeeze(1), robot.data.gravity_vec_w)
      acc["torso"] += torch.rad2deg(torch.atan2(g[:, 0], -g[:, 2]))
      cxy = robot.data.data.subtree_com[:, robot.data.indexing.root_body_id, :2]
      d = cxy - feet[..., :2].mean(1); yaw = robot.data.heading_w
      acc["cfwd"] += torch.cos(yaw) * d[:, 0] + torch.sin(yaw) * d[:, 1]
      fq = robot.data.site_quat_w[:, fcfg.site_ids] if hasattr(robot.data, "site_quat_w") else None
      if fq is not None:
        fg = quat_apply_inverse(fq.reshape(-1, 4), robot.data.gravity_vec_w.repeat_interleave(2, 0)).reshape(N, 2, 3)
        acc["footpitch"] += torch.rad2deg(torch.atan2(fg[..., 0], -fg[..., 2])).mean(1)
      if sc is not None:
        try:
          acc["coll"] += sc.data.force.norm(dim=-1).reshape(N, -1).max(1).values
        except Exception:
          pass
      qacc += robot.data.joint_pos[:, jidx]
      for name in TERMS:
        tc = rm.get_term_cfg(name)
        acc[name] += tc.func(u, **tc.params) * tc.weight
  for k in acc:
    acc[k] /= cnt
  qacc /= cnt
  print(f"\npolicy {os.path.basename(os.path.dirname(a.policy))}/{os.path.basename(a.policy)}  (arms at rest, 0.6 kg per hand)")
  hdr = f"{'cmd':>5} {'reached':>7} {'falls':>5} {'torso':>6} {'CoMfwd':>7} {'footp':>6} | {'hipP':>6} {'knee':>6} {'ankP':>6} {'(to soft lim: hip/knee/ank)':>27} | " + " ".join(f"{t[:10]:>10}" for t in TERMS)
  print(hdr)
  for i, H0 in enumerate(a.heights):
    m = slice(i * per, (i + 1) * per)
    q = qacc[m].mean(0).cpu().numpy()
    ql = q[:3]
    # margin to the soft limit in the direction of squatting (hip -, knee +, ankle -)
    mh, mk, ma = ql[0] - soft[0, 0], soft[1, 1] - ql[1], ql[2] - soft[2, 0]
    print(f"{H0:5.2f} {float(acc['h'][m].mean()):7.3f} {int(fell[m].sum()):5d} {float(acc['torso'][m].mean()):+5.1f}° "
          f"{100*float(acc['cfwd'][m].mean()):+6.1f}cm {float(acc['footpitch'][m].mean()):+5.1f}° | {ql[0]:+6.2f} {ql[1]:+6.2f} {ql[2]:+6.2f} "
          f"{mh:+8.2f} {mk:+8.2f} {ma:+8.2f}   | " + " ".join(f"{float(acc[t][m].mean()):+10.3f}" for t in TERMS))
  print("(reached = training height metric; torso + = forward; footp = sole pitch, + = toe down; joint values left leg, rad;"
        " margins > 0 = inside the soft limit; reward columns = weighted per-step-rate values, negative = cost)")

# ---- Part B: kinematic reach -------------------------------------------------
mm = u.sim.mj_model
md = mujoco.MjData(mm)
def jadr(name):
  for cand in (f"robot/{name}", name):
    try:
      return mm.joint(cand).qposadr[0]
    except KeyError:
      pass
  raise KeyError(name)
adr = {n: jadr(n) for n in PITCH}
site = None
for cand in ("robot/left_foot", "left_foot"):
  try:
    site = mm.site(cand).id; break
  except KeyError:
    pass
root_body = mm.body("robot/pelvis").id if "robot/pelvis" in [mm.body(i).name for i in range(mm.nbody)] else 1
arm_names = [n for n in jn if any(k in n for k in ("shoulder", "elbow", "wrist"))]
arm_q = demo.pose_vec(arm_names, demo.REST, demo.REST, "cpu").numpy()
base = mm.qpos0.copy()
for n, v in zip(arm_names, arm_q):
  base[jadr(n)] = v
def evaluate(theta, hip, knee, ank):
  q = base.copy()
  q[3:7] = [math.cos(theta / 2), 0.0, math.sin(theta / 2), 0.0]
  for s in ("left", "right"):
    q[adr[f"{s}_hip_pitch_joint"]] = hip; q[adr[f"{s}_knee_joint"]] = knee; q[adr[f"{s}_ankle_pitch_joint"]] = ank
  md.qpos[:] = q
  mujoco.mj_kinematics(md, mm) if False else mujoco.mj_kinematics(mm, md)
  mujoco.mj_comPos(mm, md)
  xm = md.site_xmat[site].reshape(3, 3)
  foot_pitch = math.atan2(-xm[2, 0], xm[0, 0])
  com = md.subtree_com[root_body]
  sp = md.site_xpos[site]
  height = md.xpos[root_body][2] - sp[2] + 0.02
  return foot_pitch, height, com[0] - sp[0]
def solve_ankle(theta, hip, knee):
  ank = -(theta + hip + knee)
  for _ in range(3):
    fp, _, _ = evaluate(theta, hip, knee, ank)
    fp2, _, _ = evaluate(theta, hip, knee, ank + 1e-3)
    dfd = (fp2 - fp) / 1e-3
    if abs(dfd) < 1e-6:
      break
    ank -= fp / dfd
  return ank
best = {"soft": None, "hard": None}
for theta in np.radians(np.arange(0, 62.5, 2.5)):
  for hip in np.linspace(hard[0, 0], 0.3, 60):
    for knee in np.linspace(0.0, hard[1, 1], 60):
      ank = solve_ankle(theta, hip, knee)
      fp, hgt, cx = evaluate(theta, hip, knee, ank)
      if abs(fp) > math.radians(2) or abs(cx) > 0.03:
        continue
      for key, lim in (("soft", soft), ("hard", hard)):
        if lim[0, 0] <= hip <= lim[0, 1] and lim[1, 0] <= knee <= lim[1, 1] and lim[2, 0] <= ank <= lim[2, 1]:
          if best[key] is None or hgt < best[key][0]:
            best[key] = (hgt, math.degrees(theta), hip, knee, ank, cx)
print("\nkinematic reach (feet flat, symmetric, |CoM - foot site| <= 3 cm, arms at rest, no payload):")
for key in ("soft", "hard"):
  b = best[key]
  if b:
    print(f"  within {key} limits: min height {b[0]:.3f} m at pelvis pitch {b[1]:.0f} deg, hip {b[2]:+.2f}, knee {b[3]:+.2f}, ankle {b[4]:+.2f}, CoM {100*b[5]:+.1f} cm")
  else:
    print(f"  within {key} limits: no feasible pose found")
