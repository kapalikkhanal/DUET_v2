"""Backward vs forward walking, then a sudden stop.

Per env: stand 2 s, walk at v for 5 s, command zero for 3 s. Groups of envs at
v = -0.5, -0.3, +0.3, +0.5 m/s. Flat ground, no pushes, no obs noise, arms at
rest, +0.6 kg per hand (the real Dex3 + gripper + camera). Reports, per speed:
achieved speed, torso pitch (+ = leaning forward), whole-body CoM fore-aft
offset from mid-feet while walking, and after the stop: how far the pelvis
keeps travelling, how far the feet move (catch steps), peak backward lean,
time to settle, falls.
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
from mjlab.utils.lab_api.math import quat_apply_inverse

p = argparse.ArgumentParser(); p.add_argument("--policy", required=True); p.add_argument("--stand", type=float, default=0.78)
a = p.parse_args()
TASK, DEV, N = "Unitree-G1-23Dof-Duet-Rough", "cuda:0", 128
SPEEDS = [-0.5, -0.3, 0.3, 0.5]
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
arm = u.action_manager.get_term("upper_body_pose"); tw = u.command_manager.get_term("twist"); bh = u.command_manager.get_term("base_height")
tid = u.reward_manager.get_term_cfg("body_orientation_l2").params["asset_cfg"].body_ids
fcfg = u.reward_manager.get_term_cfg("track_base_height").params["asset_cfg"]
rest = demo.pose_vec(arm.joint_names, demo.REST, demo.REST, DEV)
V = torch.tensor([SPEEDS[i * len(SPEEDS) // N] for i in range(N)], device=DEV)
dt = u.step_dt; T_WALK0, T_STOP, T_END = 2.0, 7.0, 10.0
rec = {k: [] for k in ("pitch", "cfwd", "vx", "px", "feet", "fell")}
with torch.inference_mode():
  obs, _ = env.reset()
  for k in range(int(T_END / dt)):
    t = k * dt
    walking = T_WALK0 <= t < T_STOP
    tw.vel_command_b[:, 0] = V if walking else 0.0
    tw.vel_command_b[:, 1:] = 0.0
    tw.is_standing_env[:] = not walking
    bh._squat_target[:, 0] = a.stand; bh._walk_target[:, 0] = a.stand; bh._height[:, 0] = a.stand
    arm.set_external_target(rest.unsqueeze(0).expand(N, -1).contiguous())
    obs, _, _, _ = env.step(policy(obs))
    g = quat_apply_inverse(robot.data.body_link_quat_w[:, tid, :].squeeze(1), robot.data.gravity_vec_w)
    rec["pitch"].append(torch.atan2(g[:, 0], -g[:, 2]).cpu())
    cxy = robot.data.data.subtree_com[:, robot.data.indexing.root_body_id, :2]
    feet = robot.data.site_pos_w[:, fcfg.site_ids, :2]
    yaw = robot.data.heading_w; c, s = torch.cos(yaw), torch.sin(yaw); d = cxy - feet.mean(1)
    rec["cfwd"].append((c * d[:, 0] + s * d[:, 1]).cpu())
    rec["vx"].append(robot.data.root_link_lin_vel_b[:, 0].cpu())
    pxy = robot.data.root_link_pos_w[:, :2]
    rec["px"].append(pxy.cpu()); rec["feet"].append(feet.cpu())
    rec["fell"].append(u.termination_manager.get_term("fell_over").cpu())
t = np.arange(len(rec["pitch"])) * dt
P = torch.stack(rec["pitch"]).numpy(); CF = torch.stack(rec["cfwd"]).numpy(); VX = torch.stack(rec["vx"]).numpy()
PX = torch.stack(rec["px"]).numpy(); FT = torch.stack(rec["feet"]).numpy(); FELL = torch.stack(rec["fell"]).numpy()
w = lambda t0, t1: (t >= t0) & (t < t1)
stand_p = P[w(1, 2)].mean(0)
i_stop = np.searchsorted(t, T_STOP)
after = slice(i_stop, len(t))
travel = np.linalg.norm(PX[-1] - PX[i_stop], axis=-1)                      # pelvis travel after the stop
feet_mv = np.linalg.norm(FT[-1] - FT[i_stop], axis=-1).max(1)               # largest foot displacement after stop
peak_back = (P[after] - stand_p).min(0)                                     # most backward lean after stop
spd = np.abs(VX[after])
settle = np.array([(np.argmax(spd[:, e] < 0.05) if (spd[:, e] < 0.05).any() else len(spd)) * dt for e in range(N)])
print(f"\npolicy {os.path.basename(os.path.dirname(a.policy))}/{os.path.basename(a.policy)}")
print(f"{'cmd vx':>6} {'achieved':>8} {'lean walk':>9} {'CoM walk':>8} | {'travel':>7} {'foot mv':>7} {'peak back':>9} {'settle':>6} {'falls':>5}")
for v0 in SPEEDS:
  sel = np.isclose(V.cpu().numpy(), v0)
  print(f"{v0:+6.1f} {VX[w(4, 7)][:, sel].mean():+8.2f} {math.degrees((P[w(4, 7)] - stand_p)[:, sel].mean()):+8.1f}° "
        f"{100*CF[w(4, 7)][:, sel].mean():+6.1f}cm | {100*travel[sel].mean():5.1f}cm {100*feet_mv[sel].mean():5.1f}cm "
        f"{math.degrees(peak_back[sel].mean()):+8.1f}° {settle[sel].mean():5.2f}s {int(FELL[:, sel].sum()):5d}")
print(f"absolute torso pitch (+ = forward): standing {math.degrees(stand_p.mean()):+.1f} deg; walking " + ", ".join(f"{v0:+.1f}: {math.degrees(P[w(4, 7)][:, np.isclose(V.cpu().numpy(), v0)].mean()):+.1f}" for v0 in SPEEDS))
print("(lean: torso pitch vs standing, + = forward; CoM: whole-body CoM ahead(+)/behind(-) mid-feet; "
      "travel/foot mv/peak back/settle: during the 3 s after the stop command)")
