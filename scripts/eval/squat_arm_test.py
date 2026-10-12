"""Deep squat + arms forward: does the torso straighten and the robot move back?

Per env: stand, squat to H, hold (baseline), arms to POSE over 1 s, hold 4 s
(measured), arms back. 96 envs, flat ground, no pushes, no obs noise,
training-random hand/torso payload. Works with a .pt (actor) or a deployed
.onnx. Reports torso pitch change, pelvis fore-aft drift, foot travel, CoM
support ratio and falls, by squat height and hand payload.
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
from src.tasks.common.mdp.rewards import _com_support_ratio

p = argparse.ArgumentParser()
p.add_argument("--policy", required=True); p.add_argument("--stand", type=float, default=0.78)
p.add_argument("--pose", default="forward"); p.add_argument("--n", type=int, default=96)
p.add_argument("--hand-payload", type=float, nargs=2, default=None)
p.add_argument("--heights", type=float, nargs="+", default=[0.24, 0.30, 0.36])
a = p.parse_args()
TASK = "Unitree-G1-23Dof-Duet-Rough"; DEV = "cuda:0"
POSES = {
  "forward": {"pitch": -1.2, "roll": 0.10, "yaw": 0.0, "elbow": 0.30, "wrist": 0.0},
  "straight": {"pitch": -1.5, "roll": 0.10, "yaw": 0.0, "elbow": 0.00, "wrist": 0.0},
  "wide": {"pitch": 0.0, "roll": 1.2, "yaw": 0.0, "elbow": 0.30, "wrist": 0.0},
}
HEIGHTS = a.heights

cfg = demo.build_cfg(TASK, "plane", 0.5, 0.6, 320, 240)
cfg.scene.num_envs = a.n
if a.hand_payload is not None:
  cfg.events["hand_payload"].params["ranges"] = tuple(a.hand_payload)
env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=cfg, device=DEV), clip_actions=None)
u = env.unwrapped; robot = u.scene["robot"]
if a.policy.endswith(".onnx"):
  import onnxruntime as ort
  sess = ort.InferenceSession(a.policy, providers=["CPUExecutionProvider"])
  def policy(obs):
    x = obs["actor"].cpu().numpy().astype(np.float32)
    return torch.tensor(np.concatenate([sess.run(None, {"obs": x[i:i+1]})[0] for i in range(len(x))]), device=DEV)
else:
  ag = load_rl_cfg(TASK)
  runner = (load_runner_cls(TASK) or MjlabOnPolicyRunner)(env, asdict(ag), device=DEV)
  runner.load(a.policy, load_cfg={"actor": True}, strict=True, map_location=DEV)
  policy = runner.get_inference_policy(device=DEV)

arm = u.action_manager.get_term("upper_body_pose"); tw = u.command_manager.get_term("twist")
bh = u.command_manager.get_term("base_height")
tid = u.reward_manager.get_term_cfg("body_orientation_l2").params["asset_cfg"].body_ids
fcfg = u.reward_manager.get_term_cfg("track_base_height").params["asset_cfg"]
names = arm.joint_names
rest = demo.pose_vec(names, demo.REST, demo.REST, DEV); goal = demo.pose_vec(names, POSES[a.pose], POSES[a.pose], DEV)
H = torch.tensor([HEIGHTS[i * len(HEIGHTS) // a.n] for i in range(a.n)], device=DEV)
# per-env added hand mass (both hands), read back from the model
try:
  ids = [robot.indexing.body_ids[i] for i, n in enumerate(robot.body_names) if "rubber_hand" in n]
  hand_mass = torch.as_tensor(u.sim.model.body_mass)[:, ids].sum(1)
except Exception as exc:
  print("hand mass readback unavailable:", exc); hand_mass = None

dt = u.step_dt; T = 14.0
def lerp(x0, x1, t0, t1, t):
  s = min(1.0, max(0.0, (t - t0) / (t1 - t0))); s = 0.5 - 0.5 * math.cos(math.pi * s); return x0 + s * (x1 - x0)
rec = {k: [] for k in ("t", "pitch", "x", "fx", "com", "h", "cfwd")}
fell = torch.zeros(a.n, dtype=torch.bool, device=DEV)
obs = env.get_observations()
with torch.inference_mode():
  for k in range(int(T / dt)):
    t = k * dt
    tw.vel_command_b[:] = 0.0; tw.is_standing_env[:] = True
    h = torch.where(torch.tensor(t < 2.0, device=DEV), torch.full_like(H, a.stand),
                    a.stand + (H - a.stand) * lerp(0.0, 1.0, 2.0, 4.5, t))
    bh._squat_target[:, 0] = h; bh._walk_target[:, 0] = h; bh._height[:, 0] = h
    s = lerp(0.0, 1.0, 6.5, 7.5, t) if t < 11.5 else lerp(1.0, 0.0, 11.5, 12.5, t)
    arm.set_external_target((rest + s * (goal - rest)).unsqueeze(0).expand(a.n, -1).contiguous())
    obs, _, dones, _ = env.step(policy(obs))
    fell |= u.termination_manager.get_term("fell_over")
    g = quat_apply_inverse(robot.data.body_link_quat_w[:, tid, :].squeeze(1), robot.data.gravity_vec_w)
    rec["t"].append(t); rec["pitch"].append(torch.atan2(g[:, 0], -g[:, 2]).cpu())
    rec["x"].append(robot.data.root_link_pos_w[:, 0].cpu())
    rec["fx"].append(robot.data.site_pos_w[:, fcfg.site_ids, :2].cpu())
    r, ds = _com_support_ratio(u, "feet_ground_contact", fcfg, 0.09, 0.04); rec["com"].append(r.cpu())
    cxy = robot.data.data.subtree_com[:, robot.data.indexing.root_body_id, :2]
    dxy = cxy - robot.data.site_pos_w[:, fcfg.site_ids, :2].mean(1); yaw = robot.data.heading_w
    rec["cfwd"].append((torch.cos(yaw) * dxy[:, 0] + torch.sin(yaw) * dxy[:, 1]).cpu())
    rec["h"].append((robot.data.root_link_pos_w[:, 2] - robot.data.site_pos_w[:, fcfg.site_ids, 2].min(1).values + 0.02).cpu())
t = np.array(rec["t"]); P = torch.stack(rec["pitch"]).numpy(); X = torch.stack(rec["x"]).numpy()
CF = torch.stack(rec["cfwd"]).numpy(); F = torch.stack(rec["fx"]).numpy(); C = torch.stack(rec["com"]).numpy(); Hm = torch.stack(rec["h"]).numpy()
w = lambda t0, t1: (t >= t0) & (t < t1)
stand_p, squat_p, armfwd_p = P[w(1, 2)].mean(0), P[w(5.5, 6.5)].mean(0), P[w(8.5, 11.5)].mean(0)
sgn = 1.0 if (squat_p - stand_p).mean() > 0 else -1.0  # make "forward lean" positive
i0, i1 = np.searchsorted(t, 6.5), np.searchsorted(t, 11.5)
dx = X[i1] - X[i0]
foot = np.linalg.norm(F[i1] - F[i0], axis=-1).max(1)
hm = hand_mass.cpu().numpy() if hand_mass is not None else np.zeros(a.n)
print(f"\npolicy {os.path.basename(os.path.dirname(a.policy))}/{os.path.basename(a.policy)}  pose={a.pose}  stand={a.stand}")
print(f"{'height':>6} {'payload':>8} {'n':>3} {'h meas':>7} {'lean squat':>10} {'lean arms':>9} {'d lean':>7} {'pelvis dx':>9} {'foot mv':>8} {'CoM max':>8} {'CoMfwd sq':>9} {'CoMfwd arm':>10} {'falls':>5}")
heavy = hm > np.median(hm)
for H0 in HEIGHTS:
  for lab, m in (("light", ~heavy), ("heavy", heavy)):
    sel = (np.isclose(H.cpu().numpy(), H0)) & m
    if not sel.any(): continue
    print(f"{H0:6.2f} {lab:>8} {sel.sum():3d} {Hm[w(5.5,6.5)][:, sel].mean():7.3f} "
          f"{math.degrees(sgn*squat_p[sel].mean()):9.1f}° {math.degrees(sgn*armfwd_p[sel].mean()):8.1f}° "
          f"{math.degrees(sgn*(armfwd_p-squat_p)[sel].mean()):+6.1f}° {100*dx[sel].mean():+7.1f}cm "
          f"{100*foot[sel].mean():6.1f}cm {C[w(8.5,11.5)][:, sel].max(0).mean():8.2f} {100*CF[w(5.5,6.5)][:, sel].mean():+8.1f}cm {100*CF[w(8.5,11.5)][:, sel].mean():+9.1f}cm {int(fell.cpu().numpy()[sel].sum()):5d}")
print(f"(lean: torso forward-lean angle; d lean < 0 = straightens back; pelvis dx < 0 = moved backward; "
      f"CoM max: 1.0 = edge of support; hand payload median split at {np.median(hm):.2f} kg total)")
