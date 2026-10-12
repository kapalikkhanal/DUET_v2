"""Kinematic squat reach on the G1 23-DOF model (no policy): flat feet vs heel lift, by joint-limit fraction.

Symmetric legs, arms at rest, no payload. Height = pelvis above the foot site + 0.02 (training metric).
Flat feet: sole level, CoM within +-3 cm of the foot site. Heel lift: sole pitched toe-down by 10-40 deg,
standing on the forefoot, CoM within +-2 cm of the toe line.
"""
import math, os, sys
os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np, mujoco
import mjlab.tasks, src.tasks  # noqa
import duet_demo_video as demo
from mjlab.envs import ManagerBasedRlEnv
cfg = demo.build_cfg("Unitree-G1-23Dof-Duet-PoultrySurface", "plane", 0.5, 0.6, 320, 240)
env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0"); robot = env.scene["robot"]
mm = env.sim.mj_model; md = mujoco.MjData(mm)
J = lambda n: mm.joint(f"robot/{n}").qposadr[0]
names = [f"{s}_{j}_joint" for s in ("left", "right") for j in ("hip_pitch", "knee", "ankle_pitch")]
jn = robot.joint_names; idx = [jn.index(n) for n in names[:3]]
hard = robot.data.joint_pos_limits[0, idx].cpu().numpy(); mid = hard.mean(1); half = (hard[:, 1] - hard[:, 0]) / 2
site = mm.site("robot/left_foot").id; ankle = mm.body("robot/left_ankle_roll_link").id; pelvis = mm.body("robot/pelvis").id
# toe and heel: extreme capsule ends of the left foot in the ankle frame
ends = []
for i in range(1, 8):
  g = mm.geom(f"robot/left_foot{i}_collision"); R = np.empty(9); mujoco.mju_quat2Mat(R, g.quat)
  ax = R.reshape(3, 3)[:, 2] * g.size[1]; ends += [g.pos - ax, g.pos + ax]
ends = np.array(ends); toe = ends[ends[:, 0].argmax()]; heel = ends[ends[:, 0].argmin()]
arm = [n for n in jn if any(k in n for k in ("shoulder", "elbow", "wrist"))]
base = mm.qpos0.copy()
for n, v in zip(arm, demo.pose_vec(arm, demo.REST, demo.REST, "cpu").numpy()):
  base[J(n)] = v
def ev(th, hip, knee, ank):
  q = base.copy(); q[3:7] = [math.cos(th / 2), 0, math.sin(th / 2), 0]
  for s in ("left", "right"):
    q[J(f"{s}_hip_pitch_joint")] = hip; q[J(f"{s}_knee_joint")] = knee; q[J(f"{s}_ankle_pitch_joint")] = ank
  md.qpos[:] = q; mujoco.mj_kinematics(mm, md); mujoco.mj_comPos(mm, md)
  xm = md.site_xmat[site].reshape(3, 3); fp = math.atan2(-xm[2, 0], xm[0, 0])
  R = md.xmat[ankle].reshape(3, 3); tw = md.xpos[ankle] + R @ toe
  return fp, md.xpos[pelvis][2] - md.site_xpos[site][2] + 0.02, md.subtree_com[pelvis][0], md.site_xpos[site][0], tw
def solve(th, hip, knee, target):
  ank = -(th + hip + knee) + target
  for _ in range(3):
    f1 = ev(th, hip, knee, ank)[0] - target; f2 = ev(th, hip, knee, ank + 1e-3)[0] - target
    if abs(f2 - f1) < 1e-9: break
    ank -= f1 * 1e-3 / (f2 - f1)
  return ank
print(f"foot: toe {100*toe[0]:+.1f} cm, heel {100*heel[0]:+.1f} cm (ankle frame x); hard limits hip [{hard[0,0]:+.2f},{hard[0,1]:+.2f}] knee [{hard[1,0]:+.2f},{hard[1,1]:+.2f}] ankle [{hard[2,0]:+.2f},{hard[2,1]:+.2f}]")
for frac in (0.90, 0.95, 0.97, 1.00):
  lim = np.stack([mid - frac * half, mid + frac * half], 1)
  for heel_deg in (0, 10, 20, 30, 40):
    best = None
    for th in np.radians(np.arange(0, 62.5, 2.5)):
      for hip in np.linspace(lim[0, 0], 0.3, 50):
        for knee in np.linspace(0.0, lim[1, 1], 50):
          ank = solve(th, hip, knee, math.radians(heel_deg))
          if not lim[2, 0] <= ank <= lim[2, 1]: continue
          fp, h, cx, sx, tw = ev(th, hip, knee, ank)
          if abs(fp - math.radians(heel_deg)) > math.radians(2): continue
          ok = abs(cx - sx) <= 0.03 if heel_deg == 0 else abs(cx - tw[0]) <= 0.02
          if ok and (best is None or h < best[0]):
            best = (h, math.degrees(th), hip, knee, ank)
    tag = "flat feet" if heel_deg == 0 else f"heels up {heel_deg:2d} deg"
    print(f"limits {int(frac*100):3d}% of range, {tag:>16}: " + (f"min height {best[0]:.3f} m (pelvis pitch {best[1]:.0f} deg, hip {best[2]:+.2f}, knee {best[3]:+.2f}, ankle {best[4]:+.2f})" if best else "infeasible"))
