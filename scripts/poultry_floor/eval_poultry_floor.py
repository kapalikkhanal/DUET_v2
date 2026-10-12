"""Batch evaluation of a walking policy on the simulated broiler-house litter floor.

The floor built by make_poultry_floor.py (same height maps, cake strip and feeder
pans) becomes an mjlab terrain; the PoultrySurface litter drag model adds a loose
friable layer of depth d on top of it (groups of robots at d = 3, 5, 8 cm; 8 cm is
deeper than the 0-6 cm used in training). Robots start at random positions and
headings in the aisle / cake-strip half of the floor and run the same scripted
routine, carrying 0.6 kg per hand:

   0- 2 s stand            2-10 s forward 0.4 m/s     10-14 s turn 0.6 rad/s
  14-20 s backward 0.3     20-25 s sideways 0.25      25-27 s stand
  27-33 s squat to 0.25 m (arms out wide from 29 s)   33-35 s stand up
  35-40 s forward 0.5 m/s

Reported per litter depth: falls (and on which surface), speed reached per
segment, foot-swing apex above the loose-litter surface, share of swing time
with the toe inside the loose layer, stance sinkage into the compacted litter,
and the squat height reached.

  python scripts/poultry_floor/eval_poultry_floor.py --policy <model_N.pt | policy.onnx> [--n 192]
  python scripts/poultry_floor/eval_poultry_floor.py --policy ... --video out.mp4   # one robot, rendered
"""
from __future__ import annotations

import argparse, json, math, os, sys
from dataclasses import asdict, dataclass, replace
os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch, mujoco
import mjlab.tasks, src.tasks  # noqa
import duet_demo_video as demo
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.terrains.terrain_generator import TerrainGeometry, TerrainOutput
from src.tasks.common.terrains import SoftFlatTerrainCfg

ASSETS = os.path.expanduser("~/Mujoco_kapalikkhanal/unitree_robots/g1/meshes/poultry_floor")


@dataclass(kw_only=True)
class PoultryFarmTerrainCfg(SoftFlatTerrainCfg):
  """The generated litter floor as one mjlab sub-terrain.

  Subclasses SoftFlatTerrainCfg so the PoultrySurface action treats the whole
  tile as a litter (soft) tile: drag layer and litter metrics apply.
  """
  asset_dir: str = ASSETS
  contact: str = "floor"     # "floor" (as generated: preset), "training" (softest trained floor), "rigid"
  pans: bool = True
  flat: bool = False         # replace the litter relief with a flat surface (ablation)
  # MuJoCo Warp drops contacts above 50 per geom pair; the 19 cm foot capsules overflow
  # that on 4 cm cells. Training heightfields used 10 cm cells, so collide on a coarser
  # block-averaged grid (factor k of the generated resolution).
  coarsen: int = 2

  def function(self, difficulty, spec, rng):
    meta = json.load(open(os.path.join(self.asset_dir, "floor_meta.json")))
    p = meta["params"]
    body = spec.body("terrain")
    L, W, res = p["length"], p["width"], p["resolution"]
    geoms = []

    def hfield(h, cx, cy, half_y, solref, solimp, friction, rgba):
      k = max(1, self.coarsen)
      if k > 1:  # block average, keeping the outer edge so the extent is unchanged
        ny, nx = (h.shape[0] - 1) // k * k + 1, (h.shape[1] - 1) // k * k + 1
        h = h[:ny, :nx][::k, ::k] * 0.5 + 0.5 * np.pad(h[:ny, :nx], ((0, k), (0, k)), mode="edge")[
          k // 2::k, k // 2::k][: (ny - 1) // k + 1, : (nx - 1) // k + 1]
      lo, hi = float(h.min()), float(h.max())
      span = max(hi - lo, 1e-4)
      hf = spec.add_hfield(name=f"poultry_{len(geoms)}", size=[L / 2, half_y, span, 0.05],
                           nrow=h.shape[0], ncol=h.shape[1],
                           userdata=((h - lo) / span).flatten().astype(np.float32).tolist())
      g = body.add_geom(type=mujoco.mjtGeom.mjGEOM_HFIELD, hfieldname=hf.name, pos=[cx, cy, lo])
      g.priority = 2  # above the G1 foot geoms (priority 1), as the training soft floor
      g.solref = tuple(solref); g.solimp = tuple(solimp); g.friction = (friction, 0.005, 0.0001)
      geoms.append(TerrainGeometry(geom=g, hfield=hf, color=rgba))

    litter = np.load(os.path.join(self.asset_dir, "litter.npy"))
    if self.flat:
      litter = np.zeros_like(litter); litter[0, 0] = 1e-3  # hfield needs a non-zero range
    solref, solimp = p["friable_solref"], p["friable_solimp"]
    if self.contact == "training":  # SoftFlatTerrainCfg at difficulty 1 (softest trained floor)
      solref, solimp = (0.05, 2.0), (0.6, 0.9, 0.02, 0.5, 2.0)
    elif self.contact in ("built_up", "fresh_deep"):  # floor presets (poultry_floor.PRESETS)
      import poultry_floor as pf
      solref, solimp = pf.PRESETS[self.contact]["friable_solref"], pf.PRESETS[self.contact]["friable_solimp"]
    elif self.contact == "rigid":  # hard floor (protocol baseline)
      solref, solimp = (0.02, 1.0), (0.9, 0.95, 0.001, 0.5, 2.0)
    hfield(litter, L / 2, W / 2, W / 2, solref, solimp, p["friable_friction"], (0.24, 0.18, 0.12, 1))
    cake = np.load(os.path.join(self.asset_dir, "cake_strip.npy"))
    ys = -W / 2 + np.arange(litter.shape[0]) * res
    rows = np.where(np.abs(ys - p["drinker_y"]) <= p["cake_width"] / 2 + 0.1)[0]
    cy = (ys[rows[0]] + ys[rows[-1]]) / 2 + W / 2
    hfield(cake, L / 2, cy, (rows[-1] - rows[0]) * res / 2, p["cake_solref"], p["cake_solimp"], p["cake_friction"], (0.14, 0.11, 0.08, 1))
    for (fx, fy) in (meta["feeders"] if self.pans else []):
      g = body.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[p["pan_radius"], p["pan_height"] / 2, 0],
                        pos=[fx - p["x0"], fy + W / 2, p["pan_height"] / 2 - 0.01])
      geoms.append(TerrainGeometry(geom=g, color=(0.8, 0.12, 0.1, 1)))
    # Spawn origin = floor point (0, 0) in tile coordinates.
    return TerrainOutput(origin=np.array([-p["x0"], W / 2, 0.0]), geometries=geoms)


SEGMENTS = [  # (name, t0, t1, vx, vy, wz, height, arms)
  ("stand", 0, 2, 0, 0, 0, 0.78, "rest"), ("forward 0.4", 2, 10, 0.4, 0, 0, 0.78, "rest"),
  ("turn 0.6", 10, 14, 0, 0, 0.6, 0.78, "rest"), ("backward 0.3", 14, 20, -0.3, 0, 0, 0.78, "rest"),
  ("sideways 0.25", 20, 25, 0, 0.25, 0, 0.78, "rest"), ("stand", 25, 27, 0, 0, 0, 0.78, "rest"),
  ("squat 0.25", 27, 29, 0, 0, 0, 0.25, "rest"), ("squat 0.25 arms wide", 29, 33, 0, 0, 0, 0.25, "wide"),
  ("stand up", 33, 35, 0, 0, 0, 0.78, "rest"), ("forward 0.5", 35, 40, 0.5, 0, 0, 0.78, "rest"),
]
DEPTHS = [0.03, 0.05, 0.08]


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--policy", required=True)
  ap.add_argument("--n", type=int, default=192)
  ap.add_argument("--assets", default=ASSETS)
  ap.add_argument("--video", default=None)
  ap.add_argument("--video-depth", type=float, default=0.05)
  ap.add_argument("--json", default=None)
  ap.add_argument("--contact", choices=("floor", "built_up", "fresh_deep", "training", "rigid"), default="floor",
                  help="litter contact: as generated (preset in floor_meta.json), the softest trained floor, or rigid")
  ap.add_argument("--no-pans", action="store_true", help="remove the feeder pans (obstacle ablation)")
  ap.add_argument("--flat", action="store_true", help="flatten the litter relief (roughness ablation)")
  ap.add_argument("--no-drag", action="store_true", help="disable the loose-litter drag layer")
  ap.add_argument("--coarsen", type=int, default=2, help="collision grid = generated resolution x this (2 -> 8 cm)")
  a = ap.parse_args()
  TASK, DEV = "Unitree-G1-23Dof-Duet-PoultrySurface", "cuda:0"
  meta = json.load(open(os.path.join(a.assets, "floor_meta.json"))); P = meta["params"]
  N = 1 if a.video else a.n - a.n % len(DEPTHS)

  cfg = demo.build_cfg(TASK, "plane", 0.5, 0.6, 1280, 720)
  cfg.scene.num_envs = N
  cfg.episode_length_s = 1e6
  for k in ("push_robot", "stop_backward_walkers", "randomize_terrain"):
    cfg.events.pop(k, None)
  cfg.events["hand_payload"].params["ranges"] = (0.6, 0.6)
  cfg.events["reset_base"].params["pose_range"] = (
    {"x": (0, 0), "y": (0, 0), "z": (0, 0), "yaw": (0, 0)} if a.video else
    {"x": (-3.0, 7.0), "y": (-0.8, 2.4), "z": (0, 0), "yaw": (-math.pi, math.pi)})
  cfg.curriculum = {}
  from mjlab.terrains.terrain_generator import TerrainGeneratorCfg
  cfg.scene.terrain.terrain_type = "generator"
  cfg.scene.terrain.terrain_generator = TerrainGeneratorCfg(
    size=(P["length"], P["width"]), num_rows=1, num_cols=1, curriculum=True, border_width=2.0,
    sub_terrains={"poultry_farm": PoultryFarmTerrainCfg(proportion=1.0, asset_dir=a.assets, contact=a.contact,
                                                        pans=not a.no_pans, flat=a.flat, coarsen=a.coarsen)})
  cfg.scene.terrain.max_init_terrain_level = 0
  # Height-field collisions produce many more foot contacts than flat ground.
  cfg.sim.nconmax, cfg.sim.njmax = 512, 3000
  cfg.actions["poultry_surface"].litter_depth_mode = "random"
  cfg.actions["poultry_surface"].litter_depth_range = (0.0, 0.0)
  cfg.actions["poultry_surface"].litter_enabled = not a.no_drag
  if a.video:
    cfg.viewer.distance, cfg.viewer.elevation, cfg.viewer.azimuth = 3.0, -15.0, 150.0

  env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=cfg, device=DEV, render_mode="rgb_array" if a.video else None), clip_actions=None)
  u = env.unwrapped; robot = u.scene["robot"]
  if a.policy.endswith(".onnx"):
    import onnxruntime as ort
    sess = ort.InferenceSession(a.policy, providers=["CPUExecutionProvider"])
    def policy(obs):
      x = obs["actor"].cpu().numpy().astype(np.float32)
      return torch.tensor(np.concatenate([sess.run(None, {"obs": x[i:i + 1]})[0] for i in range(len(x))]), device=DEV)
  else:
    runner = (load_runner_cls(TASK) or MjlabOnPolicyRunner)(env, asdict(load_rl_cfg(TASK)), device=DEV)
    runner.load(a.policy, load_cfg={"actor": True}, strict=True, map_location=DEV)
    policy = runner.get_inference_policy(device=DEV)

  surf = u.action_manager.get_term("poultry_surface")
  depth = torch.tensor([a.video_depth] if a.video else [DEPTHS[i * len(DEPTHS) // N] for i in range(N)], device=DEV)
  orig_reset = surf.reset
  def reset(env_ids=None):  # keep each robot's litter depth through fall resets
    orig_reset(env_ids); surf.litter_depth.copy_(depth)
  surf.reset = reset; surf.litter_depth.copy_(depth)
  arm = u.action_manager.get_term("upper_body_pose"); tw = u.command_manager.get_term("twist"); bh = u.command_manager.get_term("base_height")
  POSE = {"rest": demo.pose_vec(arm.joint_names, demo.REST, demo.REST, DEV), "wide": demo.pose_vec(arm.joint_names, demo.WIDE, demo.WIDE, DEV)}
  fcfg = u.reward_manager.get_term_cfg("track_base_height").params["asset_cfg"]
  # Height map lookup (floor frame) for sinkage.
  lit = torch.tensor(np.load(os.path.join(a.assets, "litter.npy")), device=DEV)
  res, x0, y0 = P["resolution"], P["x0"], -P["width"] / 2
  def surface_z(xy):
    gx = ((xy[..., 0] - x0) / res).clamp(0, lit.shape[1] - 1.001); gy = ((xy[..., 1] - y0) / res).clamp(0, lit.shape[0] - 1.001)
    ix, iy = gx.floor().long(), gy.floor().long(); fx, fy = gx - ix, gy - iy
    z00, z01 = lit[iy, ix], lit[iy, ix + 1]; z10, z11 = lit[iy + 1, ix], lit[iy + 1, ix + 1]
    return (z00 * (1 - fx) + z01 * fx) * (1 - fy) + (z10 * (1 - fx) + z11 * fx) * fy
  origin = u.scene.env_origins
  dt = u.step_dt; T = SEGMENTS[-1][2]
  nseg = len(SEGMENTS)
  st = {k: torch.zeros(N, nseg, device=DEV) for k in ("spd", "nspd", "air", "scoop", "sink", "nsink", "apex", "napex", "falls", "fall_cake", "fall_pan")}
  pans_xy = torch.tensor(meta["feeders"] if not a.no_pans else [[1e3, 1e3]], device=DEV)
  hreach = torch.zeros(N, device=DEV); nh = 0
  peak = torch.full((N, 2), -1.0, device=DEV); slen = torch.zeros(N, 2, device=DEV)
  toe = surf.toe_index
  writer = None
  if a.video:
    import imageio.v2 as imageio
    writer = imageio.get_writer(a.video, fps=25, codec="libx264", quality=8, macro_block_size=1)
  with torch.inference_mode():
    obs, _ = env.reset()
    for k in range(int(T / dt)):
      t = k * dt
      si = next(i for i, s in enumerate(SEGMENTS) if s[1] <= t < s[2])
      _, _, _, vx, vy, wz, h, arms = SEGMENTS[si]
      moving = abs(vx) + abs(vy) + abs(wz) > 0
      tw.vel_command_b[:, 0] = vx; tw.vel_command_b[:, 1] = vy; tw.vel_command_b[:, 2] = wz
      tw.is_standing_env[:] = not moving
      bh._squat_target[:, 0] = h; bh._walk_target[:, 0] = h; bh._height[:, 0] = h
      arm.set_external_target(POSE[arms].unsqueeze(0).expand(N, -1).contiguous())
      xy_floor = robot.data.root_link_pos_w[:, :2] - origin[:, :2]
      obs, _, _, _ = env.step(policy(obs))
      fell = u.termination_manager.get_term("fell_over").float()
      st["falls"][:, si] += fell
      st["fall_cake"][:, si] += fell * ((xy_floor[:, 1] - P["drinker_y"]).abs() < P["cake_width"] / 2).float()
      st["fall_pan"][:, si] += fell * (torch.cdist(xy_floor, pans_xy).min(1).values < 0.6).float()
      if writer is not None and k % 2 == 0:
        writer.append_data(u.render())
      if t - SEGMENTS[si][1] < 1.0:  # settle 1 s into each segment
        continue
      v = robot.data.root_link_lin_vel_b; w = robot.data.root_link_ang_vel_b
      if vx:
        st["spd"][:, si] += v[:, 0] / vx * (1 - fell)
      elif vy:
        st["spd"][:, si] += v[:, 1] / vy * (1 - fell)
      elif wz:
        st["spd"][:, si] += w[:, 2] / wz * (1 - fell)
      st["nspd"][:, si] += (1 - fell)
      if h < 0.5 and arms == "wide":
        feet = robot.data.site_pos_w[:, fcfg.site_ids]
        hreach += robot.data.root_link_pos_w[:, 2] - feet[..., 2].min(1).values + 0.02; nh += 1
      s = surf.sample()
      contact, air = s["contact"], s["air"]
      toe_clear = s["clearance"].gather(-1, toe[None, :, None].expand(N, -1, -1)).squeeze(-1)
      st["air"][:, si] += air.float().sum(1); st["scoop"][:, si] += (air & (toe_clear < 0)).float().sum(1)
      pts = s["points"]  # [N, feet, P, 3] world
      low_i = pts[..., 2].argmin(-1, keepdim=True)
      low = pts.gather(2, low_i[..., None].expand(-1, -1, -1, 3)).squeeze(2)  # [N, 2, 3]
      lz = surface_z(low[..., :2] - origin[:, None, :2]) + origin[:, None, 2]
      st["sink"][:, si] += (contact.float() * (lz - low[..., 2]).clamp(min=0)).sum(1); st["nsink"][:, si] += contact.float().sum(1)
      lowl = s["lowest_litter"]
      peak = torch.where(air, torch.maximum(peak, lowl), peak); slen = torch.where(air, slen + dt, slen)
      ended = contact & (slen > 0.12)
      st["apex"][:, si] += (peak * ended).sum(1); st["napex"][:, si] += ended.float().sum(1)
      peak = torch.where(contact, torch.full_like(peak, -1.0), peak); slen = torch.where(contact, torch.zeros_like(slen), slen)
  if writer is not None:
    writer.close(); print(f"video -> {a.video}")
    return
  out = {}
  print(f"\npolicy {a.policy}\nfloor {a.assets} (seed {P['seed']}), {N} robots, routine {T:.0f} s, 0.6 kg per hand; "
        f"contact={a.contact} pans={not a.no_pans} flat={a.flat} drag={not a.no_drag}")
  for g, d0 in enumerate(DEPTHS):
    m = depth == d0; n = int(m.sum()); mins = n * T / 60
    falls = st["falls"][m].sum().item(); cake = st["fall_cake"][m].sum().item(); pan = st["fall_pan"][m].sum().item()
    print(f"\n== loose litter {100 * d0:.0f} cm over the compacted floor ({n} robots): {falls:.0f} falls "
          f"({falls / mins:.3f} per robot-minute; {cake:.0f} on the cake strip, {pan:.0f} within 0.6 m of a feeder pan)")
    print(f"{'segment':>22} {'speed':>6} {'apex':>7} {'toe in':>7} {'sinkage':>8} {'falls':>5}")
    rows = []
    for i, s in enumerate(SEGMENTS):
      def r(a_, b_):
        return float(st[a_][m, i].sum() / st[b_][m, i].sum().clamp(min=1))
      spd = r("spd", "nspd") if any(s[3:6]) else float("nan")
      apex = r("apex", "napex"); scoop = r("scoop", "air"); sink = r("sink", "nsink")
      f = st["falls"][m, i].sum().item()
      print(f"{s[0]:>22} {('%5.0f%%' % (100 * spd)) if spd == spd else '     -'} {100 * apex:5.1f}cm {100 * scoop:5.0f}% {100 * sink:6.2f}cm {f:5.0f}")
      rows.append(dict(segment=s[0], speed=spd, apex_cm=100 * apex, toe_in_share=scoop, sinkage_cm=100 * sink, falls=f))
    out[f"{d0}"] = dict(robots=n, falls=falls, falls_per_robot_min=falls / mins, falls_on_cake=cake, segments=rows,
                        squat_reached_m=float(hreach[m].sum() / max(nh, 1) / max(n, 1) * N) if nh else None)
    if nh:
      print(f"   squat at 0.25 m command, arms wide: reached {float(hreach[m].mean() / nh):.3f} m")
  print("\n(speed = reached / commanded along the command axis; apex = mean swing peak of the lowest sole point above the loose-litter "
        "surface; toe in = share of swing time with the toe inside the loose layer; sinkage = stance depth of the lowest sole point "
        "below the compacted litter surface)")
  if a.json:
    json.dump(out, open(a.json, "w"), indent=1)


if __name__ == "__main__":
  main()
