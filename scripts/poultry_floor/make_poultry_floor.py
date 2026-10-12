"""Build the poultry-litter floor for the Unitree MuJoCo simulator and for mjlab.

Writes, into the simulator's robot folder (default
~/Mujoco_kapalikkhanal/unitree_robots/g1):
  meshes/poultry_floor/          height maps (.bin, .npy), textures, floor_meta.json
  poultry_litter_floor.xml       the floor only (include it in any scene)
  scene_23dof_poultry.xml        G1 23-DOF on the floor (simulate -s scene_23dof_poultry.xml)
  scene_poultry_floor_only.xml   floor preview without the robot
Existing files there are not modified.

Then verifies the result: loads both scenes, ray-casts the floor against the
generated height map (orientation and offsets), and drops the chips to check
they settle.

  python scripts/poultry_floor/make_poultry_floor.py [--preset built_up|fresh_deep] [--chips 150] [--seed 7] [--out DIR]
"""
import argparse, os, sys
from dataclasses import replace
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import poultry_floor as pf

ap = argparse.ArgumentParser()
ap.add_argument("--out", default=os.path.expanduser("~/Mujoco_kapalikkhanal/unitree_robots/g1"))
ap.add_argument("--chips", type=int, default=150, help="loose shaving chips (150 keeps the CPU simulator >1.8x real time; 0 for fastest)")
ap.add_argument("--seed", type=int, default=7)
ap.add_argument("--preset", choices=sorted(pf.PRESETS), default="built_up",
                help="built_up: compacted litter, ~2 cm sinkage (within training); fresh_deep: ~4 cm (stress test)")
ap.add_argument("--no-verify", action="store_true")
a = ap.parse_args()

p = replace(pf.PoultryFloorParams(), chips=a.chips, seed=a.seed, **pf.PRESETS[a.preset])
asset_dir = os.path.join(a.out, "meshes", "poultry_floor")
res = pf.write_assets(p, asset_dir)
hm = res["hm"]
# Absolute asset paths: for files in an <include>d MJCF, MuJoCo 3.6 applies meshdir to
# hfields only when loading from the model directory, and the simulator loads by absolute
# path from its build folder. Re-run this script on each machine to regenerate the paths.
cake_strip, (r0, r1), floor_xml = pf.floor_mjcf(p, hm, hfield_dir=os.path.abspath(asset_dir), texture_dir=os.path.abspath(asset_dir))
pf._write_hfield_bin(os.path.join(asset_dir, "cake_strip.bin"), cake_strip)
np.save(os.path.join(asset_dir, "cake_strip.npy"), cake_strip)
with open(os.path.join(a.out, "poultry_litter_floor.xml"), "w") as f:
  f.write(floor_xml)

scene = """<mujoco model="g1_23dof poultry litter scene">
  <include file="g1_23dof.xml"/>
  <include file="poultry_litter_floor.xml"/>
  <statistic center="2 0 0.5" extent="4.0"/>
  <visual>
    <headlight diffuse="0.45 0.43 0.40" ambient="0.35 0.33 0.30" specular="0 0 0"/>
    <rgba haze="0.35 0.32 0.28 1"/>
    <global azimuth="-150" elevation="-18" offwidth="1920" offheight="1080"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.35 0.33 0.30" rgb2="0.12 0.11 0.10" width="512" height="3072"/>
  </asset>
</mujoco>
"""
with open(os.path.join(a.out, "scene_23dof_poultry.xml"), "w") as f:
  f.write(scene)
preview = """<mujoco model="poultry litter floor preview">
  <compiler meshdir="meshes"/>
  <include file="poultry_litter_floor.xml"/>
  <statistic center="2 0 0.2" extent="5.0"/>
  <visual><global azimuth="-150" elevation="-25" offwidth="1920" offheight="1080"/></visual>
</mujoco>
"""
with open(os.path.join(a.out, "scene_poultry_floor_only.xml"), "w") as f:
  f.write(preview)
print(f"wrote {asset_dir}/, poultry_litter_floor.xml, scene_23dof_poultry.xml, scene_poultry_floor_only.xml in {a.out}")
print(f"litter surface {1000 * hm['litter'].min():.0f}..{1000 * hm['litter'].max():.0f} mm, "
      f"cake crust {1000 * hm['cake'][hm['cake_mask'] > 0.9].mean():+.0f} mm mean, {len(hm['feeders'])} feeder pans, {p.chips} chips")

if not a.no_verify:
  import mujoco
  m = mujoco.MjModel.from_xml_path(os.path.join(a.out, "scene_poultry_floor_only.xml"))
  d = mujoco.MjData(m); mujoco.mj_forward(m, d)
  rng = np.random.default_rng(0)
  lit_id, cake_id = m.geom("litter").id, m.geom("cake").id
  errs, cake_hits, cake_exp = [], 0, 0
  for _ in range(400):
    x = rng.uniform(p.x0 + 0.2, p.x0 + p.length - 0.2); y = rng.uniform(-p.width / 2 + 0.2, p.width / 2 - 0.2)
    gid = np.array([-1], dtype=np.int32)
    dist = mujoco.mj_ray(m, d, np.array([x, y, 2.0]), np.array([0, 0, -1.0]), np.array([1, 0, 0, 0, 0, 0], np.uint8), 1, -1, gid)  # floor group only
    ix = int(round((x - hm["x"][0]) / p.resolution)); iy = int(round((y - hm["y"][0]) / p.resolution))
    on_cake = hm["cake_mask"][iy, ix] > 0.98
    near_pan = any((x - fx) ** 2 + (y - fy) ** 2 < (p.pan_radius + 0.05) ** 2 for fx, fy in hm["feeders"])
    if near_pan:
      continue
    zhit = 2.0 - dist
    if on_cake:
      cake_exp += 1; cake_hits += int(gid[0] == cake_id)
      errs.append(zhit - float(hm["cake"][iy, ix]))
    elif hm["cake_mask"][iy, ix] < 0.01:
      errs.append(zhit - float(hm["litter"][iy, ix]))
  errs = np.abs(np.array(errs))
  # Bilinear hfield interpolation differs from the cell value on steep pit rims, so judge
  # by the median: a mirrored map gives ~13 mm, the correct one ~1 mm.
  print(f"VERIFY ray-cast vs height map: median |err| {1000 * np.median(errs):.2f} mm, p95 {1000 * np.percentile(errs, 95):.2f} mm "
        f"over {len(errs)} points ({'OK' if np.median(errs) < 0.003 else 'CHECK ORIENTATION'})")
  print(f"VERIFY cake strip at y={p.drinker_y}: {cake_hits}/{cake_exp} rays hit the cake geom")
  for _ in range(int(2.0 / m.opt.timestep)):
    mujoco.mj_step(m, d)
  if p.chips:
    zs = np.array([d.xpos[m.body(f'chip_{i}').id][2] for i in range(p.chips)])
    vs = np.array([np.linalg.norm(d.cvel[m.body(f'chip_{i}').id]) for i in range(p.chips)])
    print(f"VERIFY chips after 2 s: z {1000 * zs.min():.0f}..{1000 * zs.max():.0f} mm, max speed {vs.max():.3f} (settled if ~0)")
  m2 = mujoco.MjModel.from_xml_path(os.path.join(a.out, "scene_23dof_poultry.xml"))
  print(f"VERIFY robot scene loads: {m2.nbody} bodies, {m2.ngeom} geoms, timestep {m2.opt.timestep}")
