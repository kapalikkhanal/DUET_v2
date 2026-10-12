"""Procedural broiler-house litter floor for MuJoCo.

One generator, two consumers:
  * a standalone MJCF floor (``poultry_litter_floor.xml``) that the Unitree
    sim2sim simulator loads next to the G1 model, so the deployed controller
    (g1_ctrl) walks on it unchanged;
  * height maps (``*.npy``) that ``eval_poultry_floor.py`` turns into an mjlab
    terrain for batch policy evaluation with the litter drag model.

Physical basis (see docs/poultry_floor.md for sources and assumptions):
  * litter depth 3-6 in (7.5-15 cm) is recommended (Mississippi State Univ.
    Extension P3515); built-up litter can be deeper;
  * friable litter is managed near 25% moisture; a compacted wet "cake"
    (>= 40% moisture) forms over the surface near drinkers, feeders and fans;
  * fresh pine shavings ~98 kg/m^3 (6.1 lb/ft^3, NC State); used litter
    densifies (up to ~270 kg/m^3 after three flocks, rice hulls);
  * surface: mottled dark used litter with light fresh-shaving flakes, lumps
    from scratching, dust-bath hollows, litter banked at the side walls,
    feeder pans and drinker lines running the length of the house.
Sinkage and friction values are modelling assumptions (no measured litter
terramechanics data were found); they are documented with each parameter.

Frame: x along the house, y across it, z up. The mean friable-litter surface
is z = 0, so the robot spawns as on flat ground.
"""
from __future__ import annotations

import json
import os
import struct
from dataclasses import asdict, dataclass, field

import numpy as np


@dataclass
class PoultryFloorParams:
  # Floor patch (a slice of a ~150 m x 15 m house).
  length: float = 16.0          # m, along x
  width: float = 6.0            # m, across y
  x0: float = -4.0              # m, x of the near end (robot starts at x = 0)
  # 4 cm cells: at 2 cm, MuJoCo Warp's hfield collision overflowed its 50-contacts-per-pair
  # limit ~270 times per substep (contacts silently dropped). Sub-foot roughness is
  # carried by the texture instead.
  resolution: float = 0.04      # m per height-map cell
  seed: int = 7

  # Litter surface roughness: (wavelength m, std m) octaves.
  octaves: tuple = ((1.5, 0.012), (0.30, 0.006), (0.12, 0.003))
  # Dust-bath hollows: count per m^2, radius range m, depth range m, rim height m.
  pit_density: float = 0.25
  pit_radius: tuple = (0.12, 0.25)
  pit_depth: tuple = (0.02, 0.04)
  pit_rim: float = 0.010
  # Litter banked against the side curtains (windrow), m, and its width.
  windrow_height: float = 0.02
  windrow_width: float = 0.4

  # Equipment lines (y positions, m).
  drinker_y: float = 1.25
  feeder_y: float = -1.25
  feeder_spacing: float = 0.76  # m between pans
  pan_radius: float = 0.17
  pan_height: float = 0.07
  drinker_height: float = 0.40  # nipple line above litter
  feeder_pipe_height: float = 0.50

  # Cake under the drinker line (spillage): strip width, crust rise, cracks.
  cake_width: float = 0.70
  cake_rise: float = 0.008
  crack_density: float = 3.0    # cracks per m of strip
  crack_depth: float = 0.005

  # Contact of the friable (loose) litter. Two presets (see PRESETS):
  #   built_up   - compacted built-up litter, ~2 cm stance sinkage under the G1;
  #                equals the softest floor seen in training (SoftFlatTerrainCfg, t=1).
  #   fresh_deep - fresh deep shavings, ~3.5-4.5 cm sinkage; the training soft floor
  #                extrapolated 1.5x (beyond training: a stress test).
  # Friction of dry wood shavings ~0.6 (assumed).
  preset: str = "built_up"
  friable_solref: tuple = (0.05, 2.0)
  friable_solimp: tuple = (0.6, 0.9, 0.02, 0.5, 2.0)
  friable_friction: float = 0.6
  # Cake: hard, wet, slick crust (~0.3 cm sinkage), friction 0.35.
  cake_solref: tuple = (0.02, 1.0)
  cake_solimp: tuple = (0.9, 0.95, 0.001, 0.5, 2.0)
  cake_friction: float = 0.35

  # Loose shavings: free chips scattered in the walking aisle.
  chips: int = 150   # 150 keeps the CPU simulator >1.8x real time (400: 0.8x)
  chip_half: tuple = (0.015, 0.010, 0.002)  # m (3 x 2 x 0.4 cm)
  chip_mass: float = 0.0005                 # kg (~210 kg/m^3)
  chip_region: tuple = (-0.5, 8.0, -0.8, 0.8)  # x0, x1, y0, y1
  chip_clear_radius: float = 0.45  # m kept clear around the robot spawn (0, 0)


PRESETS = {
  "built_up": dict(preset="built_up", friable_solref=(0.05, 2.0), friable_solimp=(0.6, 0.9, 0.02, 0.5, 2.0)),
  "fresh_deep": dict(preset="fresh_deep", friable_solref=(0.065, 2.5), friable_solimp=(0.45, 0.875, 0.0295, 0.5, 2.0)),
}


def _fbm(rng, nx, ny, res, octaves):
  """Sum of band-limited Gaussian noise octaves via FFT filtering."""
  out = np.zeros((ny, nx))
  kx = np.fft.fftfreq(nx, d=res)[None, :]
  ky = np.fft.fftfreq(ny, d=res)[:, None]
  k = np.sqrt(kx ** 2 + ky ** 2)
  for wavelength, std in octaves:
    f0 = 1.0 / wavelength
    filt = np.exp(-0.5 * ((k - f0) / (0.5 * f0)) ** 2)
    n = np.fft.ifft2(np.fft.fft2(rng.standard_normal((ny, nx))) * filt).real
    out += std * n / (n.std() + 1e-12)
  return out


def build_heightmaps(p: PoultryFloorParams) -> dict:
  """Return height maps (row = y, column = x, metres) and layout info."""
  rng = np.random.default_rng(p.seed)
  nx, ny = int(round(p.length / p.resolution)) + 1, int(round(p.width / p.resolution)) + 1
  xs = p.x0 + np.arange(nx) * p.resolution
  ys = -p.width / 2 + np.arange(ny) * p.resolution
  X, Y = np.meshgrid(xs, ys)

  litter = _fbm(rng, nx, ny, p.resolution, p.octaves)
  # Dust-bath hollows with a raised rim of kicked-out litter; denser near feeders.
  area = p.length * p.width
  n_pits = rng.poisson(p.pit_density * area)
  for _ in range(n_pits):
    if rng.random() < 0.4:
      cy = p.feeder_y + rng.normal(0, 0.5)
    else:
      cy = rng.uniform(-p.width / 2 + 0.3, p.width / 2 - 0.3)
    cx = rng.uniform(xs[0] + 0.3, xs[-1] - 0.3)
    r = rng.uniform(*p.pit_radius); d = rng.uniform(*p.pit_depth)
    q = ((X - cx) ** 2 + (Y - cy) ** 2) / r ** 2
    litter -= d * np.exp(-2.5 * q) * (q < 1.6)
    litter += p.pit_rim * np.exp(-((np.sqrt(q) - 1.15) ** 2) / 0.05)
  # Windrows against the side curtains.
  edge = p.width / 2 - np.abs(Y)
  litter += p.windrow_height * np.clip(1 - edge / p.windrow_width, 0, 1) ** 2

  # Cake strip under the drinker line: smooth plateau with cracks.
  dy = np.abs(Y - p.drinker_y)
  mask = np.clip((p.cake_width / 2 - dy) / 0.05 + 0.5, 0, 1)  # 5 cm feathered edge
  smooth = _fbm(rng, nx, ny, p.resolution, ((2.0, 0.006), (0.5, 0.002)))
  cake = smooth + p.cake_rise
  n_cracks = rng.poisson(p.crack_density * p.length)
  for _ in range(n_cracks):
    cx = rng.uniform(xs[0], xs[-1]); ang = rng.uniform(0, np.pi); ln = rng.uniform(0.1, 0.4)
    t = np.clip(((X - cx) * np.cos(ang) + (Y - p.drinker_y) * np.sin(ang)) / ln, -0.5, 0.5)
    px, py = cx + t * ln * np.cos(ang), p.drinker_y + t * ln * np.sin(ang)
    dist = np.sqrt((X - px) ** 2 + (Y - py) ** 2)
    cake -= p.crack_depth * np.exp(-(dist / 0.008) ** 2)
  # Friable litter is pushed 1.5 cm below the crust where the cake lies, so a
  # foot on the strip contacts the cake geom first.
  litter_under = litter - 0.015 * mask

  litter_under -= litter_under[mask < 0.01].mean()  # mean friable surface at z = 0
  cake += 0.0
  feeders = [(float(x), p.feeder_y) for x in np.arange(xs[0] + 0.4, xs[-1] - 0.2, p.feeder_spacing)]
  return dict(x=xs, y=ys, litter=litter_under.astype(np.float32), cake=cake.astype(np.float32),
              cake_mask=mask.astype(np.float32), feeders=feeders)


def _write_hfield_bin(path, h):
  """MuJoCo custom hfield format: int32 nrow, int32 ncol, float32 data."""
  with open(path, "wb") as f:
    f.write(struct.pack("ii", h.shape[0], h.shape[1]))
    f.write(h.astype(np.float32).tobytes())


def _texture(rng, n=1024, cake=False):
  """Procedural litter texture (RGB uint8), 1 px = 2 mm, ~2 m per tile.

  Used litter is dark brown and mottled; fresh shaving flakes (light tan,
  1-2 cm, irregular) are sparse on top; droppings and wet specks are darker.
  """
  base = np.array([34, 27, 20] if cake else [56, 41, 29], float)
  mott = _fbm(rng, n, n, 1.0 / n, ((0.25, 1.0), (0.05, 0.6), (0.01, 0.3)))
  img = base[None, None, :] * (1 + 0.20 * mott[..., None])
  if not cake:
    yy, xx = np.mgrid[-8:9, -8:9]
    for col, count, rad in (([178, 150, 105], 1100, (2.0, 6.0)), ([22, 17, 12], 1500, (1.0, 3.0))):
      for _ in range(count):
        cy, cx = rng.integers(8, n - 8, 2)
        a, b = rng.uniform(*rad), rng.uniform(rad[0] * 0.4, rad[1] * 0.6)
        th = rng.uniform(0, np.pi)
        u = (xx * np.cos(th) + yy * np.sin(th)) / a; v = (-xx * np.sin(th) + yy * np.cos(th)) / b
        m = (u ** 2 + v ** 2) < 1 + 0.3 * rng.standard_normal(u.shape)
        patch = img[cy - 8:cy + 9, cx - 8:cx + 9]
        patch[m] = 0.35 * patch[m] + 0.65 * np.array(col) * rng.uniform(0.75, 1.1)
  return np.clip(img, 0, 255).astype(np.uint8)


def write_assets(p: PoultryFloorParams, out_dir: str) -> dict:
  """Write height maps (.bin for MuJoCo, .npy for mjlab), textures and metadata."""
  from PIL import Image
  os.makedirs(out_dir, exist_ok=True)
  hm = build_heightmaps(p)
  rng = np.random.default_rng(p.seed + 1)
  for name in ("litter", "cake"):
    h = hm[name]
    _write_hfield_bin(os.path.join(out_dir, f"{name}.bin"), h)
    np.save(os.path.join(out_dir, f"{name}.npy"), h)
  np.save(os.path.join(out_dir, "cake_mask.npy"), hm["cake_mask"])
  Image.fromarray(_texture(rng)).save(os.path.join(out_dir, "litter_tex.png"))
  Image.fromarray(_texture(rng, cake=True)).save(os.path.join(out_dir, "cake_tex.png"))
  meta = dict(params=asdict(p), feeders=hm["feeders"],
              x_range=[float(hm["x"][0]), float(hm["x"][-1])], y_range=[float(hm["y"][0]), float(hm["y"][-1])],
              litter_range=[float(hm["litter"].min()), float(hm["litter"].max())],
              cake_range=[float(hm["cake"].min()), float(hm["cake"].max())])
  with open(os.path.join(out_dir, "floor_meta.json"), "w") as f:
    json.dump(meta, f, indent=1)
  return dict(hm=hm, meta=meta)


def _f(v):
  return " ".join(f"{x:g}" for x in v)


def floor_mjcf(p: PoultryFloorParams, hm: dict, hfield_dir: str, texture_dir: str) -> str:
  """MJCF of the floor alone (assets + worldbody), for <include> in a scene."""
  rng = np.random.default_rng(p.seed + 2)
  cx = p.x0 + p.length / 2
  lit, cak = hm["litter"], hm["cake"]
  lmin, lmax = float(lit.min()), float(lit.max())
  cmin, cmax = float(cak.min()), float(cak.max())
  # Cake hfield covers only the strip around the drinker line.
  strip = p.cake_width / 2 + 0.1
  ys = hm["y"]; rows = np.where(np.abs(ys - p.drinker_y) <= strip)[0]
  r0, r1 = int(rows[0]), int(rows[-1])
  cake_strip_h = cak[r0:r1 + 1]
  # Cake below the mask edge is pushed under the litter so it never pokes through.
  cake_strip_h = np.where(hm["cake_mask"][r0:r1 + 1] > 0.02, cake_strip_h, cmin - 0.03)
  cmin = float(cake_strip_h.min())
  return cake_strip_h, (r0, r1), f"""<mujoco model="poultry_litter_floor">
  <!-- Generated by scripts/poultry_floor/make_poultry_floor.py (seed {p.seed}).
       Broiler-house litter slice ({p.preset} preset): {p.length:g} m x {p.width:g} m, friable litter (soft contact,
       friction {p.friable_friction}), caked strip under the drinker line (hard, friction {p.cake_friction}),
       dust-bath hollows, wall windrows, feeder pans, drinker/feeder pipes, {p.chips} loose shavings. -->
  <asset>
    <texture name="litter_tex" type="2d" file="{texture_dir}/litter_tex.png"/>
    <texture name="cake_tex" type="2d" file="{texture_dir}/cake_tex.png"/>
    <material name="litter_mat" texture="litter_tex" texuniform="true" texrepeat="0.5 0.5" specular="0.05" shininess="0.05"/>
    <material name="cake_mat" texture="cake_tex" texuniform="true" texrepeat="0.5 0.5" specular="0.4" shininess="0.6" reflectance="0.05"/>
    <material name="pan_mat" rgba="0.80 0.12 0.10 1"/>
    <material name="pipe_mat" rgba="0.75 0.75 0.78 1"/>
    <material name="curtain_mat" rgba="0.55 0.52 0.46 1"/>
    <material name="chip_mat" rgba="0.82 0.70 0.50 1"/>
    <hfield name="litter_hf" file="{hfield_dir}/litter.bin" size="{p.length / 2:g} {p.width / 2:g} {lmax - lmin:.5f} 0.05"/>
    <hfield name="cake_hf" file="{hfield_dir}/cake_strip.bin" size="{p.length / 2:g} {(r1 - r0) * p.resolution / 2:g} {cmax - cmin:.5f} 0.02"/>
  </asset>
  <worldbody>
    <light name="house_light_1" pos="{cx - 4:g} 0 3.2" dir="0 0 -1" diffuse="0.55 0.5 0.42" specular="0.1 0.1 0.1" castshadow="false"/>
    <light name="house_light_2" pos="{cx + 4:g} 0 3.2" dir="0 0 -1" diffuse="0.55 0.5 0.42" specular="0.1 0.1 0.1" castshadow="false"/>
    <geom name="concrete" type="plane" size="0 0 0.05" pos="0 0 {lmin - 0.08:.4f}" rgba="0.45 0.45 0.43 1"/>
    <geom name="litter" type="hfield" hfield="litter_hf" pos="{cx:g} 0 {lmin:.5f}" material="litter_mat"
          priority="1" friction="{p.friable_friction} 0.005 0.0001" solref="{_f(p.friable_solref)}" solimp="{_f(p.friable_solimp)}"/>
    <geom name="cake" type="hfield" hfield="cake_hf" pos="{cx:g} {(ys[r0] + ys[r1]) / 2:.4f} {cmin:.5f}" material="cake_mat"
          priority="1" friction="{p.cake_friction} 0.005 0.0001" solref="{_f(p.cake_solref)}" solimp="{_f(p.cake_solimp)}"/>
{_equipment(p, hm)}
{_chips(p, hm, rng)}
  </worldbody>
</mujoco>
"""


def _equipment(p, hm):
  cx = p.x0 + p.length / 2
  g = []
  for i, (x, y) in enumerate(hm["feeders"]):
    g.append(f'    <geom name="feeder_pan_{i}" type="cylinder" size="{p.pan_radius:g} {p.pan_height / 2:g}" '
             f'pos="{x:.3f} {y:g} {p.pan_height / 2 - 0.01:g}" material="pan_mat" group="1"/>')
    g.append(f'    <geom name="feeder_drop_{i}" type="cylinder" size="0.02 {(p.feeder_pipe_height - p.pan_height) / 2:g}" '
             f'pos="{x:.3f} {y:g} {(p.feeder_pipe_height + p.pan_height) / 2:g}" material="pipe_mat" contype="0" conaffinity="0" group="1"/>')
  g.append(f'    <geom name="feeder_pipe" type="capsule" size="0.03" fromto="{p.x0:g} {p.feeder_y:g} {p.feeder_pipe_height:g} '
           f'{p.x0 + p.length:g} {p.feeder_y:g} {p.feeder_pipe_height:g}" material="pipe_mat" contype="0" conaffinity="0" group="1"/>')
  g.append(f'    <geom name="drinker_pipe" type="capsule" size="0.015" fromto="{p.x0:g} {p.drinker_y:g} {p.drinker_height:g} '
           f'{p.x0 + p.length:g} {p.drinker_y:g} {p.drinker_height:g}" material="pipe_mat" contype="0" conaffinity="0" group="1"/>')
  for i, x in enumerate(np.arange(p.x0 + 0.15, p.x0 + p.length, 0.30)):
    g.append(f'    <geom name="nipple_{i}" type="cylinder" size="0.006 0.03" pos="{x:.3f} {p.drinker_y:g} '
             f'{p.drinker_height - 0.04:g}" rgba="0.9 0.6 0.1 1" contype="0" conaffinity="0" group="1"/>')
  for s in (-1, 1):
    g.append(f'    <geom name="curtain_{"l" if s > 0 else "r"}" type="box" size="{p.length / 2:g} 0.02 1.2" '
             f'pos="{cx:g} {s * (p.width / 2 + 0.03):g} 1.1" material="curtain_mat" group="1"/>')
  return "\n".join(g)


def _chips(p, hm, rng):
  if p.chips <= 0:
    return ""
  xs, ys, lit = hm["x"], hm["y"], hm["litter"]
  x0, x1, y0, y1 = p.chip_region
  out = []
  hx, hy, hz = p.chip_half
  for i in range(p.chips):
    while True:  # chips overlapping the feet at spawn make the solver unstable
      x, y = rng.uniform(x0, x1), rng.uniform(y0, y1)
      if x * x + y * y > p.chip_clear_radius ** 2:
        break
    ix = int(np.clip(round((x - xs[0]) / p.resolution), 0, len(xs) - 1))
    iy = int(np.clip(round((y - ys[0]) / p.resolution), 0, len(ys) - 1))
    z = float(lit[iy, ix]) + hz + 0.004
    yaw = rng.uniform(0, np.pi)
    out.append(f'    <body name="chip_{i}" pos="{x:.3f} {y:.3f} {z:.4f}" euler="0 0 {yaw:.3f}">'
               f'<freejoint/><geom type="box" size="{hx:g} {hy:g} {hz:g}" mass="{p.chip_mass:g}" material="chip_mat" '
               f'contype="2" conaffinity="1" condim="3" friction="0.6 0.005 0.0001" group="1" priority="2" '
               f'solref="0.005 1" solimp="0.9 0.95 0.001 0.5 2"/></body>')
  return "\n".join(out)
