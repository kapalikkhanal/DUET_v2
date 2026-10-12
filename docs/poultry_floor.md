# Simulated broiler-house litter floor

A procedural, MuJoCo-based model of a broiler-house floor covered in poultry litter. It is used to test the DUET walking policy before field trials, in two ways:

- **Sim-to-sim with the real controller.** The Unitree MuJoCo simulator loads `scene_23dof_poultry.xml`, and `g1_ctrl` with `keyboard.py` drives the robot on the floor exactly as on hardware.
- **Batch evaluation.** The same floor becomes an mjlab terrain. Many robots run a scripted routine on it with the trained policy in the loop, which gives the numbers for a paper.

Code:
- `scripts/poultry_floor/poultry_floor.py` — floor model and parameters;
- `scripts/poultry_floor/make_poultry_floor.py` — builder and checks;
- `scripts/poultry_floor/eval_poultry_floor.py` — batch evaluation.

## What a broiler-house floor is like

| Property | Value used | Source / basis |
|---|---|---|
| Litter depth | 7.5-15 cm (3-6 in) recommended; deeper built-up litter exists | Mississippi State Univ. Extension, *Broiler Litter: Odor and Moisture Concerns* (P3515): "Insufficient or excess litter depth (3-6 inches works best)" |
| Bedding material | Pine shavings (the usual choice in the US) | Wikipedia, *Poultry litter*; NC State Poultry Extension |
| Bulk density | Fresh pine shavings ~98 kg/m³ (6.1 lb/ft³); densifies with use (rice hulls up to ~270 kg/m³ after three flocks) | NC State Poultry Extension newsletter, July 2004 |
| Moisture | Friable litter managed near 25%; "cake" at 35-45% or more | MSU P3515; ThePoultrySite, *Litter Quality and Broiler Performance* (20-25% target) |
| Cake | Compacted, wet crust over the surface near drinkers, feeders and fans; forms a multilayer of friable litter and caked layers | MSU P3515 |
| Look | Dark, mottled used litter with light fresh-shaving flakes; feeder-pan lines and drinker lines along the house; birds cluster at the lines | Photos: Wikimedia Commons *Broiler house.jpg*, *Broiler chicks.jpg* (USDA) |

No published measurements of litter sinkage or friction under a legged robot were found. These two quantities are modelling assumptions and are reported as such. They are bracketed by two presets, below.

## Floor model

The floor is a 16 m x 6 m slice of a house: x along the house, y across it. The robot spawns at (0, 0), on the mean litter surface (z = 0).

| Element | Model |
|---|---|
| Compacted litter surface | Heightfield, 4 cm cells. Three roughness scales: 1.5 m wavelength (1.2 cm std), 0.3 m (0.6 cm), 0.12 m (0.3 cm). Finer roughness is texture only. |
| Dust-bath hollows | 0.25 per m², 40% of them near the feeder line. Radius 0.12-0.25 m, depth 2-4 cm, with a raised rim of kicked-out litter (+1 cm). |
| Windrows | Litter banked 2 cm against the side curtains, over a 0.4 m width. |
| Friable litter contact | Soft MuJoCo contact, friction 0.6. `built_up` (default): solref (0.05, 2.0), solimp (0.6, 0.9, 0.02), about 2 cm stance sinkage; this equals the softest floor seen in training. `fresh_deep`: solref (0.065, 2.5), solimp (0.45, 0.875, 0.0295), about 3.5-4.5 cm sinkage; this is beyond training, a stress test. |
| Cake strip | Separate heightfield 0.7 m wide under the drinker line (y = +1.25 m). Smooth plateau 0.8 cm above the litter, with 0.5 cm cracks. Hard contact (solref 0.02, 1.0), wet and slick (friction 0.35). |
| Loose layer (mjlab only) | The PoultrySurface drag model: velocity drag on sole points below the loose-litter surface, at depth d above the floor. Evaluated at d = 3, 5 and 8 cm (training used 0-6 cm). |
| Loose shavings (simulator only) | 150 free chips by default (`--chips N`; 150 runs at 1.8x real time or better on CPU, 400 at only 0.8x, 0 at 8.4x) (3 x 2 x 0.4 cm, 0.5 g, about 210 kg/m³) in the walking aisle, kept 0.45 m clear of the spawn point. Stiff chip contact, so they rest on the soft litter. |
| Equipment | 21 feeder pans (r = 0.17 m, h = 7 cm, with collision) every 0.76 m at y = -1.25 m. Feeder pipe at 0.50 m and drinker pipe with nipples at 0.40 m (visual only). Side curtains at y = ±3 m. |

**Checks** (`make_poultry_floor.py` runs these every time):
- The floor matches the generated height map to a 1-2 mm median, against about 13 mm for a mirrored map.
- The cake geom is the surface on the strip.
- The chips settle.
- The robot scene loads, and a 3 s uncontrolled drop onto litter and chips is numerically stable.

**Implementation notes:**
- **Collision grid:** MuJoCo Warp keeps at most 50 contacts per geom pair and drops the rest. The training G1 foot is seven capsules, up to 19 cm long.
  - On a 2 cm heightfield this overflowed about 270 times per physics substep; on 4 cm, about 0.2 times.
  - Those contact losses made the robots fall: the first evaluation results were artefacts and were discarded.
  - The batch evaluation therefore collides on an 8 cm grid (the 4 cm map block-averaged; `--coarsen 2`), which overflows zero times, close to the 10 cm cells of the training heightfields.
  - The CPU deploy simulator keeps 4 cm: its feet are four 5 mm spheres, and a drop test raised no solver warnings.
- **Asset paths:** they are absolute (MuJoCo resolves files in `<include>`d models differently depending on the working directory). Re-run the builder on each machine.
- **Deploy simulator feet:** the Unitree model uses four 5 mm contact spheres per foot, while mjlab uses capsule soles. Sinkage in the two simulators differs somewhat.

## Running it

Build the floor. This writes into `~/Mujoco_kapalikkhanal/unitree_robots/g1` and doesn't modify existing files.

```bash
cd ~/rbbist.lab/DUET_poultry_v2      # or wherever this repo lives
python scripts/poultry_floor/make_poultry_floor.py                         # built_up preset (default)
python scripts/poultry_floor/make_poultry_floor.py --preset fresh_deep     # deep fresh shavings
python scripts/poultry_floor/make_poultry_floor.py --chips 0               # no loose chips (faster)
```

Sim-to-sim with the deployed controller, in three terminals. This is the same workflow as the flat-floor test; only the scene changes.

```bash
# 1  simulator on the litter floor
cd ~/Mujoco_kapalikkhanal/simulate/build && ./unitree_mujoco -s scene_23dof_poultry.xml
# 2  controller (policy folder chosen as usual)
cd <DUET>/deploy/robots/g1_23dof/build && ./g1_ctrl
# 3  keyboard teleop
python <DUET>/keyboard.py
```

Floor preview without the robot:

```bash
python -m mujoco.viewer --mjcf ~/Mujoco_kapalikkhanal/unitree_robots/g1/scene_poultry_floor_only.xml
```

Batch evaluation (GPU), from the repo root with `PYTHONPATH=.`:

```bash
python scripts/poultry_floor/eval_poultry_floor.py --policy <model_N.pt | policy.onnx> --n 96 --json out.json
#   --contact built_up | fresh_deep | training | rigid   litter softness (default: as generated)
#   --no-pans --flat --no-drag                            ablations
python scripts/poultry_floor/eval_poultry_floor.py --policy <...> --video poultry_walk.mp4   # one robot, rendered
```
