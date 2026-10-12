# DUET

**Decoupled Upper- and Lower-Body Policies for VLA-Agnostic Loco-Manipulation on a Humanoid Robot**

A reinforcement-learning policy controls the twelve leg joints and the waist yaw of a
23-DOF Unitree G1, tracking commanded planar velocity, turning rate and a pelvis height
that spans a 0.73 m stand down to a 0.18 m crouch. The ten arm joints are excluded from
its action space and enter only as observed joint angles.

That controller is the single writer of the robot's low-level command topic. On every
cycle it copies the arm section of its outgoing message from a second topic
(`rt/arm_targets`) that any other process may publish to. Any policy able to write joint
targets can therefore move the arms while the legs keep balance, with no coordination
between the two and no shared training. We use this to run two separately trained
vision-language-action models, GR00T N1.7 and π0.5, on one unchanged locomotion
checkpoint.

https://github.com/bae-air-lab/DUET/raw/main/Robot.mp4

---

## DUET v2: walking and squatting on poultry-house litter

v2 adapts the DUET locomotion policy to broiler-house floors: soft, loose litter over a hard base, with caked crusts near the drinker lines. Starting from the rough-terrain policy, it adds:

- **The `Unitree-G1-23Dof-Duet-PoultrySurface` task.** It trains in a simulated litter layer (drag on the feet below the litter surface), with:
  - random litter depth of 0-6 cm per episode;
  - a lift-first swing (target 9-11 cm above the floor, rising with litter depth) and a penalty on scooping litter;
  - stronger backward walking (capped at 0.6 m/s);
  - squats balanced over the feet with the arms forward or wide;
  - a terrain mix weighted toward flat and litter tiles.
- **Deeper squats.** The hip-pitch, knee and ankle-pitch joints may use 97% of their range (other joints keep 90%), and the squat floor is 0.22 m. The policy reaches about 0.25 m, against 0.30 m before.
- **Training stability fixes.** The input normalisers are frozen when a run resumes, training clips raw actions, and a penalty discourages raw actions above 8. Without these, rare huge actions corrupted training or would trip the robot's runaway guard (|action| > 15 for 60 ms).
- **A simulated broiler-house floor** for testing before field trials (`scripts/poultry_floor/`, documented in [docs/poultry_floor.md](docs/poultry_floor.md)):
  - an uneven litter surface, dust-bath hollows, a caked strip under the drinker line, feeder pans and loose shavings;
  - the same floor runs in the Unitree MuJoCo sim-to-sim simulator (real `g1_ctrl` controller) and in mjlab for batch evaluation.
  - Results: [docs/poultry_floor_results.md](docs/poultry_floor_results.md).

### Checkpoints

| Policy | Files | Notes |
|---|---|---|
| **poultry_v3_squat_91500** (recommended) | `checkpoints/poultry_v3_squat_91500/model_91500.pt`, `exported/poultry_v3_squat_91500/` | Deepest squat (about 0.25 m); foot lift about 8 cm on flat ground; walks backward at 104% of command; zero extreme-action events in 6 million simulated steps |
| poultry_v3_squat_93000 | `checkpoints/poultry_v3_squat_93000/`, `exported/poultry_v3_squat_93000/` | Best litter clearance and foot lift; squat about 1 cm shallower; backward speed 86% |
| poultry_v2_90000 | `checkpoints/poultry_v2_90000/`, `exported/poultry_v2_90000/` | Before the deeper-squat fine-tune (squat about 0.30 m) |
| rough_50k | `checkpoints/rough_50k/model_50000.pt` | The rough-terrain policy all poultry runs start from |
| rough_squatreach_ft_50000 | `checkpoints/rough_squatreach_ft_50000/`, `exported/rough_squatreach_ft_50000/` | Rough policy with arm-reach squats; deployed as `v1_squatreach_ft50000` |

- **Exported policies:** each `exported/<name>/` folder holds `policy.onnx` (71 inputs, 13 outputs) and `deploy_contract.json`. Where the deploy config changed, it also holds `params/deploy.yaml`.
- **Default deploy folder:** `deploy/robots/g1_23dof/config/policy/velocity/v2_poultry_squat_91500/` holds the recommended policy. `g1_ctrl` loads the last-sorted policy folder, so this one is used by default.
- **Deploy configs:** all three carry a per-joint target clip and the runaway guard (`safety:`).

### Quick start (v2)

```bash
# train / fine-tune (single GPU; output goes through the log filter, because MuJoCo Warp prints
# one "ccd_iterations" warning per contact and would otherwise produce a huge console log)
bash scripts/finetune_poultry_v2.sh checkpoints/poultry_v3_squat_91500/model_91500.pt my_run 4096 100000

# build the poultry floor for the Unitree MuJoCo simulator, then run sim-to-sim:
python scripts/poultry_floor/make_poultry_floor.py --out <unitree_mujoco>/unitree_robots/g1
<unitree_mujoco>/simulate/build/unitree_mujoco -s scene_23dof_poultry.xml   # terminal 1
deploy/robots/g1_23dof/build/g1_ctrl                                       # terminal 2
python keyboard.py                                                         # terminal 3

# batch evaluation on the floor, and the per-skill tests (foot lift, backward walking,
# squats with arm motion, litter depth, extreme actions)
PYTHONPATH=. python scripts/poultry_floor/eval_poultry_floor.py --policy checkpoints/poultry_v3_squat_91500/model_91500.pt
PYTHONPATH=. python scripts/eval/action_extremes.py checkpoints/poultry_v3_squat_91500/model_91500.pt
```

- **Training records** (`training_records/`): progress tables, evaluations, decisions and reports from the v2 runs. This includes the problems found along the way: the normaliser collapse, the extreme-action drift, and contact overflow on fine heightfields.

---

## Layout

| path | what it is |
|---|---|
| `src/tasks/duet/` | the DUET task: rewards, commands, curricula, symmetry, ablation switches |
| `src/tasks/common/mdp/` | MDP terms shared between tasks |
| `src/tasks/velocity/config/g1_23dof/` | the whole-body baseline the paper compares against |
| `src/assets/robots/unitree_g1/` | G1 and G1-23DOF model and constants |
| `deploy/robots/g1_23dof/` | the C++ controller that runs the exported policy on hardware |
| `exported/` | the deployed checkpoint, its ONNX export and its deployment contract |
| `vla_bridge/` | the scripts that publish VLA arm targets to `rt/arm_targets` |
| `scripts/` | training, play, export, evaluation and ablation tooling |

Only the Unitree G1 is included here. The upstream
[unitree_rl_mjlab](https://github.com/unitreerobotics/unitree_rl_mjlab) carries the other
robots.

## Install

Requires a CUDA GPU for training. MuJoCo Warp does the simulation.

```bash
git clone https://github.com/bae-air-lab/DUET.git
cd DUET
pip install -e .            # pulls mjlab==1.2.0 and mujoco-warp==3.5.0
```

## Train

```bash
PYTHONPATH=. python scripts/train.py Unitree-G1-23Dof-Duet-Flat \
    --agent.seed 1 --env.scene.num-envs 4096 --agent.run-name duet
```

`PYTHONPATH=.` matters on a machine where an editable `src` package from a sibling
checkout is installed; without it the scripts import that checkout's tasks.

Checkpoints land in `logs/rsl_rl/DUET_G1_23dof/<run>/`. Resuming continues the command,
arm-pose and entropy curricula where they stopped — every curriculum is keyed to the
global environment step counter, which is written into the checkpoint.

Watch a trained policy:

```bash
python scripts/play.py --task Unitree-G1-23Dof-Duet-Flat \
    --checkpoint-file logs/rsl_rl/DUET_G1_23dof/<run>/model_15500.pt
```

The whole-body baseline is `Unitree-G1-23Dof-Flat`, trained the same way.

The arm-robustness training recipe (asynchronous trapezoidal arm trajectories over
a safe workspace, staged curriculum, pelvis-anchor deadband, whole-body CoM support
term, diagnostics) is described in
[`documents/duet/arm_robustness.md`](documents/duet/arm_robustness.md). The eight
digital-twin acceptance scenarios run with:

```bash
PYTHONPATH=. python scripts/duet_arm_scenarios.py \
    --checkpoint logs/rsl_rl/DUET_G1_23dof/<run>/model_XXXX.pt
```

### Ablations

Each variant removes exactly one design choice and holds everything else fixed:

```
Unitree-G1-23Dof-Duet-Abl-Reference          full method
Unitree-G1-23Dof-Duet-Abl-NoSymmetry         no mirror symmetry
Unitree-G1-23Dof-Duet-Abl-NoHeightCmd        no height command
Unitree-G1-23Dof-Duet-Abl-NoArmCurriculum    no upper-body pose curriculum
```

`scripts/duet_ablation_table.py`, `duet_ablation_curves.py`, `duet_robustness_envelope.py`
and `duet_vs_wholebody.py` reproduce the paper's tables and figures.

## The deployed policy

`exported/` holds the checkpoint the paper deploys and evaluates on hardware:

| file | |
|---|---|
| `policy.onnx` | `obs [1, 71] → actions [1, 13]`, with the deployment contract in its metadata |
| `deploy_contract.json` | joint order, PD gains, action scales, observation layout, command ranges |
| `model_15500.pt` | the RL checkpoint, for resuming or fine-tuning |
| `env.yaml`, `agent.yaml` | the exact configuration it was trained under |

The checkpoint was selected by evaluating every saved checkpoint from iteration 10000
onward under a fixed protocol and ranking by fall rate first and tracking error second,
not by taking the final checkpoint — the last checkpoint of the run was the weakest of
those evaluated.

Export your own and verify it against the config the C++ controller will load:

```bash
python scripts/export_duet_onnx.py \
    --task Unitree-G1-23Dof-Duet-Flat \
    --checkpoint logs/rsl_rl/DUET_G1_23dof/<run>/model_15500.pt \
    --out-dir exported/my_export \
    --deploy-yaml deploy/robots/g1_23dof/config/policy/velocity/v0/params/deploy.yaml

python scripts/check_deploy_consistency.py \
    --onnx exported/my_export/policy.onnx \
    --deploy-yaml deploy/robots/g1_23dof/config/policy/velocity/v0/params/deploy.yaml
```

The checker exits 0 when the export and the deployment config agree and 1 when they do
not. Run it before putting a policy on the robot: the ONNX file alone records nothing
about the gains or joint order it was trained against, so a mismatch is otherwise
invisible until hardware.

## Deploy

```bash
cd deploy/robots/g1_23dof
mkdir build && cd build && cmake .. && make -j
```

The controller loads the exported policy, holds the finite-state machine and the
joint-level gains, and is the only writer of `rt/lowcmd`. It fills the thirteen
lower-body entries from the policy and the ten arm entries from the most recent message
on `rt/arm_targets`. If that stream goes quiet for 200 ms the arms fall back to an
operator-selected pose.

## Running a VLA on top

`vla_bridge/` holds the two scripts that make a manipulation policy drive the arms. Each
constructs `G1_23_ArmController(lowcmd_topic="rt/arm_targets")` instead of writing
`rt/lowcmd`, so the RL controller stays the sole writer of the robot's command message:

- `eval_g1_groot_2cam_hybrid.py` — GR00T N1.7
- `eval_g1_pi_2cam_hybrid.py` — π0.5

They are the whole integration. Everything else they use is a dependency:

1. [Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T) for the policy server
   (`gr00t.policy.server_client`)
2. [unitree_lerobot](https://github.com/unitreerobotics/unitree_lerobot) for the arm and
   hand controllers and the image server
3. `unitree_sdk2py` for DDS

**The bridge needs our modifications to `unitree_lerobot`.** They are in
`vla_bridge/unitree_lerobot_eval_robot.patch` and touch nine files under
`unitree_lerobot/eval_robot/` — the arm controller's configurable command topic, the
image server, and the Dex3-1 hand gains. Stock upstream will not work:

```bash
cd /path/to/unitree_lerobot
git apply /path/to/DUET/vla_bridge/unitree_lerobot_eval_robot.patch
```

Vision is a head-mounted RealSense D455 for the wide view over the scene and a
wrist-mounted D405 for close range during grasping.

Run the locomotion controller first, then the bridge. The locomotion policy is never
restarted, never told which model is publishing, and never receives the arm targets or a
task description.

## Licence

This repository builds on [unitree_rl_mjlab](https://github.com/unitreerobotics/unitree_rl_mjlab)
and [mjlab](https://github.com/mujocolab/mjlab). See `LICENCE` and `doc/license/` for the
upstream terms, including those for the bundled ONNX Runtime binaries in
`deploy/thirdparty/`.
