# Task: PoultrySurface v2 (litter depth per episode, litter-relative lift target, barn-weighted terrain mix)

You're working on DUET, a reinforcement-learning locomotion project for a Unitree G1 23-DOF humanoid. It uses mjlab 1.2, MuJoCo Warp and rsl_rl PPO.

Your job: implement three changes to the `Unitree-G1-23Dof-Duet-PoultrySurface` task, verify them, and report back.
- Do not start a long training run.
- Do not touch the training run that is in progress.
- Do not commit or push.

## Background: what has happened so far

### What the policy is for
- **Where:** the robot will work in poultry barns, on bedding litter (shavings or straw) over a hard floor.
- **What it senses:** the walking policy is blind. Its 71 inputs are the IMU, the commands, the gait phase, joint positions and velocities, its previous action and the height command; it has no terrain sensing. The critic also gets a terrain height scan.
- **What it controls:** the 13 lower-body joints. The arms are driven externally by teleoperation or manipulation. On the real robot each wrist carries about 0.9–1 kg (Dex3 hand, gripper and camera).
- **What it must do:**
  - walk forward up to 1.0 m/s, backward (capped at 0.6 m/s), sideways, and turn;
  - squat (commanded down to 0.24 m pelvis height);
  - stay balanced while the arms move.
- **The poultry-specific goal:** march-like steps that lift the feet clear instead of shuffling, so the robot doesn't scoop litter and raise dust.

### History
1. **Rough 50k.** The DUET rough-terrain task (`Unitree-G1-23Dof-Duet-Rough`) was trained to 50,000 iterations on flat, soft-floor, heightfield-slope, random-rough and wave tiles, with a 2x-stretched curriculum.
   - Value-loss blow-ups at about 10.8k and 37k came from robots spawning inside heightfield tiles. They were fixed with flat-patch spawning and a per-step reward floor of -5 (runner `reward_floor`).
   - The result, `model_50000.pt`, is called "Rough 50k".
2. **Real-robot findings** (checkpoint 35k):
   - with the arms forward in a deep squat, the robot drifted backward and straightened its back;
   - backward walking was weak: it leaned back, walked slower than commanded, and stopped abruptly.
3. **Commit e063b5e** added PoultrySurface, a separate task built on the Rough config. It has a lift-first swing:
   - foot-lift rewards `poultry_clearance`, `poultry_plow`, `poultry_swing_height` and `poultry_foot_level`;
   - a litter model on soft-floor tiles: a drag force on sole points below the litter surface, proportional to depth times velocity (strength 500–1500, capped at 150 N).
4. **Commit e8836e7** (pushed as origin/poultry_terrain) prepared a 100k run:
   - backward speed capped at -0.6 m/s, a penalty for walking slower than the backward command, and random stops while walking backward;
   - a squat centre-of-mass centring penalty;
   - more deep squats and arm-reach poses, including wide arms;
   - the reward floor, and 100k iterations.
5. **The current run (Oct 9–10)**, resumed from Rough 50k on PoultrySurface:
   - **50k to 55k:**
     - backward speed went from 76% to 86–97% of the command;
     - the centre of mass moved from 6–8 cm to 4–6 cm behind the feet;
     - foot lift (the lowest sole point at the top of each swing, on flat ground) went from 1.8 to 5.2 cm.
   - **55k to 60k:**
     - lift stopped improving (4.7 cm at 60k);
     - "boost 1" was applied at 60k: `poultry_clearance` -2.0 -> -3.0 and `poultry_swing_height` -0.06 -> -0.12.
   - **By 65k:** lift was 6.6 cm (forward 7.1, backward 6.5, sideways 6.7).
     - Training falls rose about 20%, all of it on heightfield tiles (slopes, random rough) and none on flat or soft tiles.
     - Reason: the policy is blind, so the higher swing is used everywhere, but it is only rewarded on flat and soft tiles.
   - **At 67,041 the whole batch collapsed** (every robot falling).
     - Cause: a few huge actions entered rsl_rl's EmpiricalNormalization, which never stops updating. They inflated the previous-action input scaling 35-fold.
     - Fix: the actor and critic input normalisers are now frozen on load (`freeze_obs_normalizer_on_load` in runner.py and rl_cfg.py, on for PoultrySurface only).
   - **From 66k the policy had also started producing rare enormous raw actions:** the largest |a| was 89 at 66k, 331 at 67k and 1,024 at 68k, against at most 13.7 up to 65.5k.
     - Training now clips actions to +-15 (`clip_actions = 15.0` in `poultry_surface_ppo_runner_cfg`). That matches the real robot's runaway guard, which goes passive when |raw action| > 15 for 60 ms.
     - Training was restarted from 65k.
   - **The current run,** `2026-10-10_08-34-16_poultry_100k_lift1c`, is training to 100k right now and is monitored by another session. **Leave it alone.**
6. **Measured at 65k against Rough 50k,** with 0.6 kg per hand, on the softest training floor:

   | Litter depth | Rough 50k: share of swing time with the toe inside the litter | 65k |
   |---|---|---|
   | 2 cm | 87–100% | 3–17% |
   | 4 cm | 87–100% | 24–54% |
   | 6 cm | 76–100% | 42–87% |

   - **Falls:** 65k doesn't fall forward up to 10 cm of litter; Rough 50k falls forward from 8 cm.
   - **Soft floor:** neither policy falls even at twice the softest training floor.
   - **Terrain level:** the training level sits around 2.4 out of 9.

## Why these changes
- **Litter depth is tied to the terrain row.** `PoultrySurfaceAction.terrain_at()` sets depth = `max_layer_height * row / (num_rows - 1)` on soft columns, with `max_layer_height = 0.04`. With the terrain level around 2.4, the policy mostly trains in about 1 cm of litter. That's why it ploughs at 4–6 cm.
- **The clearance target is 9 cm above the hard floor, penalised in both directions** (`|lowest - 0.09|`), so lifting higher than 9 cm is punished. Meanwhile `poultry_plow` wants the sole at least 4 cm above the litter surface. Once litter deeper than about 5 cm is sampled, these two terms conflict.
- **40% of tiles are heightfield** (slopes, rough, waves), and that's where all the extra falls happen. The barn is flat with litter.

## Changes to make (PoultrySurface only)

### A. Litter depth sampled per episode, independent of the terrain row
In `src/tasks/duet/mdp/poultry_surface.py`:
- Add to `PoultrySurfaceActionCfg`:
  - `litter_depth_mode: str = "row"` ("row" is the current behaviour and stays the default);
  - `litter_depth_range: tuple[float, float] = (0.0, 0.04)`.
  - Validate both: the mode must be "row" or "random", and the range must satisfy 0 <= lo <= hi.
- In "random" mode, give each env its own depth, drawn uniformly from `litter_depth_range`:
  - sample it in `__init__`, and again in `reset()` for the env ids being reset;
  - in `terrain_at()`, use it on soft columns instead of the row formula;
  - flat and heightfield columns still get no litter, and `litter_enabled=False` still means zero;
  - `terrain_at()` receives points shaped [num_envs, feet, points, 3], so broadcast the per-env depth over that.
- "Row" mode must keep working exactly as it does now. The existing evaluation scripts set `max_layer_height` and rely on it.

In `src/tasks/duet/config/g1_23dof/poultry_surface.py`:
- For training, use `litter_depth_mode="random"` and `litter_depth_range=(0.0, 0.06)`. If the real barn depth is known, use that depth plus about 2 cm instead of 0.06.
- Make sure a fixed depth can be pinned for play and evaluation (a range of (d, d)).

### B. Lift target measured from the litter surface, with a kinematic cap
- Per-env target: `target = clamp(layer_depth + above_litter, min_target, max_target)`, with defaults above_litter=0.05, min_target=0.09, max_target=0.11.
  - With a level foot, the leg can't lift much above about 10 cm, and the current lift is about 7 cm.
  - Effect: litter from 0 to 4 cm keeps the 9 cm target; 6 cm of litter gives 11 cm.
- Use this target in `poultry_clearance` and `poultry_swing_height`; both currently use a constant `target_height=0.09`.
  - Keep the constant behaviour as the default when the new parameters aren't given.
  - PoultrySurface opts in to the new target.
- Leave these as they are:
  - the foot-speed weighting in `poultry_clearance`: it is the anti-scoop part, penalising the foot travelling horizontally while low;
  - `poultry_plow` (4 cm above the litter surface);
  - `poultry_foot_level`.

### C. Barn-weighted terrain mix (PoultrySurface only)
- **Current mix** (shared `DUET_TERRAINS_CFG`, 20 columns): flat 0.2, soft_floor 0.4, hf_pyramid_slope 0.1, hf_pyramid_slope_inv 0.1, random_rough 0.1, wave_terrain 0.1.
- **New mix for PoultrySurface:** flat 0.15, soft_floor 0.60, hf_pyramid_slope 0.05, hf_pyramid_slope_inv 0.05, random_rough 0.05, wave_terrain 0.10.
  - Heightfield drops from 40% to 25%.
  - Every value is a multiple of 0.05, so each sub-terrain maps to whole columns.
- **How to build it:**
  - make a copy of the generator and its sub-terrain configs (`dataclasses.replace`) inside the PoultrySurface env config;
  - do not edit `DUET_TERRAINS_CFG` or `_ROUGH`;
  - keep the existing flat-patch spawn settings on the heightfield tiles.
- **Check the masks:** `PoultrySurfaceAction`'s flat and soft column masks are derived from the proportions. Confirm they still match the columns the terrain generator actually builds.

## Must not change
- **Other tasks:** Rough, Flat and every other DUET task must resolve to identical configs. Verify by dumping the resolved env and agent configs before and after and diffing them.
- **Current PoultrySurface settings:**
  - the boost-1 weights (-3.0 and -0.12);
  - the backward cap, penalty and stop events;
  - the squat terms;
  - reward floor -5, `freeze_obs_normalizer_on_load=True`, `clip_actions=15.0`.
- **Interfaces:** the observation and action layouts (71 actor inputs, 283 critic inputs, 13 actions). Existing checkpoints and the ONNX deploy interface must stay compatible.
- **Deploy:** nothing under `deploy/`.

## Where to work
- **The in-progress run** uses `/home/wolftech/rbbist.lab/DUET_poultry_e063b5e`: branch `poultry-surface-100k`, commit e8836e7, plus uncommitted fixes in `src/tasks/duet/config/g1_23dof/poultry_surface.py`, `src/tasks/duet/config/g1_23dof/rl_cfg.py` and `src/tasks/duet/rl/runner.py`.
  - Do not edit files there, stop processes there, or write into its `logs/rsl_rl/...` run folders.
- **Create a new worktree and branch from that exact state:**
  ```bash
  git -C /home/wolftech/rbbist.lab/DUET_poultry_e063b5e worktree add -b poultry-surface-v2 /home/wolftech/rbbist.lab/DUET_poultry_v2 e8836e7
  git -C /home/wolftech/rbbist.lab/DUET_poultry_e063b5e diff -- src | git -C /home/wolftech/rbbist.lab/DUET_poultry_v2 apply
  ```
  Then confirm the three files are byte-identical to the originals.
- **Python:** use `/home/wolftech/rbbist.lab/miniconda3/envs/duet/bin/python`, run from the worktree root with `PYTHONPATH=.`.
- **Logging:** MuJoCo Warp prints about 100k "ccd_iterations" lines per iteration. Any training command must pipe through `python -u /home/wolftech/rbbist.lab/DUET_poultry_e063b5e/logs/monitor/poultry_compact_log.py OUT.log`, or the console log grows by about 10 GB an hour. A resume launcher that does this is at `/home/wolftech/rbbist.lab/DUET_poultry_e063b5e/logs/monitor/resume_poultry.sh`; adapt the paths for the new worktree.

## Verification (include every output in your report)
1. **Unit checks** (on CPU or a tiny GPU env):
   - in random mode, each env's depth falls inside the range and is re-drawn on reset;
   - soft columns get that env's depth, while flat and heightfield columns get 0;
   - row mode reproduces the old formula exactly;
   - `litter_enabled=False` gives 0;
   - the per-env target equals the clamp formula.
2. **Config audit:** the before/after diffs for every task except PoultrySurface are empty. For PoultrySurface, list every changed field.
3. **Terrain layout:** print the column-to-sub-terrain mapping and the flat and soft masks. Confirm 3 flat, 12 soft, 1 slope, 1 inverted slope, 1 random rough and 2 wave columns.
4. **Smoke run:** resume 64 envs for 3 iterations.
   - Copy (don't move) `/home/wolftech/rbbist.lab/DUET_poultry_e063b5e/logs/rsl_rl/DUET_G1_23dof_poultry/2026-10-10_02-19-39_poultry_100k_lift1/model_65000.pt` into a folder under the new worktree's `logs/rsl_rl/DUET_G1_23dof_poultry/`, and resume from it.
   - Confirm these startup lines: "per-step reward floor -5.0 installed", and "observation normalizer frozen" for both the actor and the critic.
   - Confirm the reward table shows the expected terms and weights, and that `params/agent.yaml` has `clip_actions: 15.0`.
   - Confirm `Episode_Metrics/poultry_layer_height_m` reflects the new depths: about 3 cm on soft tiles, about 1.8 cm averaged over all robots.
   - Delete the smoke run folder afterwards.
   - The GPU is shared with the training run, so keep smoke runs tiny and short.
5. **Row-mode compatibility:** `/home/wolftech/rbbist.lab/DUET_poultry_e063b5e/logs/monitor/eval/litter_sweep.py` uses row mode. Check that it still runs unchanged against the new worktree's code.

## Deliverables
- Don't commit or push; the work will be reviewed first.
- Report:
  - the files changed, with a short description of each change;
  - the verification outputs;
  - any deviations from this spec, and why;
  - a ready-to-use launch command for fine-tuning to 100k from a given checkpoint (don't launch it).
