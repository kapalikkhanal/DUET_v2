# PoultrySurface 100k: final report (2026-10-11)

## Recommendation
- **Deploy `exported/poultry_v2_90000`** (in the main checkout: `/home/wolftech/rbbist.lab/DUET/exported/poultry_v2_90000/policy.onnx`, plus `deploy_contract.json`).
  - It passes all 19 deploy checks against `deploy/robots/g1_23dof/config/policy/velocity/v1_squatreach_ft50000/params/deploy.yaml`. That file has the joint clip and the runaway guard.
  - Its config hash, 190203de148d70a7, matches the v1 deployment.
  - The ONNX file reproduces the checkpoint on the lift test.
- **Backup: `exported/poultry_v2_92000`.** It had 0 extreme-action events in 6 million test steps, but is a little weaker in litter and squats.
- **Do not deploy 93000-100000.** See Safety.

## How it improved

| test (flat ground unless noted; 0.6 kg per hand) | Rough 50k | 65k | **90k** |
|---|---|---|---|
| Foot lift (lowest sole point at swing apex), fwd / back / side | 2.5 / 1.6 / 1.4 cm | 7.1 / 6.5 / 6.7 | **8.1 / 7.9 / 7.9** |
| Foot skimming under 2 cm, per metre walked | 90-200 cm | 39-63 | **31-36** |
| Toe inside the litter during swing, 4 cm deep (fwd / back / side) | 87-100% | 51 / 24 / 54% | **37 / 26 / 37%** |
| Toe inside the litter during swing, 6 cm deep | 76-100% | 87 / 42 / 86% | **69 / 41 / 60%** |
| Falls in up to 8 cm of litter | forward 3 per min at 8 cm | backward 0.2 per min from 6 cm | **0** |
| Forward speed in up to 6 cm of litter | 87-101% | 76-95% | **90-93%** |
| Backward speed reached at -0.5 / -0.3 command | 76 / 53% | 90 / 97% | **106 / 117%** |
| Centre of mass behind the feet, walking backward | 7.7 / 6.1 cm | 4.8 / 4.1 | **3.5 / 3.3** |
| Torso while walking backward (absolute, + = forward) | +2.2 deg | +6.7 | +5.4 to +5.8 |
| Squat, arms forward: worst centre-of-mass offset (0.24-0.60 m) | 2.6 cm | 1.0 | 1.1 |
| Squat, arms wide: worst centre-of-mass offset | 2.1 cm | 1.9 | **1.2** |
| Soft floor, up to 2x the softest training floor | no falls | no falls | no falls |

- **Still weak:** sideways tracking (70-79% of a 0.3 m/s command in litter or on soft floor); a 0.24 m squat command bottoms out at 0.31 m.

## Safety: runaway actions (why the last checkpoints are excluded)
- **What happens:** the policy's raw outputs are unbounded. In rare states it outputs huge actions (hundreds to tens of thousands). The real robot's guard goes passive when |a| > 15 for 60 ms.
- **First form, 66k-85k:** one ankle pushing its target ever further past a limit it was already at. Addressed with `action_excess_l2` and by raising the training clip to 30 so the penalty can see the excess.
- **Second form, 93k-100k:** every joint goes extreme at once, mostly once the robot is already strongly tilted. Events per 6 million test steps: 90k 2, 91k 1, 92k 0, 94k about 22, 96k 20, 98k 45, 100k 13. From 98k, some events start before the robot is tilted.
- **Why training doesn't see it:** the training clips (actions at 30, previous-action input at 15) hide it, so reward and falls stay excellent.
- **90k's two events** both began with the robot tilted past 30 deg, and both robots fell within 0.5 s anyway.

## What changed (worktree `/home/wolftech/rbbist.lab/DUET_poultry_v2`, branch `poultry-surface-v2`, all uncommitted)
- **v2, from the other agent:**
  - litter depth drawn at random from 0-6 cm per episode on soft tiles;
  - lift target = clamp(litter + 5 cm, 9, 11 cm);
  - terrain mix 15% flat / 60% soft / 25% heightfield.
- **Kept from the 100k run:**
  - boost 1;
  - observation normalisers frozen on load (fixes the 67k collapse);
  - reward floor -5.
- **Added overnight:**
  - `action_excess_l2` (raw |a| above 8, weight -0.2);
  - training `clip_actions` 30;
  - previous-action observation clipped to +-15;
  - boost 2 (clearance -4.0, swing height -0.20; from 84k).
- **Training path:** Rough 50k -> 60k (boost 1) -> 65k -> v2 to 74.5k -> v2b to 84k -> v2c (boost 2) to 100k.

## Suggested next steps
1. **Fix runaways at the source:** penalise the raw policy mean inside the PPO loss (a bounds loss on |mean| > ~10). Rewards only see clipped actions, so a reward term can't fully do it. Then re-check with `logs/monitor/eval/action_extremes.py`. The goal is 0 events.
2. **Sideways tracking.**
3. **Measure the real barn litter depth** and set `litter_depth_range` to match.
4. **Commit `poultry-surface-v2`** once reviewed.

## Files
- **Test results:** `/home/wolftech/rbbist.lab/DUET_poultry_e063b5e/logs/poultry_100k_evals.md`.
- **Progress tables:** `/home/wolftech/rbbist.lab/DUET_poultry_v2/logs/poultry_v2_progress.md` and `/home/wolftech/rbbist.lab/DUET_poultry_e063b5e/logs/poultry_100k_progress.md`.
- **Plan and decision log:** `/home/wolftech/rbbist.lab/DUET_poultry_v2/logs/poultry_v2_plan.md`.
- **Test scripts:** `/home/wolftech/rbbist.lab/DUET_poultry_e063b5e/logs/monitor/eval/`.
