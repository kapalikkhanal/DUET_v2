# PoultrySurface 100k: checkpoint evaluations

All tests: flat ground, 0.6 kg per hand, no pushes, deterministic policy. Foot lift = lowest point of the foot at the top of each swing; skim = horizontal travel with the foot under 2 cm, per metre walked.

## Foot lift (walking +0.4 / -0.4 / sideways 0.3 m/s)

| checkpoint | fwd | back | side | mean | skim fwd / back / side (cm per m) |
|---|---|---|---|---|---|
| Rough 50k (start) | 2.5 | 1.6 | 1.4 | 1.8 | 90-200 |
| 51000 | 5.4 | 4.1 | 3.6 | 4.37 | 43 / 55 / 66 |
| 52000 | 5.9 | 3.9 | 3.8 | 4.53 | 41 / 55 / 74 |
| 53000 | 5.7 | 4.3 | 3.9 | 4.63 | 33 / 50 / 69 |
| 54500 | 6.1 | 5.1 | 4.7 | 5.30 | 40 / 40 / 51 |
| 55000 | 6.3 | 4.8 | 4.4 | 5.17 | 39 / 49 / 63 |
| 59500 | 6.2 | 3.7 | 4.3 | 4.73 | 37 / 49 / 61 |
| 60000 | 5.5 | 3.9 | 4.7 | 4.70 | 41 / 60 / 55 |

**60k decision:** mean of 59.5k/60k = 4.72 cm vs 5.23 cm at 54.5k/55k (no gain in 5k) -> boost 1 applied, resumed from model_60000.pt (poultry_clearance -2.0 -> -3.0, poultry_swing_height -0.06 -> -0.12).

| checkpoint (boost 1) | fwd | back | side | mean | skim fwd / back / side (cm per m) |
|---|---|---|---|---|---|
| 64500 | 7.0 | 6.1 | 6.2 | 6.43 | 43 / 48 / 51 |
| 65000 | 7.1 | 6.5 | 6.7 | 6.77 | 39 / 44 / 47 |

**65k:** mean 6.60 cm vs 4.72 cm at 60k (+1.9 cm in 5k) -> boost 1 kept; no revert criterion hit.


## Backward walking (speed reached; centre of mass relative to the feet, minus = behind)

| checkpoint | -0.5 cmd | -0.3 cmd | CoM at -0.5 / -0.3 | lean after stop |
|---|---|---|---|---|
| Rough 50k | 0.38 (76%) | 0.16 (53%) | -7.7 / -6.1 cm | -1.1 / -1.3 deg |
| 55000 | 0.43 (86%) | 0.29 (97%) | -5.8 / -4.1 cm | -0.2 / -0.2 deg |
| 60000 | 0.45 (90%) | 0.28 (93%) | -5.1 / -3.7 cm | -1.0 / -1.1 deg |
| 65000 (boost 1) | 0.45 (90%) | 0.29 (97%) | -4.8 / -4.1 cm | -1.0 / -1.0 deg |

## Wide-arm squats (CoM fore-aft with arms out at 0.45 / 0.50 / 0.55 / 0.60 m; pelvis drift)

| checkpoint | CoM (cm) | pelvis drift (cm) | falls |
|---|---|---|---|
| Rough 50k | -2.0 / -1.7 / -0.8 / -0.1 | -1.1 to -1.2 | 0 |
| 55000 | -1.7 / -0.9 / -0.3 / +0.2 | -1.7 to -2.1 | 0 |
| 60000 | -1.2 / -1.6 / -1.1 / -1.1 | -1.5 to -2.0 | 0 |
| 65000 (boost 1) | -1.0 / -1.1 / -0.6 / +0.1 | -1.5 to -2.4 | 0 |

## Deploy-safety: extreme actions (training conditions, 2048 robots x 1450 steps, with policy noise)

The robot's runaway guard goes passive when |raw action| > 15 for 60 ms. A checkpoint must show 0 here to be deployable.

| checkpoint | max abs action | per million steps above 15 | above 100 |
|---|---|---|---|
| 60000 | 12.1 | 0 | 0 |
| 65000 | 12.4 | 0 | 0 |
| 65500 | 13.7 | 0 | 0 |
| 66000 | 88.7 | 1.3 | 0 |
| 66500 | 23.3 | 0.7 | 0 |
| 67000 | 331 | 9.4 | 2.4 |
| 68000 (lift1b) | 1024 | 19.2 | 9.4 |

66000 onward (run lift1) and all of lift1b are NOT deployable. Training resumed from 65000 with actions clipped to +-15.

## 2026-10-10 09:20: switch to PoultrySurface v2

Candidate checkpoint 66000 from run lift1c (resumed from 65000 with actions clipped):
- Extreme actions: max 17.8; 6 above 15 (2.0 per million steps). **Fails the deploy-safety check.** Clipping keeps training stable but doesn't stop the raw outputs from creeping up after about 65.5k.
- Backward: 96% / 97% of command; centre of mass -5.2 / -4.0 cm. Forward 0.33 / 0.50. After a stop: travel 9.0 / 2.0 cm. Slightly better than 65000.
- Wide-arm squat centre of mass (0.30 / 0.45 / 0.50 / 0.55 / 0.60 m): -1.1 / -0.7 / -0.8 / -0.9 / -1.0 cm.
- Forward-arm squat centre of mass (0.24 / 0.30 / 0.40 / 0.50 / 0.60 m): -1.0 / -1.2 / -0.5 / -1.0 / -0.9 cm. Worse than 65000.
- Decision: not better than 65000, so v2 starts from 65000.

Deleted (user request), from DUET_poultry_e063b5e/logs/rsl_rl/DUET_G1_23dof_poultry:
- run lift1b, all of it (collapse and extreme-action run);
- run lift1c, all of it (superseded; its 66000 failed the safety check);
- lift1 model_66000-67500 and its policy.onnx (exported at the corrupted 67500).

v2 run: /home/wolftech/rbbist.lab/DUET_poultry_v2/logs/rsl_rl/DUET_G1_23dof_poultry/2026-10-10_09-23-55_poultry_v2. Its table is DUET_poultry_v2/logs/poultry_v2_progress.md.

The safety test (`logs/monitor/eval/action_extremes.py`) now also counts GUARD TRIPS: |a| > 15 for 3 consecutive steps (60 ms), which is what makes the robot go passive.

## v2 checks

| v2 checkpoint | max abs action | above 15 | guard trips | notes |
|---|---|---|---|---|
| 66000 | 11.5 | 0 | 0 | clean (65000 was 12.4); table row: reward 81.7, falls 0.30, value loss 0.11 |
| 70000 | 12.9 | 0 | 0 | clean |

### v2 at 70000 vs 65000 (same tests)
- **Foot lift on flat ground:** 69500 7.9 / 6.9 / 7.2, 70000 8.0 / 6.7 / 7.2. Mean 7.32 cm vs 6.60 cm at 65k (+0.7). Skim 36-49 cm per m.
- **Litter, toe inside the litter during swing (fwd / back / side):**

  | depth | 65k | 70k |
  |---|---|---|
  | 2 cm | 16 / 3 / 17% | 14 / 5 / 10% |
  | 4 cm | 51 / 24 / 54% | 38 / 34 / 43% |
  | 6 cm | 87 / 42 / 86% | 68 / 54 / 79% |
  | 8 cm | 99 / 50 / 96% | 97 / 70 / 95% |

  - Forward speed in 4-8 cm of litter: 94% (65k: 71-88%). Backward 81-88% (65k: 95-126%).
  - No falls up to 8 cm in any direction; no backward falls up to 10 cm (65k: 0.2-0.4 per minute from 6 cm).
- **Soft floor:** no falls. Lift 6.8 cm at the softest training floor (65k: 5.7). Backward 93-104% at every softness.
- **Backward walking:** -0.5 command -> 0.48 (96%), -0.3 -> 0.30 (100%). Centre of mass -5.5 / -4.5 cm (65k: -4.8 / -4.1). Torso lean -1.9 deg (65k: -0.6). After a stop: travel 13.0 / 3.7 cm.
- **Squats, forward arms** (centre of mass, 0.24 / 0.30 / 0.40 / 0.50 / 0.60 m): -1.2 / -0.4 / -0.6 / -0.8 / -0.5 cm (65k: -1.0 / +0.2 / -0.2 / -0.3 / +0.4).
- **Squats, wide arms:** -1.1 / -0.9 / -0.3 / -0.2 / -1.0 cm (65k: -1.9 / -0.8 / -0.8 / -0.9 / -0.4). The torso straightens only 0.6-2.5 deg when the arms go wide (65k: 3-5.5). No falls.
- **Safety:** max action 12.9, 0 guard trips.
- **Verdict:** better than 65k on lift, litter and soft floor. Backward lean and centre of mass about 0.5 cm worse. Squats about the same. Continue; 70000 is the new best deployable candidate.

| v2 checkpoint | max abs action | above 15 | guard trips | notes |
|---|---|---|---|---|
| 71000 | 11.3 | 0 | 0 | |
| 72000 | 12.7 | 0 | 0 | |
| 73000 | 12.4 | 0 | 0 | |
| 74000 | 12.0 | 0 | 0 | |
| 74500 | 11.4 | 0 | 0 | backward 102 / 107%, centre of mass -3.4 / -3.2 cm, after-stop travel 7.3 / 2.3 cm (best so far); lift 7.3 / 6.7 / 6.7 |
| 75000 | 17.4 | 12 | 1 | FAIL (left ankle pitch 10 times, right knee 2). Also: backward falls 0.9-8.8 per min in 8-12 cm of litter; forward 87%; torso lean back -2.7 to -6.2 deg. Deleted. |

75000 failed, so `action_excess_l2` (|a| > 8, weight -0.05) was added and training resumed from 74500 (run v2b).

| v2b checkpoint | max abs action | above 15 | guard trips | notes |
|---|---|---|---|---|
| 75000 | 13.8 | 0 | 0 | the same iteration in v2 (no penalty) was 17.4 / 12 / 1 |
| 75500 | 12.7 | 0 | 0 | above 10: 3 per million steps |

The penalty is active but tiny on average (about -1e-5 per second): it only touches the rare tail, as designed.
| 78000 | 11.0 | 0 | 0 | |
| 79000 | 11.3 | 0 | 0 | |
| 80000 | 13.2 | 0 | 0 | |

### v2b at 80000 vs v2 at 70000
- **Foot lift on flat ground:** 79500 7.9 / 7.9 / 7.5, 80000 8.1 / 6.4 / 7.5. Mean 7.55 cm vs 7.32 (+0.2).
- **Litter, toe inside the litter during swing (fwd / back / side):**

  | depth | 70k | 80k |
  |---|---|---|
  | 2 cm | 14 / 5 / 10% | 11 / 14 / 18% |
  | 4 cm | 38 / 34 / 43% | 38 / 40 / 44% |
  | 6 cm | 68 / 54 / 79% | 62 / 60 / 67% |
  | 8 cm | 97 / 70 / 95% | 95 / 67 / 87% |

  - No falls up to 8 cm. Forward speed 91-98% up to 6 cm. Sideways 62-72% (70k: 77-83%).
- **Soft floor:** no falls. Forward 92-98% up to the softest training floor. Sideways 64-69% (70k: 75-81%).
- **Backward walking:** 94% / 93% of command. Centre of mass **-2.0 / -2.8 cm (best so far**; 70k: -5.5 / -4.5). Torso lean -2.6 / -3.5 deg. After a stop: peak back lean -4.4 / -4.1 deg (70k: -2.0).
- **Forward walking:** torso leans back -7.3 / -8.0 deg relative to standing. This has grown since 70k (65k -3.7 / -5.0, 70k -3.1 / -3.2, 75k -5.7 / -6.2). WATCH.
- **Squats, forward arms** (centre of mass): -0.5 / -0.5 / -0.4 / -0.9 / -0.8 cm (best so far). Wide arms: -2.0 / -0.8 / +0.3 / -0.6 / +0.1 cm. No falls. A 0.24 command reaches 0.29 m.
- **Verdict:** about even with 70k overall. Better backward centre of mass and squats; worse sideways tracking and more back lean while walking forward. Continue without changes; watch forward lean and sideways at 85k.

### v2b at 85000 (and 84000-84500)
- **Lift:** 84500 7.6 / 6.3 / 6.5, 85000 7.8 / 7.6 / 6.9. Mean 7.12 cm (80k: 7.55), so flat.
- **Safety:**
  - 84000: max 11.1, clean;
  - 84500: max 15.8, 3 above 15, no trip;
  - 85000: max 17.8, 4 above 15, **1 guard trip** (left ankle pitch). FAIL; deleted.
- **Backward (85000):** 114% / 130% (overshoot). Centre of mass -2.4 / -2.9 cm. Lean +0.5 / -0.2 deg relative to standing; standing torso pitch +6.4 deg absolute.
- **Squats (85000):** forward arms -0.0 / -0.1 / +0.2 / -0.5 / -0.9 cm (best yet); wide arms -1.1 / -0.5 / +0.1 / -0.7 / +0.3 cm.
- **Litter, 4 cm deep:** 43 / 27 / 47%; 6 cm: 72 / 43 / 78%. Forward speed 80-84%.
- **Torso, absolute** (+ = forward), walking at 0.3 / 0.5 m/s: 50k +1.8 / +1.1, 65k +3.9 / +2.5, 70k +2.3 / +2.2, 80k +3.7 / +2.6 deg. The torso stays slightly forward while walking forward; only the standing posture leans further forward (80k +10.7). Not a problem.
- **Then:** resumed from 84000 with boost 2 and a working action penalty (clip 30, input clip 15, weight -0.2). Run v2c.

| v2c checkpoint | max abs action | above 15 | guard trips | notes |
|---|---|---|---|---|
| 85000 | 11.8 | 0 | 0 | |
| 86000 | 16.4 | 2 | 0 | right ankle pitch; above 10 rose to 45 per million steps (boost 2 pushes the ankle). Watch at 87k. |
| 86500 | 12.8 | 0 | 0 | above 10: 13.5 per million |
| 87000 | 11.3 | 0 | 0 | above 10: 5.7 per million. The penalty pulled the tail back after the 86k bump; boost 2 kept. |
| 88000 | 12.6 | 0 | 0 | |
| 89000 | 12.8 | 0 | 0 | |
| 89500 | 14.1 | 0 | 0 | |
| 90000 | 11.7 | 0 | 0 | |

### v2c at 90000 (boost 2 since 84k) vs 80k / 85k
- **Foot lift on flat ground:** 89500 8.6 / 7.3 / 8.0, 90000 8.1 / 7.9 / 7.9. Mean **7.97 cm** (85k 7.12, 80k 7.55). Skim 31-36 cm per m at 90000 (best so far).
- **Litter, toe inside the litter during swing (fwd / back / side):**

  | depth | 80k | 90k |
  |---|---|---|
  | 2 cm | 11 / 14 / 18% | 13 / 5 / 10% |
  | 4 cm | 38 / 40 / 44% | **37 / 26 / 37%** |
  | 6 cm | 62 / 60 / 67% | 69 / 41 / 60% |
  | 8 cm | 95 / 67 / 87% | 99 / 48 / 83% |

  - No falls up to 8 cm. Speed in up to 6 cm of litter: forward 90-93%, backward 101-102%, sideways 70-79%.
- **Soft floor:** no falls. Forward 89-96%, backward 109-115%, sideways 73-76%.
- **Backward walking:** 106% / 117% of command (slight overshoot). Centre of mass -3.5 / -3.3 cm (80k -2.0 / -2.8, 70k -5.5 / -4.5). After a stop: travel 10.8 / 4.1 cm, settle 0.50 / 0.35 s. Absolute torso: standing +8.0 deg; walking backward +5.4-5.8 deg.
- **Squats, forward arms** (centre of mass): -0.2 / -0.6 / -1.1 / -0.7 / -0.8 cm. Wide arms: -1.1 / -0.4 / -0.8 / -1.2 / -0.6 cm. No falls. A 0.24 command reaches 0.31 m (80k: 0.29).
- **Safety:** 88k-90k max 11.7-14.1, 0 above 15, 0 guard trips.
- **Verdict:** boost 2 works (+0.85 cm lift, better 4 cm clearance) and is kept. The backward centre of mass at -0.5 is 1.1 cm behind 85k, but within the roughly 1 cm checkpoint-to-checkpoint scatter (74.5k -3.4, 75k -4.9, 80k -2.0, 85k -2.4) and still better than 70k, so no revert. 90000 is the new best overall candidate.
| 91000 | 14.9 | 0 | 0 | |
| 92000 | 14.9 | 0 | 0 | |
| 93000 | 287 | 20 | 1 | ALL leg joints at once (runaway mode, not the ankle drift) |
| 94000 | 1265 / 1241 | 142 / 130 | 8 / 8 | all joints; 11 events, 9 began with torso tilt above 30 deg, 8 fell within 0.5 s |
| 95000 | 60 / 3786 | 2 / 370 | 0 / 7 | two runs: very noisy (a few events cause many extreme steps); 19 events, 17 began with tilt above 30 deg |

### v2c at 95000
- **Foot lift on flat ground:** 94500 8.8 / 8.1 / 8.0, 95000 8.6 / 8.6 / 7.9. Mean **8.33 cm (best so far)**. Skim 32-42 cm per m.
- **Litter, toe inside (fwd / back / side):** 4 cm 39 / 27 / 39%; 6 cm 61 / 43 / 63%. Speed in up to 6 cm: forward 76-78%, backward 106-115%, sideways 73-76%. 10 cm: 2.3 forward falls per minute.
- **Backward walking:** 102% / 117% of command. Centre of mass -4.4 / -4.1 cm (90k -3.5 / -3.3). Absolute torso walking backward +5.4-5.9 deg.
- **Safety:** runaway events at 93k-95k, mostly once the robot is already falling (tilt above 30 deg at onset). Not deployable under the strict rule.
- **Note:** the training clips (30 for actions, 15 for the previous-action input) hide these events from training, so the table rows stay normal.

### Longer safety test (2048 robots x 2950 steps = 6.0 million steps, training conditions with pushes, with policy noise)
An "event" is the start of an extreme-action episode (|a| > 15); tilt is the torso angle at onset.

| checkpoint | events | tilt above 30 deg at onset | fell within 0.5 s | max abs action | guard trips |
|---|---|---|---|---|---|
| 90000 | 2 | 2 | 2 | 1042 | 2 |
| 91000 | 1 | 1 | 1 | 15.5 | 0 |
| **92000** | **0** | 0 | 0 | **13.2** | **0** |
| 94000 (3 million steps) | 11 | 9 | 8 | 1241 | 8 |
| 95000 (3 million steps) | 19 | 17 | 7 | 3786 | 7 |

- The runaways start almost only once the robot is already strongly tilted.
- But at 95k, 12 of 19 such robots did not fall within 0.5 s in simulation (the clipped actions let them recover). On the real robot the guard would go passive and the robot would fall.
- So events count as safety failures. 92000 is the cleanest checkpoint.
| 96000 | 20 | 19 | 19 | 5544 | 10 |

### 92000 full test (vs 90000)
- **Lift:** 92000 8.6 / 7.0 / 7.9, 92500 8.5 / 7.0 / 7.7. Mean 7.78 cm (90k 7.97). Skim 25-37 cm per m (best).
- **Litter, toe inside (fwd / back / side):** 4 cm 36 / 29 / 41%; 6 cm 62 / 46 / 62%.
  - Forward speed in litter 76-82% (90k 90-93%).
  - Backward in 8 cm of litter: 0.9 falls per minute at 150% speed (90k: 0 falls, 120%).
- **Soft floor:** forward 73% at the softest training floor (90k 89%).
- **Backward walking:** 104 / 110% of command. Centre of mass -4.4 / -3.8 cm (90k -3.5 / -3.3).
- **Squats, centre of mass:** forward arms -0.8 / -1.6 / -0.5 / -1.2 / -1.3 cm; wide -1.8 / -1.5 / -0.7 / -1.3 / -0.4 cm. Slightly worse than 90k. No falls.
- **Export:** `/home/wolftech/rbbist.lab/DUET/exported/poultry_v2_92000/` (policy.onnx and deploy_contract.json). 19 / 19 deploy checks pass against v1 deploy.yaml; hash 190203de148d70a7.
- **Export of 90000:** `/home/wolftech/rbbist.lab/DUET/exported/poultry_v2_90000/`; 19 / 19 deploy checks pass; hash 190203de148d70a7. The exported ONNX reproduces the checkpoint on the lift test (8.1 / 7.7 / 7.9 cm vs 8.1 / 7.9 / 7.9).
| 98000 | 45 | 34 | 43 | 35913 | 40 |

The runaway mode grows quickly after 92k: 0 events at 92k, 11 at 94k (3 million steps), 20 at 96k, 45 at 98k. At 98k, 11 events began with tilt below 30 deg. Not deployable. The training clips hide it, so reward and falls still look excellent.
| 100000 | 13 | 7 | 13 | 6658 | 9 |

**Final:** not deployable from 93k onward. Recommended: 90000 (exported). Backup: 92000 (exported, 0 events).

## Deeper-squat fine-tune (v3_squat, from 90000; floor 0.22 m, 97% band on the squat joints)

### 92000 (floor fully at 0.22 from here)
- **Depth reached** (command -> reached; 90k in brackets): 0.30 -> 0.289 (0.325); 0.24 -> **0.256** (0.309); 0.22 -> **0.251**; 0.20 (untrained) -> 0.247 (0.299).
  - Joints at 0.22: hip pitch -2.42 (97% limit -2.45), knee +2.74 (97% 2.84), ankle pitch -0.76.
  - Torso +25 deg. Centre of mass -1.6 cm. No falls, no self-collision.
- **Lift:** 8.2 / 7.8 / 7.6 cm (90k 8.1 / 7.9 / 7.9).
- **Backward walking:** 90 / 90% of command; centre of mass -3.4 / -3.1 cm (90k -3.5 / -3.3). Forward 0.27 / 0.41 (90k 0.30 / 0.46).
- **Wide-arm squat** centre of mass at 0.22 / 0.30 / 0.40 / 0.50 / 0.60 m: -1.2 / +0.1 / -0.5 / -0.3 / -0.1 cm. No falls.
- **Safety (6 million steps):** max action 13.3, 0 events, 0 trips.

### v3_squat checkpoints 91000-92500 (vs 90000)

| ckpt | reached at 0.24 / 0.22 cmd | lift fwd / back / side | backward -0.5: speed, centre of mass | forward 0.3 / 0.5 | safety (6 million steps) |
|---|---|---|---|---|---|
| 90000 | 0.309 / - | 8.1 / 7.9 / 7.9 | 106%, -3.5 cm | 0.30 / 0.46 | 2 events (both during falls) |
| 91000 | 0.261 / 0.257 | 8.0 / 7.2 / 7.4 | 82%, -2.2 cm | 0.26 / 0.42 | 1 event (not tilted), max 15.6, 0 trips |
| **91500** | 0.257 / **0.251** | **8.7 / 7.2 / 7.7** | **104%, -3.1 cm** | **0.29 / 0.47** | **0 events, max 14.5** |
| 92000 | 0.256 / 0.251 | 8.2 / 7.8 / 7.6 | 90%, -3.4 cm | 0.27 / 0.41 | 0 events, max 13.3 |
| 92500 | 0.261 / 0.256 | 8.1 / 7.7 / 7.3 | 90%, -5.1 cm | 0.26 / 0.43 | 0 events, max 13.0 |

- **91500, litter, toe inside (fwd / back / side):** 4 cm 36 / 27 / 38% (90k 37 / 26 / 37); 6 cm 65 / 50 / 65% (90k 69 / 41 / 60).
  - Forward speed 93-94% up to 8 cm (90k 90-93%). No falls up to 8 cm.
  - 10-12 cm (beyond the 6 cm trained): 3-11 forward falls per minute (90k 0.4-1.4).
- **91500 wide-arm squat** centre of mass at 0.22-0.60 m: within 1.3 cm. No falls.
- **Best of 90-92.5k: 91500.**
| 93000 | 0.265 / 0.262 | **9.2 / 7.4 / 8.0** | 86%, -3.3 cm (-0.3: 83%, -1.7) | 0.30 / 0.45 | 2 events (both tilted, both fell), max 16.5, 0 trips |

- **93000, litter, toe inside (fwd / back / side):** 4 cm **25 / 20 / 31%** (best so far); 6 cm 51 / 44 / 57%. No forward falls up to 10 cm.
- **93000 wide-arm squat:** within 1.0 cm.
- **Verdict:** 93000 has the best litter clearance and lift, but squats 1 cm shallower than 91500, and extreme actions are starting to appear again (2 events during falls). **91500** stays the pick for squat depth plus a clean safety record.

### Exports (2026-10-12; training stopped at 93430)
- **Exports:** `/home/wolftech/rbbist.lab/DUET/exported/poultry_v3_squat_91500/` and `.../poultry_v3_squat_93000/`. Each has policy.onnx, deploy_contract.json and params/deploy.yaml.
- **deploy.yaml:** a copy of v1_squatreach_ft50000's with only `base_height.range` changed, [0.24, 0.78] -> [0.22, 0.78] (joint clip, safety guard and standstill 0.78 unchanged).
- **Checks:** 19 / 19 pass against it; hash 190203de148d70a7.
- **ONNX check (91500):** 0.253 m at a 0.22 command (checkpoint 0.251).
