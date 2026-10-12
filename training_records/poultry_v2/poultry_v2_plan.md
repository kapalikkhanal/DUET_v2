# PoultrySurface v2 run (started 2026-10-10 09:23)

- **Code:** worktree `/home/wolftech/rbbist.lab/DUET_poultry_v2`, branch poultry-surface-v2 (uncommitted). It is e8836e7 plus:
  - boost 1 (clearance -3.0, swing height -0.12);
  - frozen observation normalisers;
  - action clipping at 15;
  - v2 changes: random litter depth 0-6 cm per episode on soft tiles; lift target = clamp(litter + 5 cm, 9, 11 cm); terrain mix 15% flat / 60% soft / 25% slopes, rough and waves.
- **Start point:** model_65000.pt from run poultry_100k_lift1, the best checkpoint so far (lift 6.6 cm, backward 90-97%, clean actions).
- **Run:** `logs/rsl_rl/DUET_G1_23dof_poultry/2026-10-10_09-23-55_poultry_v2`; trainer PID 2059223; log `logs/poultry_v2.log`; table `logs/poultry_v2_progress.md`.

## Checks
- **Every 1k (table row):** value-loss spikes (alert above 5), action noise (alert at or above 0.26), falls, reward.
- **Every 5k (70k, 75k, ...):** run `/home/wolftech/rbbist.lab/DUET_poultry_e063b5e/logs/monitor/eval/`:
  - `gait_clearance.py`, `backward_test.py`, `squat_arm_test.py` (forward and wide), `action_extremes.py` (needs 0 guard trips to be deployable);
  - at 10k marks also `litter_sweep.py --sweep litter`.
  - Compare against 65000.
- **Collapse or degradation:** stop, then resume from the last checkpoint that passes the safety check.
- **If guard trips appear across checkpoints:** consider a penalty on raw action size beyond about 12. That needs the raw actions in the env: clip in the action term rather than the wrapper, and clip the last-action observation.

## Log
- 2026-10-10 16:00, 75k: first bad checkpoint (max action 17.4, 1 guard trip, mostly left ankle pitch). Backward walking in 8-12 cm of litter also became unstable. 71k-74.5k all clean.
  - Added `action_excess_l2`: raw |a| beyond 8, squared, weight -0.05.
  - Resumed from model_74500.pt at 16:17 as `2026-10-10_16-17-03_poultry_v2b`. Trainer PID 2106219. model_75000.pt deleted.
  - 74500 backward test: 102 / 107% of the command; centre of mass -3.4 / -3.2 cm (best so far).

## Overnight rules (user authorised autonomous decisions, 2026-10-10 21:20)
- **Torso check (done):** measured absolutely, the torso leans slightly forward while walking in every checkpoint (+2.6 to +3.7 deg walking forward). The "back lean" was relative to standing, and standing now leans +10.7 deg forward from the centre-of-mass centring. Walking backward leans +7-8 deg forward. No posture fix needed.
- **85k check:** foot lift (84500, 85000), safety (84000-85000), litter sweep, backward, squats.
  - **Boost 2** if lift gained less than 0.3 cm over 80k (7.55 cm) and safety passes:
    - `poultry_clearance` -3.0 -> -4.0;
    - `poultry_swing_height` -0.12 -> -0.20;
    - resume from 85000 (or the newest safe checkpoint) as run poultry_v2c.
  - Otherwise continue and re-check at 90k.
- **After a boost:** safety check at +2.5k; full check at +5k.
  - Revert to the old weights, resuming from the newest safe pre-boost checkpoint, if:
    - falls in the table stay at or above 0.45 for two rows;
    - lift and 4 cm litter clearance haven't improved;
    - backward or squat centre of mass is more than 1 cm worse.
  - On any guard trip, resume from the newest safe checkpoint.
- **End (100k):**
  - rank candidates (70k, 80k, 85k, 90k, 95k, 100k) with the full suite; the safety check must pass;
  - export the best with `scripts/export_duet_onnx.py --task Unitree-G1-23Dof-Duet-PoultrySurface --deploy-yaml /home/wolftech/rbbist.lab/DUET/deploy/robots/g1_23dof/config/policy/velocity/v1_squatreach_ft50000/params/deploy.yaml`, into `/home/wolftech/rbbist.lab/DUET/exported/poultry_v2_<iter>/`;
  - write the morning summary.
- 2026-10-10 23:25, 85k:
  - Lift flat (84500 6.8, 85000 7.4; mean 7.1 vs 7.55 at 80k).
  - 85000 failed the safety check: max action 17.8, 1 guard trip, left ankle pitch. 84500 reached 15.8 with no trip; 84000 is clean.
  - Why the penalty was weak: the wrapper's 15 clip hid everything above 15 from it.
  - Changes:
    - `clip_actions` 15 -> 30;
    - last-action observation clipped to +-15 (actor and critic);
    - `action_excess_l2` -0.05 -> -0.2;
    - boost 2: clearance -3 -> -4, swing height -0.12 -> -0.20.
  - Resumed from model_84000.pt as `2026-10-10_23-25-42_poultry_v2c`. Trainer PID 2152469.
  - Next: safety check at 86.5k; full check at 89k-90k.
  - If the safety check fails: revert boost 2, keep the penalty fix, and resume from the newest safe checkpoint.
  - If lift and litter clearance don't improve by 90k: revert boost 2.
- 2026-10-11 03:45, 90k:
  - Boost 2 works: lift 7.97 cm (85k 7.12); 4 cm litter 37 / 26 / 37% (80k 38 / 40 / 44%); safety clean from 86.5k to 90k.
  - Kept. 90000 is the new best candidate.
  - Next: 95k (safety, lift, litter), then the final ranking and export at 100k.

## Deeper squat fine-tune (2026-10-12 10:16, user request: squat floor 0.22 m, 97% joint band, nothing else changed)
- **Cause found:** at 90% soft limits, hip pitch (-2.26) and ankle pitch (-0.80) hold a flat-footed, balanced squat at 0.289 m (kinematic floor; the 90k policy reached 0.288-0.309).
  - At 97%: 0.224 m. At the hard limits: 0.206 m. Heels-up squats are not lower.
- **Changes** (PoultrySurface only; Rough and other tasks unchanged):
  - `joint_pos_limits` -> `joint_pos_limits_widened` (same -10 weight; hip pitch, knee and ankle pitch use 97% of range; every other joint keeps 90%; the entity soft limits that bound arm targets and reset sampling are untouched);
  - squat floor 0.24 -> 0.22 m, ramped over iterations 90k-92k; `deep_squat_range` (0.22, 0.65).
- **Run:** from v2c model_90000.pt to 98000, as `2026-10-12_10-16-27_poultry_v3_squat`. Trainer PID 2469374. Table `logs/poultry_v3_squat_progress.md`.
- **Checks at 92k, 94k, 96k and 98k:**
  - `squat_depth_diag.py` (depth reached, joint margins, self-collision);
  - `action_extremes.py` (6 million steps; 0 events wanted);
  - `gait_clearance.py`, `backward_test.py`, wide-arm squat;
  - `litter_sweep.py` at 94k and 98k.
  - Everything else must stay at 90k's level.
- **For deployment:** a new policy also needs `deploy.yaml` base_height range [0.22, 0.78] (only together with this policy).
