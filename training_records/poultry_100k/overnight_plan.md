# Overnight plan: PoultrySurface 100k (from Rough 50k), night of 2026-10-09

The user authorised continuing or adjusting the run without asking first.

## Current run
- Trainer PID 1948020. Output goes through the log filter, PID 1970690, into `logs/poultry_100k_from_rough50k.log`.
- Run directory: `logs/rsl_rl/DUET_G1_23dof_poultry/2026-10-09_19-33-24_poultry_100k_from_rough50k_segment0`.
- Checkpoint watcher `logs/monitor/watch_ckpt_poultry.py` (PID 1987289) writes the table `logs/poultry_100k_progress.md`.
- Evaluations go in `logs/poultry_100k_evals.md`.

## Foot-lift check, every 5k (60k, 65k, ...)
- **Measure:** run `gait_clearance.py` on model_{N-500} and model_N. Flat ground, 0.6 kg in each hand, walking at +0.4, -0.4 and sideways 0.3 m/s.
- **Score:** L(N) = the mean swing apex over the 3 directions and both checkpoints.
- **Baseline:** L(55k) = mean(5.30, 5.17) = 5.23 cm.
- **Reference trend:** 51k 4.4, 52k 4.5, 53k 4.6, 54.5k 5.3, 55k 5.2 cm.
- **Flattened** means L(N) - L(N-5k) < 0.3 cm while L(N) < 8 cm.
- At 60k, 70k, 80k, 90k and 100k, also run the backward test and the wide-arm squat test.

## If foot lift has flattened: boost and resume (at most 2 boosts in total)
- **Boost 1**, in `src/tasks/duet/config/g1_23dof/poultry_surface.py`:
  - poultry_clearance -2.0 -> -3.0
  - poultry_swing_height -0.06 -> -0.12
- **Boost 2:** -3.0 -> -4.0 and -0.12 -> -0.20.
- **Steps:**
  1. `kill -INT 1948020` (or the current trainer).
  2. Edit the two weights.
  3. Run `logs/monitor/resume_poultry.sh <run dir> N poultry_100k_lift<k>`.
  4. Restart the watcher on the new run directory and log, then restart the notifier.
- **Revert** to the previous weights, resuming from the latest checkpoint, if any of these happen:
  - falls above 0.55 in two consecutive rows;
  - a value-loss alert;
  - backward speed at the -0.5 command below 80%;
  - wide-squat centre of mass more than 1 cm worse than before the boost.

## If training crashes
- Resume from the latest checkpoint with the same configuration.
- On out-of-memory, use 2048 robots.
- If it crashes again within 1k iterations: stop and report.

## Don't
Commit, push, touch the main checkout or deploy/, or delete checkpoints.

## Log
- 2026-10-10 02:15, 60k: foot lift 4.72 cm vs 5.23 cm at 55k, so it's flat. Applied boost 1 and resumed from model_60000.pt.
  - New run directory: `2026-10-10_02-19-39_poultry_100k_lift1`. Trainer PID 2009359; log `logs/poultry_100k_lift1.log`; watcher PID 2009602.
  - Pre-boost reference for the revert rules:
    - backward speed at -0.5: 0.45 (90%);
    - wide-squat centre of mass, arms out: -1.2 / -1.6 / -1.1 / -1.1 cm;
    - falls about 0.3-0.41.
- 2026-10-10 05:34, 65k: foot lift 6.60 cm (64.5k 6.43, 65k 6.77) vs 4.72 cm at 60k, so boost 1 works and stays.
  - Backward speed at -0.5: 90%. Squat centre of mass better than at 60k. Falls in the table 0.46-0.47, up from 0.33-0.41.
  - Boost 2 at 70k only if lift is flat again AND training falls stay at or below about 0.45; otherwise hold the current weights.
- 2026-10-10 07:04, 67,041: collapse. An outlier batch of actions blew up the last-action normaliser (std x35); falls about 42 per step. Diagnosed by comparing model_67000 with model_67500.
  - Fix: freeze the actor and critic observation normalisers on load (new `freeze_obs_normalizer_on_load`, on for PoultrySurface only).
  - Resumed from model_67000.pt at 07:28 as `2026-10-10_07-28-51_poultry_100k_lift1b`, boost 1 kept. Trainer PID 2043494.
- 2026-10-10 08:12-08:28: value-loss spikes after the 67k resume (1,298, 436, 55,684); reward and falls normal. Measured actions in the training conditions:
  - Largest action: 60k 12.1, 65k 12.4, 65.5k 13.7, 66k 88.7, 66.5k 23, 67k 331, 68k 1,024.
  - The policy started producing rare huge actions at 66k.
- Stopped at 08:28 and resumed from model_65000.pt with training actions clipped to +-15 (`poultry_surface_ppo_runner_cfg`) plus the frozen normalisers.
  - Run: `2026-10-10_08-34-16_poultry_100k_lift1c`. Trainer PID 2050228.
  - New check every 5k: `action_extremes.py` must show 0 actions above 15 (deploy-safety gate; the robot's guard trips above 15).
- Unusable checkpoints: lift1 66000-67000 (outliers) and all of lift1b.
