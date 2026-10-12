# Poultry-litter floor evaluation (2026-10-12)

## Setup
- **Floor:** the floor from `scripts/poultry_floor/make_poultry_floor.py` (seed 7; 16 m x 6 m; 8 cm collision grid; zero contact overflows in every run).
- **Robots:** 96 per condition, 32 for each loose-litter depth (3, 5, 8 cm). They start at random positions and headings in the aisle and cake-strip half of the floor, carrying 0.6 kg per hand.
- **Routine (40 s, the same for every robot):** forward 0.4 m/s, turn 0.6 rad/s, backward 0.3 m/s, sideways 0.25 m/s, squat to 0.25 m with arms out wide, stand up, forward 0.5 m/s.
- **Precision:** one fall per depth group equals 0.047 falls per robot-minute (21.3 robot-minutes per group). At about 5 falls, the Poisson 1-sigma is about ±0.1.

## Policy 91500 (deeper-squat fine-tune)

| Floor | Falls per robot-min (3 / 5 / 8 cm loose litter) | Forward speed reached (0.4 m/s) | Stance sinkage, walking | Toe inside the loose layer, forward | Swing apex above the loose surface |
|---|---|---|---|---|---|
| Rigid flat, no litter (routine baseline) | 0.19 / 0.14 / 0.23 | 95 / 93 / 96% | 0.6 cm | 0% | 8.7-8.8 cm (above the floor) |
| **Built-up litter** (default; softness within training) | 0.28 / 0.23 / 0.42 | 95 / 95 / 99% | 1.0-1.1 cm | 16 / 40 / 72% | 5.7 / 3.3 / 1.0 cm |
| Fresh deep shavings (1.5x softer than training) | 0.38 / 0.94 / 2.06 | 88 / 92 / 91% | 1.8-2.4 cm | 19 / 44 / 74% | 4.5 / 2.8 / 0.2 cm |

- **Squat at a 0.25 m command, arms wide:** reached 0.255-0.257 m on built-up litter, and 0.251-0.254 m on rigid ground.
- **Falls by location:** few happened on the cake strip (0-1 per group) or near the feeder pans (0-3).
- **Fresh deep shavings at 8 cm:** most falls happen walking forward (16 of 44, and 15 more at 0.5 m/s).

## Policy 90000 vs 91500 (built-up litter)

| Policy | Falls per robot-min (3 / 5 / 8 cm) | Forward speed reached | Squat reached at the 0.25 m command |
|---|---|---|---|
| 90000 | 0.23 / 0.05 / 0.47 | 91 / 89 / 99% | 0.30 m |
| 91500 | 0.28 / 0.23 / 0.42 | 95 / 95 / 99% | 0.256 m |

## Reading
- **Built-up litter (the modelled compacted floor of a working house):** the policy is close to its rigid-floor baseline. The fall rate stays within about 2x of baseline up to 8 cm of loose litter, and speed tracking is unchanged.
- **Foot clearance shrinks with depth:** the foot clears 3 cm of loose litter, but at 8 cm the toe spends about 70% of each swing inside the loose layer. Scooping/dust would be expected there.
- **Fresh deep shavings** (about 2 cm sinkage, beyond the training range) is where performance degrades: falls rise to 2 per robot-minute at 8 cm. This is the candidate for further training (a softer contact range, deeper loose litter).
- **Policy choice:** 91500 and 90000 are statistically similar on litter. 91500 squats about 5 cm deeper.

## Files
- **Results per condition:** `91500_rigid_baseline`, `91500_built_up`, `91500_fresh_deep`, `90000_built_up` (`.log` per-segment tables, `.json`).
- **Video:** `poultry_walk_91500.mp4` (one robot, built-up litter, 5 cm loose layer).
- **Re-run:** `run_study.sh`.
- **Superseded:** `superseded_2cm_4cm/` holds runs on 2 cm and 4 cm collision grids, invalid because of contact overflow. Do not use.
