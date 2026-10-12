# 2026-10-12_10-16-27_poultry_v3_squat

One row per 1000-iteration checkpoint; every metric is the mean over the 10 iterations ending at that checkpoint. 'apex cm' is the training metric (averaged over all robots, standing/slope robots count as 0), so it reads well below the real foot lift; use it for trend. Rows written at startup are backfilled from the compact log (time = backfill time).

| iter | time | reward | falls | std | value loss | lin err | yaw err | h err | terrain | apex cm | plow x1e3 | bwd | sq com | ETA |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 91000 | 10-12 10:54 | 83.2 | 0.25 | 0.19 | 0.108 | 0.232 | 0.261 | 0.020 | 2.71 | 3.51 | 7.17 | -0.0236 | -0.0316 | 4.4 h (10-12 15:18) |
| 92000 | 10-12 11:33 | 83.5 | 0.30 | 0.19 | 0.116 | 0.235 | 0.261 | 0.020 | 2.70 | 3.63 | 7.22 | -0.0223 | -0.0352 | 3.9 h (10-12 15:25) |
| 93000 | 10-12 12:29 | 84.2 | 0.23 | 0.19 | 0.101 | 0.235 | 0.262 | 0.020 | 2.64 | 3.89 | 7.48 | -0.0227 | -0.0314 | 3.1 h (10-12 15:37) |

**Trainer exited: STOPPED** at 2026-10-12 12:51; last logged iteration 93430, last checkpoint model_93000.pt

```
    obs, rewards, dones, extras = inner_step(actions)
                                  ~~~~~~~~~~^^^^^^^^^
  File "/home/wolftech/rbbist.lab/miniconda3/envs/duet/lib/python3.13/site-packages/mjlab/rl/vecenv_wrapper.py", line 77, in step
    obs_dict, rew, terminated, truncated, extras = self.env.step(actions)
                                                   ~~~~~~~~~~~~~^^^^^^^^^
  File "/home/wolftech/rbbist.lab/miniconda3/envs/duet/lib/python3.13/site-packages/mjlab/envs/manager_based_rl_env.py", line 388, in step
    self.reward_buf = self.reward_manager.compute(dt=self.step_dt)
                      ~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "/home/wolftech/rbbist.lab/miniconda3/envs/duet/lib/python3.13/site-packages/mjlab/managers/reward_manager.py", line 126, in compute
    value = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
KeyboardInterrupt
[compact_log] stdin closed; ccd warnings since last record: 264607
```
