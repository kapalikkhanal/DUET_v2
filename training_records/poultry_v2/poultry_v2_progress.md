# 2026-10-10_09-23-55_poultry_v2

One row per 1000-iteration checkpoint; every metric is the mean over the 10 iterations ending at that checkpoint. 'apex cm' is the training metric (averaged over all robots, standing/slope robots count as 0), so it reads well below the real foot lift; use it for trend. Rows written at startup are backfilled from the compact log (time = backfill time).

| iter | time | reward | falls | std | value loss | lin err | yaw err | h err | terrain | apex cm | plow x1e3 | bwd | sq com | ETA |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 66000 | 10-10 10:03 | 81.7 | 0.30 | 0.19 | 0.111 | 0.240 | 0.266 | 0.021 | 2.60 | 3.26 | 7.85 | -0.0218 | -0.0394 | 21.2 h (10-11 07:16) |
| 67000 | 10-10 10:42 | 80.9 | 0.33 | 0.20 | 0.112 | 0.241 | 0.281 | 0.021 | 2.56 | 3.28 | 7.80 | -0.0211 | -0.0368 | 20.6 h (10-11 07:21) |
| 68000 | 10-10 11:20 | 79.9 | 0.31 | 0.20 | 0.117 | 0.240 | 0.279 | 0.022 | 2.61 | 3.38 | 7.59 | -0.0243 | -0.0361 | 19.9 h (10-11 07:17) |
| 69000 | 10-10 11:58 | 82.4 | 0.25 | 0.19 | 0.114 | 0.246 | 0.275 | 0.022 | 2.59 | 3.60 | 8.36 | -0.0227 | -0.0369 | 19.4 h (10-11 07:23) |
| 70000 | 10-10 12:36 | 80.8 | 0.34 | 0.20 | 0.119 | 0.242 | 0.276 | 0.023 | 2.57 | 3.36 | 8.05 | -0.0261 | -0.0395 | 18.8 h (10-11 07:26) |
| 71000 | 10-10 13:17 | 81.3 | 0.29 | 0.20 | 0.125 | 0.241 | 0.278 | 0.020 | 2.58 | 3.47 | 8.82 | -0.0222 | -0.0341 | 18.1 h (10-11 07:21) |
| 72000 | 10-10 13:55 | 81.0 | 0.29 | 0.20 | 0.114 | 0.244 | 0.279 | 0.022 | 2.54 | 3.46 | 8.18 | -0.0201 | -0.0388 | 17.7 h (10-11 07:37) |
| 73000 | 10-10 14:34 | 82.3 | 0.29 | 0.20 | 0.107 | 0.242 | 0.271 | 0.021 | 2.58 | 3.41 | 8.02 | -0.0208 | -0.0387 | 17.1 h (10-11 07:39) |
| 74000 | 10-10 15:14 | 83.3 | 0.19 | 0.19 | 0.105 | 0.238 | 0.265 | 0.022 | 2.54 | 3.56 | 8.39 | -0.0187 | -0.0417 | 16.2 h (10-11 07:27) |
| 75000 | 10-10 15:52 | 82.3 | 0.30 | 0.20 | 0.123 | 0.245 | 0.280 | 0.022 | 2.64 | 3.49 | 8.61 | -0.0226 | -0.0363 | 15.7 h (10-11 07:34) |

**Trainer exited: STOPPED** at 2026-10-10 16:16; last logged iteration 75330, last checkpoint model_75000.pt

```
    obs_dict, rew, terminated, truncated, extras = self.env.step(actions)
                                                   ~~~~~~~~~~~~~^^^^^^^^^
  File "/home/wolftech/rbbist.lab/miniconda3/envs/duet/lib/python3.13/site-packages/mjlab/envs/manager_based_rl_env.py", line 389, in step
    self.metrics_manager.compute()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/wolftech/rbbist.lab/miniconda3/envs/duet/lib/python3.13/site-packages/mjlab/managers/metrics_manager.py", line 97, in compute
    value = term_cfg.func(self._env, **term_cfg.params)
  File "/home/wolftech/rbbist.lab/DUET_poultry_v2/src/tasks/common/mdp/metrics.py", line 54, in torso_pitch_abs
    return torch.atan2(g[:, 0], -g[:, 2]).abs()
           ~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^
KeyboardInterrupt
[compact_log] stdin closed; ccd warnings since last record: 182484
```

**75000 failed the safety check** (max |a| 17.4, 1 guard trip, ankle pitch; also worse litter/backward). 71k-74.5k clean. Added action_excess_l2 (raw |a| beyond 8, weight -0.05) and resumed from model_74500.pt as run 2026-10-10_16-17-03_poultry_v2b (log logs/poultry_v2b.log). model_75000.pt deleted.

| iter | time | reward | falls | std | value loss | lin err | yaw err | h err | terrain | apex cm | plow x1e3 | bwd | sq com | ETA |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 76000 | 10-10 17:17 | 82.4 | 0.26 | 0.19 | 0.111 | 0.240 | 0.276 | 0.024 | 2.73 | 3.52 | 8.36 | -0.0242 | -0.0352 | 16.3 h (10-11 09:38) |
| 77000 | 10-10 17:57 | 82.1 | 0.24 | 0.19 | 0.112 | 0.237 | 0.275 | 0.020 | 2.68 | 3.85 | 8.07 | -0.0204 | -0.0352 | 14.8 h (10-11 08:45) |
| 78000 | 10-10 18:35 | 81.1 | 0.32 | 0.20 | 0.123 | 0.237 | 0.276 | 0.020 | 2.60 | 3.64 | 7.55 | -0.0215 | -0.0370 | 13.6 h (10-11 08:10) |
| 79000 | 10-10 19:13 | 83.5 | 0.24 | 0.19 | 0.115 | 0.237 | 0.260 | 0.020 | 2.56 | 3.60 | 8.31 | -0.0200 | -0.0354 | 13.5 h (10-11 08:44) |
| 80000 | 10-10 19:51 | 82.3 | 0.29 | 0.19 | 0.127 | 0.234 | 0.270 | 0.022 | 2.56 | 3.64 | 7.67 | -0.0190 | -0.0345 | 12.5 h (10-11 08:22) |
| 81000 | 10-10 20:36 | 81.2 | 0.30 | 0.20 | 0.112 | 0.241 | 0.273 | 0.021 | 2.63 | 3.60 | 7.56 | -0.0227 | -0.0344 | 11.9 h (10-11 08:28) |
| 82000 | 10-10 21:14 | 81.1 | 0.33 | 0.20 | 0.115 | 0.236 | 0.280 | 0.020 | 2.60 | 3.47 | 7.76 | -0.0211 | -0.0340 | 11.2 h (10-11 08:25) |
| 83000 | 10-10 21:53 | 81.8 | 0.26 | 0.20 | 0.113 | 0.234 | 0.273 | 0.021 | 2.52 | 3.53 | 8.12 | -0.0225 | -0.0387 | 10.6 h (10-11 08:30) |
| 84000 | 10-10 22:32 | 81.8 | 0.26 | 0.20 | 0.118 | 0.239 | 0.268 | 0.022 | 2.61 | 3.47 | 8.22 | -0.0222 | -0.0353 | 10.0 h (10-11 08:31) |
| 85000 | 10-10 23:10 | 81.2 | 0.32 | 0.20 | 0.116 | 0.242 | 0.273 | 0.020 | 2.68 | 3.42 | 7.84 | -0.0209 | -0.0356 | 9.5 h (10-11 08:40) |

**Trainer exited: STOPPED** at 2026-10-10 23:25; last logged iteration 85220, last checkpoint model_85000.pt

```
    self.action_manager.apply_action()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/wolftech/rbbist.lab/miniconda3/envs/duet/lib/python3.13/site-packages/mjlab/managers/action_manager.py", line 183, in apply_action
    term.apply_actions()
    ~~~~~~~~~~~~~~~~~~^^
  File "/home/wolftech/rbbist.lab/DUET_poultry_v2/src/tasks/duet/mdp/poultry_surface.py", line 232, in apply_actions
    points, velocity, _, _, height = self.geometry()
                                     ~~~~~~~~~~~~~^^
  File "/home/wolftech/rbbist.lab/DUET_poultry_v2/src/tasks/duet/mdp/poultry_surface.py", line 221, in geometry
    rotation = matrix_from_quat(data.body_link_quat_w[:, self.body_ids])
KeyboardInterrupt
[compact_log] stdin closed; ccd warnings since last record: 346847
```

**85000 failed the safety check** (max |a| 17.8, 1 guard trip, left ankle pitch; 84500 max 15.8, no trip; 84000 clean). Lift flat at 7.1-7.5 cm. Resumed from model_84000.pt as run 2026-10-10_23-25-42_poultry_v2c with: boost 2 (clearance -4.0, swing -0.20), action_excess_l2 -0.05 -> -0.2, clip_actions 15 -> 30, last-action input clipped to +-15. model_85000.pt deleted.

| iter | time | reward | falls | std | value loss | lin err | yaw err | h err | terrain | apex cm | plow x1e3 | bwd | sq com | ETA |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 86000 | 10-11 00:41 | 81.3 | 0.28 | 0.19 | 0.114 | 0.242 | 0.267 | 0.022 | 2.59 | 3.63 | 7.57 | -0.0205 | -0.0385 | 8.8 h (10-11 09:27) |
| 87000 | 10-11 01:23 | 80.5 | 0.33 | 0.19 | 0.121 | 0.237 | 0.274 | 0.021 | 2.52 | 3.47 | 7.14 | -0.0213 | -0.0388 | 8.2 h (10-11 09:34) |
| 88000 | 10-11 02:04 | 82.4 | 0.26 | 0.19 | 0.113 | 0.239 | 0.270 | 0.019 | 2.52 | 3.57 | 7.47 | -0.0240 | -0.0379 | 7.5 h (10-11 09:36) |
| 89000 | 10-11 02:42 | 81.6 | 0.30 | 0.19 | 0.109 | 0.241 | 0.270 | 0.020 | 2.59 | 3.71 | 7.56 | -0.0251 | -0.0355 | 6.8 h (10-11 09:32) |
| 90000 | 10-11 03:20 | 80.7 | 0.28 | 0.20 | 0.105 | 0.237 | 0.271 | 0.021 | 2.54 | 3.63 | 6.96 | -0.0223 | -0.0391 | 6.2 h (10-11 09:35) |
| 91000 | 10-11 04:06 | 82.7 | 0.25 | 0.19 | 0.106 | 0.234 | 0.264 | 0.021 | 2.52 | 3.75 | 8.06 | -0.0258 | -0.0317 | 5.6 h (10-11 09:45) |
| 92000 | 10-11 04:44 | 83.1 | 0.28 | 0.19 | 0.108 | 0.244 | 0.266 | 0.020 | 2.44 | 3.79 | 7.29 | -0.0216 | -0.0318 | 5.0 h (10-11 09:47) |
| 93000 | 10-11 05:22 | 82.7 | 0.29 | 0.19 | 0.110 | 0.239 | 0.266 | 0.021 | 2.52 | 3.50 | 7.42 | -0.0232 | -0.0348 | 4.4 h (10-11 09:44) |
| 94000 | 10-11 06:00 | 83.3 | 0.23 | 0.19 | 0.108 | 0.233 | 0.258 | 0.020 | 2.53 | 3.70 | 7.45 | -0.0223 | -0.0382 | 3.8 h (10-11 09:46) |
| 95000 | 10-11 06:38 | 84.5 | 0.25 | 0.18 | 0.119 | 0.234 | 0.249 | 0.020 | 2.58 | 3.69 | 7.15 | -0.0213 | -0.0350 | 3.1 h (10-11 09:46) |
| 96000 | 10-11 07:42 | 84.1 | 0.21 | 0.18 | 0.143 | 0.233 | 0.254 | 0.019 | 2.38 | 3.82 | 7.41 | -0.0243 | -0.0352 | 2.5 h (10-11 10:13) |
| 97000 | 10-11 08:24 | 84.3 | 0.24 | 0.18 | 0.155 | 0.238 | 0.269 | 0.020 | 2.50 | 3.71 | 6.99 | -0.0223 | -0.0379 | 1.9 h (10-11 10:17) |
| 98000 | 10-11 09:02 | 83.5 | 0.23 | 0.18 | 0.171 | 0.233 | 0.246 | 0.020 | 2.55 | 3.44 | 7.27 | -0.0215 | -0.0385 | 1.3 h (10-11 10:18) |
| 99000 | 10-11 09:44 | 83.5 | 0.25 | 0.19 | 0.102 | 0.237 | 0.260 | 0.021 | 2.50 | 3.71 | 7.31 | -0.0221 | -0.0349 | 0.6 h (10-11 10:22) |
| 100000 | 10-11 10:22 | 84.2 | 0.24 | 0.18 | 0.105 | 0.232 | 0.252 | 0.019 | 2.57 | 3.66 | 7.00 | -0.0183 | -0.0365 | 0.0 h (10-11 10:22) |

**Trainer exited: FINISHED** at 2026-10-11 10:22; last logged iteration 100000, last checkpoint model_100000.pt
