# 2026-10-09_19-33-24_poultry_100k_from_rough50k_segment0

One row per 1000-iteration checkpoint; every metric is the mean over the 10 iterations ending at that checkpoint. 'apex cm' is the training metric (averaged over all robots, standing/slope robots count as 0), so it reads well below the real foot lift; use it for trend. Rows written at startup are backfilled from the compact log (time = backfill time).

| iter | time | reward | falls | std | value loss | lin err | yaw err | h err | terrain | apex cm | plow x1e3 | bwd | sq com | ETA |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 51000 | 10-09 23:05 | 77.7 | 0.31 | 0.21 | 0.155 | 0.262 | 0.304 | 0.024 | 2.46 | 2.06 | 4.50 | -0.0255 | -0.0396 | 34.8 h (10-11 09:51) |
| 52000 | 10-09 23:05 | 77.8 | 0.31 | 0.22 | 0.151 | 0.261 | 0.308 | 0.022 | 2.36 | 2.30 | 4.90 | -0.0239 | -0.0417 | 35.0 h (10-11 10:07) |
| 53000 | 10-09 23:05 | 76.9 | 0.39 | 0.22 | 0.151 | 0.270 | 0.316 | 0.027 | 2.45 | 2.07 | 4.52 | -0.0232 | -0.0421 | 30.5 h (10-11 05:33) |
| 54000 | 10-09 23:05 | 76.7 | 0.40 | 0.21 | 0.141 | 0.266 | 0.306 | 0.024 | 2.44 | 2.03 | 4.39 | -0.0248 | -0.0365 | 29.8 h (10-11 04:53) |
| 55000 | 10-09 23:05 | 78.9 | 0.30 | 0.21 | 0.144 | 0.250 | 0.296 | 0.024 | 2.44 | 1.88 | 4.08 | -0.0236 | -0.0450 | 32.8 h (10-11 07:55) |
| 56000 | 10-09 23:38 | 79.0 | 0.31 | 0.21 | 0.154 | 0.251 | 0.301 | 0.023 | 2.55 | 2.12 | 4.30 | -0.0245 | -0.0379 | 32.2 h (10-11 07:52) |
| 57000 | 10-10 00:18 | 77.4 | 0.38 | 0.21 | 0.146 | 0.256 | 0.304 | 0.025 | 2.47 | 1.97 | 4.43 | -0.0274 | -0.0385 | 27.8 h (10-11 04:07) |
| 58000 | 10-10 00:57 | 77.2 | 0.29 | 0.21 | 0.156 | 0.262 | 0.311 | 0.028 | 2.46 | 2.05 | 4.07 | -0.0255 | -0.0427 | 28.8 h (10-11 05:45) |
| 59000 | 10-10 01:36 | 76.7 | 0.41 | 0.21 | 0.145 | 0.264 | 0.315 | 0.029 | 2.42 | 2.04 | 4.24 | -0.0279 | -0.0377 | 26.5 h (10-11 04:09) |
| 60000 | 10-10 02:15 | 77.0 | 0.41 | 0.21 | 0.167 | 0.265 | 0.311 | 0.025 | 2.48 | 2.04 | 4.04 | -0.0300 | -0.0353 | 25.9 h (10-11 04:07) |

**Trainer exited: STOPPED** at 2026-10-10 02:19; last logged iteration 60060, last checkpoint model_60000.pt

```
    ~~~~~~~~~~~~~~^^
  File "/home/wolftech/rbbist.lab/miniconda3/envs/duet/lib/python3.13/site-packages/mjlab/sim/sim.py", line 355, in sense
    ctx.finalize()
    ~~~~~~~~~~~~^^
  File "/home/wolftech/rbbist.lab/miniconda3/envs/duet/lib/python3.13/site-packages/mjlab/sensor/sensor_context.py", line 118, in finalize
    sensor.postprocess_rays()
    ~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/wolftech/rbbist.lab/miniconda3/envs/duet/lib/python3.13/site-packages/mjlab/sensor/raycast_sensor.py", line 764, in postprocess_rays
    ] + self._cached_world_rays[hit_mask] * self._distances[hit_mask].unsqueeze(-1)
                                            ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^
KeyboardInterrupt
[compact_log] stdin closed; ccd warnings since last record: 114596
```

**Boost 1 applied at 60000** (2026-10-10 02:19): poultry_clearance -2.0 -> -3.0, poultry_swing_height -0.06 -> -0.12; resumed from model_60000.pt as run 2026-10-10_02-19-39_poultry_100k_lift1 (log logs/poultry_100k_lift1.log). Reason: real foot lift 4.72 cm at 59.5k/60k vs 5.23 cm at 54.5k/55k.

| iter | time | reward | falls | std | value loss | lin err | yaw err | h err | terrain | apex cm | plow x1e3 | bwd | sq com | ETA |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 61000 | 10-10 02:58 | 77.8 | 0.33 | 0.21 | 0.142 | 0.258 | 0.309 | 0.023 | 2.51 | 2.48 | 4.25 | -0.0253 | -0.0376 | 25.4 h (10-11 04:24) |
| 62000 | 10-10 03:37 | 76.7 | 0.40 | 0.21 | 0.133 | 0.256 | 0.308 | 0.023 | 2.43 | 2.49 | 4.07 | -0.0246 | -0.0382 | 24.6 h (10-11 04:11) |
| 63000 | 10-10 04:16 | 78.2 | 0.41 | 0.21 | 0.159 | 0.250 | 0.292 | 0.023 | 2.45 | 2.36 | 4.19 | -0.0236 | -0.0411 | 23.8 h (10-11 04:03) |
| 64000 | 10-10 04:55 | 75.5 | 0.46 | 0.22 | 0.151 | 0.274 | 0.317 | 0.023 | 2.34 | 2.26 | 3.98 | -0.0204 | -0.0419 | 23.4 h (10-11 04:16) |
| 65000 | 10-10 05:34 | 75.7 | 0.47 | 0.22 | 0.135 | 0.253 | 0.309 | 0.023 | 2.35 | 2.58 | 4.04 | -0.0272 | -0.0404 | 22.7 h (10-11 04:19) |
| 66000 | 10-10 06:14 | 76.0 | 0.47 | 0.21 | 0.173 | 0.256 | 0.305 | 0.022 | 2.34 | 2.43 | 3.71 | -0.0237 | -0.0410 | 22.1 h (10-11 04:23) |
| 67000 | 10-10 07:02 | 80.1 | 0.32 | 0.20 | 0.132 | 0.248 | 0.292 | 0.023 | 2.37 | 2.33 | 4.13 | -0.0205 | -0.0422 | 33.3 h (10-11 16:20) |

**ALERT vloss** at it=67050 (10-10 07:04): reward=-15.28 vloss=98.0102 std=0.20 falls=42.1250


**ALERT std** at it=67450 (10-10 07:22): reward=43.05 vloss=0.5372 std=0.26 falls=1.6667


**Trainer exited: STOPPED** at 2026-10-10 07:25; last logged iteration 67520, last checkpoint model_67500.pt

```
    self.action_manager.apply_action()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/wolftech/rbbist.lab/miniconda3/envs/duet/lib/python3.13/site-packages/mjlab/managers/action_manager.py", line 183, in apply_action
    term.apply_actions()
    ~~~~~~~~~~~~~~~~~~^^
  File "/home/wolftech/rbbist.lab/DUET_poultry_e063b5e/src/tasks/duet/mdp/poultry_surface.py", line 215, in apply_actions
    points, velocity, _, _, height = self.geometry()
                                     ~~~~~~~~~~~~~^^
  File "/home/wolftech/rbbist.lab/DUET_poultry_e063b5e/src/tasks/duet/mdp/poultry_surface.py", line 204, in geometry
    rotation = matrix_from_quat(data.body_link_quat_w[:, self.body_ids])
KeyboardInterrupt
[compact_log] stdin closed; ccd warnings since last record: 60499
```

**Collapse at 67041 (2026-10-10 07:04)**: an outlier batch of actions blew up the last-action observation normalizer (std x35), so every robot fell (falls ~42/step). Rows past 67000 from run lift1 are invalid. Fixed by freezing the actor/critic observation normalizers on load (runner.py, rl_cfg.py, poultry_surface.py); resumed from model_67000.pt as run 2026-10-10_07-28-51_poultry_100k_lift1b (log logs/poultry_100k_lift1b.log), boost 1 weights kept.

| iter | time | reward | falls | std | value loss | lin err | yaw err | h err | terrain | apex cm | plow x1e3 | bwd | sq com | ETA |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 68000 | 10-10 08:07 | 81.8 | 0.33 | 0.19 | 0.153 | 0.251 | 0.278 | 0.021 | 2.62 | 2.61 | 4.33 | -0.0223 | -0.0333 | 20.6 h (10-11 04:45) |

**ALERT vloss** at it=68120 (10-10 08:12): reward=80.95 vloss=1297.7074 std=0.20 falls=0.3750


**Trainer exited: STOPPED** at 2026-10-10 08:28; last logged iteration 68350, last checkpoint model_68000.pt

```
    self.action_manager.apply_action()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/wolftech/rbbist.lab/miniconda3/envs/duet/lib/python3.13/site-packages/mjlab/managers/action_manager.py", line 183, in apply_action
    term.apply_actions()
    ~~~~~~~~~~~~~~~~~~^^
  File "/home/wolftech/rbbist.lab/DUET_poultry_e063b5e/src/tasks/duet/mdp/poultry_surface.py", line 215, in apply_actions
    points, velocity, _, _, height = self.geometry()
                                     ~~~~~~~~~~~~~^^
  File "/home/wolftech/rbbist.lab/DUET_poultry_e063b5e/src/tasks/duet/mdp/poultry_surface.py", line 204, in geometry
    rotation = matrix_from_quat(data.body_link_quat_w[:, self.body_ids])
KeyboardInterrupt
[compact_log] stdin closed; ccd warnings since last record: 85082
```

**Restart from 65000 (2026-10-10 08:34)**: from 66000 on the policy produced rare huge actions (max |a| 89 at 66k, 331 at 67k, 1024 at 68k; 65000/65500 max 12.4/13.7), driving value-loss spikes up to 55,684 after the 67k resume. Rows 68000+ above are from a degrading policy. Resumed from model_65000.pt with training actions clipped to +-15 (= deploy runaway guard) plus the frozen normalizers, as run 2026-10-10_08-34-16_poultry_100k_lift1c (log logs/poultry_100k_lift1c.log).

| iter | time | reward | falls | std | value loss | lin err | yaw err | h err | terrain | apex cm | plow x1e3 | bwd | sq com | ETA |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|

**Trainer exited: STOPPED** at 2026-10-10 09:19; last logged iteration 66010, last checkpoint model_66000.pt

```
    set_const_0(m, d)
    ~~~~~~~~~~~^^^^^^
  File "/home/wolftech/rbbist.lab/miniconda3/envs/duet/lib/python3.13/site-packages/mujoco_warp/_src/io.py", line 2039, in set_const_0
    wp.launch(_extract_dof_A_diag, dim=d.nworld, inputs=[dofid, result_vec], outputs=[dof_A_diag])
    ~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/wolftech/rbbist.lab/miniconda3/envs/duet/lib/python3.13/site-packages/warp/_src/context.py", line 7577, in launch
    runtime.verify_cuda_device(device)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^
  File "/home/wolftech/rbbist.lab/miniconda3/envs/duet/lib/python3.13/site-packages/warp/_src/context.py", line 5714, in verify_cuda_device
    def verify_cuda_device(self, device: DeviceLike = None) -> None:
KeyboardInterrupt
[compact_log] stdin closed; ccd warnings since last record: 60313
```
