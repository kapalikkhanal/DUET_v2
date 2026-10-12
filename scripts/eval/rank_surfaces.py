"""duet_rank_checkpoints.py protocol, on three surfaces, one env per surface.

Same protocol as scripts/duet_rank_checkpoints.py (full trained distribution
pinned, curriculum off, obs corruption on, pushes and payload on, deterministic
policy, ranked by falls first), with the terrain swapped per condition.
"""
import os, sys, math, re, json
from dataclasses import asdict, replace
os.environ.setdefault("MUJOCO_GL", "egl")
import torch
import mjlab.tasks, src.tasks  # noqa
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.lab_api.math import quat_apply_inverse
from src.tasks.duet.config.g1_23dof.env_cfgs import pin_duet_full_distribution
from src.tasks.common.terrains import SoftFlatTerrainCfg

TASK = "Unitree-G1-23Dof-Duet-Rough"
N_ENV, STEPS, DEV = int(os.environ.get("N_ENV", 512)), 700, "cuda:0"
SEEDS = [int(x) for x in os.environ.get("SEEDS", "0").split(",")]
ONLY = os.environ.get("CONDS")
CONDS = {"flat": None, "soft_d0.5_f0.6": (0.5, 0.6), "soft_d1.0_f0.3": (1.0, 0.3)}


def make(cond):
  cfg = load_env_cfg(TASK, play=False)
  cfg.scene.num_envs = N_ENV
  cfg.episode_length_s = 20.0
  pin_duet_full_distribution(cfg)
  cfg.curriculum = {}
  t = cfg.scene.terrain
  if CONDS[cond] is None:
    t.terrain_type, t.terrain_generator = "plane", None
  else:
    d, f = CONDS[cond]
    t.terrain_generator = replace(t.terrain_generator, size=(30.0, 30.0), num_rows=1, num_cols=1,
      curriculum=True, difficulty_range=(d, d),
      sub_terrains={"soft": SoftFlatTerrainCfg(proportion=1.0, friction_range=(f, f))})
    t.max_init_terrain_level = 0
  ag = load_rl_cfg(TASK)
  env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=cfg, device=DEV), clip_actions=ag.clip_actions)
  runner = (load_runner_cls(TASK) or MjlabOnPolicyRunner)(env, asdict(ag), device=DEV)
  return env, runner


def run(env, runner, ckpt, seed):
  if ckpt.endswith(".onnx"):
    import numpy as np, onnxruntime as ort
    sess = ort.InferenceSession(ckpt, providers=["CPUExecutionProvider"])
    def pol(obs):
      x = obs["actor"].cpu().numpy().astype(np.float32)
      return torch.tensor(np.concatenate([sess.run(None, {"obs": x[i:i+1]})[0] for i in range(len(x))]), device=DEV)
  else:
    runner.load(ckpt, load_cfg={"actor": True}, strict=True, map_location=DEV)
    pol = runner.get_inference_policy(device=DEV)
  torch.manual_seed(seed)
  with torch.inference_mode():
    obs, _ = env.reset()
  u = env.unwrapped; rb = u.scene["robot"]
  fid = u.reward_manager.get_term_cfg("track_base_height").params["asset_cfg"].site_ids
  tid = u.reward_manager.get_term_cfg("body_orientation_l2").params["asset_cfg"].body_ids
  falls = torch.zeros(N_ENV, device=DEV)
  acc = {k: 0.0 for k in ("vxy", "yaw", "h", "tilt", "jit")}; cnt = 0.0; prev = prev2 = None
  with torch.inference_mode():
    for _ in range(STEPS):
      a = pol(obs)
      obs, _, dones, _ = env.step(a)
      falls += u.termination_manager.get_term("fell_over").float()
      valid = (~dones.bool()).float(); w = float(valid.sum())
      if w > 0:
        cmd = u.command_manager.get_command("twist")
        lin, ang = rb.data.root_link_lin_vel_b, rb.data.root_link_ang_vel_b
        acc["vxy"] += float((torch.norm(cmd[:, :2] - lin[:, :2], dim=1) * valid).sum())
        acc["yaw"] += float(((cmd[:, 2] - ang[:, 2]).abs() * valid).sum())
        hc = u.command_manager.get_command("base_height")[:, 0]
        bh = rb.data.root_link_pos_w[:, 2] - rb.data.site_pos_w[:, fid, 2].min(1).values
        acc["h"] += float(((bh - hc + 0.02).abs() * valid).sum())
        g = quat_apply_inverse(rb.data.body_link_quat_w[:, tid, :].squeeze(1), rb.data.gravity_vec_w)
        acc["tilt"] += float((torch.acos((-g[:, 2]).clamp(-1, 1)) * valid).sum())
        if prev2 is not None:
          acc["jit"] += float(((a - 2 * prev + prev2).abs().mean(1) * valid).sum())
        cnt += w
      prev2, prev = prev, a.clone()
  env_min = N_ENV * STEPS * u.step_dt / 60.0
  r = {"falls_per_env_min": float(falls.sum()) / env_min, "vxy_err": acc["vxy"] / cnt,
       "yaw_err": acc["yaw"] / cnt, "h_err": acc["h"] / cnt,
       "tilt_deg": math.degrees(acc["tilt"] / cnt), "jitter": acc["jit"] / cnt}
  r["score"] = 10 * r["falls_per_env_min"] + r["vxy_err"] + 0.5 * r["yaw_err"] + r["h_err"]
  return r



if __name__ == '__main__':
  ckpts = sys.argv[2:]; out_json = sys.argv[1]
  results = {}
  for cond in [c for c in CONDS if not ONLY or c in ONLY.split(",")]:
    env, runner = make(cond)
    for c in ckpts:
     for seed in SEEDS:
      m = re.search(r"model_(\d+)\.pt$", c)
      it = m.group(1) if m else os.path.basename(os.path.dirname(c))
      r = run(env, runner, c, seed); results.setdefault(cond, {})[f"{it}_s{seed}"] = r
      print(f"[{cond}] iter {it} seed {seed}: falls/min {r['falls_per_env_min']:.3f} vxy {r['vxy_err']:.3f} "
            f"yaw {r['yaw_err']:.3f} h {r['h_err']:.4f} tilt {r['tilt_deg']:.2f} jit {r['jitter']:.3f} "
            f"score {r['score']:.3f}", flush=True)
      json.dump(results, open(out_json, "w"), indent=1)
    env.close(); del env, runner; torch.cuda.empty_cache()
  print("DONE")
