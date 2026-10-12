"""Compact, size-capped console log for a training run.

Usage:  python -u scripts/train.py ... 2>&1 | python compact_log.py OUT.log

* Drops MuJoCo Warp's per-contact ``opt.ccd_iterations ... needs to be
  increased`` device printf (it arrives ~100k times per iteration on Rough and
  produced 8.5 GB in under an hour); the count is kept in every record.
* Replaces each ~4 KB rsl_rl iteration block with one ``it=N key=value ...``
  line, written every RECORD_EVERY iterations and for the WINDOW iterations
  ending at each multiple of WINDOW_EVERY (the progress table averages those).
* Passes every other line through (startup info, warnings, tracebacks),
  suppressing a line after it has been seen MAX_REPEATS times.
* Rotates OUT.log to OUT.log.1 at MAX_BYTES, so the file can never grow
  without bound.

It must never exit before stdin closes: the trainer would die on a broken
pipe. Any internal error drops to a raw, still size-capped pass-through.
"""

from __future__ import annotations

import os
import re
import sys

RECORD_EVERY = 10
WINDOW_EVERY = 1000
WINDOW = 10
MAX_REPEATS = 5
MAX_LINE = 2000
MAX_BYTES = 50 * 1024 * 1024

ANSI = re.compile(r"\x1b\[[0-9;]*m")
HEADER = re.compile(r"Learning iteration (\d+)/(\d+)")
KV = re.compile(r"^\s*(.+?):\s+(\S+)\s*$")
SEPARATOR = re.compile(r"^\s*[#-]{10,}\s*$")
CCD = b"ccd_iterations"

# record key -> rsl_rl log key
KEYS = {
  "reward": "Mean reward",
  "eplen": "Mean episode length",
  "falls": "Episode_Termination/fell_over",
  "std": "Mean action std",
  "lin": "Episode_Metrics/lin_vel_track_error",
  "yaw": "Episode_Metrics/yaw_track_error",
  "h": "Episode_Metrics/height_track_error",
  "drift": "Episode_Metrics/idle_root_drift",
  "terrain": "Curriculum/terrain_levels",
  "push": "Curriculum/push_magnitude",
  "arm": "Curriculum/arm_curriculum_state",
  "vloss": "Mean value loss",
  "ent": "Mean entropy_coef loss",
  "sqcom": "Episode_Reward/squat_com_centering",
  "backward_cost": "Episode_Reward/backward_speed_deficit",
  "apex": "Episode_Metrics/poultry_swing_apex_m",
  "clearance": "Episode_Metrics/poultry_swing_clearance_m",
  "plow": "Episode_Metrics/poultry_plow_cost",
  "poultry_falls": "Episode_Metrics/poultry_falls",
  "it_s": "Iteration time",
  "elapsed": "Time elapsed",
  "eta": "ETA",
}


class Out:
  def __init__(self, path: str):
    self.path = path
    self.f = open(path, "a", buffering=1)
    self.size = self.f.tell()

  def write(self, line: str) -> None:
    line = line[:MAX_LINE] + "\n"
    if self.size + len(line) > MAX_BYTES:
      self.f.close()
      os.replace(self.path, self.path + ".1")
      self.f = open(self.path, "a", buffering=1)
      self.size = 0
    self.f.write(line)
    self.size += len(line)


def wanted(it: int) -> bool:
  r = it % WINDOW_EVERY
  return it % RECORD_EVERY == 0 or r == 0 or r > WINDOW_EVERY - WINDOW


def run(out: Out) -> None:
  block: dict | None = None
  ccd = 0
  seen: dict[str, int] = {}

  def finish() -> None:
    nonlocal block, ccd
    if block is None:
      return
    it = block.pop("_iter")
    if wanted(it):
      parts = [f"it={it}"]
      for short, long in KEYS.items():
        if long in block:
          parts.append(f"{short}={block[long].rstrip('s') if short == 'it_s' else block[long]}")
      parts.append(f"ccd={ccd}")
      out.write(" ".join(parts))
      ccd = 0
    block = None

  for raw in sys.stdin.buffer:
    if CCD in raw:
      ccd += 1
      continue
    line = ANSI.sub("", raw.decode("utf-8", "replace")).rstrip()
    m = HEADER.search(line)
    if m:
      finish()
      block = {"_iter": int(m.group(1))}
      continue
    if block is not None:
      if not line.strip() or SEPARATOR.match(line):
        continue
      kv = KV.match(line)
      if kv:
        block[kv.group(1).strip()] = kv.group(2)
        if kv.group(1).strip() == "ETA":
          finish()
        continue
    if not line.strip() or SEPARATOR.match(line):
      continue
    n = seen.get(line, 0) + 1
    seen[line] = n
    if n <= MAX_REPEATS:
      out.write(line)
    elif n == MAX_REPEATS + 1:
      out.write(f"[compact_log] further repeats of the line above suppressed")
  finish()
  out.write(f"[compact_log] stdin closed; ccd warnings since last record: {ccd}")


def main() -> None:
  out = Out(sys.argv[1])
  try:
    run(out)
  except Exception as exc:  # never let the trainer hit a broken pipe
    try:
      out.write(f"[compact_log] parser error {exc!r}; raw pass-through from here")
    except Exception:
      pass
    for raw in sys.stdin.buffer:
      if CCD in raw:
        continue
      try:
        out.write(raw.decode("utf-8", "replace").rstrip())
      except Exception:
        pass


if __name__ == "__main__":
  main()
