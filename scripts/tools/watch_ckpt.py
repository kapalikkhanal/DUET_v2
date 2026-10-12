"""Event-driven training monitor: one table row per 1000-iteration checkpoint.

Sleeps in poll() on two kernel handles, with no timer:
  * an inotify watch on the run directory, which fires when model_N.pt is
    written (IN_CLOSE_WRITE / IN_MOVED_TO);
  * a pidfd on the trainer process, which fires when it exits.

On a checkpoint with N % EVERY == 0 it waits (inotify on the compact log
written by compact_log.py) until the record for iteration N is present, then
appends one Markdown row averaged over iterations N-AVG+1..N. On trainer exit
it writes a final status line (finished / stopped / crashed + last log lines).

Usage: watch_ckpt.py RUN_DIR COMPACT_LOG TABLE_MD TRAINER_PID
"""

from __future__ import annotations

import ctypes
import os
import re
import select
import struct
import sys
import time
from collections import deque

EVERY = int(os.environ.get("CKPT_EVERY", 1000))
AVG = 10
FINAL_ITER = int(os.environ.get("CKPT_FINAL", 100000))
START_ITER = int(os.environ.get("CKPT_START", 50000))
READ_CHUNK = 8 * 1024 * 1024

IN_MODIFY, IN_CLOSE_WRITE, IN_MOVED_TO = 0x2, 0x8, 0x80
SYS_PIDFD_OPEN = 434  # x86_64 and aarch64

CKPT = re.compile(r"^model_(\d+)\.pt$")

# (column title, compact-log key, format, scale)
COLS = [
  ("reward", "reward", "{:.1f}", 1),
  ("falls", "falls", "{:.2f}", 1),
  ("std", "std", "{:.2f}", 1),
  ("value loss", "vloss", "{:.3f}", 1),
  ("lin err", "lin", "{:.3f}", 1),
  ("yaw err", "yaw", "{:.3f}", 1),
  ("h err", "h", "{:.3f}", 1),
  ("terrain", "terrain", "{:.2f}", 1),
  ("apex cm", "apex", "{:.2f}", 100),
  ("plow x1e3", "plow", "{:.2f}", 1000),
  ("bwd", "backward_cost", "{:.4f}", 1),
  ("sq com", "sqcom", "{:.4f}", 1),
]

# Between-checkpoint alerts on any compact-log record, at most one per kind
# per ALERT_GAP iterations.
ALERTS = {
  "vloss": lambda r: float(r["vloss"]) > 5.0,
  "std": lambda r: float(r["std"]) >= 0.26,
  "nonfinite": lambda r: any(r.get(k, "0").lower() in ("nan", "inf", "-inf")
                             for k in ("reward", "vloss", "std")),
}
ALERT_GAP = 500

libc = ctypes.CDLL("libc.so.6", use_errno=True)


def stage(it: int) -> str:
  # Rough, stretched curriculum (env_cfgs.ROUGH_STRETCH).
  for bound, name in ((1500, "A"), (4500, "B"), (8500, "C"), (12500, "D")):
    if it < bound:
      return name
  return "E"


def in_window(it: int) -> bool:
  r = it % EVERY
  return r == 0 or r > EVERY - AVG


class LogReader:
  """Incremental reader of the compact log (``it=N key=value ...`` lines)."""

  def __init__(self, path: str, table: str):
    self.path = path
    self.table = table
    self.offset = 0
    self.partial = ""
    self.records: dict[int, dict] = {}
    self.latest = -1
    self.tail: deque[str] = deque(maxlen=30)
    self.last_alert: dict[str, int] = {}
    self.alerts_on = False  # off while reading history at startup

  def read(self) -> None:
    if not os.path.exists(self.path):
      return
    if os.path.getsize(self.path) < self.offset:  # rotated
      self.offset = 0
    with open(self.path, "r", errors="replace") as f:
      f.seek(self.offset)
      while chunk := f.read(READ_CHUNK):
        lines = (self.partial + chunk).split("\n")
        self.partial = lines.pop()
        for line in lines:
          self._line(line)
      self.offset = f.tell()

  def _line(self, line: str) -> None:
    if not line.startswith("it="):
      if line.strip():
        self.tail.append(line.rstrip())
      return
    rec = dict(tok.split("=", 1) for tok in line.split() if "=" in tok)
    try:
      it = int(rec["it"])
    except (KeyError, ValueError):
      return
    self.latest = max(self.latest, it)
    if in_window(it):
      self.records[it] = rec
    if not self.alerts_on:
      return
    for kind, test in ALERTS.items():
      try:
        hit = test(rec)
      except (KeyError, ValueError):
        hit = False
      if hit and it - self.last_alert.get(kind, -10**9) >= ALERT_GAP:
        self.last_alert[kind] = it
        append(self.table, f"\n**ALERT {kind}** at it={it} ({time.strftime('%m-%d %H:%M')}): "
               f"reward={rec.get('reward')} vloss={rec.get('vloss')} std={rec.get('std')} "
               f"falls={rec.get('falls')}\n")

  def window(self, n: int) -> list[dict]:
    return [self.records[i] for i in range(n - AVG + 1, n + 1) if i in self.records]


def _mean(recs: list[dict], key: str) -> float | None:
  vals = []
  for r in recs:
    try:
      vals.append(float(r[key]))
    except (KeyError, ValueError):
      pass
  return sum(vals) / len(vals) if vals else None


def fmt_row(n: int, recs: list[dict], when: str | None = None) -> str:
  # rsl_rl prints elapsed/ETA via gmtime, which wraps at 24 h; use wall-clock
  # time and an ETA computed from the mean iteration time instead.
  cells = [str(n), when or time.strftime("%m-%d %H:%M")]
  for _, key, f, scale in COLS:
    v = _mean(recs, key)
    cells.append(f.format(v * scale) if v is not None else "-")
  it_s = _mean(recs, "it_s")
  if it_s is not None and when is None:
    left = max(0, FINAL_ITER - n) * it_s
    finish = time.strftime("%m-%d %H:%M", time.localtime(time.time() + left))
    cells.append(f"{left / 3600:.1f} h ({finish})")
  else:
    cells.append("-")
  return "| " + " | ".join(cells) + " |"


def table_header() -> str:
  titles = ["iter", "time"] + [c[0] for c in COLS] + ["ETA"]
  return "| " + " | ".join(titles) + " |\n" + "|" + "---|" * len(titles) + "\n"


def rows_done(table: str) -> set[int]:
  done = set()
  if os.path.exists(table):
    for line in open(table):
      m = re.match(r"^\|\s*(\d+)\s*\|", line)
      if m:
        done.add(int(m.group(1)))
  return done


def append(table: str, text: str) -> None:
  with open(table, "a") as f:
    f.write(text + "\n")


def add_watch(fd: int, path: str, mask: int) -> int:
  wd = libc.inotify_add_watch(fd, path.encode(), mask)
  if wd < 0:
    raise OSError(ctypes.get_errno(), f"inotify_add_watch {path}")
  return wd


def main() -> None:
  run_dir, log_path, table, pid = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
  run_name = os.path.basename(run_dir.rstrip("/"))

  if not os.path.exists(table):
    with open(table, "w") as f:
      f.write(
        f"# {run_name}\n\n"
        f"One row per {EVERY}-iteration checkpoint; every metric is the mean "
        f"over the {AVG} iterations ending at that checkpoint. 'apex cm' is the "
        "training metric (averaged over all robots, standing/slope robots count "
        "as 0), so it reads well below the real foot lift; use it for trend. "
        "Rows written at startup are backfilled from the compact log (time = "
        "backfill time).\n\n"
      )
      f.write(table_header())

  pidfd = libc.syscall(SYS_PIDFD_OPEN, pid, 0)
  if pidfd < 0:
    append(table, f"\n**Trainer PID {pid} not running at monitor start.**")
    return

  ifd = libc.inotify_init1(os.O_CLOEXEC)
  if ifd < 0:
    raise OSError(ctypes.get_errno(), "inotify_init1")
  dir_wd = add_watch(ifd, run_dir, IN_CLOSE_WRITE | IN_MOVED_TO)
  # The log is watched for the whole run: every record is checked for alerts.
  add_watch(ifd, log_path, IN_MODIFY)

  reader = LogReader(log_path, table)
  done = rows_done(table)
  pending: set[int] = set()

  # Backfill checkpoints already on disk (monitor restarted mid-run).
  for name in os.listdir(run_dir):
    m = CKPT.match(name)
    # The resume iteration's own save holds only the startup record; skip it.
    if m and int(m.group(1)) % EVERY == 0 and int(m.group(1)) > START_ITER:
      if int(m.group(1)) not in done:
        pending.add(int(m.group(1)))

  def flush_pending(force: bool = False) -> None:
    reader.read()
    for n in sorted(pending):
      if n in reader.records or reader.latest > n or force:
        append(table, fmt_row(n, reader.window(n)))
        done.add(n)
        pending.discard(n)

  flush_pending()  # backfill, if anything is pending
  reader.alerts_on = True

  poller = select.poll()
  poller.register(ifd, select.POLLIN)
  poller.register(pidfd, select.POLLIN)

  while True:
    for fd, _ in poller.poll():  # blocks until an event; no timeout
      if fd == pidfd:
        flush_pending(force=True)
        reader.read()
        ckpts = sorted(
          int(m.group(1)) for n in os.listdir(run_dir) if (m := CKPT.match(n))
        )
        last_ckpt = ckpts[-1] if ckpts else None
        crashed = any("Traceback" in l or "Error" in l for l in reader.tail)
        status = (
          "FINISHED" if last_ckpt is not None and last_ckpt >= FINAL_ITER
          else "CRASHED" if crashed else "STOPPED"
        )
        msg = (
          f"\n**Trainer exited: {status}** at {time.strftime('%Y-%m-%d %H:%M')}; "
          f"last logged iteration {reader.latest}, last checkpoint "
          f"model_{last_ckpt}.pt"
        )
        if status != "FINISHED":
          msg += "\n\n```\n" + "\n".join(list(reader.tail)[-12:]) + "\n```"
        append(table, msg)
        return
      buf = os.read(ifd, 65536)
      off = 0
      while off < len(buf):
        wd, mask, _, nlen = struct.unpack_from("iIII", buf, off)
        name = buf[off + 16 : off + 16 + nlen].rstrip(b"\0").decode()
        off += 16 + nlen
        if wd == dir_wd:
          m = CKPT.match(name)
          if m:
            n = int(m.group(1))
            if n > 0 and n % EVERY == 0 and n not in done:
              pending.add(n)
      flush_pending()


if __name__ == "__main__":
  main()
