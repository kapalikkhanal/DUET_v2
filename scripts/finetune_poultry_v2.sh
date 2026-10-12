#!/usr/bin/env bash
# Usage: bash scripts/finetune_poultry_v2.sh CHECKPOINT RUN_NAME [NUM_ENVS] [FINAL]
# Runs in the foreground; all trainer output goes through the compact logger.
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
PY=${PYTHON:-python}  # set PYTHON to your env's interpreter if needed
FILTER="$ROOT/scripts/tools/compact_log.py"
CHECKPOINT=$(realpath -- "${1:?checkpoint path required}")
NAME=${2:?run name required}
ENVS=${3:-4096}
FINAL=${4:-100000}
[[ "$NAME" =~ ^[a-zA-Z0-9_-]+$ ]] || { echo 'Use letters, digits, underscores or dashes in RUN_NAME.' >&2; exit 1; }
[[ "$ENVS" =~ ^[1-9][0-9]*$ && "$FINAL" =~ ^[1-9][0-9]*$ ]] || exit 1
test -f "$CHECKPOINT"
test -f "$FILTER"
# Read the actual resume iteration; filenames alone need not be authoritative.
START=$("$PY" - "$CHECKPOINT" <<'PY'
import sys, torch
print(int(torch.load(sys.argv[1], map_location='cpu', weights_only=False)['iter']))
PY
)
ITERS=$(( FINAL - START + 1 ))
(( ITERS > 0 )) || { echo 'FINAL must be at least the checkpoint iteration.' >&2; exit 1; }
cd "$ROOT"
LOAD_RUN="v2_source_${NAME}"
# A unique source directory prevents overwriting an earlier staged checkpoint.
mkdir -p logs/rsl_rl/DUET_G1_23dof_poultry
mkdir "logs/rsl_rl/DUET_G1_23dof_poultry/$LOAD_RUN"
cp -- "$CHECKPOINT" "logs/rsl_rl/DUET_G1_23dof_poultry/$LOAD_RUN/model_${START}.pt"
echo "Resuming iteration $START through $FINAL ($ITERS iterations), $ENVS environments."
PYTHONPATH=. PYTHONUNBUFFERED=1 WANDB_CONSOLE=off CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
  "$PY" scripts/train.py Unitree-G1-23Dof-Duet-PoultrySurface \
  --agent.resume True --agent.load-run "$LOAD_RUN" --agent.load-checkpoint "model_${START}.pt" \
  --agent.max-iterations "$ITERS" --agent.seed 1 --agent.logger tensorboard --agent.upload-model False \
  --env.scene.num-envs "$ENVS" --agent.run-name "$NAME" \
  2>&1 | "$PY" -u "$FILTER" "$ROOT/logs/${NAME}.log"
