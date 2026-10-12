#!/usr/bin/env bash
# Poultry-floor study: 96 robots per condition, 40 s routine.
cd "$(dirname "$0")/../.."   # repo root
export PYTHONPATH=. MUJOCO_GL=egl
PY=${PYTHON:-python}
P915=checkpoints/poultry_v3_squat_91500/model_91500.pt
P90=checkpoints/poultry_v2_90000/model_90000.pt
O=training_records/poultry_floor_eval
run() { name=$1; shift; $PY scripts/poultry_floor/eval_poultry_floor.py --n 96 --json $O/$name.json "$@" > $O/$name.log 2>&1; echo "$name done: $(grep -c 'height field collision overflow' $O/$name.log) overflow warnings"; }
run 91500_rigid_baseline --policy $P915 --contact rigid --flat --no-drag --no-pans
run 91500_built_up       --policy $P915 --contact built_up
run 91500_fresh_deep     --policy $P915 --contact fresh_deep
run 90000_built_up       --policy $P90  --contact built_up
echo STUDY_DONE
