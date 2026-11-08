#!/bin/bash
# Evaluation of a planner adapter, one shard per GPU, then the merge (report in <out>).
#   bash scripts/eval_planner_8gpu.sh <planner data dir> <adapter dir> <out dir> [eval_planner.py args...]
# e.g. sweep every 3 s:          ... $D out/G/adapter out/G/adapter/sweep_every3 --sweep --ask-every 3 --per-task 200
#      chains every 3 s:         ... $D out/G/adapter out/G/adapter/chains --chains 60 --every 3 --confirm 2
#      chains, threshold 0.8:    ... $D out/G/adapter out/G/adapter/chains --chains 60 --every 3 --confirm 2 --threshold 0.8
# A planner on another base model also needs --base <base model dir>.
set -euo pipefail
cd "$(dirname "$0")/.."
D=$1; ADAPTER=$2; OUT=$3; shift 3
PY=${PLANNER_PYTHON:-python}
NGPU=${NGPU:-8}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
pids=()
for ((k = 0; k < NGPU; k++)); do
  CUDA_VISIBLE_DEVICES=$k "$PY" policy/longhorizon/eval_planner.py --data "$D" --adapter "$ADAPTER" --out "$OUT" \
      --shard "$k/$NGPU" "$@" > "$OUT.shard$k.log" 2>&1 &
  pids+=($!)
done
mkdir -p "$OUT"
for pid in "${pids[@]}"; do wait "$pid"; done
"$PY" policy/longhorizon/eval_planner.py --data "$D" --adapter "$ADAPTER" --out "$OUT" --merge "$NGPU" "$@"
