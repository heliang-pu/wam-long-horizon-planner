#!/bin/bash
# LoRA fine-tuning of the planner on one machine's GPUs.
#   bash scripts/train_planner_8gpu.sh <planner data dir> <out dir> [train_planner.py args...]
# e.g. G:  bash scripts/train_planner_8gpu.sh $D out/G --state --progress --samples samples_mem.jsonl
#      G2: bash scripts/train_planner_8gpu.sh $D out/G2 --state --progress --samples samples_mem2.jsonl --past 20,10,3,2,1
set -euo pipefail
cd "$(dirname "$0")/.."
D=$1; OUT=$2; shift 2
PY=${PLANNER_PYTHON:-python}
NGPU=${NGPU:-8}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
"$PY" -m torch.distributed.run --nproc_per_node="$NGPU" --master_port "${MASTER_PORT:-29612}" \
    policy/longhorizon/train_planner.py --data "$D" --out "$OUT" --save-every 2000 "$@"
