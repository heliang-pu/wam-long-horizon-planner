#!/bin/bash
# The high-level planner's interpreter (policy/longhorizon/planner_worker.py, train_planner.py, eval_planner.py):
# RynnBrain1.1-2B (Qwen3.5-2B) needs transformers >= 5.2, LoRA needs peft. Its own venv, so Phoenix's (which the
# bench's worker runs in) is not touched; torch is Phoenix's pin (2.7.1 + CUDA 12.8), from uv's cache when it is there.
#
#   bash policy/longhorizon/setup_planner_env.sh            # -> $WS_ROOT/venvs/planner (PLANNER_VENV to change)
#   UV_DEFAULT_INDEX=https://mirrors.aliyun.com/pypi/simple/ TORCH_FIND_LINKS=https://mirrors.aliyun.com/pytorch-wheels/cu128/ \
#       bash policy/longhorizon/setup_planner_env.sh        # the mirrors the 4090 uses
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
WS_ROOT="${WS_ROOT:-$(cd "$REPO_ROOT/.." && pwd)}"
# a worktree under .worktrees/<name> has the workspace two levels further up
[[ "$(basename "$(dirname "$REPO_ROOT")")" == ".worktrees" ]] && WS_ROOT="${WS_ROOT_OVERRIDE:-$(cd "$REPO_ROOT/../../.." && pwd)}"
VENV="${PLANNER_VENV:-$WS_ROOT/venvs/planner}"
TORCH_FIND_LINKS="${TORCH_FIND_LINKS:-https://download.pytorch.org/whl/cu128}"

uv venv "$VENV" --python 3.11
uv pip install --python "$VENV/bin/python" --find-links "$TORCH_FIND_LINKS" \
    "torch==2.7.1+cu128" "torchvision==0.22.1+cu128"
uv pip install --python "$VENV/bin/python" \
    "transformers==5.20.2" "flash-linear-attention" "peft" "accelerate" "pillow" "av" "pandas" "pyarrow" "numpy" "pyyaml" "pytest"
"$VENV/bin/python" -c "import torch, transformers, peft; print('planner venv:', torch.__version__, transformers.__version__, peft.__version__, torch.cuda.is_available())"
