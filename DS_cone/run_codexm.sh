#!/bin/bash
# Portable launcher. Override PYTHON or DEVICE from the environment if needed.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]:-$0}")" && pwd -P)"
cd "$SCRIPT_DIR"

PYTHON="${PYTHON:-python3}"
DEVICE="${DEVICE:-cuda:0}"

exec "$PYTHON" main.py --dataset codex-m\
    --cuda True\
    --device "$DEVICE"\
    --batch_size 512\
    --max_grad_norm 3.0\
    --nneg 200\
    --npos 1\
    --margin 1.08\
    --max_norm 5.\
    --lr 0.005\
    --gamma 0.9\
    --step_size 40\
    --num_epochs 200\
    --dim 256\
    --valid_steps 25\
    --early_stop 20\
    --optimizer radam\
    --noise_reg 0.05\
