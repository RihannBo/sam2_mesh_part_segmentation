#!/usr/bin/env bash
# Run training with rotation augmentation in the background
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

if [ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]; then
  # shellcheck source=/dev/null
  source "${HOME}/miniconda3/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV:-seg3d_env}"
elif [ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]; then
  # shellcheck source=/dev/null
  source "${HOME}/anaconda3/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV:-seg3d_env}"
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

mkdir -p "${REPO_ROOT}/logs"

LOG_FILE="${REPO_ROOT}/logs/training_rotaug_$(date +%Y%m%d_%H%M%S).log"
CONFIG="${REPO_ROOT}/sam2/sam2/configs/sam2.1_training/sam2.1_multiview_finetune.yaml"
CKPT="${REPO_ROOT}/checkpoints/best.pt"

nohup python -m seg3d.training.run_trainer "$CONFIG" "$CKPT" > "$LOG_FILE" 2>&1 &

PID=$!

echo "Training started in background!"
echo "Process ID: $PID"
echo "Log file: $LOG_FILE"
echo ""
echo "Monitor (filtered):"
echo "  tail -f \"$LOG_FILE\" | grep -E 'Epoch|Train|Val|loss|ERROR|Error|Exception|Traceback|Finished|Starting'"
echo ""
echo "Stop: kill $PID"
