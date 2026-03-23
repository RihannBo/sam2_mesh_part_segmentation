#!/usr/bin/env bash
# Resume training from best checkpoint with reduced learning rates
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

SESSION_NAME="${SESSION_NAME:-training_resume}"
BEST_CHECKPOINT="${BEST_CHECKPOINT:-${REPO_ROOT}/checkpoints/best.pt}"
RESUME_CONFIG="${RESUME_CONFIG:-${REPO_ROOT}/sam2/sam2/configs/sam2.1_training/sam2.1_multiview_finetune_resume.yaml}"

# Check if best checkpoint exists
if [ ! -f "$BEST_CHECKPOINT" ]; then
    echo "Error: Best checkpoint not found at $BEST_CHECKPOINT"
    echo "   Set BEST_CHECKPOINT or ensure training has written this file."
    exit 1
fi

# Optional conda
if [ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]; then
  # shellcheck source=/dev/null
  source "${HOME}/miniconda3/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV:-seg3d_env}"
elif [ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]; then
  # shellcheck source=/dev/null
  source "${HOME}/anaconda3/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV:-seg3d_env}"
fi

# Check if screen session already exists
if screen -list | grep -q "$SESSION_NAME"; then
    echo "Screen session '$SESSION_NAME' already exists!"
    echo "To attach: screen -r $SESSION_NAME"
    echo "To kill it first: screen -X -S $SESSION_NAME quit"
    exit 1
fi

echo "Resuming training from: $BEST_CHECKPOINT"
echo "Using config: $RESUME_CONFIG"
echo ""

# Start screen in detached mode with training command
screen -dmS "$SESSION_NAME" bash -c "
cd '${REPO_ROOT}'
export PYTHONPATH='${REPO_ROOT}/src':\"\$PYTHONPATH\"
export CUDA_VISIBLE_DEVICES=\"\${CUDA_VISIBLE_DEVICES:-0}\"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -m seg3d.training.run_trainer '${RESUME_CONFIG}' '${BEST_CHECKPOINT}'
exec bash
"

echo "Training resumed in screen session: $SESSION_NAME"
echo ""
echo "Useful commands:"
echo "  Attach to session:    screen -r $SESSION_NAME"
echo "  List all sessions:    screen -ls"
echo "  Detach (while inside): Press Ctrl+A then D"
echo "  Kill session:         screen -X -S $SESSION_NAME quit"
echo ""
echo "Monitor training: tail -f ${REPO_ROOT}/checkpoints/log.csv"
