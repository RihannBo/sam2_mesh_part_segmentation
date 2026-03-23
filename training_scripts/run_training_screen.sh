#!/usr/bin/env bash
# Run training off-screen with screen
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

SESSION_NAME="${SESSION_NAME:-training}"

# Optional conda (set CONDA_ENV to your env name, or comment this block if using system Python)
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

# Start screen in detached mode with training command
screen -dmS "$SESSION_NAME" bash -c "
cd '${REPO_ROOT}'
export PYTHONPATH='${REPO_ROOT}/src':\"\$PYTHONPATH\"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -m seg3d.training.run_trainer
exec bash
"

echo "Training started in screen session: $SESSION_NAME"
echo ""
echo "Useful commands:"
echo "  Attach to session:    screen -r $SESSION_NAME"
echo "  List all sessions:    screen -ls"
echo "  Detach (while inside): Press Ctrl+A then D"
echo "  Kill session:         screen -X -S $SESSION_NAME quit"
