#!/usr/bin/env bash
# Run training off-screen with nohup
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

LOG_FILE="${REPO_ROOT}/training.log"

nohup bash -c "cd '${REPO_ROOT}' && export PYTHONPATH='${REPO_ROOT}/src':\"\$PYTHONPATH\" && PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -m seg3d.training.run_trainer" > "$LOG_FILE" 2>&1 &

echo "Training started in background. PID: $!"
echo "Logs: $LOG_FILE"
echo "Monitor: tail -f $LOG_FILE"
