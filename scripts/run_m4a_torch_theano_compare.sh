#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="$ROOT_DIR/results/compare"
TS="$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

TORCH_LOG="$LOG_DIR/m4a_torch_${TS}.log"
THEANO_LOG="$LOG_DIR/m4a_theano_${TS}.log"

echo "[INFO] Root: $ROOT_DIR"
echo "[INFO] Torch log: $TORCH_LOG"
echo "[INFO] Theano log: $THEANO_LOG"

# echo "[INFO] Running Torch experiment (experiment/m4a_torch.yml)"
# (
#   cd "$ROOT_DIR"
#   /usr/bin/time -p env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run python run_config.py experiment/m4a_torch.yml
# ) 2>&1 | tee "$TORCH_LOG"

echo "[INFO] Running Theano experiment in conda env 'srec' (experiment/m4a_theano.yml)"
(
  cd "$ROOT_DIR"
  if ! command -v conda >/dev/null 2>&1; then
    echo "[ERROR] conda command not found."
    exit 1
  fi
  # Some conda activate/deactivate hooks reference optional vars that may be
  # unset; temporarily relax nounset to avoid false-positive aborts.
  set +u
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate srec
  set -u
  # Force CPU backend for Theano to avoid GPU/cuDNN runtime dependency in this comparison run.
  /usr/bin/time -p env THEANO_FLAGS=device=cpu python run_config.py experiment/m4a_theano.yml
) 2>&1 | tee "$THEANO_LOG"

if grep -q "Falling back to GRU4RecTorch" "$THEANO_LOG"; then
  echo "[ERROR] Theano run fell back to GRU4RecTorch. Check the 'srec' environment and Theano installation."
  exit 2
fi

echo "[INFO] Finished. Logs saved in $LOG_DIR"
