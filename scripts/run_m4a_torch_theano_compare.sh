#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="$ROOT_DIR/results/compare"
TS="$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

TORCH_STRICT_LOG="$LOG_DIR/m4a_torch_theano_strict_${TS}.log"
TORCH_OPT_LOG="$LOG_DIR/m4a_torch_optimized_${TS}.log"
THEANO_LOG="$LOG_DIR/m4a_theano_${TS}.log"
SUMMARY_MD="$LOG_DIR/summary_${TS}.md"

run_with_optional_timeout() {
  if [ "${QUICK_RUN:-1}" = "1" ]; then
    timeout "${QUICK_TIMEOUT_SEC:-60}" "$@" || {
      rc="$?"
      if [ "$rc" -eq 124 ]; then
        echo "[WARN] QUICK_RUN timeout reached for command: $*"
      else
        return "$rc"
      fi
    }
  else
    "$@"
  fi
}

extract_pair_s() {
  local file="$1"
  local value
  value="$(grep -Eo '[0-9]+\.[0-9]+pair/s' "$file" | tail -n 1 | sed 's/pair\/s//' || true)"
  if [ -z "$value" ]; then
    value="N/A"
  fi
  echo "$value"
}

echo "[INFO] Root: $ROOT_DIR"
echo "[INFO] Torch strict log: $TORCH_STRICT_LOG"
echo "[INFO] Torch optimized log: $TORCH_OPT_LOG"
echo "[INFO] Theano log: $THEANO_LOG"
echo "[INFO] Summary: $SUMMARY_MD"

echo "[INFO] Running Torch strict parity experiment (experiment/m4a_torch_theano_strict.yml)"
(
  cd "$ROOT_DIR"
  run_with_optional_timeout /usr/bin/time -p /home/tamak/srec-refactor/.venv/bin/python run_config.py experiment/m4a_torch_theano_strict.yml
) 2>&1 | tee "$TORCH_STRICT_LOG"

echo "[INFO] Running Torch optimized experiment (experiment/m4a_torch_optimized.yml)"
(
  cd "$ROOT_DIR"
  run_with_optional_timeout /usr/bin/time -p /home/tamak/srec-refactor/.venv/bin/python run_config.py experiment/m4a_torch_optimized.yml
) 2>&1 | tee "$TORCH_OPT_LOG"

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
  if [ "${USE_GPU:-0}" = "1" ]; then
    export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
    export THEANO_FLAGS="device=cuda0,floatX=float32,dnn.base_path=$CONDA_PREFIX,dnn.include_path=$CONDA_PREFIX/include,dnn.library_path=$CONDA_PREFIX/lib"
    echo "[INFO] USE_GPU=1: running Theano with GPU (cuda0)"
  else
    export THEANO_FLAGS="device=cpu,floatX=float32"
    echo "[INFO] USE_GPU is not 1: running Theano with CPU"
  fi
  set -u
  run_with_optional_timeout /usr/bin/time -p python run_config.py experiment/m4a_theano.yml
) 2>&1 | tee "$THEANO_LOG"

if grep -q "Falling back to GRU4RecTorch" "$THEANO_LOG"; then
  echo "[ERROR] Theano run fell back to GRU4RecTorch. Check the 'srec' environment and Theano installation."
  exit 2
fi

TORCH_STRICT_PAIR_S="$(extract_pair_s "$TORCH_STRICT_LOG")"
TORCH_OPT_PAIR_S="$(extract_pair_s "$TORCH_OPT_LOG")"
THEANO_PAIR_S="$(extract_pair_s "$THEANO_LOG")"

{
  echo "# M4A Torch vs Theano Summary ($TS)"
  echo
  echo "| Run | pair/s | Log |"
  echo "|---|---:|---|"
  echo "| Torch strict parity | $TORCH_STRICT_PAIR_S | $TORCH_STRICT_LOG |"
  echo "| Torch optimized | $TORCH_OPT_PAIR_S | $TORCH_OPT_LOG |"
  echo "| Theano (${USE_GPU:-0}=>1 means GPU) | $THEANO_PAIR_S | $THEANO_LOG |"
} > "$SUMMARY_MD"

echo "[INFO] Finished. Logs saved in $LOG_DIR"
echo "[INFO] Summary written to $SUMMARY_MD"
