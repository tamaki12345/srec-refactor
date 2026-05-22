#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_VERSION="3.11"
ENV_DIR=".venv"
TORCH_TARGET="cpu" # cpu or cu121

print_help() {
  cat <<EOF
Usage: ./scripts/setup_uv_env.sh [options]

Options:
  --python <version>    Python version for uv venv (default: 3.11)
  --env-dir <path>      Virtualenv directory (default: .venv)
  --torch <target>      Torch target: cpu or cu121 (default: cpu)
  --help                Show this help message

Examples:
  ./scripts/setup_uv_env.sh
  ./scripts/setup_uv_env.sh --torch cu121
  ./scripts/setup_uv_env.sh --python 3.11 --env-dir .venv-cu121 --torch cu121
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --python)
      PYTHON_VERSION="$2"
      shift 2
      ;;
    --env-dir)
      ENV_DIR="$2"
      shift 2
      ;;
    --torch)
      TORCH_TARGET="$2"
      shift 2
      ;;
    --help|-h)
      print_help
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      print_help
      exit 1
      ;;
  esac
done

if [[ "$TORCH_TARGET" != "cpu" && "$TORCH_TARGET" != "cu121" ]]; then
  echo "Invalid --torch value: ${TORCH_TARGET}. Use cpu or cu121." >&2
  exit 1
fi

UV_BIN="$(command -v uv || true)"
if [[ -z "$UV_BIN" && -x "$HOME/.local/bin/uv" ]]; then
  UV_BIN="$HOME/.local/bin/uv"
fi

if [[ -z "$UV_BIN" ]]; then
  echo "uv not found. Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  UV_BIN="$HOME/.local/bin/uv"
fi

if [[ ! -x "$UV_BIN" ]]; then
  echo "uv installation failed or uv is not executable." >&2
  exit 1
fi

cd "$PROJECT_DIR"

echo "[1/5] Creating venv at ${ENV_DIR} (Python ${PYTHON_VERSION})"
"$UV_BIN" venv --python "$PYTHON_VERSION" "$ENV_DIR"

PYTHON_BIN="${ENV_DIR}/bin/python"
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python executable not found at ${PYTHON_BIN}" >&2
  exit 1
fi

echo "[2/5] Upgrading packaging tools"
"$UV_BIN" pip install --python "$PYTHON_BIN" -U pip setuptools wheel

echo "[3/5] Installing pinned core dependencies"
"$UV_BIN" pip install --python "$PYTHON_BIN" -r requirements.refactor.txt

echo "[4/5] Installing PyTorch stack (${TORCH_TARGET})"
if [[ "$TORCH_TARGET" == "cpu" ]]; then
  "$UV_BIN" pip install --python "$PYTHON_BIN" -r requirements.torch.cpu.txt
else
  "$UV_BIN" pip install --python "$PYTHON_BIN" -r requirements.torch.cu121.txt
fi

echo "[5/5] Verifying setup"
"$PYTHON_BIN" -V
"$PYTHON_BIN" -c "import torch, numpy, pandas; print('torch', torch.__version__); print('cuda', torch.cuda.is_available())"

echo
echo "Setup complete."
echo "Activate with: source ${ENV_DIR}/bin/activate"
