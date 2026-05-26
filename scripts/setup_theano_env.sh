#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-srec_theano_gpu}"
PY_VER="${PY_VER:-3.8}"
if [[ -z "${CUDA_HOME:-}" ]]; then
	if [[ -x "/usr/local/cuda-12/bin/nvcc" ]]; then
		CUDA_HOME="/usr/local/cuda-12"
	else
		CUDA_HOME="/usr/local/cuda"
	fi
fi
THEANO_FLAGS_BASE="device=cuda0,floatX=float32,dnn.enabled=False,force_device=True,blas.ldflags="

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK_DIR="${WORK_DIR:-/tmp/libgpuarray_build_${USER}}"

if ! command -v conda >/dev/null 2>&1; then
	echo "[ERROR] conda command not found." >&2
	exit 1
fi

if [[ ! -x "${CUDA_HOME}/bin/nvcc" ]]; then
	echo "[ERROR] nvcc not found at ${CUDA_HOME}/bin/nvcc. Set CUDA_HOME correctly." >&2
	exit 1
fi

# conda activate in non-interactive shell
set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
set -u

echo "[1/8] Create/activate conda env: ${ENV_NAME} (python=${PY_VER})"
# conda activate/deactivate hooks may reference optional vars (e.g., CONDA_BACKUP_CXX)
# and can fail under nounset. Temporarily disable set -u for these operations.
set +u
if conda env list | awk '{print $1}' | grep -Fxq "${ENV_NAME}"; then
	echo "[INFO] conda env '${ENV_NAME}' already exists. Reusing it."
else
	conda create -n "${ENV_NAME}" "python=${PY_VER}" -y
fi
conda activate "${ENV_NAME}"
set -u

echo "[2/8] Install Python build/runtime deps"
python -m pip install -U "pip<25" "setuptools<70" "wheel<0.45"
# pygpu/libgpuarray is not compatible with Cython 3.x.
# Theano 1.0.5 references np.bool, so NumPy must be <1.24.
python -m pip install -U "cython<3" "numpy<1.24" scipy pandas pyyaml tqdm
# Avoid easy_install pulling modern sdist-only deps that break with legacy setup.py flows.
python -m pip install -U "mako<1.3.11" "MarkupSafe<3"

echo "[3/8] Install native build tools in conda env"
# libgpuarray build requires cmake/make/compiler toolchain.
set +u
# libgpuarray's CMakeLists sets old policies that break on CMake 4.x.
# Pin to CMake 3.x for compatibility.
conda install -n "${ENV_NAME}" -y "cmake<4" make pkg-config git gcc_linux-64 gxx_linux-64
# Driver 535 + system CUDA 12.5 can trigger PTX version mismatch in Theano/pygpu.
# Prefer conda cudatoolkit 11.8 runtime (libnvrtc) for stable runtime JIT.
conda install -n "${ENV_NAME}" -y -c conda-forge "cudatoolkit=11.8"
# Python.h in this env may require crypt.h; provide it from conda.
conda install -n "${ENV_NAME}" -y -c conda-forge libxcrypt
set -u

if [[ ! -f "${CONDA_PREFIX}/include/crypt.h" ]]; then
	echo "[ERROR] Missing ${CONDA_PREFIX}/include/crypt.h after libxcrypt install." >&2
	echo "        Try: conda install -n ${ENV_NAME} -y -c conda-forge libxcrypt" >&2
	exit 1
fi

export CUDA_HOME
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

echo "[4/8] Prepare libgpuarray sources"
mkdir -p "${WORK_DIR}"
cd "${WORK_DIR}"
if [[ -d libgpuarray/.git ]]; then
	git -C libgpuarray fetch --all --tags
	git -C libgpuarray reset --hard origin/master
else
	git clone https://github.com/Theano/libgpuarray.git
fi

echo "[5/8] Build/install libgpuarray C library"
rm -rf libgpuarray/build
mkdir -p libgpuarray/build
cd libgpuarray/build
cmake .. \
	-DCMAKE_BUILD_TYPE=Release \
	-DCMAKE_INSTALL_PREFIX="${CONDA_PREFIX}" \
	-DCMAKE_POLICY_VERSION_MINIMUM=3.5
make -j"$(nproc)"
make install

echo "[6/8] Build/install pygpu Python package"
cd "${WORK_DIR}/libgpuarray"
python setup.py build_ext -I"${CONDA_PREFIX}/include" -L"${CONDA_PREFIX}/lib"
# setup.py install --no-deps is not supported in this project setup script.
# Use pip with no-deps/no-build-isolation to avoid pulling incompatible modern deps.
python -m pip install . --no-deps --no-build-isolation

echo "[7/8] Install Theano"
python -m pip install -U "Theano==1.0.5"
# run_config.py imports skopt unconditionally.
python -m pip install -U "scikit-optimize" "dill"

echo "[8/8] GPU smoke test + experiment run"
THEANO_FLAGS="${THEANO_FLAGS_BASE},print_active_device=True" \
python - <<'PY'
import theano
import theano.tensor as T

f = theano.function([], T.constant(1.0))
print("ok", f())
print("device:", theano.config.device)
PY

cd "${ROOT_DIR}"
THEANO_FLAGS="${THEANO_FLAGS_BASE}" \
python run_config.py experiment/m4a_theano.yml