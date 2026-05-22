# session-rec refactor: uv environment guide

## 1. Goal

This repository was originally built around Python 3.5/3.7 + Theano/TensorFlow 1.x.
For the refactor, we standardize on a modern environment centered on PyTorch.

Selected baseline:

- Python: 3.11
- Package manager / virtualenv: uv
- Deep learning framework: PyTorch 2.x (CPU or CUDA)

Why this choice:

- Python 3.11 has broad package support and strong performance/stability.
- uv is fast and reproducible for environment creation and dependency install.
- PyTorch 2.x is the practical modern target for rewriting legacy neural models.

## 2. Install uv

Linux/macOS:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Reload shell and verify:

```bash
uv --version
```

## 3. Create virtual environment

From repository root:

```bash
cd /home/tamaki/srec-refactor
uv venv --python 3.11 .venv
source .venv/bin/activate
```

## 4. Install core dependencies (common)

```bash
uv pip install -U pip setuptools wheel
uv pip install \
	numpy pandas scipy scikit-learn pyyaml networkx numexpr tables \
	scikit-optimize psutil pympler python-dateutil pytz dill \
	python-telegram-bot
```

## 5. Install PyTorch

Choose one of the following.

CPU:

```bash
uv pip install torch torchvision torchaudio \
	--index-url https://download.pytorch.org/whl/cpu
```

NVIDIA GPU (CUDA 12.1 wheels):

```bash
uv pip install torch torchvision torchaudio \
	--index-url https://download.pytorch.org/whl/cu121
```

If your CUDA stack differs, replace `cu121` with the matching PyTorch wheel index.

## 6. Verify environment

```bash
python -V
python -c "import torch, numpy, pandas; print('torch', torch.__version__)"
python -c "import torch; print('cuda', torch.cuda.is_available())"
```

## 7. Run existing scripts with uv environment

After activation:

```bash
python run_config.py conf/example_next.yml
python run_preprocessing.py conf/preprocess/window/rsc15.yml
```

Or without manual activation:

```bash
uv run python run_config.py conf/example_next.yml
uv run python run_preprocessing.py conf/preprocess/window/rsc15.yml
```

## 8. Dependency management during refactor

When adding a new package:

```bash
uv pip install <package>
```

When syncing on another machine (using a frozen list):

```bash
uv pip freeze > requirements.lock.txt
uv pip install -r requirements.lock.txt
```

## 9. Notes on legacy models

- Old code paths that require Theano / TensorFlow 1.x are not part of this modern baseline.
- Refactoring target is migration to PyTorch-based implementations.
- If legacy reproduction is required, use a separate legacy environment (Python 3.7 + old dependencies) and keep it isolated from this refactor environment.

## 10. One-command setup for other PCs

A reusable script is provided to reproduce the same environment:

```bash
cd /home/tamaki/srec-refactor
chmod +x scripts/setup_uv_env.sh
./scripts/setup_uv_env.sh
```

GPU setup (CUDA 12.1):

```bash
./scripts/setup_uv_env.sh --torch cu121
```

Custom Python or venv path:

```bash
./scripts/setup_uv_env.sh --python 3.11 --env-dir .venv-refactor
```

This script performs:

- uv installation (if missing)
- venv creation via uv
- pinned dependency installation from `requirements.refactor.txt`
- pinned PyTorch installation (CPU or cu121)
- final import/version sanity check

