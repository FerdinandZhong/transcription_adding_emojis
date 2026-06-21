#!/usr/bin/env bash
# Create a Python venv and install GPU-ready dependencies for emoji-asr training.
#
# Usage (from repo root or anywhere):
#   bash setup/install_gpu.sh
#
# Options (environment variables):
#   VENV_DIR=.venv          virtualenv path (default: <repo>/.venv)
#   PYTHON=python3.11       Python interpreter used to create venv
#   TORCH_CUDA=cu124        PyTorch CUDA wheel tag: cu124, cu121, cu118, or cpu
#   SKIP_VERIFY=1           skip post-install verification
#
# Example (CUDA 12.1):
#   TORCH_CUDA=cu121 bash setup/install_gpu.sh

set -euo pipefail

# Avoid corporate/local proxy breaking PyTorch wheel downloads.
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy || true

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-$ROOT/.venv}"
PYTHON="${PYTHON:-python3}"
TORCH_CUDA="${TORCH_CUDA:-cu124}"
TORCH_INDEX="https://download.pytorch.org/whl/${TORCH_CUDA}"

echo "==> Repo:      $ROOT"
echo "==> Venv:      $VENV_DIR"
echo "==> Python:    $PYTHON"
echo "==> Torch CUDA: $TORCH_CUDA ($TORCH_INDEX)"

if ! command -v "$PYTHON" >/dev/null 2>&1; then
  echo "ERROR: $PYTHON not found. Install Python 3.9+ or set PYTHON=..." >&2
  exit 1
fi

cd "$ROOT"

if [[ ! -d "$VENV_DIR" ]]; then
  echo "==> Creating virtual environment..."
  "$PYTHON" -m venv "$VENV_DIR"
else
  echo "==> Reusing existing venv at $VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

# CDSW and some platforms export PIP_USER=1 globally; that breaks venv installs.
unset PIP_USER PYTHONUSERBASE
export PIP_NO_CACHE_DIR="${PIP_NO_CACHE_DIR:-1}"

pip_install() {
  python -m pip install --no-user "$@"
}

echo "==> Upgrading pip..."
pip_install --upgrade pip setuptools wheel

echo "==> Installing PyTorch (CUDA wheel)..."
pip_install torch torchvision torchaudio --index-url "$TORCH_INDEX"

echo "==> Installing project dependencies..."
pip_install -r "$ROOT/setup/requirements-gpu.txt"

echo "==> Installing emoji-asr package (editable)..."
pip_install -e "$ROOT"

if [[ "${SKIP_VERIFY:-0}" != "1" ]]; then
  echo "==> Verifying environment..."
  python "$ROOT/setup/verify_env.py"
fi

cat <<EOF

Done.

Activate the environment:
  source "$VENV_DIR/bin/activate"

Quick training sanity check (lite encoder, synthetic data):
  python -m emoji_asr.experiment

Real MELD data (after copying data/processed/meld_silver_openai/):
  PYTHONPATH=src python -m emoji_asr.train --config configs/default.yaml

EOF
