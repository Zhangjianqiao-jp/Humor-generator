#!/bin/sh
set -eu

cd "$(dirname "$0")/.."

module purge
module load python/3.12.11
module load cuda/12.6.1

VENV_DIR="${VENV_DIR:-.venv-genkai}"
python3 -m venv "$VENV_DIR"
. "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip setuptools wheel
python -m pip install \
  torch==2.12.0+cu126 \
  torchvision==0.27.0+cu126 \
  --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt
python -m pip install tensorboard

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch CUDA:", torch.version.cuda)
print("CUDA available on login node:", torch.cuda.is_available())
PY

echo "Environment ready: $VENV_DIR"
