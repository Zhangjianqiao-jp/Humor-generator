#!/bin/sh
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$ROOT"

PYTHON312=${PYTHON312:-/home/app/python/3.12.11/bin/python3}
if [ ! -x "$PYTHON312" ]; then
  echo "Python 3.12.11 is unavailable: $PYTHON312" >&2
  exit 1
fi

"$PYTHON312" -m venv .venv
.venv/bin/python -m pip install --upgrade pip==26.2 setuptools==78.1.0 wheel==0.47.0
.venv/bin/python -m pip install --extra-index-url https://download.pytorch.org/whl/cu126 -r requirements.lock
.venv/bin/python -m pip install --no-deps -e .
export NLTK_DATA="$ROOT/artifacts/nltk_data"
.venv/bin/python scripts/fetch_nltk_assets.py
.venv/bin/python scripts/check_environment.py
