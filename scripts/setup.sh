#!/usr/bin/env bash
# Create a local virtual environment and install dependencies.
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python3}"

if ! "$PYTHON" -m venv --help >/dev/null 2>&1; then
  echo "python3-venv is missing. On Ubuntu/WSL run: sudo apt install -y python3-venv" >&2
  exit 1
fi

"$PYTHON" -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt

echo
echo "Setup complete. Activate with:  source .venv/bin/activate"
