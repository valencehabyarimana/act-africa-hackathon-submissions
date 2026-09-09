#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

echo "============================================================"
echo "ACT-Africa 2026 Bearing Challenge - macOS/Linux setup"
echo "============================================================"

if command -v python3 >/dev/null 2>&1; then
  PY=python3
elif command -v python >/dev/null 2>&1; then
  PY=python
else
  echo "ERROR: Python was not found. Install Python 3.11, then run this file again."
  exit 1
fi

"$PY" --version
"$PY" -m pip install -r requirements.txt
"$PY" check_environment.py

echo
echo "READY"
echo "Launch the workshop notebook with:"
echo "  $PY -m notebook starter.ipynb"
