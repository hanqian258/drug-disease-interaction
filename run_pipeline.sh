#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PY="$ROOT_DIR/.venv/bin/python"

if [[ ! -x "$VENV_PY" ]]; then
  echo "Error: virtual environment not found at .venv" >&2
  echo "Create it first: python3 -m venv .venv" >&2
  exit 1
fi

echo "[1/3] Building heterogeneous graph..."
"$VENV_PY" "$ROOT_DIR/02_Code/03_build_hetero_graph.py"

echo "[2/3] Expanding graph..."
"$VENV_PY" "$ROOT_DIR/02_Code/04_expand_graph.py"

echo "[3/3] Validating graph..."
"$VENV_PY" "$ROOT_DIR/02_Code/05_validate_graph.py"

echo "Pipeline completed successfully."
