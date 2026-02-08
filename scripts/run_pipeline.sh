#!/bin/sh
set -eu

REPO="$(cd "$(dirname "$0")/.." && pwd)"

cd "$REPO"

# If a project virtualenv exists, activate it so users don't need to set PYTHONHOME/PYTHONPATH
if [ -f "$REPO/.venv/bin/activate" ]; then
	# shellcheck disable=SC1091
	. "$REPO/.venv/bin/activate"
	PYTHON="$REPO/.venv/bin/python3"
	[ -x "$PYTHON" ] || PYTHON=python3
else
	PYTHON=python3
fi

# Ensure we can import from src/ (computed from repo root)
export PYTHONPATH="$REPO/src"

exec "$PYTHON" -m cyclemusic.main
