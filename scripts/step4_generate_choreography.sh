#!/bin/sh
set -eu

REPO="$(cd "$(dirname "$0")/.." && pwd)"
CAPDIR="${CAPTURES_DIR:-$REPO/captures}"

cd "$REPO"

if [ -f "$REPO/.venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  . "$REPO/.venv/bin/activate"
fi

export PYTHONPATH="$REPO/src"

exec python -m manage.generate_choreography "$CAPDIR"
