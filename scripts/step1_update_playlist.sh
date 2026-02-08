#!/bin/sh
set -eu

REPO="$(cd "$(dirname "$0")/.." && pwd)"

cd "$REPO"

if [ -f "$REPO/.venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  . "$REPO/.venv/bin/activate"
fi

export PYTHONPATH="$REPO/src"

exec python -m manage.manage_processing_playlist
