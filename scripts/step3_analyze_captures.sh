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

ANALYZE_FLAGS=""
if [ "${ANALYSIS_ONLY:-0}" = "1" ]; then
  ANALYZE_FLAGS="--analysis-only"
fi

found=0
for wav in "$CAPDIR"/*.wav; do
  [ -f "$wav" ] || continue
  found=1
  echo "[analyze] $wav"
  python -m analyze.analyze_track $ANALYZE_FLAGS "$wav"
done

if [ "$found" -eq 0 ]; then
  echo "[analyze] No wav files found in $CAPDIR"
fi
