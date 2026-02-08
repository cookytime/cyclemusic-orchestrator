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

ANALYZE_MOD="analyze.analyze_track"

ready() {
  id="$1"
  [ -f "$CAPDIR/$id.wav" ] || return 1
  [ -f "$CAPDIR/$id.metadata.json" ] || return 1
  return 0
}

process_one() {
  wav="$1"
  id="$(basename "$wav" .wav)"

  doneflag="$CAPDIR/$id.analyzed.ok"
  lockdir="$CAPDIR/$id.analyze.lock"
  choreo="$CAPDIR/$id.choreography.json"

  # Require both inputs
  ready "$id" || return 0

  # Already done?
  [ -f "$doneflag" ] && return 0
  [ -f "$choreo" ] && { : > "$doneflag"; return 0; }

  # Simple lock (dir creation is atomic)
  mkdir "$lockdir" 2>/dev/null || return 0
  trap 'rmdir "$lockdir" 2>/dev/null || true' EXIT

  echo "[analyze] $id starting"
  # Run as module so it finds assets under src/
  if python -m "$ANALYZE_MOD" "$CAPDIR/$id.wav"; then
    : > "$doneflag"
    echo "[analyze] $id done"
  else
    echo "[analyze] $id FAILED" >&2
  fi

  rmdir "$lockdir" 2>/dev/null || true
  trap - EXIT
}

echo "[analyze] watching $CAPDIR"

# First pass (anything already present)
for wav in "$CAPDIR"/*.wav; do
  [ -f "$wav" ] && process_one "$wav"
done

if command -v inotifywait >/dev/null 2>&1; then
  inotifywait -m -e close_write,create,move "$CAPDIR" | while read -r _ _ file; do
    case "$file" in
      *.wav)
        process_one "$CAPDIR/$file"
        ;;
      *.metadata.json)
        id="$(basename "$file" .metadata.json)"
        [ -f "$CAPDIR/$id.wav" ] && process_one "$CAPDIR/$id.wav"
        ;;
    esac
  done
else
  while true; do
    for wav in "$CAPDIR"/*.wav; do
      [ -f "$wav" ] && process_one "$wav"
    done
    sleep 2
  done
fi
