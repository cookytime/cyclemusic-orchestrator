#!/bin/sh
set -eu

REPO="/home/appuser/cyclemusic-orchestrator"
CAPDIR="${CAPTURES_DIR:-$REPO/captures}"

cd "$REPO"
. "$REPO/.venv/bin/activate"
export PYTHONPATH="$REPO/src"
source "$REPO/.env"

process_one() {
  choreo="$1"
  id="$(basename "$choreo" .choreography.json)"

  doneflag="$CAPDIR/$id.uploaded.ok"
  lockdir="$CAPDIR/$id.upload.lock"

  [ -f "$doneflag" ] && return 0

  mkdir "$lockdir" 2>/dev/null || return 0
  trap 'rmdir "$lockdir" 2>/dev/null || true' EXIT

  echo "[upload] $id starting"
  if python - <<PY
import json
from cyclemusic.config import load_config
from cyclemusic.steps.step5_upload import upload_choreo

cfg = load_config()
with open("$CAPDIR/$id.choreography.json","r") as f:
    choreography = json.load(f)
upload_choreo(cfg, [{"track_id": "$id", "choreography": choreography}])
PY
  then
    : > "$doneflag"
    echo "[upload] $id done"
  else
    echo "[upload] $id FAILED" >&2
  fi

  rmdir "$lockdir" 2>/dev/null || true
  trap - EXIT
}

echo "[upload] watching $CAPDIR"

# First pass
for f in "$CAPDIR"/*.choreography.json; do
  [ -f "$f" ] && process_one "$f"
done

if command -v inotifywait >/dev/null 2>&1; then
  inotifywait -m -e close_write,create,move "$CAPDIR" | while read -r _ _ file; do
    case "$file" in
      *.choreography.json)
        process_one "$CAPDIR/$file"
        ;;
    esac
  done
else
  while true; do
    for f in "$CAPDIR"/*.choreography.json; do
      [ -f "$f" ] && process_one "$f"
    done
    sleep 2
  done
fi
