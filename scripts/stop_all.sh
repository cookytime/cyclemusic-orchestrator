#!/bin/sh
set -eu

PATTERNS="\
supervise-daemon pipeline --start|\
supervise-daemon pipeline-analyze --start|\
supervise-daemon pipeline-upload --start|\
cyclemusic.main|\
capture.capture_playlist|\
capture_runner|\
librespot|\
ffmpeg|\
inotifywait"

do_kill() {
  _signal="$1"
  if command -v sudo >/dev/null 2>&1; then
    sudo pkill -"$_signal" -f "$PATTERNS" 2>/dev/null || true
  else
    pkill -"$_signal" -f "$PATTERNS" 2>/dev/null || true
  fi
}

echo "[stop] sending SIGTERM..."
do_kill TERM
sleep 1

if pgrep -af "$PATTERNS" >/dev/null 2>&1; then
  echo "[stop] still running, sending SIGKILL..."
  do_kill KILL
  sleep 1
fi

echo "[stop] remaining processes (if any):"
pgrep -af "$PATTERNS" || true

# Restore terminal in case any child left it in a bad state
stty sane 2>/dev/null || true
