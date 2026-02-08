#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<EOF
Usage: $0 [options] <pattern>
Options:
  -y        : assume yes (no prompt)
  -n        : dry-run (show matches only)
  -t SECS   : wait SECS for graceful shutdown before SIGKILL (default 5)
  -a        : target all users (do not restrict to current user)
  -h        : show this help

Examples:
  $0 cyclemusic
  $0 -n libre*
  $0 -y -t 10 cyclemusic
EOF
}

YES=0
DRY=0
WAIT=5
ALL_USERS=0

while getopts "ynt:ah" opt; do
  case $opt in
    y) YES=1 ;;
    n) DRY=1 ;;
    t) WAIT="$OPTARG" ;;
    a) ALL_USERS=1 ;;
    h) usage; exit 0 ;;
    *) usage; exit 2 ;;
  esac
done
shift $((OPTIND-1))

PATTERN="${1:-}"
if [[ -z "$PATTERN" ]]; then
  usage
  exit 2
fi

pgrep_args=("-f" "$PATTERN")
if [[ "$ALL_USERS" -eq 0 ]]; then
  pgrep_args=("-u" "$USER" "-f" "$PATTERN")
fi

# collect matching PIDs (may be empty)
mapfile -t PIDS < <(pgrep "${pgrep_args[@]}" || true)

if [[ ${#PIDS[@]} -eq 0 ]]; then
  echo "No matching processes for pattern: $PATTERN"
  exit 0
fi

echo "Matched PIDs: ${PIDS[*]}"
echo "Details:"
# BusyBox `ps` accepts a limited set of -o columns; use compatible ones
# Use `etime` (elapsed time) and `args` (full command line)
ps -o pid,user,etime,args "${PIDS[@]}"

if [[ "$DRY" -eq 1 ]]; then
  echo "Dry-run; not killing processes."
  exit 0
fi

if [[ "$YES" -ne 1 ]]; then
  read -r -p "Send SIGTERM to these processes? [y/N] " RESP
  case "$RESP" in
    [yY]|[yY][eE][sS]) ;;
    *) echo "Aborted."; exit 0 ;;
  esac
fi

echo "Sending SIGTERM..."
for pid in "${PIDS[@]}"; do
  kill -TERM "$pid" 2>/dev/null || true
done

echo "Waiting $WAIT seconds for graceful shutdown..."
sleep "$WAIT"

# check remaining
mapfile -t REMAIN < <(pgrep "${pgrep_args[@]}" || true)
if [[ ${#REMAIN[@]} -eq 0 ]]; then
  echo "All processes exited after SIGTERM."
  exit 0
fi

echo "Processes still running: ${REMAIN[*]}"
echo "Sending SIGKILL..."
for pid in "${REMAIN[@]}"; do
  kill -KILL "$pid" 2>/dev/null || true
done

echo "Done."
