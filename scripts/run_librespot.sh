#!/usr/bin/env bash
set -euo pipefail

# ---- Config ----
LIBRESPOT_BIN="/usr/bin/librespot"
DEVICE_NAME="CycleMusicLibrespot"

# ---- Environment ----
# ---- Run ----
# Keep stdout drained to avoid Broken pipe when running librespot standalone.
exec "${LIBRESPOT_BIN}" \
  --backend pipe \
  --name "${DEVICE_NAME}" \
  --bitrate 320 \
  | ffmpeg \
    -hide_banner \
    -loglevel error \
    -fflags +discardcorrupt \
    -err_detect ignore_err \
    -f s16le \
    -ar 44100 \
    -ac 2 \
    -i - \
    -f null \
    -
