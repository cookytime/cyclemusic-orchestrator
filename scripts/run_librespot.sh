#!/usr/bin/env bash
set -e

# ---- Config ----
LIBRESPOT_BIN="/usr/bin/librespot"
DEVICE_NAME="CycleMusicLibrespot"

# ---- Environment ----
# ---- Run ----
exec "${LIBRESPOT_BIN}" \
  --backend pipe \
  --name "${DEVICE_NAME}" \
  --bitrate 320
