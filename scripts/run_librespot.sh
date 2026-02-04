#!/usr/bin/env bash
set -e

# ---- Config ----
LIBRESPOT_BIN="/usr/bin/librespot"
DEVICE_NAME="CycleMusicLibrespot"
PULSE_SINK="librespot_sink"

# ---- Environment ----
export PULSE_SINK="${PULSE_SINK}"

# ---- Run ----
exec "${LIBRESPOT_BIN}" \
  --backend pulseaudio \
  --name "${DEVICE_NAME}" \
  --enable-oauth \
  --bitrate 320
