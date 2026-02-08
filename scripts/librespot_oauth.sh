#!/bin/sh
set -eu

ROOT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"

LIBRESPOT_DEVICE_NAME="${LIBRESPOT_DEVICE_NAME:-CycleMusicLibrespot}"
LIBRESPOT_OAUTH_PORT="${LIBRESPOT_OAUTH_PORT:-5588}"
LIBRESPOT_SYSTEM_CACHE="${LIBRESPOT_SYSTEM_CACHE:-$ROOT_DIR/logs/librespot-cache}"

mkdir -p "$LIBRESPOT_SYSTEM_CACHE"

echo "[oauth] Starting librespot for one-time login..."
echo "[oauth] Device name: $LIBRESPOT_DEVICE_NAME"
echo "[oauth] OAuth port: $LIBRESPOT_OAUTH_PORT"
echo "[oauth] System cache: $LIBRESPOT_SYSTEM_CACHE"
echo "[oauth] Follow the printed URL in your browser, then return here."

librespot \
  --backend rodio \
  --name "$LIBRESPOT_DEVICE_NAME" \
  --bitrate 320 \
  --enable-oauth \
  --oauth-port "$LIBRESPOT_OAUTH_PORT" \
  --system-cache "$LIBRESPOT_SYSTEM_CACHE"

# Restore terminal in case librespot changes terminal modes
stty sane 2>/dev/null || true
