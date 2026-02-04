#!/bin/sh
set -eu

cd /home/appuser/cyclemusic-orchestrator

# Ensure we can import from src/
export PYTHONPATH="/home/appuser/cyclemusic-orchestrator/src"

exec python3 -m cyclemusic.main
