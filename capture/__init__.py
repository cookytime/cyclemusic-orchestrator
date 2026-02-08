"""Package shim so `python -m capture.xxx` works from the repository root.

This file adjusts the package `__path__` to point at `./src/capture` so the
real implementation under `src/` is used while allowing `-m capture.xxx`.
"""
import os

# repo root is one level up from this file
REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
SRC_CAPTURE = os.path.join(REPO_ROOT, "src", "capture")
if os.path.isdir(SRC_CAPTURE):
    # Prefer the src/capture implementation
    __path__.insert(0, SRC_CAPTURE)
# Ensure the `src` directory is on sys.path so other top-level packages (like
# `manage` and `cyclemusic`) can be imported when running `-m capture.xxx`.
SRC_ROOT = os.path.join(REPO_ROOT, "src")
try:
    import sys
    if os.path.isdir(SRC_ROOT) and SRC_ROOT not in sys.path:
        sys.path.insert(0, SRC_ROOT)
except Exception:
    pass
