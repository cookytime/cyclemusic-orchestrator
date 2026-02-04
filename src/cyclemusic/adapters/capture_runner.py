from __future__ import annotations

import os
import time
import glob
import subprocess
from typing import List


def _list_wavs(captures_dir: str) -> set[str]:
    return set(glob.glob(os.path.join(captures_dir, "*.wav")))


def capture_playlist(playlist_id: str, expected: int, captures_dir: str) -> List[str]:
    """
    Runs the local capture module (vendored into this repo):
      python3 -m capture.auto_capture_playlist_only

    Autoplay/stop not implemented yet (we'll add later).
    """
    os.makedirs(captures_dir, exist_ok=True)

    before = _list_wavs(captures_dir)

    env = os.environ.copy()
    env["CAPTURES_DIR"] = captures_dir
    env["SPOTIFY_PLAYLIST_ID"] = playlist_id

    # IMPORTANT: run from orchestrator repo, with PYTHONPATH=src already set by main runner
    cmd = ["python3", "-m", "capture.auto_capture_playlist_only"]
    subprocess.run(cmd, check=True, env=env, cwd="/home/appuser/cyclemusic-orchestrator")

    time.sleep(0.5)

    after = _list_wavs(captures_dir)
    return sorted(after - before)
