from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import List, Dict, Any


ANALYZE_SCRIPT = Path("/home/appuser/cyclemusic-orchestrator/src/analyze/analyze_track.py")


def _track_id_from_wav(wav_path: Path) -> str:
    # assumes filename is "<spotify_id>.wav"
    return wav_path.stem


def analyze_files(wav_paths: List[str]) -> List[Dict[str, Any]]:
    """
    Runs: python analyze/analyze_track.py <wav>
    Expects: <wav_base>.choreography.json to be created next to the wav.

    Returns list of:
      { "track_id": "<spotify_id>", "choreography": <json> }
    """
    results: List[Dict[str, Any]] = []

    for wav in wav_paths:
        wav_path = Path(wav).resolve()
        if not wav_path.exists():
            continue

        cmd = ["python", str(ANALYZE_SCRIPT), str(wav_path)]
        subprocess.run(
            cmd,
            check=True,
            cwd="/home/appuser/cyclemusic-orchestrator",
        )

        choreo_path = wav_path.with_suffix(".choreography.json")
        if not choreo_path.exists():
            # analyzer didn't produce output; skip (or raise if you prefer)
            continue

        choreography = json.loads(choreo_path.read_text())
        results.append(
            {
                "track_id": _track_id_from_wav(wav_path),
                "choreography": choreography,
            }
        )

    return results
