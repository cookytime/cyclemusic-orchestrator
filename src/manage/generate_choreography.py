#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from analyze.analyze_track import (
    generate_track_choreography_openai,
    load_track_metadata_for_audio,
    safe_file_write,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def iter_music_map_paths(targets: list[str]) -> list[Path]:
    if not targets:
        default_dir = REPO_ROOT / "captures"
        return sorted(default_dir.glob("*.music_map.json"))

    paths: list[Path] = []
    for target in targets:
        path = Path(target)
        if path.is_dir():
            paths.extend(sorted(path.glob("*.music_map.json")))
        else:
            paths.append(path)
    return paths


def base_path_from_music_map(path: Path) -> Path:
    suffix = ".music_map.json"
    if path.name.endswith(suffix):
        return Path(str(path)[: -len(suffix)])
    return path.with_suffix("")


def generate_for_music_map(path: Path, rider_settings: dict) -> bool:
    if not path.exists():
        print(f"✗ Music map not found: {path}")
        return False

    with open(path, "r", encoding="utf-8") as f:
        music_map = json.load(f)

    base_path = base_path_from_music_map(path)
    wav_path = base_path.with_suffix(".wav")
    track_metadata = load_track_metadata_for_audio(str(wav_path))
    track_json = generate_track_choreography_openai(music_map, rider_settings, track_metadata)

    choreography_path = base_path.with_suffix(".choreography.json")
    output_data = {"track": track_json}

    with safe_file_write(choreography_path) as f:
        json.dump(output_data, f, indent=2)

    print(f"✓ Choreography saved: {choreography_path}")

    spotify_id = track_json.get("spotify_id") or (music_map.get("spotify") or {}).get("spotify_id")
    if spotify_id:
        captures_dir = base_path.parent
        spotify_named_path = captures_dir / f"{spotify_id}.choreography.json"
        if spotify_named_path.resolve() != choreography_path.resolve():
            with safe_file_write(spotify_named_path) as f:
                json.dump(output_data, f, indent=2)
            print(f"✓ Also saved: {spotify_named_path}")

    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate choreography from music_map.json outputs.",
    )
    parser.add_argument(
        "paths",
        nargs="*",
        help="Paths to .music_map.json files or directories (defaults to captures/).",
    )
    args = parser.parse_args()

    load_dotenv(REPO_ROOT / ".env")

    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not found in .env; cannot generate choreography.")
        return 1

    rider_settings = {
        "rider_level": os.environ.get("RIDER_LEVEL", "intermediate"),
        "resistance_scale": {"min": 1, "max": 24},
        "cadence_limits": {
            "seated": {"min_rpm": 60, "max_rpm": 115},
            "standing": {"min_rpm": 55, "max_rpm": 80},
        },
        "cue_spacing_s": {"min": 24, "max": 32},
    }

    targets = iter_music_map_paths(args.paths)
    if not targets:
        print("No music_map.json files found to process.")
        return 0

    failed = 0
    for path in targets:
        print(f"\nGenerating choreography for {path.name}")
        try:
            if not generate_for_music_map(path, rider_settings):
                failed += 1
        except Exception as exc:
            print(f"✗ Failed: {exc}")
            failed += 1

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
