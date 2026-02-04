#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

# Repo root is .../src; we assume PYTHONPATH=src when running
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env", override=False)

from manage.spotify_api import refresh_access_token, spotify_get_currently_playing, spotify_get


# ===== CONFIG =====
PULSE_MONITOR_SOURCE = os.environ.get("PULSE_MONITOR_SOURCE", "librespot_sink.monitor")

# Where to save captures
OUT_DIR = Path(os.environ.get("CAPTURES_DIR", str(PROJECT_ROOT / "captures")))
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Poll interval for Spotify currently-playing
POLL_SECONDS = float(os.environ.get("POLL_SECONDS", "1.0"))

# Record a tiny bit extra so we don't cut off the tail
PAD_SECONDS = float(os.environ.get("PAD_SECONDS", "1.0"))

# Playlist can be URL or ID; in orchestrator we set SPOTIFY_PLAYLIST_ID
PLAYLIST_ID_OR_URL = os.environ.get("SPOTIFY_PLAYLIST_ID") or os.environ.get("SPOTIFY_PLAYLIST_URL")
if not PLAYLIST_ID_OR_URL:
    raise SystemExit("Missing SPOTIFY_PLAYLIST_ID (or SPOTIFY_PLAYLIST_URL)")

def extract_playlist_id(value: str) -> str:
    m = re.search(r"playlist[:/]([a-zA-Z0-9]+)", value)
    if m:
        return m.group(1)
    if re.fullmatch(r"[a-zA-Z0-9]+", value):
        return value
    raise ValueError(f"Invalid Spotify playlist URL or ID: {value}")

PLAYLIST_ID = extract_playlist_id(PLAYLIST_ID_OR_URL)
PLAYLIST_URI = f"spotify:playlist:{PLAYLIST_ID}"

_shutdown = False

def _stop(*_):
    global _shutdown
    _shutdown = True

def fetch_playlist_track_ids(token: str, playlist_id: str) -> set[str]:
    ids: set[str] = set()
    url = f"https://api.spotify.com/v1/playlists/{playlist_id}/tracks"
    params = {"fields": "items(track(id,type)),next", "limit": 100}
    while True:
        data = spotify_get(token, url, params=params)
        for it in data.get("items", []):
            tr = (it.get("track") or {})
            if tr.get("type") == "track" and tr.get("id"):
                ids.add(tr["id"])
        nxt = data.get("next")
        if not nxt:
            break
        url = nxt
        params = None
    return ids

def save_track_metadata(track_item: dict, metadata_path: Path) -> None:
    meta = {
        "spotify_id": track_item.get("id"),
        "name": track_item.get("name"),
        "artists": [
            {"name": a.get("name"), "id": a.get("id"), "uri": a.get("uri")}
            for a in (track_item.get("artists") or [])
        ],
        "album": {
            "name": (track_item.get("album") or {}).get("name"),
            "id": (track_item.get("album") or {}).get("id"),
            "release_date": (track_item.get("album") or {}).get("release_date"),
            "images": (track_item.get("album") or {}).get("images", []),
        },
        "duration_ms": track_item.get("duration_ms"),
        "explicit": track_item.get("explicit"),
        "uri": track_item.get("uri"),
        "external_urls": track_item.get("external_urls"),
        "isrc": (track_item.get("external_ids") or {}).get("isrc"),
        "captured_at": datetime.now().isoformat(),
    }
    metadata_path.write_text(json.dumps(meta, indent=2))

def start_ffmpeg(wav_path: Path, seconds: float) -> subprocess.Popen:
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-f", "pulse",
        "-i", PULSE_MONITOR_SOURCE,
        "-ac", "1",
        "-ar", "44100",
        "-t", f"{max(1.0, seconds):.2f}",
        "-y",
        str(wav_path),
    ]
    return subprocess.Popen(cmd)

def stop_proc(proc: subprocess.Popen | None) -> None:
    if not proc:
        return
    if proc.poll() is None:
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()

def main() -> int:
    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)

    print(f"[capture] playlist={PLAYLIST_ID} monitor={PULSE_MONITOR_SOURCE} out={OUT_DIR}", flush=True)
    # Warm up ffmpeg / pulse pipeline (helps avoid missing the first track intro)
    try:
        warmup = OUT_DIR / ".warmup.wav"
        proc = subprocess.Popen([
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-f", "pulse", "-i", PULSE_MONITOR_SOURCE,
            "-ac", "1", "-ar", "44100",
            "-t", "2.0",
            "-y", str(warmup),
        ])
        proc.wait(timeout=5)
        if warmup.exists():
            warmup.unlink()
        print("[capture] warmup complete", flush=True)
    except Exception as e:
        print(f"[capture] warmup skipped: {e}", flush=True)

    client_id = os.environ["SPOTIFY_CLIENT_ID"]
    client_secret = os.environ["SPOTIFY_CLIENT_SECRET"]
    refresh_token = os.environ["SPOTIFY_REFRESH_TOKEN"]

    token = refresh_access_token(client_id, client_secret, refresh_token)

    playlist_track_ids = fetch_playlist_track_ids(token, PLAYLIST_ID)
    print(f"[capture] playlist tracks cached: {len(playlist_track_ids)}", flush=True)

    current_track_id: str | None = None
    ffmpeg_proc: subprocess.Popen | None = None
    current_wav: Path | None = None

    while not _shutdown:
        payload = spotify_get_currently_playing(token)
        if not payload or not payload.get("is_playing"):
            stop_proc(ffmpeg_proc)
            ffmpeg_proc = None
            current_track_id = None
            time.sleep(POLL_SECONDS)
            continue

        item = payload.get("item") or {}
        track_id = item.get("id")
        if not track_id:
            time.sleep(POLL_SECONDS)
            continue

        ctx = payload.get("context") or {}
        ctx_uri = ctx.get("uri")

        # Gate 1: must be in our playlist context
        if ctx_uri != PLAYLIST_URI:
            stop_proc(ffmpeg_proc)
            ffmpeg_proc = None
            current_track_id = None
            time.sleep(POLL_SECONDS)
            continue

        # Gate 2: track must be one of the playlist tracks
        if track_id not in playlist_track_ids:
            time.sleep(POLL_SECONDS)
            continue

        progress_ms = payload.get("progress_ms") or 0
        duration_ms = item.get("duration_ms") or 0
        remaining_s = max(0.0, (duration_ms - progress_ms) / 1000.0) + PAD_SECONDS

        # If track changed, start a new capture
        if track_id != current_track_id:
            stop_proc(ffmpeg_proc)

            name = item.get("name", "Unknown Track")
            artists = ", ".join(a.get("name","") for a in (item.get("artists") or []) if a.get("name")) or "Unknown Artist"
            print(f"[capture] ▶ {artists} — {name}  (~{remaining_s:.1f}s)", flush=True)

            # Write metadata + start ffmpeg
            metadata_path = OUT_DIR / f"{track_id}.metadata.json"
            wav_path = OUT_DIR / f"{track_id}.wav"

            save_track_metadata(item, metadata_path)
            ffmpeg_proc = start_ffmpeg(wav_path, remaining_s)

            current_track_id = track_id
            current_wav = wav_path

        # If ffmpeg finished, we just idle until track changes
        if ffmpeg_proc and ffmpeg_proc.poll() is not None:
            # Optional: tiny-file guard
            if current_wav and current_wav.exists() and current_wav.stat().st_size < 100_000:
                print(f"[capture] ⚠️ tiny capture: {current_wav.name}", flush=True)
            ffmpeg_proc = None

        time.sleep(POLL_SECONDS)

    print("[capture] stopping", flush=True)
    stop_proc(ffmpeg_proc)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
