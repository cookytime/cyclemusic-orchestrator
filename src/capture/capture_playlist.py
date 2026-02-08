#!/usr/bin/env python3
from __future__ import annotations


# --- New version: librespot pipe integration ---
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
from manage.spotify_api import refresh_access_token, spotify_get_currently_playing, spotify_get

PROJECT_ROOT = Path(__file__).resolve().parent.parent
# Try loading .env from the project root and from the current working directory
# (some users invoke the module from different cwd locations). Load whichever
# exists and print which file was loaded to aid debugging.
candidates = [PROJECT_ROOT / ".env", Path.cwd() / ".env"]
loaded = False
for p in candidates:
    if p.exists():
        load_dotenv(p, override=False)
        print(f"[debug] loaded .env from {p}")
        loaded = True
        break
if not loaded:
    print("[debug] no .env file found in project root or cwd")

OUT_DIR = Path(os.environ.get("CAPTURES_DIR", str(PROJECT_ROOT / "captures")))
OUT_DIR.mkdir(parents=True, exist_ok=True)
POLL_SECONDS = float(os.environ.get("POLL_SECONDS", "1.0"))
PAD_SECONDS = float(os.environ.get("PAD_SECONDS", "1.0"))
PLAYLIST_ID_OR_URL = os.environ.get("SPOTIFY_PLAYLIST_ID") or os.environ.get("SPOTIFY_PLAYLIST_URL")
if not PLAYLIST_ID_OR_URL:
    raise SystemExit("Missing SPOTIFY_PLAYLIST_ID (or SPOTIFY_PLAYLIST_URL)")
_env_keepalive = os.environ.get("LIBRESPOT_KEEPALIVE")
if _env_keepalive is None:
    LIBRESPOT_KEEPALIVE = True
else:
    LIBRESPOT_KEEPALIVE = _env_keepalive.lower() in ("1", "true", "yes")
LIBRESPOT_OAUTH_PORT = os.environ.get("LIBRESPOT_OAUTH_PORT")

def extract_playlist_id(value: str) -> str:
    m = re.search(r"playlist[:/]([a-zA-Z0-9]+)", value)
    if m:
        return m.group(1)
    if re.fullmatch(r"[a-zA-Z0-9]+", value):
        return value
    raise ValueError(f"Invalid Spotify playlist URL or ID: {value}")

PLAYLIST_ID = extract_playlist_id(PLAYLIST_ID_OR_URL)
PLAYLIST_URI = f"spotify:playlist:{PLAYLIST_ID}"

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

def start_librespot_pipe(track_id: str, wav_path: Path, duration_s: float, device_name: str = "CycleMusicLibrespot"):
    # Start librespot with pipe backend and ffmpeg to write to wav
    cmd_librespot = [
        "librespot",
        "--backend", "pipe",
        "--name", device_name,
        "--bitrate", "320",
        # Add more options as needed
    ]
    if LIBRESPOT_OAUTH_PORT:
        cmd_librespot += ["--oauth-port", str(LIBRESPOT_OAUTH_PORT)]
    cmd_ffmpeg = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-f", "s16le",
        "-ar", "44100",
        "-ac", "2",
        "-t", f"{duration_s:.2f}",
        "-y",
        "-i", "-",
        str(wav_path),
    ]
    # Start librespot and ffmpeg, piping librespot's stdout to ffmpeg's stdin
    librespot_proc = subprocess.Popen(cmd_librespot, stdout=subprocess.PIPE)
    ffmpeg_proc = subprocess.Popen(cmd_ffmpeg, stdin=librespot_proc.stdout)
    return librespot_proc, ffmpeg_proc

def start_librespot_keepalive(device_name: str = "CycleMusicLibrespot"):
    cmd_librespot = [
        "librespot",
        "--backend", "pipe",
        "--name", device_name,
        "--bitrate", "320",
    ]
    if LIBRESPOT_OAUTH_PORT:
        cmd_librespot += ["--oauth-port", str(LIBRESPOT_OAUTH_PORT)]
    cmd_ffmpeg = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "warning",
        "-f", "s16le",
        "-ar", "44100",
        "-ac", "2",
        "-i", "-",
        "-f", "null",
        "-",
    ]
    librespot_proc = subprocess.Popen(cmd_librespot, stdout=subprocess.PIPE)
    ffmpeg_proc = subprocess.Popen(cmd_ffmpeg, stdin=librespot_proc.stdout)
    return librespot_proc, ffmpeg_proc

def stop_proc(proc: subprocess.Popen | None) -> None:
    if not proc:
        return
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()

def is_librespot_running() -> bool:
    try:
        out = subprocess.check_output(["pgrep", "-af", "librespot"], stderr=subprocess.DEVNULL)
        return bool(out.strip())
    except subprocess.CalledProcessError:
        return False
    except Exception:
        return False

def main() -> int:
    client_id = os.environ["SPOTIFY_CLIENT_ID"]
    client_secret = os.environ["SPOTIFY_CLIENT_SECRET"]
    refresh_token = os.environ["SPOTIFY_REFRESH_TOKEN"]


    token = refresh_access_token(client_id, client_secret, refresh_token)

    # Fetch playlist tracks and print details
    playlist_track_ids = fetch_playlist_track_ids(token, PLAYLIST_ID)
    print(f"[capture] playlist tracks cached: {len(playlist_track_ids)}", flush=True)
    # ...existing code...

    current_track_id: str | None = None
    librespot_proc: subprocess.Popen | None = None
    ffmpeg_proc: subprocess.Popen | None = None
    keepalive_librespot: subprocess.Popen | None = None
    keepalive_ffmpeg: subprocess.Popen | None = None
    current_wav: Path | None = None

    while True:
        if LIBRESPOT_KEEPALIVE:
            if keepalive_librespot is None or keepalive_librespot.poll() is not None:
                if is_librespot_running():
                    print("[capture] librespot already running — skipping keepalive start", flush=True)
                else:
                    try:
                        keepalive_librespot, keepalive_ffmpeg = start_librespot_keepalive()
                    except Exception as e:
                        print(f"[capture] failed to start librespot keepalive: {e}", flush=True)
                        keepalive_librespot = None
                        keepalive_ffmpeg = None

        payload = spotify_get_currently_playing(token)
        ctx_uri = None
        track_id = None
        if payload:
            ctx = payload.get("context") or {}
            ctx_uri = ctx.get("uri")
            item = payload.get("item") or {}
            track_id = item.get("id")
        # ...existing code...

        if not payload or not payload.get("is_playing"):
            stop_proc(librespot_proc)
            stop_proc(ffmpeg_proc)
            librespot_proc = None
            ffmpeg_proc = None
            current_track_id = None
            time.sleep(POLL_SECONDS)
            continue

        if not track_id:
            time.sleep(POLL_SECONDS)
            continue

        # Gate 1: must be in our playlist context
        if ctx_uri != PLAYLIST_URI:
            stop_proc(librespot_proc)
            stop_proc(ffmpeg_proc)
            librespot_proc = None
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

        # If track changed, start a new librespot/ffmpeg pipe
        if track_id != current_track_id:
            name = item.get("name", "Unknown Track")
            artists = ", ".join(a.get("name","") for a in (item.get("artists") or []) if a.get("name")) or "Unknown Artist"
            print(f"[capture] ▶ {artists} — {name}  (~{remaining_s:.1f}s)", flush=True)

            stop_proc(librespot_proc)
            stop_proc(ffmpeg_proc)

            # Write metadata + start librespot/ffmpeg
            metadata_path = OUT_DIR / f"{track_id}.metadata.json"
            wav_path = OUT_DIR / f"{track_id}.wav"

            save_track_metadata(item, metadata_path)
            if LIBRESPOT_KEEPALIVE:
                stop_proc(keepalive_ffmpeg)
                stop_proc(keepalive_librespot)
                keepalive_librespot = None
                keepalive_ffmpeg = None
            librespot_proc, ffmpeg_proc = start_librespot_pipe(track_id, wav_path, remaining_s)

            current_track_id = track_id
            current_wav = wav_path

        # If ffmpeg finished, we just idle until track changes
        if ffmpeg_proc and ffmpeg_proc.poll() is not None:
            # Optional: tiny-file guard
            if current_wav and current_wav.exists() and current_wav.stat().st_size < 100_000:
                print(f"[capture] ⚠️ tiny capture: {current_wav.name}", flush=True)
            ffmpeg_proc = None
            librespot_proc = None
            if LIBRESPOT_KEEPALIVE:
                if is_librespot_running():
                    print("[capture] librespot already running — skipping keepalive restart", flush=True)
                else:
                    try:
                        keepalive_librespot, keepalive_ffmpeg = start_librespot_keepalive()
                    except Exception as e:
                        print(f"[capture] failed to restart librespot keepalive: {e}", flush=True)
                        keepalive_librespot = None
                        keepalive_ffmpeg = None

        time.sleep(POLL_SECONDS)

    print("[capture] stopping", flush=True)
    stop_proc(librespot_proc)
    stop_proc(ffmpeg_proc)
    stop_proc(keepalive_ffmpeg)
    stop_proc(keepalive_librespot)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
