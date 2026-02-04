
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
from typing import List, Optional

from dotenv import load_dotenv
from manage.spotify_api import refresh_access_token, spotify_get_currently_playing, spotify_get


print(f"[DEBUG] Python executable: {sys.executable}")
print(f"[DEBUG] PYTHONPATH: {os.environ.get('PYTHONPATH')}")
print(f"[DEBUG] All environment variables:")
for k, v in os.environ.items():
    print(f"    {k}={v}")

# ===== CONFIG =====
PULSE_MONITOR_SOURCE = os.environ.get("PULSE_MONITOR_SOURCE", "librespot_sink.monitor")

def extract_playlist_id(value: str) -> str:
  m = re.search(r"playlist[:/]([a-zA-Z0-9]+)", value)
  if m:
    return m.group(1)
  if re.fullmatch(r"[a-zA-Z0-9]+", value):
    return value
  raise ValueError(f"Invalid Spotify playlist URL or ID: {value}")

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

def stop_proc(proc: Optional[subprocess.Popen]) -> None:
  if not proc:
    return
  if proc.poll() is None:
    proc.send_signal(signal.SIGINT)
    try:
      proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
      proc.kill()

def capture_playlist_tracks(
  playlist_id_or_url: str,
  captures_dir: str,
  poll_seconds: float = 1.0,
  pad_seconds: float = 1.0,
  client_id: Optional[str] = None,
  client_secret: Optional[str] = None,
  refresh_token: Optional[str] = None,
) -> None:
  """
  Captures tracks from a Spotify playlist using the same logic as auto_capture_playlist_only.py.
  """
  # Load environment and config
  PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
  load_dotenv(PROJECT_ROOT / ".env", override=False)

  OUT_DIR = Path(captures_dir)
  OUT_DIR.mkdir(parents=True, exist_ok=True)

  PLAYLIST_ID = extract_playlist_id(playlist_id_or_url)
  PLAYLIST_URI = f"spotify:playlist:{PLAYLIST_ID}"

  if client_id is None:
    client_id = os.environ["SPOTIFY_CLIENT_ID"]
  if client_secret is None:
    client_secret = os.environ["SPOTIFY_CLIENT_SECRET"]
  if refresh_token is None:
    refresh_token = os.environ["SPOTIFY_REFRESH_TOKEN"]

  token = refresh_access_token(client_id, client_secret, refresh_token)
  playlist_track_ids = fetch_playlist_track_ids(token, PLAYLIST_ID)
  print(f"[capture] playlist tracks cached: {len(playlist_track_ids)}", flush=True)

  current_track_id: str | None = None
  ffmpeg_proc: subprocess.Popen | None = None
  current_wav: Path | None = None

  _shutdown = False
  def _stop(*_):
    nonlocal _shutdown, ffmpeg_proc
    _shutdown = True
    # Ensure all child processes are killed on shutdown
    stop_proc(ffmpeg_proc)

  signal.signal(signal.SIGTERM, _stop)
  signal.signal(signal.SIGINT, _stop)

  try:
    while not _shutdown:
      payload = spotify_get_currently_playing(token)
      if not payload or not payload.get("is_playing"):
        stop_proc(ffmpeg_proc)
        ffmpeg_proc = None
        current_track_id = None
        time.sleep(poll_seconds)
        continue

      item = payload.get("item") or {}
      track_id = item.get("id")
      if not track_id:
        time.sleep(poll_seconds)
        continue

      ctx = payload.get("context") or {}
      ctx_uri = ctx.get("uri")

      # Gate 1: must be in our playlist context
      if ctx_uri != PLAYLIST_URI:
        stop_proc(ffmpeg_proc)
        ffmpeg_proc = None
        current_track_id = None
        time.sleep(poll_seconds)
        continue

      # Gate 2: track must be one of the playlist tracks
      if track_id not in playlist_track_ids:
        time.sleep(poll_seconds)
        continue

      progress_ms = payload.get("progress_ms") or 0
      duration_ms = item.get("duration_ms") or 0
      remaining_s = max(0.0, (duration_ms - progress_ms) / 1000.0) + pad_seconds

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

      time.sleep(poll_seconds)
  finally:
    print("[capture] stopping", flush=True)
    stop_proc(ffmpeg_proc)

if __name__ == "__main__":
  import argparse

  parser = argparse.ArgumentParser(description="Capture Spotify playlist tracks to WAV files.")
  parser.add_argument("playlist", type=str, help="Spotify playlist URL or ID")
  parser.add_argument("--output-dir", type=str, default="captures", help="Directory to save captured WAV files")
  parser.add_argument("--poll-seconds", type=float, default=1.0, help="Seconds between polling Spotify API")
  parser.add_argument("--pad-seconds", type=float, default=1.0, help="Seconds to pad at end of capture")
  args = parser.parse_args()

  capture_playlist_tracks(
    playlist_id_or_url=args.playlist,
    captures_dir=args.output_dir,
    poll_seconds=args.poll_seconds,
    pad_seconds=args.pad_seconds,
  )
