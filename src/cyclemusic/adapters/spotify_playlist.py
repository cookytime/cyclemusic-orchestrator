import os
import re
import base64
import requests
from dotenv import load_dotenv

load_dotenv()

SPOTIFY_PLAYLIST_ID_OR_URL = os.environ.get("SPOTIFY_PLAYLIST_ID") or os.environ.get("SPOTIFY_PLAYLIST_URL")
SPOTIFY_CLIENT_ID = os.environ["SPOTIFY_CLIENT_ID"]
SPOTIFY_CLIENT_SECRET = os.environ["SPOTIFY_CLIENT_SECRET"]
SPOTIFY_REFRESH_TOKEN = os.environ["SPOTIFY_REFRESH_TOKEN"]

if not SPOTIFY_PLAYLIST_ID_OR_URL:
    raise RuntimeError("Missing SPOTIFY_PLAYLIST_ID (or SPOTIFY_PLAYLIST_URL)")


def extract_playlist_id(value: str) -> str:
    m = re.search(r"playlist[:/]([a-zA-Z0-9]+)", value)
    if m:
        return m.group(1)
    if re.fullmatch(r"[a-zA-Z0-9]+", value):
        return value
    raise ValueError(f"Invalid Spotify playlist URL or ID: {value}")


PLAYLIST_ID = extract_playlist_id(SPOTIFY_PLAYLIST_ID_OR_URL)


def refresh_access_token() -> str:
    token_url = "https://accounts.spotify.com/api/token"
    basic = base64.b64encode(f"{SPOTIFY_CLIENT_ID}:{SPOTIFY_CLIENT_SECRET}".encode()).decode()

    data = {
        "grant_type": "refresh_token",
        "refresh_token": SPOTIFY_REFRESH_TOKEN,
    }

    resp = requests.post(
        token_url,
        data=data,
        headers={
            "Authorization": f"Basic {basic}",
            "Content-Type": "application/x-www-form-urlencoded",
        },
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["access_token"]


def replace_processing_playlist(missing_tracks) -> str:
    """
    Replace Spotify playlist contents with tracks missing choreography.
    missing_tracks: Base44 track dicts with spotify_id
    Returns playlist_id.
    """
    # Only valid Spotify track IDs (22 chars)
    track_ids = [
        t["spotify_id"]
        for t in missing_tracks
        if t.get("spotify_id") and re.fullmatch(r"[A-Za-z0-9]{22}", t["spotify_id"])
    ]
    uris = [f"spotify:track:{tid}" for tid in track_ids]

    token = refresh_access_token()
    url = f"https://api.spotify.com/v1/playlists/{PLAYLIST_ID}/tracks"

    resp = requests.put(
        url,
        json={"uris": uris},
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        timeout=30,
    )
    resp.raise_for_status()
    return PLAYLIST_ID
