"""Shared utilities for Base44 API interactions."""

import os
import requests
from dotenv import load_dotenv

load_dotenv()

API_BASE_URL = "https://app.base44.com/api"
APP_ID = os.environ["BASE44_APP_ID"]
API_KEY = os.environ["BASE44_API_KEY"]
ENTITY_TYPE = "Track"


def make_api_request(api_path: str, method: str = "GET", data=None, params=None):
    """Make authenticated API request to Base44."""
    url = f"{API_BASE_URL}/{api_path.lstrip('/')}"
    headers = {
        "api_key": API_KEY,
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    method = method.upper()
    if method == "GET":
        resp = requests.get(url, headers=headers, params=params, timeout=30)
    else:
        resp = requests.request(method, url, headers=headers, json=data, params=params, timeout=30)

    resp.raise_for_status()
    # Some endpoints may return empty body on success; guard for that.
    if not resp.content:
        return None
    return resp.json()


def get_all_tracks():
    """Fetch all tracks from Base44."""
    try:
        return make_api_request(f"apps/{APP_ID}/entities/{ENTITY_TYPE}") or []
    except requests.exceptions.RequestException as e:
        print(f"  ✗ Error fetching tracks: {e}")
        return []


def filter_tracks_needing_choreography(tracks):
    """
    Tracks that have:
    - a spotify_id
    - missing choreography (None) OR empty list
    """
    needing = []
    for track in tracks:
        spotify_id = track.get("spotify_id")
        choreography = track.get("choreography")

        if not spotify_id:
            continue

        if (not choreography) or (isinstance(choreography, list) and len(choreography) == 0):
            needing.append(track)

    return needing


def get_track_spotify_ids_needing_choreography(tracks):
    """Return spotify_id strings for tracks needing choreography."""
    ids = []
    for track in tracks:
        spotify_id = track.get("spotify_id")
        choreography = track.get("choreography")

        if not spotify_id:
            continue

        if (not choreography) or (isinstance(choreography, list) and len(choreography) == 0):
            ids.append(spotify_id)

    return ids


def update_track_choreography(track_id: str, choreography):
    """
    Update a Track entity with generated choreography.

    NOTE: This assumes Base44 supports PATCH on:
      apps/{APP_ID}/entities/Track/{track_id}
    If your API uses a different update route, tell me the correct one and I’ll adjust.
    """
    try:
        # Send all fields directly, no 'track' wrapper
        if isinstance(choreography, dict) and "track" in choreography and len(choreography) == 1:
            payload = choreography["track"]
        else:
            payload = choreography
        import json as _json
        print(f"[DEBUG] update_track_choreography: PUT apps/{APP_ID}/entities/{ENTITY_TYPE}/{track_id}")
        print(f"[DEBUG] Payload: {_json.dumps(payload, indent=2)}")
        return make_api_request(
            f"apps/{APP_ID}/entities/{ENTITY_TYPE}/{track_id}",
            method="PUT",
            data=payload,
        )
    except requests.exceptions.RequestException as e:
        print(f"  ✗ Error updating track {track_id}: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"[DEBUG] API error response: {e.response.text}")
        return None


def get_track(track_id: str):
    """Fetch a single track (optional helper)."""
    try:
        return make_api_request(f"apps/{APP_ID}/entities/{ENTITY_TYPE}/{track_id}")
    except requests.exceptions.RequestException as e:
        print(f"  ✗ Error fetching track {track_id}: {e}")
        return None
