"""Shared utilities for Base44 API interactions (clean repo version)."""

import os
import requests
from dotenv import load_dotenv

load_dotenv()

API_BASE_URL = "https://app.base44.com/api"
APP_ID = os.environ["BASE44_APP_ID"]
API_KEY = os.environ["BASE44_API_KEY"]
ENTITY_TYPE = "Track"


def make_api_request(api_path, method="GET", data=None, params=None):
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
    if not resp.content:
        return None
    return resp.json()


def get_all_tracks():
    """Fetch all tracks from Base44."""
    try:
        entities = make_api_request(f"apps/{APP_ID}/entities/{ENTITY_TYPE}")
        return entities or []
    except requests.exceptions.RequestException as e:
        print(f"  ✗ Error fetching tracks: {e}")
        return []


def filter_tracks_needing_choreography(tracks):
    """
    Tracks that need choreography:
    - have spotify_id
    - choreography missing or empty list
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


def update_track_choreography(track_id: str, choreography):
    """
    Update a Track entity with generated choreography.

    Assumes PATCH:
      apps/{APP_ID}/entities/Track/{track_id}

    If your Base44 update route differs (PUT /update /actions), tell me and we’ll adjust.
    """
    payload = {"choreography": choreography}
    try:
        return make_api_request(
            f"apps/{APP_ID}/entities/{ENTITY_TYPE}/{track_id}",
            method="PATCH",
            data=payload,
        )
    except requests.exceptions.RequestException as e:
        print(f"  ✗ Error updating track {track_id}: {e}")
        return None
