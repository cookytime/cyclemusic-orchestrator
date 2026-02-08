# src/common/spotify_api.py (NEW - Consolidated Spotify API module)
"""Unified Spotify API client with retry logic and proper error handling."""
from __future__ import annotations

import time
from typing import Any

import requests

# API retry constants
MAX_RETRIES = 3
RETRY_BACKOFF_S = 1.0
API_TIMEOUT_S = 30


def refresh_access_token(
    client_id: str,
    client_secret: str,
    refresh_token: str,
    timeout: int = API_TIMEOUT_S,
) -> str:
    """Refresh Spotify access token with retry logic."""
    def make_request():
        response = requests.post(
            "https://accounts.spotify.com/api/token",
            data={
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
                "client_id": client_id,
                "client_secret": client_secret,
            },
            timeout=timeout,
        )
        response.raise_for_status()
        return response.json()["access_token"]
    
    result = _retry_with_backoff(make_request)
    if result is None:
        raise RuntimeError("Spotify token refresh returned no access_token")
    return str(result)


def spotify_get(
    token: str, url: str, params: dict[str, Any] | None = None, timeout: int = API_TIMEOUT_S
) -> dict[str, Any]:
    """Make a GET request to the Spotify API with retry logic."""
    def make_request():
        response = requests.get(
            url,
            headers={"Authorization": f"Bearer {token}"},
            params=params,
            timeout=timeout,
        )
        if response.status_code == 401:
            raise PermissionError("Spotify token expired/unauthorized")
        response.raise_for_status()
        return response.json()
    
    result = _retry_with_backoff(make_request)
    if result is None:
        raise RuntimeError("Spotify GET returned no JSON")
    return result


def spotify_post(
    token: str, url: str, data: dict[str, Any] | None = None, timeout: int = API_TIMEOUT_S
) -> dict[str, Any] | None:
    """Make a POST request to the Spotify API with retry logic."""
    def make_request():
        response = requests.post(
            url,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
            json=data,
            timeout=timeout,
        )
        if response.status_code == 401:
            raise PermissionError("Spotify token expired/unauthorized")
        response.raise_for_status()
        return response.json() if response.text else None
    
    return _retry_with_backoff(make_request)


def spotify_put(
    token: str, url: str, data: dict[str, Any] | None = None, timeout: int = API_TIMEOUT_S
) -> dict[str, Any] | None:
    """Make a PUT request to the Spotify API with retry logic."""
    def make_request():
        response = requests.put(
            url,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
            json=data,
            timeout=timeout,
        )
        if response.status_code == 401:
            raise PermissionError("Spotify token expired/unauthorized")
        response.raise_for_status()
        return response.json() if response.text else None
    
    return _retry_with_backoff(make_request)


def spotify_delete(
    token: str, url: str, data: dict[str, Any] | None = None, timeout: int = API_TIMEOUT_S
) -> None:
    """Make a DELETE request to the Spotify API with retry logic."""
    def make_request():
        response = requests.delete(
            url,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
            json=data,
            timeout=timeout,
        )
        if response.status_code == 401:
            raise PermissionError("Spotify token expired/unauthorized")
        response.raise_for_status()
    
    _retry_with_backoff(make_request)


def spotify_get_currently_playing(
    token: str, timeout: int = API_TIMEOUT_S
) -> dict[str, Any] | None:
    """Get the user's currently playing track with retry and rate limit handling."""
    url = "https://api.spotify.com/v1/me/player/currently-playing"
    
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(
                url,
                headers={"Authorization": f"Bearer {token}"},
                timeout=timeout,
            )
            
            if response.status_code == 204:
                return None
            if response.status_code == 401:
                raise PermissionError("Spotify token expired/unauthorized")
            
            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After", "")
                if attempt < MAX_RETRIES - 1:
                    wait_time = int(retry_after) if retry_after.isdigit() else RETRY_BACKOFF_S * (2 ** attempt)
                    time.sleep(wait_time)
                    continue
                return None
            
            if response.status_code >= 500:
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_BACKOFF_S * (2 ** attempt))
                    continue
                return None
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            if attempt == MAX_RETRIES - 1:
                raise
            time.sleep(RETRY_BACKOFF_S * (2 ** attempt))
    
    return None


def _retry_with_backoff(func, max_retries=MAX_RETRIES, backoff=RETRY_BACKOFF_S):
    """Retry helper with exponential backoff."""
    for attempt in range(max_retries):
        try:
            return func()
        except (requests.exceptions.RequestException, Exception) as e:
            if attempt == max_retries - 1:
                raise
            wait_time = backoff * (2 ** attempt)
            time.sleep(wait_time)
