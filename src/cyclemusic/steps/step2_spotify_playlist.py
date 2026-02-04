import sys
import os

print(f"[DEBUG] Python executable: {sys.executable}")
print(f"[DEBUG] PYTHONPATH: {os.environ.get('PYTHONPATH')}")
print(f"[DEBUG] All environment variables:")
for k, v in os.environ.items():
    print(f"    {k}={v}")

from cyclemusic.adapters.spotify_playlist import replace_processing_playlist

def sync_playlist(cfg, missing_tracks):
    return replace_processing_playlist(missing_tracks)
