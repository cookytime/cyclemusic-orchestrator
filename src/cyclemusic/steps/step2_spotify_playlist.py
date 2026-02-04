from cyclemusic.adapters.spotify_playlist import replace_processing_playlist

def sync_playlist(cfg, missing_tracks):
    return replace_processing_playlist(missing_tracks)
