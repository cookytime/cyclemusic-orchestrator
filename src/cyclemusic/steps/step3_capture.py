from cyclemusic.adapters.capture_runner import capture_playlist

def capture_tracks(cfg, playlist_id, missing_tracks):
    return capture_playlist(playlist_id=playlist_id, expected=len(missing_tracks), captures_dir=cfg.captures_dir)
