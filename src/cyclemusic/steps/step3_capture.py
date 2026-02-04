import cyclemusic.adapters.capture_runner as runner

def capture_tracks(cfg, playlist_id, missing_tracks):
    # You can pass cfg.captures_dir and playlist_id directly
    return runner.capture_playlist_tracks(
        playlist_id_or_url=playlist_id,
        captures_dir=cfg.captures_dir
    )
