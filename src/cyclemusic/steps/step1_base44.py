from cyclemusic.adapters.base44_utils import get_all_tracks, filter_tracks_needing_choreography

def fetch_missing(cfg):
    tracks = get_all_tracks()
    missing = filter_tracks_needing_choreography(tracks)
    return missing[:cfg.batch_size]
