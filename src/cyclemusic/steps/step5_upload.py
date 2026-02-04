from cyclemusic.adapters.base44_utils import get_all_tracks, update_track_choreography

def upload_choreo(cfg, results):
    """
    results: [{ "track_id": "<spotify_id>", "choreography": <json> }, ...]
    We map spotify_id -> Base44 Track entity id, then PATCH Base44.
    """
    tracks = get_all_tracks()
    # Base44 typically uses "id" for entity id; keep fallback keys just in case
    spotify_to_base44 = {}
    for t in tracks:
        sid = t.get("spotify_id")
        eid = t.get("id") or t.get("_id") or t.get("entity_id")
        if sid and eid:
            spotify_to_base44[sid] = eid

    updated = 0
    missing = 0

    for item in results:
        spotify_id = item.get("track_id")
        choreography = item.get("choreography")

        if not spotify_id or choreography is None:
            continue

        base44_id = spotify_to_base44.get(spotify_id)
        if not base44_id:
            print(f"[upload] ⚠️ no Base44 Track found for spotify_id={spotify_id}")
            missing += 1
            continue

        update_track_choreography(base44_id, choreography)
        updated += 1
        print(f"[upload] ✓ updated Base44 track_id={base44_id} (spotify_id={spotify_id})")

    print(f"[upload] done: updated={updated} missing={missing}")
