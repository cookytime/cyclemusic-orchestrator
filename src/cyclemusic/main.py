import time
import signal

from cyclemusic.log import setup_logging, log
from cyclemusic.config import load_config

from cyclemusic.steps.step1_base44 import fetch_missing
from cyclemusic.steps.step2_spotify_playlist import sync_playlist

_shutdown = False

def sleep_interruptible(seconds: float, step: float = 0.1):
    """Sleep in small increments so Ctrl-C/SIGTERM reacts quickly."""
    remaining = seconds
    while remaining > 0 and not _shutdown:
        time.sleep(min(step, remaining))
        remaining -= step

def _stop(*_):
    global _shutdown
    _shutdown = True

def main() -> int:
    setup_logging()
    cfg = load_config()

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)

    log.info("scheduler starting (capture/analyze/upload are external services)")

    while not _shutdown:
        try:
            missing_tracks = fetch_missing(cfg)

            if not missing_tracks:
                log.info("no missing choreography tracks; sleeping %ss", cfg.poll_seconds)
                sleep_interruptible(5)
                continue

            log.info("found %d tracks missing choreography", len(missing_tracks))

            playlist_id = sync_playlist(cfg, missing_tracks)
            log.info("synced spotify processing playlist %s", playlist_id)

            # Nothing else to do here—capture service will record when you press play,
            # analyze/upload services will process files as they appear.
            log.info("READY: press play on the processing playlist. Rechecking in %ss", cfg.poll_seconds)
            sleep_interruptible(5)

        except Exception:
            log.exception("scheduler loop failed; retrying soon")
            time.sleep(5)

    log.info("scheduler exiting")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
