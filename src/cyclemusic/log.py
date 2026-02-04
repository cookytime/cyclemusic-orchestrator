import logging
import os

def setup_logging():
    log_dir = os.getenv("LOG_DIR", "/home/appuser/cyclemusic-orchestrator/logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "worker.log")

    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_path),
        ],
    )

log = logging.getLogger("cyclemusic")
