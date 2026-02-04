from dataclasses import dataclass
import os
from dotenv import load_dotenv

load_dotenv()

@dataclass(frozen=True)
class Config:
    poll_seconds: int
    batch_size: int
    captures_dir: str

def load_config() -> Config:
    # Fail loud on missing secrets
    os.environ["BASE44_APP_ID"]
    os.environ["BASE44_API_KEY"]
    os.environ["SPOTIFY_CLIENT_ID"]
    os.environ["SPOTIFY_CLIENT_SECRET"]
    os.environ["SPOTIFY_REFRESH_TOKEN"]
    os.environ["SPOTIFY_PLAYLIST_ID"]

    return Config(
        poll_seconds=float(os.getenv("POLL_SECONDS", "5")),
        batch_size=int(os.getenv("BATCH_SIZE", "25")),
        captures_dir=os.getenv("CAPTURES_DIR", "/home/appuser/cyclemusic-orchestrator/captures"),
    )
