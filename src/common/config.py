# src/common/config.py (NEW - Centralized configuration)
"""Centralized configuration management."""
import os
from pathlib import Path
from dotenv import load_dotenv

# Detect project root
_current = Path(__file__).resolve()
PROJECT_ROOT = _current.parent.parent.parent

# Load environment variables
load_dotenv(PROJECT_ROOT / ".env", override=False)

# Directory paths (relative to project root)
CAPTURES_DIR = Path(os.getenv("CAPTURES_DIR", str(PROJECT_ROOT / "captures")))
QUEUE_DIR = Path(os.getenv("QUEUE_DIR", str(PROJECT_ROOT / "queue")))
LOG_DIR = Path(os.getenv("LOG_DIR", str(PROJECT_ROOT / "logs")))

# Ensure directories exist
CAPTURES_DIR.mkdir(parents=True, exist_ok=True)
QUEUE_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# API configuration
BASE44_APP_ID = os.environ.get("BASE44_APP_ID")
BASE44_API_KEY = os.environ.get("BASE44_API_KEY")
SPOTIFY_CLIENT_ID = os.environ.get("SPOTIFY_CLIENT_ID")
SPOTIFY_CLIENT_SECRET = os.environ.get("SPOTIFY_CLIENT_SECRET")
SPOTIFY_REFRESH_TOKEN = os.environ.get("SPOTIFY_REFRESH_TOKEN")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# Validate required environment variables
_required_vars = [
    "BASE44_APP_ID",
    "BASE44_API_KEY",
    "SPOTIFY_CLIENT_ID",
    "SPOTIFY_CLIENT_SECRET",
    "SPOTIFY_REFRESH_TOKEN",
]

_missing = [v for v in _required_vars if not os.environ.get(v)]
if _missing:
    raise RuntimeError(f"Missing required environment variables: {', '.join(_missing)}")
