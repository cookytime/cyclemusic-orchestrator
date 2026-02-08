CycleMusic Orchestrator
======================

Quick setup and run instructions for development.

Prerequisites
-
- Python 3.10+ (3.12 recommended)
- ffmpeg
- librespot (required; capture is pipe-only)

Setup
-
1. Create a virtual environment and install dependencies:

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

2. Copy the example env and fill in secrets:

```bash
cp .env.example .env
# edit .env to add SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET, SPOTIFY_REFRESH_TOKEN
```

3. Install and configure `librespot` on the machine; capture is pipe-only.

Running
-
- Run the full pipeline:

```bash
./scripts/run_pipeline.sh
```

- Manual step-by-step (recommended if you want to run each stage yourself):

```bash
# 1) Update the processing playlist with missing choreography tracks
./scripts/step1_update_playlist.sh

# 2) Start capture (this launches librespot with --backend pipe and waits for playback)
./scripts/step2_capture_playlist.sh

# 3) Analyze captured wavs (analysis + choreography by default)
./scripts/step3_analyze_captures.sh
```

- Optional: if you want to split analysis and choreography into separate steps:

```bash
# Analyze only (writes *.music_map.json, skips choreography)
ANALYSIS_ONLY=1 ./scripts/step3_analyze_captures.sh

# Generate choreography from existing *.music_map.json outputs
./scripts/step4_generate_choreography.sh
```

- Run the capture watcher (will use `.env`):

```bash
./scripts/watch_analyze.sh
./scripts/watch_upload.sh
```

Capture mode
-
- This project uses a pipe capture via `librespot --backend pipe` and `ffmpeg` by default. `LIBRESPOT_PIPE` is enabled by default and pipe capture is the only supported capture mode.

Notes
-
- If you run scripts directly from the `scripts/` directory they will auto-activate `.venv` (if present) and set `PYTHONPATH` so you don't need to export it manually.
