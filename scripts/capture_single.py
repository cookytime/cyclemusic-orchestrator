#!/usr/bin/env python3
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from cyclemusic.adapters import capture_runner as cr


def main():
    import argparse

    p = argparse.ArgumentParser(description="One-shot librespot+ffmpeg capture for debugging")
    p.add_argument("track", help="Arbitrary track id label for log filenames")
    p.add_argument("--seconds", type=float, default=10.0, help="Duration to capture in seconds")
    args = p.parse_args()

    wav = Path("captures") / f"{args.track}.wav"
    wav.parent.mkdir(parents=True, exist_ok=True)

    lp = None
    fp = None
    try:
        lp, fp = cr.start_librespot_pipe(args.track, wav, args.seconds)
        print("Started librespot pid:", getattr(lp, "pid", None))
        print("ffmpeg pid:", getattr(fp, "pid", None))
        print("librespot log:", getattr(lp, "_logfile", None) and lp._logfile.name)
        print("ffmpeg out:", getattr(fp, "_outfile", None) and fp._outfile.name)
        print("ffmpeg err:", getattr(fp, "_errfile", None) and fp._errfile.name)
        time.sleep(max(0.1, args.seconds + 1.0))
    except Exception as e:
        print("Capture failed:", e)
    finally:
        cr.stop_proc(fp)
        cr.stop_proc(lp)


if __name__ == "__main__":
    raise SystemExit(main())
