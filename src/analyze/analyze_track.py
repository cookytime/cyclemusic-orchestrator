#!/usr/bin/env python3
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Consolidated path setup
SRC_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = SRC_ROOT.parent
load_dotenv(REPO_ROOT / ".env", override=False)

SCHEMA_DIR = SRC_ROOT / "schemas"
TRACK_SCHEMA_PATH = SCHEMA_DIR / "track_schema.json"

if not TRACK_SCHEMA_PATH.exists():
    raise FileNotFoundError(f"track_schema.json not found at {TRACK_SCHEMA_PATH}")

PROMPT_DIR = SRC_ROOT / "prompts"
SYSTEM_PROMPT_PATH = PROMPT_DIR / "choreography_system.txt"
USER_PROMPT_PATH = PROMPT_DIR / "choreography_user.txt"

if not SYSTEM_PROMPT_PATH.exists():
    raise FileNotFoundError(f"Missing system prompt: {SYSTEM_PROMPT_PATH}")
if not USER_PROMPT_PATH.exists():
    raise FileNotFoundError(f"Missing user prompt: {USER_PROMPT_PATH}")

SYSTEM_PROMPT = SYSTEM_PROMPT_PATH.read_text()
USER_PROMPT_TEMPLATE = USER_PROMPT_PATH.read_text()

def _reexec_with_venv() -> None:
    if __name__ != "__main__":
        return
    if os.environ.get("VIRTUAL_ENV"):
        return
    start = Path(__file__).resolve().parent
    venv_python = None
    for parent in (start, *start.parents):
        candidate = parent / ".venv" / "bin" / "python"
        if candidate.exists():
            venv_python = candidate
            break
    if venv_python is None:
        return
    if Path(sys.executable).resolve() == venv_python.resolve():
        return
    os.execv(str(venv_python), [str(venv_python), *sys.argv])

_reexec_with_venv()

import copy
import json
import re
import sys
import time
from string import Template
from contextlib import contextmanager

import librosa
import librosa.beat
import librosa.effects
import librosa.segment
import numpy as np
import requests
from mutagen._file import File as MutagenFile
from openai import OpenAI

# Audio processing constants
AUDIO_SAMPLE_RATE = 22050
AUDIO_HOP_LENGTH = 512
TRIM_SILENCE_DB = 30

# Peak detection constants
PEAK_PRE_MAX = 20
PEAK_POST_MAX = 20
PEAK_PRE_AVG = 20
PEAK_POST_AVG = 20
PEAK_DELTA = 0.5
PEAK_WAIT = 20

# Anchor detection constants
DROP_ENERGY_THRESHOLD = 0.7
DROP_CONFIDENCE = 0.85
PEAK_CONFIDENCE = 0.7
DROP_PRIORITY_BONUS = 0.25
MAX_ANCHORS_PER_MIN = 2.0
MIN_ANCHOR_SPACING_S = 6.0
SNAP_TOLERANCE_S = 0.15

# Block timing constants
FIRST_BLOCK_WITHIN_S = 10.0
LAST_BLOCK_WITHIN_S = 20.0
TARGET_GAP_S = 34.0
MAX_GAP_S = 55.0

# API retry constants
MAX_RETRIES = 3
RETRY_BACKOFF_S = 1.0
API_TIMEOUT_S = 30


class NumpyEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.integer):
            return int(o)
        elif isinstance(o, np.floating):
            return float(o)
        elif isinstance(o, np.ndarray):
            return o.tolist()
        return super(NumpyEncoder, self).default(o)


@contextmanager
def safe_file_write(path: Path):
    """Context manager for safe atomic file writes."""
    tmp_path = path.with_suffix(path.suffix + '.tmp')
    try:
        with open(tmp_path, 'w', encoding='utf-8') as f:
            yield f
        tmp_path.replace(path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def retry_with_backoff(func, max_retries=MAX_RETRIES, backoff=RETRY_BACKOFF_S):
    """Retry decorator with exponential backoff."""
    for attempt in range(max_retries):
        try:
            return func()
        except (requests.exceptions.RequestException, Exception) as e:
            if attempt == max_retries - 1:
                raise
            wait_time = backoff * (2 ** attempt)
            print(f"  ⚠️ Attempt {attempt + 1} failed: {e}. Retrying in {wait_time}s...")
            time.sleep(wait_time)


def _allow_null(schema: dict) -> dict:
    sch = copy.deepcopy(schema)
    t = sch.get("type")
    if isinstance(t, str):
        if t != "null":
            sch["type"] = [t, "null"]
    elif isinstance(t, list):
        if "null" not in t:
            sch["type"] = t + ["null"]
    if "enum" in sch and None in sch["enum"]:
        sch["enum"] = ["null" if v is None else v for v in sch["enum"]]
    return sch


def normalize_for_openai_json_schema(schema: dict) -> dict:
    sch = copy.deepcopy(schema)
    sch.pop("name", None)
    sch.pop("rls", None)

    def walk(node: dict) -> dict:
        node = copy.deepcopy(node)
        if node.get("type") == "object" and isinstance(node.get("properties"), dict):
            props = node["properties"]
            for k, v in list(props.items()):
                props[k] = walk(v)
                props[k] = _allow_null(props[k])
            node["required"] = list(props.keys())
            node["additionalProperties"] = False
            node["properties"] = props
        if node.get("type") == "array" and isinstance(node.get("items"), dict):
            node["items"] = walk(node["items"])
        return node

    sch = walk(sch)
    return sch


def load_track_schema() -> dict:
    with open(TRACK_SCHEMA_PATH, 'r', encoding='utf-8') as f:
        raw = json.load(f)
    return normalize_for_openai_json_schema(raw)


def load_track_metadata_for_audio(file_path: str) -> dict:
    base_path = os.path.splitext(file_path)[0]
    metadata_path = f"{base_path}.metadata.json"
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Failed to load track metadata from {metadata_path}: {e}")
    return {}


def generate_track_choreography_openai(music_map: dict, rider_settings: dict, track_metadata: dict) -> dict:
    client = OpenAI(timeout=API_TIMEOUT_S)
    track_schema = load_track_schema()
    system_text = SYSTEM_PROMPT

    track_metadata_json = json.dumps(track_metadata, indent=2, default=str)
    music_map_json = json.dumps(music_map, indent=2, cls=NumpyEncoder)

    user_prompt = (
        USER_PROMPT_TEMPLATE
        .replace("{{TRACK_METADATA_JSON}}", track_metadata_json)
        .replace("{{MUSIC_MAP_JSON}}", music_map_json)
    )

    def make_request():
        return client.chat.completions.create(
            model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": system_text},
                {"role": "user", "content": user_prompt},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "Track", "schema": track_schema, "strict": True},
            },
        )

    resp = retry_with_backoff(make_request)
    out_text = resp.choices[0].message.content
    if out_text is None:
        raise ValueError("OpenAI response did not contain any content.")
    track_data = json.loads(out_text)

    dur_s = float((music_map.get("metadata") or {}).get("duration_s") or 0.0)
    if dur_s > 0:
        track_data["duration_minutes"] = round(dur_s / 60.0, 2)

    track_data = clean_choreography_timestamps(
        track_data, duration_s_override=dur_s if dur_s > 0 else None
    )

    return track_data


_TS_RE = re.compile(r"^\s*(\d+):(\d+)\s*$")


def ts_to_seconds_loose(ts: str) -> int | None:
    m = _TS_RE.match(ts or "")
    if not m:
        return None
    mm = int(m.group(1))
    ss = int(m.group(2))
    mm += ss // 60
    ss = ss % 60
    return mm * 60 + ss


def seconds_to_ts(sec: int) -> str:
    mm = sec // 60
    ss = sec % 60
    return f"{mm}:{ss:02d}"


from bisect import bisect_left


def snap_to_next_downbeat(t: float, downbeats: list[float]) -> float:
    if not downbeats:
        return float(t)
    i = bisect_left(downbeats, t - 1e-6)
    if i >= len(downbeats):
        return float(downbeats[-1])
    return float(downbeats[i])


def build_allowed_block_starts(
    music_map: dict,
    first_within_s: float = FIRST_BLOCK_WITHIN_S,
    last_within_s: float = LAST_BLOCK_WITHIN_S,
    target_gap_s: float = TARGET_GAP_S,
    max_gap_s: float = MAX_GAP_S,
) -> list[str]:
    duration_s = float((music_map.get("metadata") or {}).get("duration_s") or 0.0)
    downbeats = list((music_map.get("global") or {}).get("downbeats_s") or [])
    timeline = list(music_map.get("timeline") or [])
    anchors = list(music_map.get("anchors") or [])

    if duration_s <= 0 and downbeats:
        duration_s = float(downbeats[-1])
    if duration_s <= 0:
        duration_s = 240.0

    candidates = set()
    candidates.add(snap_to_next_downbeat(0.0, downbeats))
    candidates.add(snap_to_next_downbeat(min(first_within_s, duration_s), downbeats))
    candidates.add(snap_to_next_downbeat(max(0.0, duration_s - last_within_s), downbeats))

    for seg in timeline:
        start = float(seg.get("start_s") or 0.0)
        end = float(seg.get("end_s") or start)
        candidates.add(snap_to_next_downbeat(start, downbeats))
        if end - start > 50.0:
            mid = start + (end - start) / 2.0
            candidates.add(snap_to_next_downbeat(mid, downbeats))

    for a in anchors:
        t = float(a.get("time_s") or 0.0)
        candidates.add(snap_to_next_downbeat(t, downbeats))

    cand = sorted(t for t in candidates if 0.0 <= t <= duration_s)

    filled = [cand[0]] if cand else [snap_to_next_downbeat(0.0, downbeats)]
    for t in cand[1:]:
        prev = filled[-1]
        gap = t - prev
        while gap > max_gap_s:
            new_t = snap_to_next_downbeat(prev + target_gap_s, downbeats)
            if new_t <= prev + 0.5:
                break
            filled.append(new_t)
            prev = new_t
            gap = t - prev
        filled.append(t)

    dedup = []
    for t in sorted(set(filled)):
        if not dedup or abs(t - dedup[-1]) > 0.75:
            dedup.append(t)

    if duration_s < 180:
        desired_min, desired_max = 5, 6
    elif duration_s < 240:
        desired_min, desired_max = 6, 8
    else:
        desired_min, desired_max = 7, 9

    if len(dedup) > desired_max:
        first = dedup[0]
        last = dedup[-1]
        middle = dedup[1:-1]
        keep_mid = max(0, desired_max - 2)
        if keep_mid <= 0:
            selected = [first, last]
        else:
            if keep_mid == 1:
                selected_mid = [middle[len(middle) // 2]] if middle else []
            else:
                idxs = [round(i * (len(middle) - 1) / (keep_mid - 1)) for i in range(keep_mid)]
                selected_mid = [middle[int(i)] for i in sorted(set(int(x) for x in idxs)) if middle]
            selected = [first] + selected_mid + [last]
        dedup = selected

    while len(dedup) < desired_min and len(dedup) >= 2:
        gaps = [(dedup[i + 1] - dedup[i], i) for i in range(len(dedup) - 1)]
        gaps.sort(reverse=True)
        gap, i = gaps[0]
        insert_t = snap_to_next_downbeat(dedup[i] + gap / 2.0, downbeats)
        if insert_t <= dedup[i] + 0.5 or insert_t >= dedup[i + 1] - 0.5:
            break
        dedup = sorted(set(dedup + [insert_t]))

    return [seconds_to_ts(int(round(t))) for t in dedup]


def clean_choreography_timestamps(track: dict, duration_s_override: float | None = None) -> dict:
    duration_s = int(round(float(duration_s_override or 0)))
    if duration_s <= 0:
        duration_s = int(round(float(track.get("duration_minutes", 0) or 0) * 60))
    if duration_s <= 0:
        duration_s = 10_000

    cleaned = []
    for item in track.get("choreography", []) or []:
        raw_ts = item.get("timestamp")
        sec = ts_to_seconds_loose(raw_ts)
        if sec is None:
            continue
        if sec > duration_s:
            continue
        item = dict(item)
        item["timestamp"] = seconds_to_ts(sec)
        cleaned.append(item)

    cleaned.sort(key=lambda x: ts_to_seconds_loose(x["timestamp"]) or 0)
    for i, item in enumerate(cleaned):
        t0 = ts_to_seconds_loose(item["timestamp"]) or 0
        t1 = (ts_to_seconds_loose(cleaned[i + 1]["timestamp"]) if i + 1 < len(cleaned) else duration_s)
        if t1 is None:
            t1 = duration_s
        item["duration_seconds"] = max(10, min(90, int(t1 - t0)))

    track = dict(track)
    track["choreography"] = cleaned
    return track


def _ensure_scalar_float(val):
    if isinstance(val, (np.ndarray, list)):
        if len(val) == 0:
            return 0.0
        return float(val[0])
    return float(val)


class TrackAnalyzer:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.y = None
        self.sr = AUDIO_SAMPLE_RATE
        self.duration = 0.0
        self.hop_length = AUDIO_HOP_LENGTH
        self.time_offset = 0.0
        self.rms = None
        self.spectral_centroid = None
        self.onset_env = None
        self.chroma = None
        self.tempo = 0.0
        self.beats = None
        self.beat_times = None
        self.downbeat_times = None
        self.key = None
        self.max_anchors_per_min = MAX_ANCHORS_PER_MIN
        self.min_anchor_spacing_s = MIN_ANCHOR_SPACING_S
        self.snap_anchors_to_downbeat = True
        self.snap_tolerance_s = SNAP_TOLERANCE_S
        self.drop_priority_bonus = DROP_PRIORITY_BONUS

    def _to_global_time(self, t):
        if isinstance(t, (list, np.ndarray)):
            return [float(x) + self.time_offset for x in t]
        return float(t) + self.time_offset

    def load_audio(self):
        try:
            print("Loading audio file...")
            y_raw, self.sr = librosa.load(self.file_path, sr=self.sr, mono=True)

            print("Trimming silence...")
            self.y, trim_indices = librosa.effects.trim(y_raw, top_db=TRIM_SILENCE_DB)
            self.time_offset = trim_indices[0] / self.sr

            self.duration = float(librosa.get_duration(y=self.y, sr=self.sr))
            print(f"Analysis Duration: {self.duration:.2f}s (Offset: {self.time_offset:.2f}s)")

            print("Computing features (RMS, Centroid, Onset)...")
            self.rms = librosa.feature.rms(y=self.y, hop_length=self.hop_length)[0]
            self.spectral_centroid = librosa.feature.spectral_centroid(
                y=self.y, sr=self.sr, hop_length=self.hop_length
            )[0]
            self.onset_env = librosa.onset.onset_strength(
                y=self.y, sr=self.sr, hop_length=self.hop_length
            )
            self.chroma = librosa.feature.chroma_cqt(
                y=self.y, sr=self.sr, hop_length=self.hop_length
            )

            print("Extracting beat grid...")
            self.tempo, self.beats = librosa.beat.beat_track(
                onset_envelope=self.onset_env,
                sr=self.sr,
                hop_length=self.hop_length,
            )
            self.tempo = float(
                self.tempo[0] if isinstance(self.tempo, (list, np.ndarray)) else self.tempo
            )

            local_beat_times = librosa.frames_to_time(
                self.beats, sr=self.sr, hop_length=self.hop_length
            )

            if len(self.beats) > 4:
                candidates = self.beats[:4]
                candidates = candidates[candidates < len(self.rms)]
                if len(candidates) > 0:
                    energies = self.rms[candidates]
                    phase_offset = np.argmax(energies)
                else:
                    phase_offset = 0
            else:
                phase_offset = 0

            local_downbeats = local_beat_times[phase_offset::4]
            self.beat_times = self._to_global_time(local_beat_times)
            self.downbeat_times = self._to_global_time(local_downbeats)

            print("Audio loading complete.")
        except Exception as e:
            print(f"Error loading audio: {e}")
            raise

    def detect_key(self):
        try:
            maj_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
            min_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

            if self.chroma is None:
                raise ValueError("Chroma feature not computed. Run load_audio() first.")
            chroma_mean = np.mean(self.chroma, axis=1)

            maj_corrs = [np.corrcoef(chroma_mean, np.roll(maj_profile, i))[0, 1] for i in range(12)]
            min_corrs = [np.corrcoef(chroma_mean, np.roll(min_profile, i))[0, 1] for i in range(12)]

            max_maj = np.max(maj_corrs)
            max_min = np.max(min_corrs)

            pitch_classes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

            if max_maj > max_min:
                key_idx = np.argmax(maj_corrs)
                self.key = f"{pitch_classes[key_idx]} Major"
            else:
                key_idx = np.argmax(min_corrs)
                self.key = f"{pitch_classes[key_idx]} Minor"

        except Exception as e:
            print(f"Key detection failed: {e}")
            self.key = "Unknown"

    def get_metadata(self):
        self.detect_key()
        meta = {
            "title": None,
            "artist": None,
            "duration_s": round(float(self.duration), 2),
            "sample_rate": int(self.sr),
            "bpm": round(float(self.tempo), 1),
            "key": self.key,
            "bpm_confidence": 0.0,
        }

        try:
            audio_tags = MutagenFile(self.file_path)
            if audio_tags:
                tags = audio_tags.tags
                if tags:
                    def get_tag(keys):
                        for k in keys:
                            if k in tags:
                                val = tags[k]
                                return str(val[0]) if isinstance(val, list) else str(val)
                        return None

                    meta["title"] = get_tag(["TIT2", "title", "TITLE"]) or os.path.basename(self.file_path)
                    meta["artist"] = get_tag(["TPE1", "artist", "ARTIST"]) or "Unknown"
        except Exception as e:
            print(f"Metadata extraction warning: {e}")

        try:
            tempogram = librosa.feature.tempogram(onset_envelope=self.onset_env, sr=self.sr)
            meta["bpm_confidence"] = round(float(np.max(np.mean(tempogram, axis=1))), 2)
        except Exception:
            meta["bpm_confidence"] = 0.0

        return meta

    @staticmethod
    def _snap_to_nearest(t: float, grid: np.ndarray, threshold: float = SNAP_TOLERANCE_S) -> float:
        if grid is None or len(grid) == 0:
            return t
        idx = int(np.argmin(np.abs(grid - t)))
        nearest_val = float(grid[idx])
        dist = abs(nearest_val - t)
        if dist <= threshold:
            return nearest_val
        return t

    @staticmethod
    def _enforce_spacing(anchors, min_spacing_s: float):
        if not anchors:
            return anchors
        anchors = sorted(anchors, key=lambda a: a["time_s"])
        kept = []
        for a in anchors:
            if not kept:
                kept.append(a)
                continue
            if a["time_s"] - kept[-1]["time_s"] >= min_spacing_s:
                kept.append(a)
                continue
            prev = kept[-1]
            if a["importance"] > prev["importance"]:
                kept[-1] = a
            elif a["importance"] == prev["importance"]:
                if a["type"] == "drop" and prev["type"] != "drop":
                    kept[-1] = a
        return kept

    def get_segmentation(self):
        if self.chroma is None:
            raise ValueError("Chroma feature not computed. Run load_audio() first.")
        chroma_stack = librosa.feature.stack_memory(self.chroma, n_steps=10, delay=3)

        bounds_frames = librosa.segment.agglomerative(chroma_stack, k=8)
        local_bounds_times = librosa.frames_to_time(bounds_frames, sr=self.sr, hop_length=self.hop_length)

        local_bounds_times = np.unique(np.concatenate(([0.0], local_bounds_times, [self.duration])))
        local_bounds_times.sort()

        segments = []
        max_rms = float(np.max(self.rms) + 1e-6) if self.rms is not None else 1.0
        max_cent = float(np.max(self.spectral_centroid) + 1e-6) if self.spectral_centroid is not None else 1.0

        for i in range(len(local_bounds_times) - 1):
            start_local = float(local_bounds_times[i])
            end_local = float(local_bounds_times[i + 1])

            f_start = int(librosa.time_to_frames(start_local, sr=self.sr, hop_length=self.hop_length))
            f_end = int(librosa.time_to_frames(end_local, sr=self.sr, hop_length=self.hop_length))
            f_end = max(f_end, f_start + 1)

            seg_rms = float(np.mean(self.rms[f_start:f_end])) if self.rms is not None else 0.0
            seg_cent = float(np.mean(self.spectral_centroid[f_start:f_end])) if self.spectral_centroid is not None else 0.0

            energy = round(seg_rms / max_rms, 2)
            intensity = round(seg_cent / max_cent, 2)
            tension = round(float(energy * intensity), 2)

            intent_hint = None
            if start_local < 15.0 and energy < 0.4:
                intent_hint = "intro"
            elif end_local > (self.duration - 15.0) and energy < 0.4:
                intent_hint = "outro"
            else:
                if energy >= 0.75:
                    intent_hint = "surge"
                elif energy >= 0.4:
                    if self.rms is not None:
                        slope, _ = np.polyfit(np.arange(f_end - f_start), self.rms[f_start:f_end], 1)
                        intent_hint = "build" if slope > 0.0001 else "steady"
                    else:
                        intent_hint = "steady"
                else:
                    intent_hint = "steady"

            global_start = self._to_global_time(start_local)
            global_end = self._to_global_time(end_local)

            if isinstance(global_start, (list, np.ndarray)):
                global_start_val = float(global_start[0]) if global_start else 0.0
            else:
                global_start_val = float(global_start)
            if isinstance(global_end, (list, np.ndarray)):
                global_end_val = float(global_end[0]) if global_end else 0.0
            else:
                global_end_val = float(global_end)

            seg_downbeats = []
            if self.downbeat_times is not None and isinstance(self.downbeat_times, (list, np.ndarray)):
                seg_downbeats = [
                    round(float(t), 2)
                    for t in self.downbeat_times
                    if global_start_val <= t <= global_end_val
                ]

            segments.append({
                "start_s": round(global_start_val, 2),
                "end_s": round(global_end_val, 2),
                "energy": energy,
                "intensity": intensity,
                "tension": tension,
                "intent_hint": intent_hint,
                "downbeats_s": seg_downbeats,
            })

        return segments

    def get_anchors(self):
        anchors = []

        peak_frames = librosa.util.peak_pick(
            self.onset_env,
            pre_max=PEAK_PRE_MAX,
            post_max=PEAK_POST_MAX,
            pre_avg=PEAK_PRE_AVG,
            post_avg=PEAK_POST_AVG,
            delta=PEAK_DELTA,
            wait=PEAK_WAIT,
        )
        local_peak_times = librosa.frames_to_time(peak_frames, sr=self.sr, hop_length=self.hop_length)

        if self.rms is not None:
            rms_diff = np.diff(self.rms)
            if len(rms_diff) > 0:
                thresh = float(np.max(rms_diff) * DROP_ENERGY_THRESHOLD)
                drop_frames = np.where(rms_diff > thresh)[0]
            else:
                drop_frames = np.array([], dtype=int)
        else:
            drop_frames = np.array([], dtype=int)

        local_drop_times = librosa.frames_to_time(drop_frames, sr=self.sr, hop_length=self.hop_length)

        for t_local in local_drop_times:
            t_global = self._to_global_time(t_local)
            if isinstance(t_global, (list, np.ndarray)):
                t_global_val = float(t_global[0]) if t_global else 0.0
            else:
                t_global_val = float(t_global)

            if self.snap_anchors_to_downbeat and self.downbeat_times is not None:
                t_global_val = float(t_global_val)
                t_global = self._snap_to_nearest(t_global_val, np.array(self.downbeat_times), self.snap_tolerance_s)

            anchors.append({
                "time_s": round(_ensure_scalar_float(t_global), 2),
                "type": "drop",
                "confidence": DROP_CONFIDENCE,
                "reason": "Sudden energy spike",
            })

        for t_local in local_peak_times:
            t_global = self._to_global_time(t_local)
            if isinstance(t_global, (list, np.ndarray)):
                t_global_val = float(t_global[0]) if t_global else 0.0
            else:
                t_global_val = float(t_global)

            if self.snap_anchors_to_downbeat and self.downbeat_times is not None:
                t_global_val = float(t_global_val)
                t_global = self._snap_to_nearest(t_global_val, np.array(self.downbeat_times), self.snap_tolerance_s)

            if not any(abs(_ensure_scalar_float(t_global) - a["time_s"]) < 1.0 for a in anchors):
                anchors.append({
                    "time_s": round(_ensure_scalar_float(t_global), 2),
                    "type": "peak",
                    "confidence": PEAK_CONFIDENCE,
                    "reason": "Strong transient onset",
                })

        anchors.sort(key=lambda x: (x["time_s"], 0 if x["type"] == "drop" else 1))
        dedup = []
        seen = set()
        for a in anchors:
            key = a["time_s"]
            if key in seen:
                if a["type"] == "drop":
                    for i in range(len(dedup) - 1, -1, -1):
                        if dedup[i]["time_s"] == key:
                            dedup[i] = a
                            break
            else:
                dedup.append(a)
                seen.add(key)

        for a in dedup:
            imp = float(a["confidence"])
            if a["type"] == "drop":
                imp += self.drop_priority_bonus
            a["importance"] = round(imp, 3)

        spaced = self._enforce_spacing(dedup, self.min_anchor_spacing_s)
        max_keep = max(8, int((self.duration / 60.0) * self.max_anchors_per_min))
        spaced_sorted = sorted(spaced, key=lambda a: a["importance"], reverse=True)[:max_keep]
        spaced_sorted.sort(key=lambda a: a["time_s"])

        return spaced_sorted

    def load_spotify_metadata(self):
        base_path = os.path.splitext(self.file_path)[0]
        metadata_path = f"{base_path}.metadata.json"

        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, "r") as f:
                    return json.load(f)
            except Exception as e:
                print(f"Warning: Failed to load Spotify metadata from {metadata_path}: {e}")
                return None
        return None

    def analyze(self):
        print(f"Analyzing {self.file_path} ...")
        self.load_audio()

        output = {
            "metadata": self.get_metadata(),
            "global": {
                "tempo_bpm": round(float(self.tempo), 1),
                "beats_s": (
                    [round(float(t), 2) for t in self.beat_times]
                    if self.beat_times is not None and isinstance(self.beat_times, (list, np.ndarray))
                    else ([round(float(self.beat_times), 2)] if isinstance(self.beat_times, (float, int)) else [])
                ),
                "downbeats_s": (
                    [round(float(t), 2) for t in self.downbeat_times]
                    if self.downbeat_times is not None and isinstance(self.downbeat_times, (list, np.ndarray))
                    else ([round(float(self.downbeat_times), 2)] if isinstance(self.downbeat_times, (float, int)) else [])
                ),
            },
            "timeline": self.get_segmentation(),
            "anchors": self.get_anchors(),
        }

        spotify_metadata = self.load_spotify_metadata()
        if spotify_metadata:
            print("Merging Spotify metadata...")
            output["spotify"] = spotify_metadata
            if spotify_metadata.get("name"):
                output["metadata"]["title"] = spotify_metadata["name"]
            if spotify_metadata.get("artists") and len(spotify_metadata["artists"]) > 0:
                output["metadata"]["artist"] = ", ".join(
                    a["name"] for a in spotify_metadata["artists"] if a.get("name")
                )

        return output


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze a track and optionally generate choreography.")
    parser.add_argument("file_path", help="Path to a wav file to analyze.")
    parser.add_argument(
        "--analysis-only",
        action="store_true",
        help="Only write the music map; skip choreography generation.",
    )
    args = parser.parse_args()

    file_path = args.file_path
    if not os.path.exists(file_path):
        print("File not found.")
        sys.exit(1)

    analyzer = TrackAnalyzer(file_path)
    result = analyzer.analyze()

    base_path = os.path.splitext(file_path)[0]
    output_file = f"{base_path}.music_map.json"

    with safe_file_write(Path(output_file)) as f:
        json.dump(result, f, indent=2, cls=NumpyEncoder)

    print(f"Analysis complete. Output written to {output_file}")

    if args.analysis_only:
        print("Analysis-only mode enabled; skipping choreography generation.")
        sys.exit(0)

    load_dotenv(REPO_ROOT / ".env")

    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  OPENAI_API_KEY not found in .env - skipping choreography generation")
        sys.exit(0)

    rider_settings = {
        "rider_level": os.environ.get("RIDER_LEVEL", "intermediate"),
        "resistance_scale": {"min": 1, "max": 24},
        "cadence_limits": {
            "seated": {"min_rpm": 60, "max_rpm": 115},
            "standing": {"min_rpm": 55, "max_rpm": 80},
        },
        "cue_spacing_s": {"min": 24, "max": 32},
    }

    try:
        print("\nGenerating choreography with OpenAI...")
        track_metadata = load_track_metadata_for_audio(file_path)
        track_json = generate_track_choreography_openai(result, rider_settings, track_metadata)

        choreography_path = f"{base_path}.choreography.json"
        output_data = {"track": track_json}

        with safe_file_write(Path(choreography_path)) as f:
            json.dump(output_data, f, indent=2)

        print(f"✓ Choreography saved: {choreography_path}")

        spotify_id = track_json.get("spotify_id") or (result.get("spotify") or {}).get("spotify_id")
        if spotify_id:
            captures_dir = os.path.dirname(os.path.abspath(file_path))
            spotify_named_path = os.path.join(captures_dir, f"{spotify_id}.choreography.json")

            if os.path.abspath(spotify_named_path) != os.path.abspath(choreography_path):
                with safe_file_write(Path(spotify_named_path)) as f:
                    json.dump(output_data, f, indent=2)
                print(f"✓ Also saved: {spotify_named_path}")

        print("\n✅ Analysis and choreography generation complete!")

    except Exception as e:
        print(f"❌ Choreography generation failed: {e}")
        print("Music map was saved successfully - run choreography generation separately if needed")
        sys.exit(1)
