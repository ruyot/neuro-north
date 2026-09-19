# Recorded sessions: markers, saving/loading, and cutting trials

from __future__ import annotations

import glob
import json
import math
import os
from dataclasses import dataclass, field
from importlib.metadata import version

import numpy as np

from . import knight as stream  # stream.clean() is the filter for everything

from . import config as cfg
from .signal_quality import Quality, assess_window

SIGNAL_CONTRACT_ERROR = "Session lacks the validated signal contract; record a new calibration."


def preprocessing_meta() -> dict:
    return {
        "version": "fixed-history-v1", "mains_hz": stream.MAINS_HZ,
        "bandpass_hz": [1.0, 40.0], "order": 4,
        "history_s": cfg.FILTER_HISTORY, "latency_s": cfg.VISUAL_LATENCY,
        "gaze_s": cfg.GAZE_DURATION, "flicker_s": cfg.FLICKER_DURATION,
    }


def _scale_settings() -> tuple[float | None, str | None, int | None]:
    factor, evidence = os.getenv("EEG_UV_PER_SDK_UNIT"), os.getenv("EEG_SCALE_EVIDENCE")
    if factor is not None or evidence is not None:
        try:
            factor = float(factor)
        except (TypeError, ValueError):
            raise ValueError("EEG_UV_PER_SDK_UNIT and nonempty EEG_SCALE_EVIDENCE are required together.") from None
        if not math.isfinite(factor) or factor <= 0 or not evidence or not evidence.strip():
            raise ValueError("EEG_UV_PER_SDK_UNIT must be finite/positive with nonempty EEG_SCALE_EVIDENCE.")
        evidence = evidence.strip()
    polarity = os.getenv("LEADOFF_OFF_BIT")
    if polarity is not None and polarity not in ("0", "1"):
        raise ValueError("LEADOFF_OFF_BIT must be 0 or 1 after a controlled firmware contact check.")
    return factor, evidence, None if polarity is None else int(polarity)


def _validate_meta(meta: dict) -> None:
    """Validate new metadata without pretending an unfinished/failed run is model-ready."""
    required = {
        "schema_version", "board_id", "rate", "eeg_rows", "names", "marker_row",
        "packet_row", "timestamp_row", "timestamp_kind", "imu_rows", "imu_units_documented",
        "brainflow_version", "meegkit_version", "gain_requested", "raw_eeg_unit", "model_eeg_unit",
        "channels_requested", "bias_channels_requested", "eeg_uv_per_sdk_unit", "eeg_scale_evidence",
        "leadoff_off_bit", "display_refresh_hz", "preprocessing", "quality_baseline",
        "quality_baseline_error", "trials", "frequencies", "letters", "flicker_duration", "marker_scheme",
    }
    if not required <= meta.keys():
        raise ValueError(SIGNAL_CONTRACT_ERROR)
    fixed = {
        "schema_version": 2, "board_id": 66, "rate": 125, "gain_requested": 12,
        "packet_row": 0, "timestamp_row": 20, "marker_row": 21,
        "timestamp_kind": "host_frame_parse_unix", "raw_eeg_unit": "brainflow_native",
        "model_eeg_unit": "brainflow_native",
        "imu_rows": {"accel": [11, 12, 13], "gyro": [14, 15, 16], "mag": [17, 18, 19]},
        "imu_units_documented": ["m/s^2", "rad/s", "uT"],
        "marker_scheme": "calibration=block*10+target+1; live=-trial_id",
    }
    for key, value in fixed.items():
        if meta[key] != value or (type(value) is int and type(meta[key]) is not int):
            raise ValueError(f"Invalid session {key}: expected {value!r}, got {meta[key]!r}.")
    rows = stream.validate_eeg_channels(meta["eeg_rows"])
    if rows != meta["eeg_rows"] or any(meta[key] != rows for key in ("channels_requested", "bias_channels_requested")):
        raise ValueError("Session channel/bias membership must match ascending EEG rows.")
    names, letters, freqs = meta["names"], meta["letters"], meta["frequencies"]
    if not isinstance(names, list) or len(names) != len(rows) or any(not isinstance(n, str) or not n.strip() for n in names):
        raise ValueError("Session channel names must match EEG rows.")
    if (not isinstance(letters, list) or not letters
            or any(not isinstance(v, str) or not v.strip() for v in letters) or len(set(letters)) != len(letters)
            or not isinstance(freqs, list) or len(freqs) != len(letters)
            or any(type(f) not in (int, float) or not math.isfinite(f) or not 0 < f < meta["rate"] / 2 for f in freqs)
            or len(set(freqs)) != len(freqs)):
        raise ValueError("Session target letters/frequencies are malformed.")
    for key in ("brainflow_version", "meegkit_version"):
        if not isinstance(meta[key], str) or not meta[key].strip():
            raise ValueError(f"Invalid session {key}.")
    factor, evidence = meta["eeg_uv_per_sdk_unit"], meta["eeg_scale_evidence"]
    if factor is not None or evidence is not None:
        if (type(factor) not in (int, float) or not math.isfinite(factor) or factor <= 0
                or not isinstance(evidence, str) or not evidence.strip()):
            raise ValueError("Session EEG scale requires a finite positive factor and nonempty evidence together.")
    polarity = meta["leadoff_off_bit"]
    if polarity is not None and (type(polarity) is not int or polarity not in (0, 1)):
        raise ValueError("Session leadoff_off_bit must be null, 0 or 1.")
    refresh = meta["display_refresh_hz"]
    if refresh is not None and (type(refresh) not in (int, float) or not math.isfinite(refresh) or refresh <= 0):
        raise ValueError("Session display_refresh_hz must be null or finite and positive.")
    preprocessing = meta["preprocessing"]
    if (not isinstance(preprocessing, dict) or set(preprocessing) != set(preprocessing_meta())
            or preprocessing.get("version") != "fixed-history-v1"
            or preprocessing.get("mains_hz") not in (50, 60)
            or preprocessing.get("bandpass_hz") != [1.0, 40.0] or preprocessing.get("order") != 4
            or any(type(preprocessing[key]) not in (int, float) or not math.isfinite(preprocessing[key])
                   or preprocessing[key] <= 0 for key in ("history_s", "latency_s", "gaze_s", "flicker_s"))
            or preprocessing["latency_s"] + preprocessing["gaze_s"] > preprocessing["flicker_s"]
            or meta["flicker_duration"] != preprocessing["flicker_s"]):
        raise ValueError("Malformed fixed-history preprocessing metadata.")
    error = meta["quality_baseline_error"]
    if error is not None and (not isinstance(error, str) or not error.strip()):
        raise ValueError("Session quality_baseline_error must be null or a validation error string.")
    if meta["quality_baseline"] is not None:
        from .signal_quality import validate_baseline

        validate_baseline(meta)
        if error is not None:
            raise ValueError("A valid quality baseline cannot also have a baseline error.")
    if not isinstance(meta["trials"], dict):
        raise ValueError("Session trials must be keyed by decimal marker codes.")
    trial_ids = set()
    for code, trial in meta["trials"].items():
        try:
            marker = int(code)
        except (TypeError, ValueError):
            raise ValueError("Session trials must be keyed by decimal marker codes.") from None
        if not isinstance(code, str) or str(marker) != code or marker == 0:
            raise ValueError("Session trial marker codes must be nonzero decimal strings.")
        if not isinstance(trial, dict) or set(trial) != {"trial_id", "onset_request_monotonic_ns", "marker_submit_monotonic_ns", "dropped_frames"}:
            raise ValueError(f"Malformed trial metadata for marker {code}.")
        trial_id, requested, submitted = (trial[key] for key in ("trial_id", "onset_request_monotonic_ns", "marker_submit_monotonic_ns"))
        if (any(type(v) is not int or v <= 0 for v in (trial_id, requested, submitted))
                or submitted < requested or trial_id in trial_ids or (marker < 0 and marker != -trial_id)):
            raise ValueError(f"Invalid trial identity/software timing for marker {code}.")
        dropped = trial["dropped_frames"]
        if dropped is not None and (type(dropped) is not int or dropped < 0):
            raise ValueError(f"Invalid dropped-frame count for marker {code}.")
        trial_ids.add(trial_id)


def _require_model_contract(meta: dict) -> None:
    if meta.get("schema_version") != 2 or meta.get("quality_baseline") is None:
        raise ValueError(SIGNAL_CONTRACT_ERROR)
    for package, pinned in (("brainflow", "5.23.0"), ("meegkit", "0.2.0")):
        if meta[f"{package}_version"] != pinned or version(package) != pinned:
            raise ValueError(f"Model trials require {package}=={pinned} in the session and runtime; record a new calibration.")
    if meta["preprocessing"] != preprocessing_meta():
        raise ValueError("Preprocessing changed; record a new calibration.")
    if meta["quality_baseline"]["multiplier"] != cfg.QUALITY_MULTIPLIER:
        raise ValueError("Quality multiplier changed; record a new calibration.")
    refresh = meta["display_refresh_hz"]
    if refresh is None or abs(refresh - cfg.EXPECTED_REFRESH_HZ) > 1.0:
        raise ValueError(f"Model trials require measured display refresh within 1 Hz of {cfg.EXPECTED_REFRESH_HZ}; record a new calibration.")

def encode_marker(block: int, target: int) -> int:
    """Calibration marker: block * 10 + target + 1 (block 3, target B -> 32)."""
    return block * 10 + target + 1


def decode_marker(code: float, n_targets: int = cfg.N_TARGETS) -> tuple[int, int] | None:
    """(block, target) for a calibration marker, None for live / unknown markers."""
    if not np.isfinite(code) or code != int(code):
        return None
    code = int(code)
    target = code % 10 - 1
    if not 0 <= target < n_targets or code < 10:
        return None
    return code // 10, target

def session_meta(board) -> dict:
    factor, evidence, polarity = _scale_settings()
    meta = {
        "schema_version": 2,
        "board_id": board.board_id, "rate": board.rate,
        "eeg_rows": list(board.eeg_rows), "names": list(board.names),
        "marker_row": board.marker_row, "packet_row": board.packet_row,
        "timestamp_row": board.timestamp_row, "timestamp_kind": "host_frame_parse_unix",
        "imu_rows": board.imu_rows, "imu_units_documented": ["m/s^2", "rad/s", "uT"],
        "brainflow_version": version("brainflow"), "meegkit_version": version("meegkit"),
        "gain_requested": 12, "channels_requested": list(board.eeg_rows),
        "bias_channels_requested": list(board.eeg_rows),
        "raw_eeg_unit": "brainflow_native", "model_eeg_unit": "brainflow_native",
        "eeg_uv_per_sdk_unit": factor, "eeg_scale_evidence": evidence,
        "leadoff_off_bit": polarity, "display_refresh_hz": None,
        "preprocessing": preprocessing_meta(), "quality_baseline": None,
        "quality_baseline_error": None, "trials": {},
        "frequencies": list(cfg.STIMULUS_FREQUENCIES), "letters": list(cfg.TARGET_LETTERS),
        "flicker_duration": cfg.FLICKER_DURATION,
        "marker_scheme": "calibration=block*10+target+1; live=-trial_id",
    }
    _validate_meta(meta)
    return meta


def save_session(path: str, data: np.ndarray, meta: dict) -> None:
    """Write raw.npz + session.json; the .npz is written to a temp file then renamed."""
    os.makedirs(path, exist_ok=True)
    tmp = os.path.join(path, "raw.tmp.npz")
    np.savez(tmp, data=data)
    os.replace(tmp, os.path.join(path, "raw.npz"))
    with open(os.path.join(path, "session.json"), "w") as f:
        json.dump({**meta, "n_samples": int(data.shape[1]),
                   "n_markers": int(np.count_nonzero(data[meta["marker_row"]]))}, f, indent=2, allow_nan=False)


def load_session(path: str) -> tuple[np.ndarray, dict]:
    with open(os.path.join(path, "session.json")) as f:
        meta = json.load(f)
    if not isinstance(meta, dict):
        raise ValueError("Session metadata must be a JSON object.")
    if "schema_version" in meta:
        _validate_meta(meta)
    with np.load(os.path.join(path, "raw.npz"), allow_pickle=False) as raw:
        data = raw["data"]
    if data.ndim != 2 or not np.issubdtype(data.dtype, np.number) or np.iscomplexobj(data):
        raise ValueError("Raw session data must be a real numeric rows-by-samples array.")
    if meta.get("schema_version") == 2 and data.shape[0] != 22:
        raise ValueError("Knight IMU raw data must retain all 22 SDK rows.")
    return data, meta


def latest_session() -> str | None:
    """Newest session folder in training_data/ with at least one trial. Sessions
    aborted before the first trial are skipped, not deleted."""
    for path in sorted(glob.glob(os.path.join(cfg.TRAINING_DATA_DIR, "session_*")), reverse=True):
        info = os.path.join(path, "session.json")
        if not (os.path.exists(info) and os.path.exists(os.path.join(path, "raw.npz"))):
            continue
        with open(info) as f:
            if json.load(f).get("n_markers", 0) > 0:
                return path
    return None

def history_samples(rate: int) -> int:
    return int(round(cfg.FILTER_HISTORY * rate))


def flicker_samples(rate: int) -> int:
    return int(round(cfg.FLICKER_DURATION * rate))


def analysis_slice(rate: int) -> slice:
    """Samples after the flicker onset that get classified: skip latency, keep GAZE_DURATION."""
    start = int(round(cfg.VISUAL_LATENCY * rate))
    return slice(start, start + int(round(cfg.GAZE_DURATION * rate)))


def epoch_at(data: np.ndarray, onset: int, meta: dict, full: bool = False) -> tuple[np.ndarray | None, Quality]:
    """Admit the complete raw history+flicker interval, then filter selected EEG.

    Returns (samples, channels) or no epoch with the rejection flags. SDK marker
    submission and UI completion are software provenance, not optical timing.
    """
    if data.ndim != 2 or data.shape[0] != 22:
        raise ValueError("Epoch extraction requires all 22 Knight SDK rows.")
    rate = meta["rate"]
    hist, flick = history_samples(rate), flicker_samples(rate)
    if onset < hist or onset + flick > data.shape[1]:
        return None, Quality.INCOMPLETE
    context = data[:, onset - hist:onset + flick]
    quality = assess_window(context, meta)
    marker = data[meta["marker_row"], onset]
    trial = meta["trials"].get(str(int(marker))) if np.isfinite(marker) and marker == int(marker) else None
    if trial is None or trial["dropped_frames"] is None:
        quality |= Quality.INCOMPLETE
    elif trial["dropped_frames"] > 0:
        quality |= Quality.TIMING
    if quality != Quality.OK:
        return None, quality
    filtered = stream.clean(context[meta["eeg_rows"]], rate)[:, hist:]
    return (filtered if full else filtered[:, analysis_slice(rate)]).T, Quality.OK


@dataclass
class Trials:
    eeg: np.ndarray        # (samples, channels, trials) - meegkit's layout
    targets: np.ndarray    # target index per trial
    blocks: np.ndarray     # calibration block per trial
    rate: int
    names: list[str]
    freqs: list[float]     # the session's own targets (not today's config)
    letters: list[str]
    rows: list[int]        # the board rows this session recorded
    skipped: int = 0       # rejected calibration trials, not individual reasons
    rejection_counts: dict[str, int] = field(default_factory=dict)

    @property
    def n_targets(self) -> int:
        return len(self.freqs)


def load_trials(path: str, full: bool = False) -> Trials:
    """Every labelled calibration trial in a session, cut with epoch_at().

    Targets come from the session's own session.json, so a session recorded with
    a different frequency set still gets labelled correctly.
    """
    data, meta = load_session(path)
    _require_model_contract(meta)
    rate = meta["rate"]
    freqs, letters = meta["frequencies"], meta["letters"]
    markers = data[meta["marker_row"]]
    epochs, targets, blocks, skipped = [], [], [], 0
    rejection_counts = {}
    for onset in np.flatnonzero(markers):
        decoded = decode_marker(markers[onset], len(freqs))
        if decoded is None:
            continue
        epoch, quality = epoch_at(data, int(onset), meta, full)
        if quality != Quality.OK:
            skipped += 1
            for reason in Quality:
                if quality & reason:
                    rejection_counts[reason.name] = rejection_counts.get(reason.name, 0) + 1
            continue
        epochs.append(epoch)
        blocks.append(decoded[0])
        targets.append(decoded[1])
    stacked = np.stack(epochs, axis=2) if epochs else np.empty((0, len(meta["eeg_rows"]), 0))
    return Trials(stacked, np.array(targets), np.array(blocks), rate, meta["names"], freqs, letters,
                  list(meta["eeg_rows"]), skipped, rejection_counts)
