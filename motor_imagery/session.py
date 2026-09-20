"""Motor-imagery recordings: markers, session metadata, and cutting trials.

Sessions are saved in the same shape as the SSVEP ones - raw.npz plus
session.json, written by ssvep_training.session - so the raw EEG is kept and the
analysis window can be moved afterwards without recording again. (The tutorial's
collector saves only the filtered 1 s windows, which locks the window length in
forever.)

Cutting a trial means: filter the whole recording from sample 0 with
filters.FilterBank, then slice ANALYSIS_START..+ANALYSIS_DURATION after each
marker. Markers inside the filter's priming stretch are dropped.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

from ssvep_training.session import latest_session as _latest_session
from ssvep_training.session import load_session, save_session  # noqa: F401 (re-exported)

from . import config as cfg
from .filters import filter_recording, primed_samples


def encode_marker(block: int, class_id: int) -> int:
    """block * 10 + class + 1, the SSVEP scheme (block 3, rest -> 33)."""
    return block * 10 + class_id + 1


def decode_marker(code: float) -> tuple[int, int] | None:
    """(block, class id) for a calibration marker, None for live / unknown ones."""
    code = int(round(code))
    if code == cfg.LIVE_MARKER or code < 10:
        return None
    class_id = code % 10 - 1
    if class_id not in cfg.CLASS_NAMES:
        return None
    return code // 10, class_id


def session_meta(board, mode: str, classes: list[int]) -> dict:
    """SSVEP's session_meta with the motor-imagery facts instead of flicker ones."""
    return {
        "paradigm": "motor_imagery",
        "board_id": board.board_id,
        "rate": board.rate,
        "eeg_rows": board.eeg_rows,
        # The board reports stream.py's SSVEP names; what the cap actually wore
        # is the motor-imagery montage, so record that and keep the raw labels.
        "names": cfg.CHANNEL_NAMES[:len(board.eeg_rows)],
        "board_names": board.names,
        "marker_row": board.marker_row,
        "mode": mode,
        "classes": classes,
        "class_names": cfg.names_for(classes),
        "bands": {k: list(v) for k, v in cfg.BANDS.items()},
        "cue_duration": cfg.CUE_DURATION,
        "hold_duration": cfg.HOLD_DURATION,
        "prime_seconds": cfg.PRIME_SECONDS,
        "marker_scheme": "block*10 + class + 1 (0=left, 1=right, 2=rest, 3=both); live = %d"
                         % cfg.LIVE_MARKER,
    }


def latest_session(directory: str | None = None) -> str | None:
    return _latest_session(directory or cfg.TRAINING_DATA_DIR, cfg.SESSION_PREFIX)


def window_slice(rate: float, start: float | None = None,
                 duration: float | None = None) -> tuple[int, int]:
    """Samples after a marker that get classified, as (offset, length)."""
    start = cfg.ANALYSIS_START if start is None else start
    duration = cfg.ANALYSIS_DURATION if duration is None else duration
    return int(round(start * rate)), int(round(duration * rate))


@dataclass
class Trials:
    """Cut calibration trials, one entry per band: (trials, channels, samples)."""
    bands: dict[str, np.ndarray]
    labels: np.ndarray            # class id per trial
    blocks: np.ndarray            # calibration block per trial, for grouped CV
    rate: float
    names: list[str]
    classes: list[int]
    mode: str
    skipped: int = 0              # markers dropped: still priming, or data missing
    meta: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.labels)

    @property
    def class_names(self) -> list[str]:
        return cfg.names_for(self.classes)

    def counts(self) -> dict[str, int]:
        return {cfg.CLASS_NAMES[c]: int((self.labels == c).sum()) for c in self.classes}

    def subset(self, index) -> "Trials":
        return Trials({b: x[index] for b, x in self.bands.items()}, self.labels[index],
                      self.blocks[index], self.rate, self.names, self.classes, self.mode,
                      self.skipped, self.meta)


def load_trials(path: str, start: float | None = None, duration: float | None = None) -> Trials:
    """Every labelled trial in a session, filtered exactly as live typing filters."""
    data, meta = load_session(path)
    rate = meta["rate"]
    bands = filter_recording(data[meta["eeg_rows"]], rate)
    markers = data[meta["marker_row"]]
    offset, length = window_slice(rate, start, duration)
    prime = primed_samples(rate)

    cut: dict[str, list[np.ndarray]] = {name: [] for name in bands}
    labels, blocks, skipped = [], [], 0
    for onset in np.flatnonzero(markers):
        decoded = decode_marker(markers[onset])
        if decoded is None:
            continue
        begin, end = int(onset) + offset, int(onset) + offset + length
        if onset < prime or end > data.shape[1]:
            skipped += 1                       # still settling, or the run ended mid-trial
            continue
        for name, filtered in bands.items():
            cut[name].append(filtered[:, begin:end])
        blocks.append(decoded[0])
        labels.append(decoded[1])

    stacked = {name: (np.stack(trials) if trials else np.empty((0, len(meta["eeg_rows"]), length)))
               for name, trials in cut.items()}
    return Trials(stacked, np.array(labels, dtype=int), np.array(blocks, dtype=int), rate,
                  meta.get("names", cfg.CHANNEL_NAMES), meta.get("classes", cfg.classes_for("lrr")),
                  meta.get("mode", "clench"), skipped, meta)


def model_path(session: str) -> str:
    return os.path.join(session, cfg.MODEL_FILE)


def session_info(path: str) -> dict:
    """session.json only - the rate and channel rows, without loading raw.npz."""
    import json

    with open(os.path.join(path, "session.json")) as f:
        return json.load(f)


def channel_quality(path: str) -> list[dict]:
    """Raw-channel RMS and clipping flags for quick contact/saturation checks."""
    data, meta = load_session(path)
    eeg = data[meta["eeg_rows"]]
    centered = eeg - eeg.mean(axis=1, keepdims=True)
    rms = np.sqrt(np.mean(centered * centered, axis=1))
    clipped = np.mean(np.abs(eeg) >= 300_000, axis=1)
    names = meta.get("names", cfg.CHANNEL_NAMES)
    return [
        {"name": names[i] if i < len(names) else f"ch{i + 1}",
         "rms": float(rms[i]), "clipped": float(clipped[i])}
        for i in range(eeg.shape[0])
    ]


def quality_warnings(path: str, rms_uv: float = 50_000.0,
                     clipped_fraction: float = 0.01) -> list[str]:
    """Human-readable warnings for recordings that are unlikely to decode well."""
    warnings = []
    for ch in channel_quality(path):
        reasons = []
        if ch["clipped"] >= clipped_fraction:
            reasons.append(f"{100 * ch['clipped']:.1f}% clipped")
        if ch["rms"] >= rms_uv:
            reasons.append(f"RMS {ch['rms']:.0f} uV")
        if reasons:
            warnings.append(f"{ch['name']}: " + ", ".join(reasons))
    return warnings
