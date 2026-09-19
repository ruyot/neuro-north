# Recorded sessions: markers, saving/loading, and cutting trials

from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass

import numpy as np

from . import knight as stream  # stream.clean() is the filter for everything

from . import config as cfg

def encode_marker(block: int, target: int) -> int:
    """Calibration marker: block * 10 + target + 1 (block 3, target B -> 32)."""
    return block * 10 + target + 1


def decode_marker(code: float, n_targets: int = cfg.N_TARGETS) -> tuple[int, int] | None:
    """(block, target) for a calibration marker, None for live / unknown markers."""
    code = int(round(code))
    target = code % 10 - 1
    if code == cfg.LIVE_MARKER or not 0 <= target < n_targets or code < 10:
        return None
    return code // 10, target

def session_meta(board) -> dict:
    return {
        "board_id": board.board_id,
        "rate": board.rate,
        "eeg_rows": board.eeg_rows,
        "names": board.names,
        "marker_row": board.marker_row,
        "frequencies": cfg.STIMULUS_FREQUENCIES,
        "letters": cfg.TARGET_LETTERS,
        "flicker_duration": cfg.FLICKER_DURATION,
        "marker_scheme": "block*10 + target + 1; live trials = %d" % cfg.LIVE_MARKER,
    }


def save_session(path: str, data: np.ndarray, meta: dict) -> None:
    """Write raw.npz + session.json; the .npz is written to a temp file then renamed."""
    os.makedirs(path, exist_ok=True)
    tmp = os.path.join(path, "raw.tmp.npz")
    np.savez(tmp, data=data)
    os.replace(tmp, os.path.join(path, "raw.npz"))
    with open(os.path.join(path, "session.json"), "w") as f:
        json.dump({**meta, "n_samples": int(data.shape[1])}, f, indent=2)


def load_session(path: str) -> tuple[np.ndarray, dict]:
    with open(os.path.join(path, "session.json")) as f:
        meta = json.load(f)
    return np.load(os.path.join(path, "raw.npz"))["data"], meta


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


def epoch_at(eeg: np.ndarray, onset: int, rate: int, full: bool = False) -> np.ndarray | None:
    """Filtered trial starting at sample `onset` of `eeg` (channels x samples).

    Returns (samples, channels): the analysis window, or the whole flicker if
    `full`. None if there isn't enough history before or flicker after `onset`.
    """
    hist, flick = history_samples(rate), flicker_samples(rate)
    if onset < hist or onset + flick > eeg.shape[1]:
        return None
    filtered = stream.clean(eeg[:, onset - hist:onset + flick], rate)[:, hist:]
    return (filtered if full else filtered[:, analysis_slice(rate)]).T


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
    skipped: int = 0       # markers without enough history / flicker data

    @property
    def n_targets(self) -> int:
        return len(self.freqs)


def load_trials(path: str, full: bool = False) -> Trials:
    """Every labelled calibration trial in a session, cut with epoch_at().

    Targets come from the session's own session.json, so a session recorded with
    a different frequency set still gets labelled correctly.
    """
    data, meta = load_session(path)
    rate, eeg = meta["rate"], data[meta["eeg_rows"]]
    freqs, letters = meta["frequencies"], meta["letters"]
    markers = data[meta["marker_row"]]
    epochs, targets, blocks, skipped = [], [], [], 0
    for onset in np.flatnonzero(markers):
        decoded = decode_marker(markers[onset], len(freqs))
        if decoded is None:
            continue
        epoch = epoch_at(eeg, onset, rate, full)
        if epoch is None:
            skipped += 1
            continue
        epochs.append(epoch)
        blocks.append(decoded[0])
        targets.append(decoded[1])
    stacked = np.stack(epochs, axis=2) if epochs else np.empty((0, len(meta["eeg_rows"]), 0))
    return Trials(stacked, np.array(targets), np.array(blocks), rate, meta["names"], freqs, letters,
                  list(meta["eeg_rows"]), skipped)
