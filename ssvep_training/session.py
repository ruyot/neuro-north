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
        "processing": {"visual_latency": cfg.VISUAL_LATENCY, "gaze_duration": cfg.GAZE_DURATION,
                       "filter_history": cfg.FILTER_HISTORY, "filter_band": cfg.FILTER_BAND,
                       "cca_bands": cfg.CCA_BANDS, "confidence_threshold": cfg.CONFIDENCE_THRESHOLD,
                       "decoy_frequencies": list(cfg.DECOY_FREQUENCIES)},
        "stimulus_method": cfg.stimulus_method(),
        "experimental_layout": cfg.EXPERIMENTAL_LAYOUT,
        "target_geometry": {"x": cfg.TARGET_X, "size_norm": list(cfg.TARGET_SIZE)},
        "stimulus_details": ({"frequency_units": "direction_reversals_per_second",
                              "full_cycle_hz": [f / 2 for f in cfg.STIMULUS_FREQUENCIES],
                              "spatial_cycles": cfg.MOTION_SPATIAL_CYCLES, "phase_amplitude_cycles": .25,
                              "contrast": .8, "mask": "raisedCos"}
                             if cfg.STIMULUS_MODE == "motion" else {}),
        "montage_note": "names are configured labels; physical placement must be verified",

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
            meta = json.load(f)
            if (meta.get("n_markers", 0) > 0
                    and meta.get("stimulus_method") == cfg.stimulus_method()):
                return path
    return None

def history_samples(rate: int) -> int:
    return int(round(cfg.FILTER_HISTORY * rate))


def flicker_samples(rate: int) -> int:
    return int(round(cfg.FLICKER_DURATION * rate))


def analysis_slice(rate: int, gaze: float | None = None) -> slice:
    """Samples after the flicker onset that get classified: skip latency, keep `gaze`
    seconds (GAZE_DURATION by default). Live selections grow this until the pick
    is confident, so it is not always the configured value."""
    start = int(round(cfg.VISUAL_LATENCY * rate))
    gaze = cfg.GAZE_DURATION if gaze is None else gaze
    return slice(start, start + int(round(gaze * rate)))


def epoch_at(eeg: np.ndarray, onset: int, rate: int, full: bool = False,
             gaze: float | None = None, flicker_duration: float | None = None) -> np.ndarray | None:
    """Filtered trial starting at sample `onset` of `eeg` (channels x samples).

    Returns (samples, channels): the analysis window, or the whole flicker if
    `full`. None if there isn't enough history before or flicker after `onset`.
    """
    # Only wait for what the returned window needs: the full flicker for a
    # spectrum, but just the analysis window for classification. Live selections
    # would otherwise sit idle for the difference on every letter.
    hist = history_samples(rate)
    window = analysis_slice(rate, gaze)
    need = int(round(flicker_duration * rate)) if full and flicker_duration is not None else (flicker_samples(rate) if full else window.stop)
    if onset < hist or onset + need > eeg.shape[1]:
        return None
    filtered = stream.clean(eeg[:, onset - hist:onset + need], rate)[:, hist:]
    return (filtered if full else filtered[:, window]).T


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


def load_trials(path: str, full: bool = False, reject_bad: bool = False, channels=None) -> Trials:
    """Every labelled calibration trial in a session, cut with epoch_at().

    Targets come from the session's own session.json, so a session recorded with
    a different frequency set still gets labelled correctly.
    """
    data, meta = load_session(path)
    trial_status = None
    if reject_bad:
        status_file = os.path.join(path, 'calibration.json')
        if not os.path.exists(status_file):
            raise ValueError('Calibration has no display-quality log; collect a fresh calibration')
        with open(status_file) as source:
            trial_status = {r['marker']: r for r in json.load(source)['trials']}
    rows = list(meta['eeg_rows']) if channels is None else list(channels)
    if not rows or len(set(rows)) != len(rows) or not set(rows) <= set(meta['eeg_rows']):
        raise ValueError('Requested channels must be a unique nonempty subset of the recorded EEG channels')
    names = [meta['names'][meta['eeg_rows'].index(row)] for row in rows]
    rate, eeg = meta["rate"], data[rows]
    freqs, letters = meta["frequencies"], meta["letters"]
    markers = data[meta["marker_row"]]
    epochs, targets, blocks, skipped = [], [], [], 0
    for onset in np.flatnonzero(markers):
        decoded = decode_marker(markers[onset], len(freqs))
        if decoded is None:
            continue
        if trial_status is not None:
            from .quality import trial_transport_ok
            status = trial_status.get(int(round(markers[onset])))
            if (not status or not status['completed'] or status['late_frames']
                    or not trial_transport_ok(data, onset, rate,
                                              analysis_slice(rate).stop, history_samples(rate))):
                skipped += 1
                continue
        # Never let today's longer baseline include an old recording's dark rest.
        recorded_flicker = meta["flicker_duration"]
        gaze = min(cfg.GAZE_DURATION, recorded_flicker - cfg.VISUAL_LATENCY)
        if gaze <= 0:
            skipped += 1
            continue
        epoch = epoch_at(eeg, onset, rate, full, gaze=gaze, flicker_duration=recorded_flicker)
        if epoch is None:
            skipped += 1
            continue
        if reject_bad and (not np.isfinite(epoch).all() or np.any(np.std(epoch, axis=0) < 1e-9)):
            skipped += 1
            continue
        epochs.append(epoch)
        blocks.append(decoded[0])
        targets.append(decoded[1])
    stacked = np.stack(epochs, axis=2) if epochs else np.empty((0, len(rows), 0))
    return Trials(stacked, np.array(targets, dtype=int), np.array(blocks, dtype=int), rate, names, freqs, letters,
                  rows, skipped)
