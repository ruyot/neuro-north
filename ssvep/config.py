"""
Central configuration for the SSVEP-TRCA pipeline.

Design (frequencies, timing, filters, filter bank, montage) follows NeuroPawn's
SSVEP pipeline: https://github.com/NeuroPawn/ssvep. Adapted here for the Knight
**IMU** board on macOS and for A/B/C/D typing.

Import it everywhere as:

    from ssvep import config as cfg
"""

from __future__ import annotations

import glob
import os

from brainflow.board_shim import BoardIds

# --------------------------------------------------------------------------- #
# Hardware
# --------------------------------------------------------------------------- #
# Default serial port; every script also accepts --port.
#   macOS   -> "/dev/cu.usbserial-XXXX"   (find with: ls /dev/cu.*)
#   Windows -> "COM3", ...
SERIAL_PORT = "/dev/cu.usbserial-A5069RR4"

# Knight IMU board (66). The plain Knight (57) uses a different packet format.
BOARD_ID = BoardIds.NEUROPAWN_KNIGHT_BOARD_IMU

NUM_CHANNELS = 8
SAMPLING_RATE = 125  # Hz, fixed by the Knight board
CHANNEL_GAIN = 12    # PGA gain used in `chon_{ch}_{gain}`

# --------------------------------------------------------------------------- #
# Display
# --------------------------------------------------------------------------- #
STIMULUS_SCREEN = 0
FULLSCREEN = True
WINDOW_SIZE = [1280, 800]       # only used when FULLSCREEN is False
# The flicker frequencies assume this refresh rate. Verified: the MacBook Air
# built-in display flips at 60.0 Hz under PsychoPy.
EXPECTED_REFRESH_HZ = 60

# --------------------------------------------------------------------------- #
# Electrode montage (10-20 system), board channel 1..8 -> electrode.
# Reference / ground on the earlobes or mastoids.
# --------------------------------------------------------------------------- #
ELECTRODE_LABELS = ["Oz", "O1", "O2", "PO7", "PO8", "PO3", "PO4", "POz"]

# --------------------------------------------------------------------------- #
# Targets: one flicker frequency + letter per corner square.
# Order = top-left, top-right, bottom-left, bottom-right.
# Near-integer frame counts on 60 Hz: 6.67 ~ 9, 8.57 ~ 7, 10 = 6, 12 = 5 frames.
# --------------------------------------------------------------------------- #
STIMULUS_FREQUENCIES = [6.67, 8.57, 10.00, 12.00]  # Hz
TARGET_LETTERS = ["A", "B", "C", "D"]
N_TARGETS = len(STIMULUS_FREQUENCIES)

# --------------------------------------------------------------------------- #
# Trial timing (seconds): [cue] -> [flicker] -> [capture + rest]
# --------------------------------------------------------------------------- #
CUE_DURATION = 1.0
FLICKER_DURATION = 1.5
INTER_TRIAL_INTERVAL = 0.5

# TRCA analysis window (must match how the data was recorded).
GAZE_DURATION = 1.0       # seconds of EEG used for classification
VISUAL_LATENCY = 0.15     # skip this much at the start (cortex not locked on yet)

# --------------------------------------------------------------------------- #
# Pre-processing filters (BrainFlow)
# --------------------------------------------------------------------------- #
BANDPASS_LOW = 3.0
BANDPASS_HIGH = 48.0
BANDPASS_ORDER = 2
# Both mains notches; the one not matching your grid is harmless.
NOTCH_BANDS = [(49.0, 51.0), (59.0, 61.0)]
NOTCH_ORDER = 4

# --------------------------------------------------------------------------- #
# TRCA filter bank: meegkit format [[(passband), (stopband)], ...].
# Upper edges stay below the 62.5 Hz Nyquist limit of a 125 Hz board.
# --------------------------------------------------------------------------- #
FILTERBANK = [
    [(6, 48), (4, 60)],
    [(14, 48), (10, 60)],
    [(22, 48), (16, 60)],
    [(30, 48), (24, 60)],
    [(38, 48), (32, 60)],
    [(46, 48), (40, 60)],
]
USE_ENSEMBLE_TRCA = True

# --------------------------------------------------------------------------- #
# File paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Each calibration run gets its own folder: training_data/session_YYYYmmdd_HHMMSS/
# holding block_{block}_{trial}.csv files. Electrodes move between sessions, so
# models are trained on one session at a time (the latest by default).
TRAINING_DATA_DIR = os.path.join(PROJECT_ROOT, "training_data")
# --synthetic runs (BrainFlow's fake 250 Hz board) are kept separate so fake
# data can never be picked up as the latest real session.
SYNTHETIC_DATA_DIR = os.path.join(PROJECT_ROOT, "training_data_synthetic")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

# Rates the boards in this project stream at: Knight (125 Hz), synthetic (250 Hz).
BOARD_RATES = (125, 250)


def capture_samples(sampling_rate: int) -> int:
    """Samples grabbed after each flicker at `sampling_rate` (189 at 125 Hz)."""
    return int(round(FLICKER_DURATION * sampling_rate)) + 1


def data_root(synthetic: bool = False) -> str:
    return SYNTHETIC_DATA_DIR if synthetic else TRAINING_DATA_DIR


def latest_session_dir(synthetic: bool = False) -> str | None:
    """Most recent session_* folder with at least one complete block, or None.

    Looks in training_data/ (or training_data_synthetic/ if `synthetic`).
    Aborted runs can leave empty or partial session folders; those are skipped.
    """
    sessions = sorted(glob.glob(os.path.join(data_root(synthetic), "session_*")))
    for session in reversed(sessions):
        if all(os.path.exists(os.path.join(session, f"block_1_{t}.csv")) for t in range(1, N_TARGETS + 1)):
            return session
    return None
