"""
Central configuration for the NeuroPawn SSVEP-TRCA pipeline.

Everything that you might want to tweak (serial port, electrode count,
stimulus frequencies, timing, filter settings, key bindings, file paths)
lives here so that the rest of the code never hard-codes "magic numbers".

Import it everywhere as:

    from ssvep import config as cfg
"""

from __future__ import annotations

import os
import sys

from brainflow.board_shim import BoardIds

# --------------------------------------------------------------------------- #
# Hardware
# --------------------------------------------------------------------------- #
# Which Knight board you own. BrainFlow parses the serial packets according to
# the board id, so the wrong one gives you garbage instead of EEG:
#   "imu"   -> NEUROPAWN_KNIGHT_BOARD_IMU (22 rows, the motion-sensor variant)
#   "plain" -> NEUROPAWN_KNIGHT_BOARD     (13 rows, no motion sensor)
BOARD_VARIANT = os.environ.get("SSVEP_BOARD_VARIANT", "imu")

# Serial port the Knight board enumerates on.
#   Windows -> "COM3", "COM4", ...
#   Linux   -> "/dev/ttyUSB0", "/dev/ttyACM0", ...
#   macOS   -> "/dev/cu.usbserial-XXXX"
SERIAL_PORT = os.environ.get("SSVEP_SERIAL_PORT") or (
    "COM3" if sys.platform == "win32"
    else "/dev/cu.usbserial-0001" if sys.platform == "darwin"
    else "/dev/ttyUSB0"
)

_BOARD_IDS = {
    "imu": BoardIds.NEUROPAWN_KNIGHT_BOARD_IMU.value,
    "plain": BoardIds.NEUROPAWN_KNIGHT_BOARD.value,
    "synthetic": BoardIds.SYNTHETIC_BOARD.value,
}


def board_id(variant: str | None = None) -> int:
    """BrainFlow board id for `variant` (defaults to BOARD_VARIANT)."""
    variant = variant or BOARD_VARIANT
    try:
        return _BOARD_IDS[variant]
    except KeyError:
        raise ValueError(
            f"unknown board variant {variant!r}; expected one of "
            f"{sorted(_BOARD_IDS)}"
        ) from None


# Number of EEG channels you have wired up on the board (1-8).
NUM_CHANNELS = 8

# Nominal rate of the Knight board. Live code should prefer `board.sr` (the
# synthetic board, for instance, runs at 250 Hz).
SAMPLING_RATE = 125  # Hz

# --------------------------------------------------------------------------- #
# Display
# --------------------------------------------------------------------------- #
# Monitor index PsychoPy draws the flicker on (0 = primary, 1 = secondary).
STIMULUS_SCREEN = 0
FULLSCREEN = True
WINDOW_SIZE = [1920, 1080]      # only used when FULLSCREEN is False
# The flicker frequencies assume this refresh rate. Use a 60 Hz monitor.
EXPECTED_REFRESH_HZ = 60

# Per-channel gain used in the `chon_{ch}_{gain}` command (see board.py).
CHANNEL_GAIN = 12

# --------------------------------------------------------------------------- #
# Electrode montage (10-20 system)
# --------------------------------------------------------------------------- #
# SSVEP is generated in the primary visual cortex, so every electrode sits
# over the occipital / parieto-occipital scalp. The order below maps
# board channel 1..8 -> physical electrode. Reference/ground go on the
# earlobes or mastoids (A1/A2).
ELECTRODE_LABELS = ["Oz", "O1", "O2", "PO7", "PO8", "PO3", "PO4", "POz"]

# --------------------------------------------------------------------------- #
# Stimulus frequencies
# --------------------------------------------------------------------------- #
# One flicker frequency per on-screen target. These values were chosen so
# that a 60 Hz monitor can render them with an (almost) integer number of
# frames per cycle, which keeps the real flicker stable:
#   6.67 Hz ~ 9 frames, 8.57 Hz ~ 7 frames, 10 Hz = 6 frames, 12 Hz = 5 frames.
STIMULUS_FREQUENCIES = [6.67, 8.57, 10.00, 12.00]  # Hz
N_TARGETS = len(STIMULUS_FREQUENCIES)

# --------------------------------------------------------------------------- #
# Trial timing (seconds)
# --------------------------------------------------------------------------- #
# A single trial runs: [cue] -> [flicker] -> [capture].
CUE_DURATION = 1.0        # how long the "look here" cue is shown
FLICKER_DURATION = 1.5    # how long the targets actually flash
INTER_TRIAL_INTERVAL = 0.5  # blank rest between trials (gaze-shift time)

# TRCA analysis window (must match how the data was recorded):
GAZE_DURATION = 1.0       # length of EEG used for classification [s]
VISUAL_LATENCY = 0.15     # delay before the cortex "locks on" [s]

# Number of samples grabbed from the ring buffer after each flicker period.
# 1.5 s * 125 Hz = 187.5 -> round() gives 188, +1 margin = 189 samples, which
# covers the whole flicker window. Like SAMPLING_RATE this describes the
# Knight board; live code should size windows from `board.sr` instead.
CAPTURE_SAMPLES = int(round(FLICKER_DURATION * SAMPLING_RATE)) + 1  # 189

# --------------------------------------------------------------------------- #
# Pre-processing filters (BrainFlow)
# --------------------------------------------------------------------------- #
BANDPASS_LOW = 3.0        # Hz  - remove drift / DC
BANDPASS_HIGH = 48.0      # Hz  - keep SSVEP fundamentals + first harmonics
BANDPASS_ORDER = 2

# Mains-noise notches. Keep both so the same code works in 50 Hz and 60 Hz
# countries; the filter that does not match your grid is simply harmless.
NOTCH_BANDS = [(49.0, 51.0), (59.0, 61.0)]
NOTCH_ORDER = 4

# --------------------------------------------------------------------------- #
# TRCA filter bank
# --------------------------------------------------------------------------- #
# meegkit expects [[(passband), (stopband)], ...]. Each sub-band isolates a
# harmonic range so the ensemble classifier can exploit SSVEP harmonics.
FILTERBANK = [
    [(6, 48), (4, 60)],
    [(14, 48), (10, 60)],
    [(22, 48), (16, 60)],
    [(30, 48), (24, 60)],
    [(38, 48), (32, 60)],
    [(46, 48), (40, 60)],
]
USE_ENSEMBLE_TRCA = True   # ensemble TRCA is more accurate than plain TRCA

# --------------------------------------------------------------------------- #
# Key bindings for real-time control
# --------------------------------------------------------------------------- #
# Maps the predicted square (0..N_TARGETS-1) to a key that gets typed into
# whatever window has focus. Point it at a text box and looking at a square
# types its letter -- that is the whole speller.
#   index 0 (6.67 Hz) -> 'a'   top-left
#   index 1 (8.57 Hz) -> 'b'   top-right
#   index 2 (10.0 Hz) -> 'c'   bottom-left
#   index 3 (12.0 Hz) -> 'd'   bottom-right
# Swap these for w/s/a/d if you'd rather drive a game than spell.
KEY_MAP = {0: "a", 1: "b", 2: "c", 3: "d"}

# --------------------------------------------------------------------------- #
# File paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Where training blocks are saved / loaded from (block_{block}_{trial}.csv).
TRAINING_DATA_DIR = os.path.join(PROJECT_ROOT, "training_data")
