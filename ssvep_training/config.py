"""Fresh two-target baseline. Freeze these settings during a validation run.

These are starting parameters, not settings validated on the old recordings.
"""
from __future__ import annotations
import os

# Frequency-pair experiment: positions remain A left, B right.
STIMULUS_FREQUENCIES = [12.0, 15.0]
TARGET_LETTERS = ["A", "B"]
N_TARGETS = len(STIMULUS_FREQUENCIES)
EXPECTED_REFRESH_HZ = 60
TARGET_X = 0.70
TARGET_SIZE = (0.45, 1.0)

# Live-only visual experiments; these do not replace the calibrated default.
LAYOUT_PRESETS = {
    'current': (0.70, (0.45, 1.0), 8),
    'small': (0.70, (0.30, 0.66), 8),
    'large': (0.64, (0.60, 1.20), 8),
    'coarse': (0.70, (0.45, 1.0), 4),
}

def configure_layout(name):
    global EXPERIMENTAL_LAYOUT, TARGET_X, TARGET_SIZE, MOTION_SPATIAL_CYCLES
    if name not in LAYOUT_PRESETS:
        raise ValueError('Unknown layout preset')
    EXPERIMENTAL_LAYOUT = name
    TARGET_X, TARGET_SIZE, MOTION_SPATIAL_CYCLES = LAYOUT_PRESETS[name]
    os.environ['NEURO_LAYOUT'] = name

configure_layout(os.environ.get('NEURO_LAYOUT', 'current'))

CUE_DURATION = 1.0
FLICKER_DURATION = 2.5
INTER_TRIAL_INTERVAL = 1.5
VISUAL_LATENCY = 0.2
GAZE_DURATION = 2.0
MAX_GAZE_DURATION = GAZE_DURATION
GAZE_STEP = 0.4
FILTER_HISTORY = 4.0
FILTER_BAND = (1.0, 40.0)

# Plain CCA is the baseline. At these frequencies/40 Hz cutoff, both targets
# use two harmonics each: A 12/24 Hz, B 15/30 Hz.
# Bank scoring remains available for later comparison.
CCA_BANDS = 1
DECOY_FREQUENCIES = [11.0, 13.0, 17.5, 23.0, 26.0]
# A heuristic abstention gate, NOT a calibrated probability or significance.
# The new validation task reports accuracy, coverage, and idle false activations.
CONFIDENCE_THRESHOLD = 0.5

LIVE_MARKER = 99
FILTERBANK = [
    [(5, 40), (3, 44)],
    [(14, 40), (10, 44)],
    [(22, 40), (16, 44)],
    [(30, 40), (24, 44)],
]
USE_ENSEMBLE_TRCA = True
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
TRAINING_DATA_DIR = os.path.join(PROJECT_ROOT, "training_data")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

# IMU integration from imu-head-gestures (c0b1616 / 2001ae8).
# Relative gyro thresholds; axis/sign mapping still needs a worn-headset check.
GESTURE_THRESHOLD = 6.0
GESTURE_COOLDOWN = 0.6
GESTURE_MIN_SPEED = 0.08  # firmware stream units, not raw 16-bit counts
GESTURE_DWELL = 0.08     # sustained motion before firing
GESTURE_REARM = 0.25     # quiet motion required before another gesture
GESTURE_PROFILE_PATH = os.path.join(RESULTS_DIR, "imu_profile.json")

# Environment propagates the selected paradigm into multiprocessing workers.
STIMULUS_MODE = os.environ.get("NEURO_STIMULUS", "flicker")
def stimulus_method():
    return "motion_grating_reversal_v1" if STIMULUS_MODE == "motion" else "integer_frame_cycles_v1"

def configure_stimulus(mode):
    global STIMULUS_MODE
    if mode not in ("flicker", "motion"):
        raise ValueError("Unknown stimulus mode")
    STIMULUS_MODE = mode
    os.environ["NEURO_STIMULUS"] = mode


def stimulus_warning():
    description = ("This screen shows rapidly reversing moving patterns." if STIMULUS_MODE == "motion"
                   else "This screen FLASHES at 12-15 Hz.")
    return (description + "\n\nDo not use it if you have epilepsy or have ever had a seizure,"
            "\nand stop immediately if you feel unwell.\n\nPress SPACE to continue, Escape to quit.")
