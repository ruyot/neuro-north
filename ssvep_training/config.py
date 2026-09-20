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
