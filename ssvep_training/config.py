"""Stimulus, trial and classifier settings. Board handling and filtering come
from stream.py via knight.py."""

from __future__ import annotations

import os

# One flicker frequency + letter per square. 2 -> left/right, 4 -> corners.
# 15 and 20 Hz are whole-frame cycles on a 60 Hz screen (4 and 3 frames) and sit
# clear of the 8-13 Hz alpha band, which would otherwise fire on any square.
# 20 Hz relies on its fundamental: its 2nd harmonic (40 Hz) is stream.clean()'s cutoff.
STIMULUS_FREQUENCIES = [15.0, 20.0]  # Hz
TARGET_LETTERS = ["A", "B"]
N_TARGETS = len(STIMULUS_FREQUENCIES)
EXPECTED_REFRESH_HZ = 60

# Two-target layout in normalised units (screen = -1..1 on both axes): square
# centre x-offset, then (width, height). Further apart = less gaze crosstalk
# between targets; smaller = weaker SSVEP, since the response scales with
# stimulus area. Keep centre +/- half-width under ~0.96 or the cue outline clips.
TARGET_X = 0.70
TARGET_SIZE = (0.45, 1.0)

# Trial timing (seconds): [cue] -> [flicker] -> [rest], marker at flicker onset.
CUE_DURATION = 1.0
FLICKER_DURATION = 1.5
INTER_TRIAL_INTERVAL = 0.5

VISUAL_LATENCY = 0.15   # skipped after onset: the cortex hasn't locked on yet
GAZE_DURATION = 1.0     # length of the window that gets classified

# stream.clean() is causal and its 1 Hz high-pass needs time to settle, so every
# trial is filtered with this much EEG before the onset, then cut.
FILTER_HISTORY = 4.0

# Relative artifact admission, fitted only during the explicit quiet baseline.
QUALITY_BASELINE_SECONDS = 10.0
QUALITY_MULTIPLIER = 6.0

# TRCA filter bank (meegkit format), capped to stream.clean()'s 1-40 Hz band.
FILTERBANK = [
    [(5, 40), (3, 44)],      # widest band: every target's fundamental
    [(14, 40), (10, 44)],
    [(22, 40), (16, 44)],
    [(30, 40), (24, 44)],
]
USE_ENSEMBLE_TRCA = True
DECODERS = ("trca", "fbcca", "fbcca-car")

# Each run writes training_data/session_<time>/ with raw.npz + session.json.
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
TRAINING_DATA_DIR = os.path.join(PROJECT_ROOT, "training_data")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

# Head gestures (head.py). The SDK passes firmware float32 gyro values through;
# documented rad/s units have not been verified on this hardware. Thresholds
# remain relative to resting jitter. Below ~4 is twitchy, above ~8 wants a deliberate
# swipe. Check it with `python stream.py --real`: a swipe's `gyro pk` should
# clear the peaks between swipes by more than this factor.
GESTURE_THRESHOLD = 6.0   # multiples of resting gyro noise, not a physical unit
GESTURE_COOLDOWN = 0.6    # s, so one swipe cannot fire on every sample it spans
