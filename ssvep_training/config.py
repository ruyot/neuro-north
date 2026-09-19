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

# Calibration markers are block * 10 + target + 1 (32 = block 3, target B).
# Target N_TARGETS is "rest": look at the centre cross while every square flickers.
LIVE_MARKER = 99        # live typing trials, no label

# Idle detection: rest trials set a threshold on the TRCA score of the winning
# square. Live picks scoring under it type nothing ("neither"), so a drifting
# gaze or a look at the text doesn't become a wrong letter.
REST_TRIALS_PER_BLOCK = 1

# TRCA filter bank (meegkit format), capped to stream.clean()'s 1-40 Hz band.
FILTERBANK = [
    [(5, 40), (3, 44)],      # widest band: every target's fundamental
    [(14, 40), (10, 44)],
    [(22, 40), (16, 44)],
    [(30, 40), (24, 44)],
]
USE_ENSEMBLE_TRCA = True

# Each run writes training_data/session_<time>/ with raw.npz + session.json.
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
TRAINING_DATA_DIR = os.path.join(PROJECT_ROOT, "training_data")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
