"""Montage, trial timing, filter bands and classes for motor imagery.

Board handling still comes from ssvep_training/board.py (Knight board 66 at
125 Hz); only the electrode positions and the decoder differ.
"""

from __future__ import annotations

import os

# The electrodes are PHYSICALLY SOMEWHERE ELSE than for SSVEP: over the
# sensorimotor strip, not the occiput. Board channels 1-8 in order, per
# NeuroPawn's motor-imagery tutorial (and commit 48cd96a):
CHANNEL_NAMES = ["FC4", "C4", "CP4", "C2", "C1", "CP3", "C3", "FC3"]
# stream.py's NAMES are the SSVEP montage and stay that way, so every script
# here prints CHANNEL_NAMES at startup: if the cap says otherwise, stop.
RIGHT_HEMISPHERE = ["FC4", "C4", "CP4", "C2"]   # left-hand imagery shows up here
LEFT_HEMISPHERE = ["C1", "CP3", "C3", "FC3"]    # right-hand imagery shows up here

# Class ids follow the tutorial's, so a ported model keeps its meaning.
# "both" (3) is recorded only with --both; the speller maps it to space.
LEFT, RIGHT, REST, BOTH = 0, 1, 2, 3
CLASS_NAMES = {LEFT: "left", RIGHT: "right", REST: "rest", BOTH: "both"}
CLASS_SETS = {                      # --classes
    "lrr": [LEFT, RIGHT, REST],     # default: left / right / neither
    "lr": [LEFT, RIGHT],            # binary, for comparison
    "lrrb": [LEFT, RIGHT, REST, BOTH],
}
DEFAULT_CLASSES = "lrr"

# The two boxes the speller picks between, in class order.
BOX_CLASSES = [LEFT, RIGHT]

# Trial timing (seconds): [cue] -> [hold, marker at its first frame] -> [break].
# One selection costs CUE + HOLD + FEEDBACK ~ 4 s against SSVEP's 2.3 s, which
# is the price of not having a flicker to lock onto.
CUE_DURATION = 1.0
HOLD_DURATION = 2.0
BREAK_DURATION = (0.8, 1.4)     # randomised during calibration so cues stay unpredictable

# The slice of the hold that gets classified, relative to the marker. The first
# half second is reaction time and the ERD still building.
ANALYSIS_START = 0.5
ANALYSIS_DURATION = 1.5

# The filter bank is causal and stateful, and both calibration and live
# filtering run it from the first recorded sample - so a trial is only usable
# once the slowest band has settled. 0.05 Hz has a ~3 s time constant per pole;
# four poles plus margin is why this is 20 s and not 4 like FILTER_HISTORY.
PRIME_SECONDS = 20.0

BANDS = {
    "0.05-5": (0.05, 5.0),      # slow movement-related potentials (MRCP branch)
    "8-30": (8.0, 30.0),        # mu + beta: the band the default model uses
    "8-12": (8.0, 12.0),        # the five FBCSP / filter-bank sub-bands
    "12-16": (12.0, 16.0),
    "16-20": (16.0, 20.0),
    "20-26": (20.0, 26.0),
    "26-30": (26.0, 30.0),
}
FILTER_ORDER = 4
BROADBAND = "8-30"
SUB_BANDS = ["8-12", "12-16", "16-20", "20-26", "26-30"]
MRCP_BAND = "0.05-5"

# Markers, same scheme as the SSVEP sessions: block * 10 + class + 1.
LIVE_MARKER = 99

# Recording mode. Physical clenches score far higher than imagery and the
# difference is largely muscle, so the two never share a training set.
MODES = ["clench", "imagine"]

# Decoding. The margin is how far the best hand must beat rest before anything
# is typed; evaluate.py tunes it, since a wrong letter costs more than a retry.
DEFAULT_MODEL = "tangent"
DEFAULT_MARGIN = 0.15
MAX_FALSE_PICK = 0.10           # target: at most 10% of rest trials type something

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
TRAINING_DATA_DIR = os.path.join(PROJECT_ROOT, "training_data")
SESSION_PREFIX = "mi_session_"
MODEL_FILE = "mi_model.joblib"
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")


def classes_for(name: str) -> list[int]:
    return CLASS_SETS[name]


def names_for(classes: list[int]) -> list[str]:
    return [CLASS_NAMES[c] for c in classes]


def montage_banner(rows: list[int] | None = None) -> str:
    """The line every entry point prints before touching the board."""
    pairs = ", ".join(f"{i + 1}={n}" for i, n in enumerate(CHANNEL_NAMES))
    head = f"[montage] motor imagery expects {pairs}"
    if rows is not None and len(rows) != len(CHANNEL_NAMES):
        head += f"\n[montage] WARNING: recording {len(rows)} channels, montage lists {len(CHANNEL_NAMES)}"
    return head + "\n[montage] these are NOT stream.py's SSVEP names - check the cap before recording."
