"""
EEG pre-processing shared by calibration and live classification.

The exact same chain must run in both, or TRCA's learned spatial filters won't
match the incoming signal. Per channel, in place, all zero-phase:
  1. Detrend (constant)
  2. Band-pass 3-48 Hz
  3. Band-stop 49-51 Hz and 59-61 Hz (mains hum, 50 or 60 Hz grids)
"""

from __future__ import annotations

import numpy as np
from brainflow.data_filter import DataFilter, DetrendOperations, FilterTypes

from . import config as cfg


def filter_eeg(data: np.ndarray, channels, sampling_rate: int) -> np.ndarray:
    """Apply detrend + band-pass + notches to each EEG row of a BrainFlow matrix, in place."""
    zero_phase = FilterTypes.BUTTERWORTH_ZERO_PHASE
    for ch in channels:
        DataFilter.detrend(data[ch], DetrendOperations.CONSTANT.value)
        DataFilter.perform_bandpass(data[ch], sampling_rate, cfg.BANDPASS_LOW, cfg.BANDPASS_HIGH,
                                    cfg.BANDPASS_ORDER, zero_phase, 0)
        for low, high in cfg.NOTCH_BANDS:
            DataFilter.perform_bandstop(data[ch], sampling_rate, low, high, cfg.NOTCH_ORDER,
                                        zero_phase, 0)
    return data


def extract_channel_matrix(data: np.ndarray, channels) -> np.ndarray:
    """Filtered BrainFlow matrix -> tidy (samples, channels) array."""
    matrix = np.array([np.round(data[ch], decimals=7) for ch in channels])
    matrix[matrix == 0] = 0.0  # squash -0.0
    return matrix.T


def crop_indices(sampling_rate: int,
                 delay: float = cfg.VISUAL_LATENCY,
                 duration: float = cfg.GAZE_DURATION) -> np.ndarray:
    """Sample indices of the TRCA analysis window: skip `delay`, keep `duration` ([19:144] at 125 Hz).

    `sampling_rate` is required: it must be the rate the data was recorded at
    (125 Hz Knight, 250 Hz synthetic), never assumed.
    """
    delay_s = int(np.floor(delay * sampling_rate + 0.5))
    gaze_s = int(np.floor(duration * sampling_rate + 0.5))
    return np.arange(delay_s, delay_s + gaze_s)
