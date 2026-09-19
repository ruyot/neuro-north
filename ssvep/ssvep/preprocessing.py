"""
EEG pre-processing shared by the recorder and the real-time classifier.

The exact same filter chain must be applied when *collecting* training data
and when *classifying* live data -- otherwise the spatial filters learned by
TRCA no longer match the incoming signal. Keeping it in one function
guarantees both paths stay identical.

Filter chain (per channel, in place):
  1. Detrend (remove the DC offset / constant baseline).
  2. Band-pass 3-48 Hz : keep the SSVEP fundamentals and their first
     harmonics, drop slow drift and high-frequency muscle noise.
  3. Band-stop 49-51 Hz and 59-61 Hz : notch out mains hum (works for both
     50 Hz and 60 Hz power grids).

All filters are zero-phase (forward-backward) so they do not smear the
timing of the SSVEP response.
"""

from __future__ import annotations

import numpy as np
from brainflow.data_filter import DataFilter, FilterTypes, DetrendOperations

from . import config as cfg


def filter_eeg(data: np.ndarray, channels, sampling_rate: int) -> np.ndarray:
    """
    Apply the detrend + band-pass + notch chain to each EEG channel in place.

    Parameters
    ----------
    data : np.ndarray
        BrainFlow data matrix, shape (n_rows, n_samples). Only the rows listed
        in `channels` are modified.
    channels : list[int]
        Row indices of the EEG channels within `data`.
    sampling_rate : int
        Sampling rate in Hz.

    Returns
    -------
    np.ndarray
        The same array (filtered in place), returned for convenience.
    """
    zero_phase = FilterTypes.BUTTERWORTH_ZERO_PHASE

    for ch in channels:
        DataFilter.detrend(data[ch], DetrendOperations.CONSTANT.value)
        DataFilter.perform_bandpass(
            data[ch], sampling_rate,
            cfg.BANDPASS_LOW, cfg.BANDPASS_HIGH, cfg.BANDPASS_ORDER,
            zero_phase, 0,
        )
        for low, high in cfg.NOTCH_BANDS:
            DataFilter.perform_bandstop(
                data[ch], sampling_rate, low, high, cfg.NOTCH_ORDER,
                zero_phase, 0,
            )
    return data


def extract_channel_matrix(data: np.ndarray, channels) -> np.ndarray:
    """
    Turn a filtered BrainFlow matrix into a clean (samples, channels) array.

    Rounds to 7 decimals and squashes negative-zero to plain zero so the saved
    CSVs are tidy and reproducible.
    """
    rows = [np.round(data[ch], decimals=7) for ch in channels]
    matrix = np.array(rows)                      # (channels, samples)
    matrix[matrix == 0] = 0.0                    # kill -0.0
    return matrix.T                              # (samples, channels)


def crop_indices(sampling_rate: int,
                 delay: float = cfg.VISUAL_LATENCY,
                 duration: float = cfg.GAZE_DURATION) -> np.ndarray:
    """
    Sample indices of the analysis window used by TRCA.

    We skip the first `delay` seconds (visual latency: the cortex does not
    entrain instantly) and then keep `duration` seconds of data. At 125 Hz
    that is samples [19 : 144] -> a clean 1.0 s window.

    `sampling_rate` is deliberately required, not defaulted to
    `cfg.SAMPLING_RATE`: callers must pass the rate the data was actually
    recorded at (`board.sr` live, derived from file length on disk).
    Defaulting it is how you end up cropping 250 Hz data as if it were
    125 Hz and silently analysing the wrong half-second.
    """
    delay_s = int(np.floor(delay * sampling_rate + 0.5))
    gaze_s = int(np.floor(duration * sampling_rate + 0.5))
    return np.arange(delay_s, delay_s + gaze_s)
