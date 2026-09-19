"""The filter bank, in one place so calibration and live typing can't diverge.

Every band is a 4th-order Butterworth as second-order sections, run with
sosfilt and its state kept between calls. That makes it causal - it only ever
sees the past - and it means feeding the same stream in one lump or in 30-sample
chunks gives bit-identical output.

Which is the point: the SSVEP branch already lost accuracy once to a training /
live mismatch. Here BOTH paths use FilterBank from the first recorded sample
onwards, so a calibration trial and a live selection are filtered the same way.
Nothing zero-phase (filtfilt) may touch this data: it would shift the slow MRCP
features that training then leans on.

The cost of causal filtering from a zero state is a startup transient, so the
first cfg.PRIME_SECONDS of every recording are unusable - see primed_samples().
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfilt

from . import config as cfg


def design(low: float, high: float, rate: float, order: int = cfg.FILTER_ORDER) -> np.ndarray:
    """Bandpass SOS. 0.05 Hz at 125 Hz is a very low normalised cutoff, which is
    why this stays 4th order and gets primed rather than pushed higher."""
    nyquist = 0.5 * rate
    if high >= nyquist:
        raise ValueError(f"band {low}-{high} Hz needs a rate above {2 * high} Hz, got {rate}")
    return butter(order, [max(low, 0.001) / nyquist, high / nyquist], btype="bandpass", output="sos")


def primed_samples(rate: float) -> int:
    """Samples at the start of a recording whose filter output is still settling."""
    return int(round(cfg.PRIME_SECONDS * rate))


class FilterBank:
    """Every band of `channels` channels at once, keeping filter state.

        bank = FilterBank(rate, n_channels)
        out = bank.process(chunk)        # {band: (channels, samples)}

    Chunks must arrive in order and each sample may be passed exactly once.
    """

    def __init__(self, rate: float, channels: int, bands: dict | None = None,
                 order: int = cfg.FILTER_ORDER):
        self.rate = float(rate)
        self.channels = channels
        self.bands = bands or cfg.BANDS
        self.sos = {name: design(lo, hi, rate, order) for name, (lo, hi) in self.bands.items()}
        # zi for sosfilt along the last axis: (sections, channels, 2). Zeros, not
        # sosfilt_zi's step-response state, so priming is the only assumption.
        self.zi = {name: np.zeros((sos.shape[0], channels, 2)) for name, sos in self.sos.items()}
        self.total = 0                   # samples pushed through so far

    def process(self, chunk: np.ndarray) -> dict[str, np.ndarray]:
        """Filter (channels, samples) through every band, advancing the state."""
        if chunk.shape[0] != self.channels:
            raise ValueError(f"expected {self.channels} channels, got {chunk.shape[0]}")
        out = {}
        for name, sos in self.sos.items():
            out[name], self.zi[name] = sosfilt(sos, chunk, axis=-1, zi=self.zi[name])
        self.total += chunk.shape[1]
        return out


def filter_recording(eeg: np.ndarray, rate: float) -> dict[str, np.ndarray]:
    """Run a whole recording through a fresh bank, exactly as the live path did.

    eeg: (channels, samples) -> {band: (channels, samples)}
    """
    return FilterBank(rate, eeg.shape[0]).process(eeg)
