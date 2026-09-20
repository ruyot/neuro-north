"""Training-free FBCCA over admitted EEG windows; scores are not probabilities."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from meegkit.cca import nt_cca
from meegkit.utils.trca import bandpass

from . import config as cfg
from .knight import stream


def _combine_scores(correlations: np.ndarray) -> np.ndarray:
    """Combine (banks, trials, targets) only after every bank has contributed."""
    weights = np.arange(1, correlations.shape[0] + 1, dtype=float) ** -1.25 + 0.25
    return np.sum(weights[:, None, None] * correlations ** 2, axis=0)


class FBCCA:
    def __init__(self, rate: int, frequencies: Sequence[float], *, car: bool = False) -> None:
        if isinstance(rate, bool) or not isinstance(rate, (int, np.integer)) or rate <= 0:
            raise ValueError("rate must be a positive integer")
        frequencies = np.asarray(frequencies)
        if frequencies.ndim != 1 or not frequencies.size or frequencies.dtype.kind not in "iuf":
            raise ValueError("frequencies must be a nonempty sequence of real numbers")
        if (not np.isfinite(frequencies).all() or np.any(frequencies <= 0)
                or np.unique(frequencies).size != frequencies.size):
            raise ValueError("frequencies must be distinct, finite and positive")
        if np.any(frequencies >= rate / 4):
            raise ValueError("both reference harmonics must be below Nyquist")

        self.rate = int(rate)
        self.frequencies = tuple(float(frequency) for frequency in frequencies)
        self.car = car
        self._samples = round(self.rate * cfg.GAZE_DURATION)
        self._filterbank = tuple((tuple(wp), tuple(ws)) for wp, ws in cfg.FILTERBANK)
        self._references = []
        for frequency in self.frequencies:
            reference = stream.cca_reference(frequency, self.rate, self._samples, harmonics=2).T
            reference -= reference.mean(axis=0, keepdims=True)
            self._references.append(reference)

    def scores(self, eeg: np.ndarray) -> np.ndarray:
        """Return (trials, targets) scores, with zero rows for unusable epochs."""
        eeg = np.asarray(eeg)
        if (eeg.ndim != 3 or eeg.shape[0] != self._samples
                or eeg.shape[1] == 0 or eeg.shape[2] == 0):
            raise ValueError(f"eeg must have shape ({self._samples}, channels, trials), with nonempty channels/trials")
        if eeg.dtype.kind not in "iuf":
            raise ValueError("eeg must contain finite real numeric data")
        if self.car and eeg.shape[1] < 2:
            raise ValueError("CAR requires at least two selected EEG channels")
        eeg = np.asarray(eeg, dtype=float)
        if not np.isfinite(eeg).all():
            raise ValueError("eeg must contain finite real numeric data")

        # Test the original samples: filtering a constant can create IIR roundoff.
        usable = np.any(eeg.max(axis=0) != eeg.min(axis=0), axis=0)
        if self.car:
            original_norm = np.linalg.norm(eeg - eeg.mean(axis=0, keepdims=True), axis=(0, 1))
            eeg = eeg - eeg.mean(axis=1, keepdims=True)
            residual_norm = np.linalg.norm(eeg - eeg.mean(axis=0, keepdims=True), axis=(0, 1))
            tolerance = np.finfo(eeg.dtype).eps * max(eeg.shape[:2]) * original_norm
            usable &= residual_norm > tolerance

        correlations = np.zeros((len(self._filterbank), eeg.shape[2], len(self._references)))
        trial_indices = np.flatnonzero(usable)
        if trial_indices.size:
            selected = eeg if usable.all() else eeg[:, :, usable]
            for bank, (wp, ws) in enumerate(self._filterbank):
                filtered = bandpass(selected, self.rate, wp, ws)
                # nt_cca deliberately does not remove means from either input.
                filtered -= filtered.mean(axis=0, keepdims=True)
                for index, trial in enumerate(trial_indices):
                    epoch = filtered[:, :, index]
                    if np.linalg.norm(epoch) == 0:
                        continue
                    for target, reference in enumerate(self._references):
                        _, _, rho = nt_cca(epoch, reference)
                        if not np.isfinite(rho).all():
                            raise FloatingPointError("FBCCA produced non-finite canonical correlations")
                        correlations[bank, trial, target] = np.clip(np.max(rho), 0, 1)
        return _combine_scores(correlations)

    def predict(self, eeg: np.ndarray) -> np.ndarray:
        """Return zero-based target labels, or NaN when no bank has usable energy."""
        scores = self.scores(eeg)
        labels = scores.argmax(axis=1).astype(float)
        labels[np.all(scores == 0, axis=1)] = np.nan
        return labels


def self_check() -> None:
    """Exercise the actual decoder with deterministic numerical inputs only."""
    def requires_error(call):
        try:
            call()
        except ValueError:
            return
        raise AssertionError("Expected ValueError for malformed FBCCA input")

    rng = np.random.default_rng(0)
    rate = 125
    frequencies = cfg.STIMULUS_FREQUENCIES
    t = np.arange(125) / 125
    channel = np.arange(8)
    gain = 1 + 0.1 * channel
    offset = 0.1 * channel
    labels = np.tile([0, 1], 20)
    eeg = np.empty((125, 8, labels.size))
    for trial, label in enumerate(labels):
        phase = rng.uniform(0, 2 * np.pi)
        noise = rng.normal(0, 1, (125, 8))
        frequency = frequencies[label]
        eeg[:, :, trial] = (
            5 * gain * np.sin(2 * np.pi * frequency * t[:, None] + phase + offset)
            + 2 * gain * np.sin(4 * np.pi * frequency * t[:, None] + 2 * phase + offset / 2)
            + noise
        )
    model = FBCCA(rate, frequencies)
    before = eeg.tobytes()
    correct = np.count_nonzero(model.predict(eeg) == labels)
    assert correct >= 38, f"Only {correct}/40 separable synthetic trials classified correctly"
    assert eeg.tobytes() == before, "FBCCA mutated the caller's input"

    signal = np.sin(2 * np.pi * frequencies[0] * t + 0.37)
    duplicated = np.repeat(signal[:, None, None], 8, axis=1)
    before = duplicated.tobytes()
    assert np.array_equal(model.predict(duplicated), [0]), "Rank-deficient EEG failed"
    assert np.array_equal(model.predict(duplicated[:, :1]), [0]), "Single-channel EEG failed"
    car_model = FBCCA(rate, frequencies, car=True)
    assert np.all(car_model.scores(duplicated) == 0), "CAR amplified identical-channel roundoff"
    assert np.isnan(car_model.predict(duplicated)).all(), "CAR fabricated a target"
    assert duplicated.tobytes() == before, "CAR mutated the caller's input"

    constant = np.zeros((125, 8, 2))
    constant[:, :, 1] = 2.7 + channel
    assert np.all(model.scores(constant) == 0), "Constant EEG acquired spurious filter energy"
    assert np.isnan(model.predict(constant)).all(), "Constant EEG fabricated a target"
    # Mixed batches must keep their original trial positions when unusable rows are skipped.
    mixed = np.concatenate((constant[:, :, :1], duplicated, constant[:, :, 1:]), axis=2)
    mixed_labels = model.predict(mixed)
    assert np.isnan(mixed_labels[[0, 2]]).all() and mixed_labels[1] == 0

    correlations = np.array([[[0.60, 0.59]], [[0.60, 0.59]], [[0, 1]]])
    combined = _combine_scores(correlations)
    assert combined.shape == (1, 2)
    assert np.allclose(combined[0], [0.6913613547, 1.1717865830], rtol=1e-9, atol=1e-10)
    assert combined.argmax(axis=1).item() == 1, "Selected a partial-sum vote instead of the final score"

    for invalid in (eeg.transpose(1, 0, 2), eeg[:, :, 0], eeg[:-1, :, :1],
                    eeg[:, :, :0], eeg[:, :0, :1],
                    np.full((125, 8, 1), np.nan), np.full((125, 8, 1), np.inf),
                    eeg[:, :, :1].astype(complex), np.ones((125, 1, 1), dtype=bool),
                    np.full((125, 1, 1), "1")):
        requires_error(lambda: model.predict(invalid))
    requires_error(lambda: car_model.predict(duplicated[:, :1]))
    for invalid_rate in (0, 125.0, True):
        requires_error(lambda: FBCCA(invalid_rate, frequencies))
    for invalid_frequencies in ([], [frequencies[0]] * 2, [0], [np.nan], [np.inf],
                                [rate / 4], [complex(frequencies[0])], [str(frequencies[0])]):
        requires_error(lambda: FBCCA(rate, invalid_frequencies))

    print(f"FBCCA self-check passed: {correct}/40 separable synthetic trials.")
    print("Numerical evidence only; human accuracy and idle/no-control detection remain unvalidated.")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-check", action="store_true", help="run deterministic numerical decoder checks")
    args = parser.parse_args()
    if not args.self_check:
        parser.error("pass --self-check")
    self_check()


if __name__ == "__main__":
    main()
