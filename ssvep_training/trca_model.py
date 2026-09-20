"""
TRCA classifier

Task-Related Component Analysis learns, per target, a spatial filter (a weighted
mix of the 8 electrodes) that makes the response to that flicker as repeatable
as possible across calibration trials.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from meegkit.trca import TRCA
from meegkit.utils.trca import itr, normfit

from . import config as cfg
from .session import Trials


def build_model(rate: int) -> TRCA:
    # Start with one broad band, without tuning on the test set. meegkit 0.2
    # cascades training bands but independently filters prediction bands; using
    # one band also avoids that train/test mismatch in its multiband path.
    return TRCA(rate, [cfg.FILTERBANK[0]], cfg.USE_ENSEMBLE_TRCA)


def fit(trials: Trials) -> TRCA:
    """Train on every trial in the session (what live typing uses)."""
    missing = sorted(set(range(trials.n_targets)) - set(trials.targets.tolist()))
    if missing:
        raise ValueError("no calibration trials for " + ", ".join(trials.letters[t] for t in missing))
    if min(np.bincount(trials.targets, minlength=trials.n_targets)) < 2:
        raise ValueError('TRCA needs at least two usable training trials per target')
    if not np.isfinite(trials.eeg).all():
        raise ValueError('TRCA calibration contains nonfinite samples')
    model = build_model(trials.rate)
    model.fit(trials.eeg, trials.targets)
    return model


def load_calibration(path, channels=None):
    """Only compare/fit fresh calibration with the exact active signal settings."""
    from .session import load_session, load_trials
    _, meta = load_session(path)
    expected = {'gaze_duration': cfg.GAZE_DURATION, 'visual_latency': cfg.VISUAL_LATENCY,
                'filter_history': cfg.FILTER_HISTORY, 'filter_band': list(cfg.FILTER_BAND)}
    if (meta.get('frequencies') != list(cfg.STIMULUS_FREQUENCIES)
            or meta.get('experimental_layout', 'current') != 'current'
            or meta.get('stimulus_method') != cfg.stimulus_method()
            or meta.get('flicker_duration') != cfg.FLICKER_DURATION
            or any(meta.get('processing', {}).get(k) != v for k, v in expected.items())):
        raise ValueError('Calibration timing/frequencies/filtering differ; collect a fresh calibration')
    return load_trials(path, reject_bad=True, channels=channels)


def scores(model, epoch):
    """Expose the same template correlations used by meegkit.predict.

    These are scores, not probabilities. Retain them on center trials so an
    idle-rejection rule can be fitted and independently tested later.
    """
    from meegkit.utils.trca import bandpass
    result = np.zeros(len(model.classes))
    for band, (passing, stopping) in enumerate(model.filterbank):
        filtered = bandpass(epoch, model.sfreq, Wp=passing, Ws=stopping)
        for target in model.classes:
            weights = model.coef_[band].T if model.ensemble else model.coef_[band, target]
            x = (filtered @ weights).ravel()
            y = (model.trains[target, band] @ weights).ravel()
            correlation = np.corrcoef(x, y)[0,1] if x.std() > 0 and y.std() > 0 else 0.
            result[target] += ((band+1)**(-1.25)+.25)*correlation
    return result


def selection_seconds() -> float:
    """What one selection costs: the whole flicker plus the rest before the next."""
    return cfg.FLICKER_DURATION + cfg.INTER_TRIAL_INTERVAL


def bits_per_min(accuracy: float, n_targets: int) -> float:
    # meegkit's itr() raises at or below chance; that's 0 bits/min.
    return itr(n_targets, accuracy, selection_seconds()) if accuracy > 1 / n_targets else 0.0


@dataclass
class CVResult:
    blocks: list[int]
    accuracy: np.ndarray       # per held-out block, %
    itr: np.ndarray            # per held-out block, bits/min
    confusion: np.ndarray      # [true target, predicted target] counts over all held-out trials
    freqs: list[float]
    letters: list[str]

    @property
    def n_targets(self) -> int:
        return len(self.freqs)

    @property
    def mean_accuracy(self) -> float:
        return float(self.accuracy.mean())


def cross_validate(trials: Trials, classify=None) -> CVResult:
    """Leave-one-block-out: each block is predicted by a model trained on the others.

    `classify(train_trials, test_eeg) -> predicted targets` defaults to TRCA; other
    detectors (e.g. SNR, which ignores the training data) plug in the same way.
    """
    blocks = sorted(set(trials.blocks.tolist()))
    if len(blocks) < 2:
        raise ValueError(f"need at least 2 blocks to cross-validate, found {len(blocks)}")
    classify = classify or _trca_classify
    n = trials.n_targets
    confusion = np.zeros((n, n), dtype=int)
    accs, itrs = [], []
    for held_out in blocks:
        test = trials.blocks == held_out
        train = Trials(trials.eeg[..., ~test], trials.targets[~test], trials.blocks[~test],
                       trials.rate, trials.names, trials.freqs, trials.letters, trials.rows)
        predicted = np.asarray(classify(train, trials.eeg[..., test]))
        truth = trials.targets[test]
        for t, p in zip(truth, predicted):
            confusion[t, p] += 1
        acc = float(np.mean(predicted == truth))
        accs.append(100 * acc)
        itrs.append(bits_per_min(acc, n))
    return CVResult(blocks, np.array(accs), np.array(itrs), confusion, trials.freqs, trials.letters)


def _trca_classify(train: Trials, test_eeg: np.ndarray) -> np.ndarray:
    return fit(train).predict(test_eeg)


def print_cv(result: CVResult, title: str = "TRCA") -> None:
    letters = result.letters
    print(f"{title} - leave-one-block-out ({len(result.blocks)} blocks):")
    for block, acc, bits in zip(result.blocks, result.accuracy, result.itr):
        print(f"  block {block}: {acc:5.1f}%   {bits:5.1f} bits/min")
    if len(result.blocks) > 1:
        # meegkit's normfit takes the confidence *level* (0.95), not alpha.
        _, _, ci, _ = normfit(result.accuracy, 0.95)
        _, _, ci_itr, _ = normfit(result.itr, 0.95)
        ci_text = f"(95% CI {max(ci[0], 0):.0f}-{min(ci[1], 100):.0f}%)"
        itr_text = f"(95% CI {max(ci_itr[0], 0):.1f}-{ci_itr[1]:.1f})"
    else:
        ci_text = itr_text = ""
    print(f"  mean accuracy {result.mean_accuracy:.1f}% {ci_text}   chance {100 / result.n_targets:.0f}%")
    print(f"  mean speed    {result.itr.mean():.1f} bits/min {itr_text}   "
          f"({selection_seconds():.1f} s per selection)")

    print("\n  looked at -> predicted " + "".join(f"{l:>5}" for l in letters) + "   correct")
    for i, letter in enumerate(letters):
        row = result.confusion[i]
        pct = 100 * row[i] / row.sum() if row.sum() else 0
        print(f"  {letter} ({result.freqs[i]:5.2f} Hz)       " + "".join(f"{n:5d}" for n in row)
              + f"   {pct:5.0f}%")
