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
    return TRCA(rate, cfg.FILTERBANK, cfg.USE_ENSEMBLE_TRCA)


def _require_training_trials(trials: Trials) -> None:
    counts = [np.count_nonzero(trials.targets == t) for t in range(trials.n_targets)]
    deficient = [f"{trials.letters[t]} ({count})" for t, count in enumerate(counts) if count < 2]
    if deficient:
        raise ValueError("Need at least 2 accepted trials per target; deficient: " + ", ".join(deficient))


def fit(trials: Trials) -> TRCA:
    """Train on all admitted trials, with at least two per declared target."""
    _require_training_trials(trials)
    model = build_model(trials.rate)
    model.fit(trials.eeg, trials.targets)
    return model


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
    if len(blocks) < 3:
        raise ValueError(f"Need at least 3 blocks to cross-validate, found {len(blocks)}")
    classify = classify or _trca_classify
    n = trials.n_targets
    confusion = np.zeros((n, n), dtype=int)
    accs, itrs = [], []
    for held_out in blocks:
        test = trials.blocks == held_out
        train = Trials(trials.eeg[..., ~test], trials.targets[~test], trials.blocks[~test],
                       trials.rate, trials.names, trials.freqs, trials.letters, trials.rows)
        try:
            _require_training_trials(train)
        except ValueError as exc:
            raise ValueError(f"Cannot hold out block {held_out}: {exc}") from exc
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
