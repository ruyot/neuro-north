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
from meegkit.utils.trca import bandpass, itr, normfit

from . import config as cfg
from .session import Trials


def build_model(rate: int) -> TRCA:
    return TRCA(rate, cfg.FILTERBANK, cfg.USE_ENSEMBLE_TRCA)


def fit(trials: Trials) -> TRCA:
    """Train on every trial in the session (what live typing uses)."""
    missing = sorted(set(range(trials.n_targets)) - set(trials.targets.tolist()))
    if missing:
        raise ValueError("no calibration trials for " + ", ".join(trials.letters[t] for t in missing))
    model = build_model(trials.rate)
    model.fit(trials.eeg, trials.targets)
    return model


def scores(model: TRCA, eeg: np.ndarray) -> np.ndarray:
    """(trials, targets) fused TRCA correlation per target - what model.predict()
    takes the argmax of. eeg: (samples, channels, trials)."""
    # Same filterbank weights and filtering as meegkit's predict().
    fb_coefs = np.array([(i + 1) ** -1.25 + 0.25 for i in range(model.n_bands)])
    out = np.zeros((eeg.shape[-1], len(model.classes)))
    for trial in range(eeg.shape[-1]):
        r = np.zeros((model.n_bands, len(model.classes)))
        for fb in range(model.n_bands):
            test = bandpass(eeg[..., trial], model.sfreq, Wp=model.filterbank[fb][0],
                            Ws=model.filterbank[fb][1])
            for c in model.classes:
                w = model.coef_[fb].T if model.ensemble else model.coef_[fb, c]
                r[fb, c] = np.corrcoef((test @ w).ravel(), (model.trains[c, fb] @ w).ravel())[0, 1]
        out[trial] = fb_coefs @ r
    return out


def decide(trial_scores: np.ndarray, threshold: float | None) -> int:
    """Winning target, or -1 ("neither") when its score is under the rest threshold."""
    best = int(np.argmax(trial_scores))
    return -1 if threshold is not None and trial_scores[best] < threshold else best


@dataclass
class IdleResult:
    """Leave-one-block-out scores behind the rest threshold."""
    threshold: float
    pick_scores: np.ndarray    # winning score per held-out square trial
    pick_correct: np.ndarray   # did the winner match the cued square
    rest_scores: np.ndarray    # winning score per held-out rest trial

    @property
    def rest_ignored(self) -> float:
        return float(np.mean(self.rest_scores < self.threshold))

    @property
    def picks_kept(self) -> float:
        return float(np.mean(self.pick_scores >= self.threshold))

    @property
    def kept_accuracy(self) -> float:
        kept = self.pick_scores >= self.threshold
        return float(np.mean(self.pick_correct[kept])) if kept.any() else 0.0


def fit_idle(trials: Trials, rest: Trials) -> IdleResult:
    """Pick the rest threshold from held-out scores (in-sample ones are inflated:
    a trial is part of its own template).

    Kept = correct picks; dropped = rest trials AND wrong picks. The threshold
    maximises the average of the two hit rates, so a pick TRCA got wrong is as
    good to drop as a look at the centre."""
    blocks = sorted(set(trials.blocks.tolist()))
    if len(blocks) < 2:
        raise ValueError(f"need at least 2 blocks to set the rest threshold, found {len(blocks)}")
    pick_scores, pick_correct, rest_scores = [], [], []
    for held_out in blocks:
        test, test_rest = trials.blocks == held_out, rest.blocks == held_out
        model = fit(Trials(trials.eeg[..., ~test], trials.targets[~test], trials.blocks[~test],
                           trials.rate, trials.names, trials.freqs, trials.letters, trials.rows))
        s = scores(model, trials.eeg[..., test])
        pick_scores += s.max(axis=1).tolist()
        pick_correct += (s.argmax(axis=1) == trials.targets[test]).tolist()
        if test_rest.any():
            rest_scores += scores(model, rest.eeg[..., test_rest]).max(axis=1).tolist()
    pick_scores, pick_correct = np.array(pick_scores), np.array(pick_correct)
    rest_scores = np.array(rest_scores)

    keep = pick_scores[pick_correct]
    drop = np.concatenate([rest_scores, pick_scores[~pick_correct]])
    values = np.unique(np.concatenate([keep, drop]))
    candidates = np.concatenate([[values[0] - 1e-6], (values[:-1] + values[1:]) / 2])
    balanced = [(np.mean(keep >= t) if len(keep) else 0) + (np.mean(drop < t) if len(drop) else 0)
                for t in candidates]
    return IdleResult(float(candidates[int(np.argmax(balanced))]), pick_scores, pick_correct, rest_scores)


def print_idle(result: IdleResult) -> None:
    n_rest, n_pick = len(result.rest_scores), len(result.pick_scores)
    print(f"Idle detection - leave-one-block-out ({n_rest} rest trials):")
    print(f"  threshold {result.threshold:.3f}  (TRCA score of the winning square)")
    print(f"  rest correctly ignored  {round(result.rest_ignored * n_rest):3d}/{n_rest}  "
          f"{100 * result.rest_ignored:5.0f}%")
    print(f"  square picks kept       {round(result.picks_kept * n_pick):3d}/{n_pick}  "
          f"{100 * result.picks_kept:5.0f}%   (accuracy of the kept ones {100 * result.kept_accuracy:.0f}%)")
    print(f"  mean score: squares {result.pick_scores.mean():.3f}, rest {result.rest_scores.mean():.3f}")


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
