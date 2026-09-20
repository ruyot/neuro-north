"""CCA frequency matching with optional filter-bank scoring.

CCA always has a winning candidate; a winner alone is not evidence of gaze.
Use prospective labeled and idle trials to measure accuracy and false actions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .trca_model import CVResult

from . import config as cfg
from . import knight as stream
from .session import Trials


def harmonics_for(freqs) -> int:
    """How many harmonics every target can use equally.

    A target given more reference components has more freedom to fit noise and
    would win more often for that reason alone, so all targets get the count the
    most-constrained one can manage inside stream.clean()'s pass band.
    """
    ceiling = cfg.FILTER_BAND[1]
    return max(1, min(sum(1 for h in (1, 2, 3) if h * f < ceiling) for f in freqs))


def _cca(eeg: np.ndarray, rate: int, freqs, harmonics: int) -> np.ndarray:
    """Plain CCA: one canonical correlation per target. eeg: (channels, samples)."""
    return np.array([stream.cca_score(eeg, stream.cca_reference(f, rate, eeg.shape[1], harmonics))
                     for f in freqs])


def scores(epoch: np.ndarray, rate: int, freqs, harmonics: int, bands: int | None = None) -> np.ndarray:
    """One score per target, highest wins. epoch: (samples, channels).

    With bands > 1 this is filter-bank CCA: the same correlation computed in
    several sub-bands and combined as sum(w(n) * r_n^2). Each sub-band starts
    higher than the last, so a band that contains only the response's harmonics
    gets to speak without the much larger fundamental drowning it out.
    """
    from scipy import signal

    eeg = epoch.T
    bands = cfg.CCA_BANDS if bands is None else bands
    if bands <= 1:
        return _cca(eeg, rate, freqs, harmonics)

    total = np.zeros(len(freqs))
    used = 0
    for n in range(min(bands, len(cfg.FILTERBANK))):
        low = cfg.FILTERBANK[n][0][0]
        # Every candidate must retain a represented harmonic in this band.
        # With 15/20 Hz and H=1, 22-40 Hz contains neither reference.
        if not all(any(low <= h * f < cfg.FILTER_BAND[1]
                       for h in range(1, harmonics + 1)) for f in freqs):
            continue
        used += 1
        sos = signal.butter(4, [low, cfg.FILTER_BAND[1]], btype="band", fs=rate, output="sos")
        # filtfilt: zero phase, so a sub-band cannot shift the response in time
        # relative to the reference it is about to be correlated against.
        sub = signal.sosfiltfilt(sos, eeg, axis=1)
        weight = (n + 1) ** -1.25 + 0.25
        total += weight * _cca(sub, rate, freqs, harmonics) ** 2
    return total if used else _cca(eeg, rate, freqs, harmonics)


def evaluate(trials: Trials) -> tuple[CVResult, dict]:
    """Accuracy per block plus the mean margin (top correlation minus runner-up).

    The margin is the honest "is there a distinguishable signal" number: ~0.17
    on a session that decodes at 90%+, ~0.08 on one that decodes at chance.
    """
    from .trca_model import CVResult, bits_per_min   # pulls in meegkit; live never needs it
    harmonics = harmonics_for(trials.freqs)
    n = trials.n_targets
    confusion = np.zeros((n, n), dtype=int)
    blocks = sorted(set(trials.blocks.tolist()))
    accs, itrs, margins = [], [], []
    for block in blocks:
        sel = trials.blocks == block
        ok = 0
        for k in np.flatnonzero(sel):
            s = scores(trials.eeg[..., k], trials.rate, trials.freqs, harmonics)
            pick, truth = int(s.argmax()), int(trials.targets[k])
            confusion[truth, pick] += 1
            ok += pick == truth
            ranked = np.sort(s)
            margins.append(float(ranked[-1] - ranked[-2]) if n > 1 else 0.0)
        acc = ok / int(sel.sum())
        accs.append(100 * acc)
        itrs.append(bits_per_min(acc, n))
    sigma = np.array([confidence(trials.eeg[..., k], trials.rate, trials.freqs, harmonics)
                      for k in range(trials.eeg.shape[-1])])
    picked = np.array([int(scores(trials.eeg[..., k], trials.rate, trials.freqs, harmonics).argmax())
                       for k in range(trials.eeg.shape[-1])])
    hit = picked == trials.targets
    result = CVResult(blocks, np.array(accs), np.array(itrs), confusion,
                      trials.freqs, trials.letters)
    stats = {"margin": float(np.mean(margins)),
             "sigma": float(sigma.mean()),
             "sigma_correct": float(sigma[hit].mean()) if hit.any() else float("nan"),
             "sigma_wrong": float(sigma[~hit].mean()) if (~hit).any() else float("nan")}
    return result, stats


def confidence(epoch: np.ndarray, rate: int, freqs, harmonics: int) -> float:
    """Heuristic decoy contrast for the actual winning candidate.

    This is NOT a calibrated z-score or probability. Five colored, correlated
    frequency controls cannot establish a statistical noise distribution.
    """
    eeg = epoch.T
    choice = int(scores(epoch, rate, freqs, harmonics).argmax())
    target = _cca(eeg, rate, freqs, harmonics)[choice]
    null = _cca(eeg, rate, cfg.DECOY_FREQUENCIES, harmonics)
    if null.std() < 1e-9:
        return 0.0
    return float((target - null.mean()) / null.std())


def accept_prediction(choice: int, contrast: float, threshold: float) -> bool:
    """Zero disables the heuristic gate; invalid/absent predictions never pass."""
    return choice >= 0 and np.isfinite(contrast) and (threshold <= 0 or contrast >= threshold)
