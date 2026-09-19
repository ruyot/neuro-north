"""
Task-Related Component Analysis (TRCA) model helpers.

What TRCA does, in one paragraph
--------------------------------
Every time you stare at a target that flickers at frequency f, the visual
cortex produces the *same* stereotyped response. TRCA learns, per target, a
spatial filter (a weighted sum of the electrodes) that maximises the
reproducibility of that response across your training trials. At test time it
band-pass filters the incoming EEG into several sub-bands (the "filter bank",
so harmonics are used too), projects each sub-band through every target's
spatial filter, correlates the result with that target's template, and picks
the target with the highest combined correlation. Because it is trained on
*your* data with *your* electrodes, it is far more accurate than a generic
CCA detector.

Training data layout
--------------------
`training_data/session_*/block_{block}_{trial}.csv`, where each CSV is one
recorded flicker window of shape (samples, channels). `trial` is 1..N_TARGETS
and encodes the label (trial 1 -> target 0, trial 2 -> target 1, ...). Blocks
are independent repeats used for cross-validation.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from meegkit.trca import TRCA
from meegkit.utils.trca import itr, normfit

from . import config as cfg
from .preprocessing import crop_indices


def resolve_data_dir(data_dir: str | None) -> str:
    """Use `data_dir` if given, else the latest calibration session."""
    if data_dir is None:
        data_dir = cfg.latest_session_dir()
    if data_dir is None:
        raise FileNotFoundError(
            f"No calibration sessions in {cfg.TRAINING_DATA_DIR!r}. "
            "Run collect_training_data.py first."
        )
    return data_dir


def load_training_data(data_dir: str, n_blocks: int, n_targets: int = cfg.N_TARGETS,
                       crop: bool = True):
    """
    Load every block/trial CSV into an (samples, channels, trials) EEG cube.

    Returns
    -------
    eeg : np.ndarray, shape (gaze_samples, channels, n_blocks * n_targets)
        Cropped to the TRCA analysis window (or the full capture if crop=False).
    labels : np.ndarray, shape (n_blocks * n_targets,)
        Target index (0..n_targets-1) for each trial.
    """
    trials = []
    for block in range(1, n_blocks + 1):
        for trial in range(1, n_targets + 1):
            path = os.path.join(data_dir, f"block_{block}_{trial}.csv")
            trials.append(pd.read_csv(path).values)   # (samples, channels)

    # meegkit wants (samples, channels, trials).
    eeg = np.array(trials).transpose(1, 2, 0)

    # Label = (trial index - 1), repeated for every block.
    labels = np.array(list(range(n_targets)) * n_blocks)

    # Keep only the analysis window (skip visual latency).
    if crop:
        eeg = eeg[crop_indices()]
    return eeg, labels


def build_model() -> TRCA:
    """Create an (untrained) TRCA classifier configured from `config`."""
    return TRCA(cfg.SAMPLING_RATE, cfg.FILTERBANK, cfg.USE_ENSEMBLE_TRCA)


def fit_model(data_dir: str | None = None, n_blocks: int | None = None) -> TRCA:
    """
    Load all training data and fit a TRCA model ready for live prediction.

    If `data_dir` is None the latest session is used. If `n_blocks` is None it
    is inferred from how many block_*_1.csv files exist.
    """
    data_dir = resolve_data_dir(data_dir)
    if n_blocks is None:
        n_blocks = count_blocks(data_dir)
    if n_blocks == 0:
        raise FileNotFoundError(
            f"No training blocks found in {data_dir!r}. "
            "Run collect_training_data.py first."
        )

    eeg, labels = load_training_data(data_dir, n_blocks)
    model = build_model()
    model.fit(eeg, labels)
    print(f"TRCA fitted on {n_blocks} block(s) from {os.path.basename(data_dir)} "
          f"({eeg.shape[-1]} trials, {eeg.shape[1]} channels).")
    return model


def cross_validate(data_dir: str | None = None,
                   n_blocks: int | None = None,
                   alpha_ci: float = 0.05) -> None:
    """
    Leave-one-block-out cross-validation, reporting accuracy and ITR.

    This is the honest way to estimate how well the classifier will work: each
    block is held out once as an unseen test set while the rest train the model.
    ITR (information transfer rate) summarises speed + accuracy in bits/min.
    """
    data_dir = resolve_data_dir(data_dir)
    if n_blocks is None:
        n_blocks = count_blocks(data_dir)
    if n_blocks < 2:
        raise ValueError(f"Need at least 2 complete blocks to cross-validate, found {n_blocks}.")
    eeg, labels = load_training_data(data_dir, n_blocks)

    n_targets = cfg.N_TARGETS
    # One selection costs a whole trial cycle (the user stares for the full
    # flicker, then rests), not just the 1 s analysis window.
    selection_time = cfg.FLICKER_DURATION + cfg.INTER_TRIAL_INTERVAL
    ci = 100 * (1 - alpha_ci)

    model = build_model()
    accs = np.zeros(n_blocks)
    itrs = np.zeros(n_blocks)

    print(f"Session: {os.path.basename(data_dir)}")
    print("Leave-one-block-out cross-validation (ensemble TRCA):\n")
    for i in range(n_blocks):
        # Train on every block except block i.
        train = np.concatenate(
            (eeg[..., :i * n_targets], eeg[..., (i + 1) * n_targets:]), axis=2)
        y_train = np.concatenate(
            (labels[:i * n_targets], labels[(i + 1) * n_targets:]), axis=0)
        model.fit(train, y_train)

        # Test on the held-out block.
        test = eeg[..., i * n_targets:(i + 1) * n_targets]
        y_test = labels[i * n_targets:(i + 1) * n_targets]
        predicted = model.predict(test)

        correct = np.mean(predicted == y_test)
        accs[i] = correct * 100
        # meegkit's itr() raises at or below chance; that's 0 bits/min.
        itrs[i] = itr(n_targets, correct, selection_time) if correct > 1 / n_targets else 0.0
        print(f"  Block {i + 1}: accuracy = {accs[i]:5.1f}%   ITR = {itrs[i]:5.1f} bits/min")

    # meegkit's normfit takes the confidence *level* (0.95), not alpha (0.05).
    mu_acc, _, ci_acc, _ = normfit(accs, 1 - alpha_ci)
    mu_itr, _, ci_itr, _ = normfit(itrs, 1 - alpha_ci)
    print(f"\nMean accuracy = {mu_acc:.1f}%  "
          f"({ci:.0f}% CI: {ci_acc[0]:.1f}-{ci_acc[1]:.1f}%)   chance = {100 / n_targets:.0f}%")
    print(f"Mean ITR      = {mu_itr:.1f} bits/min  "
          f"({ci:.0f}% CI: {ci_itr[0]:.1f}-{ci_itr[1]:.1f})")


def count_blocks(data_dir: str) -> int:
    """Count how many complete blocks exist (all N_TARGETS trial files present)."""
    block = 0
    while all(os.path.exists(os.path.join(data_dir, f"block_{block + 1}_{t}.csv"))
              for t in range(1, cfg.N_TARGETS + 1)):
        block += 1
    return block
