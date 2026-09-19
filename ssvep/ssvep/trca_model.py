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
`training_data/block_{block}_{trial}.csv`, where each CSV is one recorded
flicker window of shape (samples, channels). `trial` is 1..N_TARGETS and
encodes the label (trial 1 -> target 0, trial 2 -> target 1, ...). Blocks are
independent repeats used for cross-validation.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from meegkit.trca import TRCA
from meegkit.utils.trca import itr, normfit

from . import config as cfg
from .preprocessing import crop_indices

# Rates the two boards this project records from actually stream at: the
# Knight board (125 Hz) and BrainFlow's synthetic board (250 Hz). Used to
# recognise a recorded block's rate from its length -- see `_derive_rate`.
BOARD_RATES = (125, 250)


def load_training_data(data_dir: str, n_blocks: int, n_targets: int = cfg.N_TARGETS):
    """
    Load every block/trial CSV into an (samples, channels, trials) EEG cube.

    Returns
    -------
    eeg : np.ndarray, shape (gaze_samples, channels, n_blocks * n_targets)
        Cropped to the TRCA analysis window.
    labels : np.ndarray, shape (n_blocks * n_targets,)
        Target index (0..n_targets-1) for each trial.
    sampling_rate : int
        The rate the blocks were recorded at, recovered from their length.
    """
    if n_blocks < 1:
        raise FileNotFoundError(
            f"No training blocks found in {data_dir!r}. "
            "Run collect_training_data.py first."
        )

    trials = []
    for block in range(1, n_blocks + 1):
        for trial in range(1, n_targets + 1):
            path = os.path.join(data_dir, f"block_{block}_{trial}.csv")
            trials.append(pd.read_csv(path).values)   # (samples, channels)

    # meegkit wants (samples, channels, trials).
    eeg = np.array(trials).transpose(1, 2, 0)

    # Label = (trial index - 1), repeated for every block.
    labels = np.array(list(range(n_targets)) * n_blocks)

    # The rate is not a config constant: blocks recorded from the 250 Hz
    # synthetic board must crop (and later fit) at 250 Hz, not at 125.
    sampling_rate = _derive_rate(trials[0].shape[0])

    # Keep only the analysis window (skip visual latency).
    eeg = eeg[crop_indices(sampling_rate)]
    return eeg, labels, sampling_rate


def build_model(sampling_rate: int) -> TRCA:
    """Create an (untrained) TRCA classifier at the data's sampling rate."""
    return TRCA(sampling_rate, cfg.FILTERBANK, cfg.USE_ENSEMBLE_TRCA)


def fit_model(data_dir: str = cfg.TRAINING_DATA_DIR,
              n_blocks: int | None = None) -> TRCA:
    """
    Load all training data and fit a TRCA model ready for live prediction.

    If `n_blocks` is None it is inferred from how many block_*_1.csv files exist.
    """
    if n_blocks is None:
        n_blocks = _count_blocks(data_dir)

    eeg, labels, sampling_rate = load_training_data(data_dir, n_blocks)
    model = build_model(sampling_rate)
    model.fit(eeg, labels)
    print(f"TRCA fitted on {n_blocks} block(s) "
          f"({eeg.shape[-1]} trials, {eeg.shape[1]} channels, {sampling_rate} Hz).")
    return model


def cross_validate(data_dir: str = cfg.TRAINING_DATA_DIR,
                   n_blocks: int | None = None,
                   alpha_ci: float = 0.05) -> np.ndarray:
    """
    Leave-one-block-out cross-validation, reporting accuracy and ITR.

    This is the honest way to estimate how well the classifier will work: each
    block is held out once as an unseen test set while the rest train the model.
    ITR (information transfer rate) summarises speed + accuracy in bits/min.

    Returns the per-block accuracy (%) array.
    """
    if n_blocks is None:
        n_blocks = _count_blocks(data_dir)
    eeg, labels, sampling_rate = load_training_data(data_dir, n_blocks)

    n_targets = cfg.N_TARGETS
    # One selection costs a whole trial cycle, not just the analysis window:
    # the user stares for FLICKER_DURATION, then the rest gap before the next.
    selection_time = cfg.FLICKER_DURATION + cfg.INTER_TRIAL_INTERVAL
    ci = 100 * (1 - alpha_ci)

    model = build_model(sampling_rate)
    accs = np.zeros(n_blocks)
    itrs = np.zeros(n_blocks)

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
        # itr() refuses accuracies below chance; below chance the rate is 0.
        itrs[i] = (itr(n_targets, correct, selection_time)
                   if correct >= 1 / n_targets else 0.0)
        print(f"  Block {i}: accuracy = {accs[i]:5.1f}%   ITR = {itrs[i]:5.1f} bits/min")

    # meegkit's normfit takes the confidence *level* (0.95), not alpha (0.05).
    mu_acc, _, ci_acc, _ = normfit(accs, 1 - alpha_ci)
    mu_itr, _, ci_itr, _ = normfit(itrs, 1 - alpha_ci)
    print(f"\nMean accuracy = {mu_acc:.1f}%  "
          f"({ci:.0f}% CI: {ci_acc[0]:.1f}-{ci_acc[1]:.1f}%)")
    print(f"Mean ITR      = {mu_itr:.1f} bits/min  "
          f"({ci:.0f}% CI: {ci_itr[0]:.1f}-{ci_itr[1]:.1f})")
    return accs


def _count_blocks(data_dir: str) -> int:
    """Count how many complete blocks exist by looking for block_{b}_1.csv."""
    block = 0
    while os.path.exists(os.path.join(data_dir, f"block_{block + 1}_1.csv")):
        block += 1
    return block


def _derive_rate(n_rows: int) -> int:
    """
    Recover the recording rate from one trial's row count.

    `recording.py` captures `int(round(FLICKER_DURATION * board.sr)) + 1` rows,
    so at the stock 1.5 s flicker 189 rows -> 125 Hz (Knight) and 376 rows ->
    250 Hz (synthetic).

    Inverting that arithmetic is *not* enough on its own: row count and
    duration are underdetermined, so every count divides neatly into some
    rate. A set recorded at 125 Hz and then re-read after someone edited
    FLICKER_DURATION to 2.0 would "derive" a perfectly self-consistent 94 Hz
    and crop at the wrong window. So only accept a rate a board in this
    project actually streams at.
    """
    for rate in BOARD_RATES:
        if int(round(cfg.FLICKER_DURATION * rate)) + 1 == n_rows:
            return rate
    raise ValueError(
        f"A {n_rows}-row trial does not match FLICKER_DURATION "
        f"({cfg.FLICKER_DURATION}s) at any rate this project records at "
        f"{BOARD_RATES}: expected "
        f"{[int(round(cfg.FLICKER_DURATION * r)) + 1 for r in BOARD_RATES]} "
        f"rows. Either the CSVs are not captures, or FLICKER_DURATION was "
        f"changed after they were recorded."
    )
