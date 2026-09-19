"""
check_analysis.py
=================

Offline proof that the analysis chain (load -> crop -> TRCA fit -> predict ->
cross-validate) works, with NO EEG hardware attached.

It synthesises SSVEP-like trials -- a sine at each target frequency plus its
2nd harmonic, with a per-channel amplitude/phase that is stable across blocks
(that reproducible response is exactly what TRCA learns) and fresh Gaussian
noise every trial -- writes them into a tempdir laid out like `training_data/`,
then runs the *real* library code over them.

Run:  python check_analysis.py
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

from ssvep import config as cfg
from ssvep.preprocessing import crop_indices, filter_eeg, extract_channel_matrix
from ssvep.trca_model import load_training_data, fit_model, cross_validate

N_BLOCKS = 6
NOISE_UV = 4.0   # additive Gaussian noise, microvolts RMS


def fake_trial(target: int, rng: np.random.Generator, sampling_rate: int,
               signal: bool = True) -> np.ndarray:
    """One fake SSVEP trial, shaped exactly like a recording.py capture."""
    n_rows = int(round(cfg.FLICKER_DURATION * sampling_rate)) + 1
    t = np.arange(n_rows) / sampling_rate
    trial = rng.normal(0, NOISE_UV, (n_rows, cfg.NUM_CHANNELS))

    if signal:
        freq = cfg.STIMULUS_FREQUENCIES[target]
        # Per-channel amplitude/phase: deterministic per target, so every block
        # sees the same spatial pattern (what the spatial filter locks onto).
        montage = np.random.default_rng(1000 + target)
        amp = montage.uniform(2.0, 6.0, cfg.NUM_CHANNELS)        # microvolts
        phase = montage.uniform(0, 2 * np.pi, cfg.NUM_CHANNELS)
        trial += (amp * np.sin(2 * np.pi * freq * t[:, None] + phase)
                  + 0.5 * amp * np.sin(2 * np.pi * 2 * freq * t[:, None]
                                       + 2 * phase))

    # Same filter chain the recorder runs before saving. BrainFlow wants
    # (n_rows, n_samples), so hand it the transpose.
    data = np.ascontiguousarray(trial.T)
    channels = list(range(cfg.NUM_CHANNELS))
    filter_eeg(data, channels, sampling_rate)
    return extract_channel_matrix(data, channels)


def write_fake_blocks(data_dir: str, sampling_rate: int,
                      signal: bool = True) -> None:
    """Write N_BLOCKS x N_TARGETS CSVs in the layout trca_model expects."""
    rng = np.random.default_rng(0)
    header = ",".join(cfg.ELECTRODE_LABELS[:cfg.NUM_CHANNELS])
    for block in range(1, N_BLOCKS + 1):
        for trial in range(1, cfg.N_TARGETS + 1):   # 1-based, encodes label
            path = os.path.join(data_dir, f"block_{block}_{trial}.csv")
            trial_data = fake_trial(trial - 1, rng, sampling_rate, signal)
            assert np.isfinite(trial_data).all(), "filter chain produced NaNs"
            np.savetxt(path, trial_data, delimiter=",",
                       header=header, comments="", fmt="%.7f")


def check_crop_indices() -> None:
    idx = crop_indices(125, cfg.VISUAL_LATENCY, cfg.GAZE_DURATION)
    assert len(idx) == 125, len(idx)
    assert idx[0] == 19 and idx[-1] == 143, (idx[0], idx[-1])
    # (config's comment says 188; round(187.5) is banker's-rounded to 188, +1
    # -> 189. Either way the crop must land inside the captured window.)
    assert idx[-1] < cfg.CAPTURE_SAMPLES, "crop runs past the capture window"
    # Synthetic board streams at 250 Hz -> the same call must still fit inside
    # a 250 Hz capture window. This is why crop_indices takes an explicit rate.
    idx250 = crop_indices(250, cfg.VISUAL_LATENCY, cfg.GAZE_DURATION)
    capture250 = int(round(cfg.FLICKER_DURATION * 250)) + 1
    assert len(idx250) == 250, len(idx250)
    assert idx250[0] == 38 and idx250[-1] == 287, (idx250[0], idx250[-1])
    assert idx250[-1] < capture250, (idx250[-1], capture250)
    print(f"crop_indices(125) -> {len(idx)} samples [{idx[0]}:{idx[-1] + 1}] "
          f"inside {cfg.CAPTURE_SAMPLES}")
    print(f"crop_indices(250) -> {len(idx250)} samples "
          f"[{idx250[0]}:{idx250[-1] + 1}] inside {capture250}")


def check_rate(sampling_rate: int) -> None:
    """Full load -> fit -> predict -> cross-validate round trip at one rate."""
    n_gaze = int(np.floor(cfg.GAZE_DURATION * sampling_rate + 0.5))
    data_dir = tempfile.mkdtemp(prefix=f"ssvep_{sampling_rate}_")
    try:
        write_fake_blocks(data_dir, sampling_rate)

        eeg, labels, derived = load_training_data(data_dir, N_BLOCKS)
        assert derived == sampling_rate, (derived, sampling_rate)
        assert eeg.shape == (n_gaze, cfg.NUM_CHANNELS,
                             N_BLOCKS * cfg.N_TARGETS), eeg.shape
        assert labels.tolist() == list(range(cfg.N_TARGETS)) * N_BLOCKS
        print(f"\n=== {sampling_rate} Hz ===")
        print(f"rate recovered from file length = {derived} Hz, cube {eeg.shape}")

        # Fit on blocks 1..5, predict the held-out block 6.
        model = fit_model(data_dir, N_BLOCKS - 1)
        test = eeg[..., (N_BLOCKS - 1) * cfg.N_TARGETS:]
        y_test = labels[(N_BLOCKS - 1) * cfg.N_TARGETS:]
        acc = float(np.mean(model.predict(test) == y_test))
        print(f"held-out block accuracy = {acc:.2f}")
        assert acc >= 0.9, f"{sampling_rate} Hz: expected near-perfect, got {acc}"

        accs = cross_validate(data_dir)
        assert len(accs) == N_BLOCKS, accs
        assert np.mean(accs) >= 90.0, accs

        # The script itself, end to end, on the same tempdir.
        print(f"\n$ python evaluate_trca.py --data-dir <{sampling_rate} Hz tempdir>")
        subprocess.run([sys.executable, "evaluate_trca.py", "--data-dir", data_dir],
                       check=True)
    finally:
        shutil.rmtree(data_dir, ignore_errors=True)


def check_failure_modes() -> None:
    """The paths that used to crash, or used to guess wrong, must not."""
    # Garbage EEG must degrade, not crash: pure noise drives at least one
    # block below chance, which is where itr() used to raise.
    noise_dir = tempfile.mkdtemp(prefix="ssvep_noise_")
    try:
        write_fake_blocks(noise_dir, cfg.SAMPLING_RATE, signal=False)
        print("\nPure-noise data (must survive below-chance blocks):")
        noise_accs = cross_validate(noise_dir)
        assert min(noise_accs) < 25.0, noise_accs
    finally:
        shutil.rmtree(noise_dir, ignore_errors=True)

    # Empty directory must say so, not die on an axis error.
    empty = tempfile.mkdtemp(prefix="ssvep_empty_")
    try:
        cross_validate(empty)
        raise AssertionError("empty dir should have raised")
    except FileNotFoundError as exc:
        print(f"\nempty dir -> FileNotFoundError: {str(exc)[:44]}...")
    finally:
        shutil.rmtree(empty, ignore_errors=True)

    # A row count no board could have produced must fail loudly.
    bad = tempfile.mkdtemp(prefix="ssvep_bad_")
    try:
        for trial in range(1, cfg.N_TARGETS + 1):
            np.savetxt(os.path.join(bad, f"block_1_{trial}.csv"),
                       np.zeros((200, cfg.NUM_CHANNELS)), delimiter=",",
                       header=",".join(cfg.ELECTRODE_LABELS[:cfg.NUM_CHANNELS]),
                       comments="")
        try:
            load_training_data(bad, 1)
            raise AssertionError("200-row trial should have raised")
        except ValueError as exc:
            print(f"200-row trial -> ValueError: {str(exc)[:44]}...")

        # Same files, but FLICKER_DURATION edited after recording: the old
        # float-inversion guard happily "derived" 94 Hz here.
        for trial in range(1, cfg.N_TARGETS + 1):
            np.savetxt(os.path.join(bad, f"block_1_{trial}.csv"),
                       np.zeros((189, cfg.NUM_CHANNELS)), delimiter=",",
                       header=",".join(cfg.ELECTRODE_LABELS[:cfg.NUM_CHANNELS]),
                       comments="")
        stock = cfg.FLICKER_DURATION
        cfg.FLICKER_DURATION = 2.0
        try:
            load_training_data(bad, 1)
            raise AssertionError("189 rows at 2.0s flicker should have raised")
        except ValueError as exc:
            print(f"189 rows @ 2.0s flicker -> ValueError: {str(exc)[:32]}...")
        finally:
            cfg.FLICKER_DURATION = stock
    finally:
        shutil.rmtree(bad, ignore_errors=True)


def main() -> None:
    check_crop_indices()
    for sampling_rate in (cfg.SAMPLING_RATE, 250):   # Knight, synthetic board
        check_rate(sampling_rate)
    check_failure_modes()
    print("\nOK")


if __name__ == "__main__":
    main()
