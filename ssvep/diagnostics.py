"""
"Did SSVEP actually show up?" -- spectral check of a calibration session.

TRCA accuracy alone can't tell you *why* a session went badly. This looks at
the calibration trials directly: when you looked at a square flickering at f,
was there more power at f (and its 2nd harmonic 2f) than when you looked at the
other squares?

    relative power R[letter, f] = power at f while looking at `letter`
                                  / mean power at f across all letters

A working headset gives a strong diagonal: R > 1 for the frequency you were
looking at, R < 1 for the others. No diagonal means the signal isn't reaching
the electrodes (contact / placement), not a classifier problem.
"""

from __future__ import annotations

import os

import numpy as np

from . import config as cfg

NFFT = 1024          # zero-padded FFT length -> smooth spectrum (~0.12 Hz bins)
PEAK_HALF_WIDTH = 0.3  # Hz: power at f = max within f +/- this
CLEAR, WEAK = 1.3, 1.1  # thresholds on the diagonal ratio


def trial_spectra(eeg: np.ndarray, sampling_rate: int = cfg.SAMPLING_RATE):
    """Power spectrum of every trial and channel.

    eeg: (samples, channels, trials) -> freqs (F,), power (F, channels, trials)
    """
    window = np.hanning(eeg.shape[0])[:, None, None]
    spectrum = np.fft.rfft(eeg * window, n=NFFT, axis=0)
    freqs = np.fft.rfftfreq(NFFT, 1 / sampling_rate)
    return freqs, np.abs(spectrum) ** 2


def power_at(freqs, power, f):
    """Peak power within f +/- PEAK_HALF_WIDTH (plus the 2nd harmonic if it's measurable)."""
    total = 0
    for h in (1, 2):
        if h * f + PEAK_HALF_WIDTH < freqs[-1]:
            band = np.abs(freqs - h * f) <= PEAK_HALF_WIDTH
            total = total + power[band].max(axis=0)
    return total


def relative_power(freqs, power, labels, n_targets=cfg.N_TARGETS):
    """R[looked-at target, frequency target, channel], normalised per frequency and channel."""
    absolute = np.zeros((n_targets, n_targets, power.shape[1]))
    for looked in range(n_targets):
        trials = power[..., labels == looked]
        for j, f in enumerate(cfg.STIMULUS_FREQUENCIES):
            absolute[looked, j] = power_at(freqs, trials, f).mean(axis=-1)
    return absolute / absolute.mean(axis=0, keepdims=True)


def report(eeg: np.ndarray, labels: np.ndarray, session_name: str, plot_path: str | None = None) -> dict:
    """Print the SSVEP check for a session and optionally save the spectrum plot."""
    freqs, power = trial_spectra(eeg)
    rel = relative_power(freqs, power, labels)      # (looked, freq, channel)
    rel_mean = rel.mean(axis=-1)                    # average over channels
    diag = np.diag(rel_mean)

    letters, fs = cfg.TARGET_LETTERS, cfg.STIMULUS_FREQUENCIES
    print("SSVEP check: relative power at each flicker frequency (+2nd harmonic)")
    print("1.00 = average for that frequency; the diagonal (*) should stand out\n")
    print("  looking at  | " + " ".join(f"{f:>7.2f}Hz" for f in fs) + " | verdict")
    for i, (letter, f) in enumerate(zip(letters, fs)):
        cells = " ".join(f"{'*' if i == j else ' '}{rel_mean[i, j]:7.2f}  " for j in range(len(fs)))
        largest = rel_mean[i, i] >= rel_mean[:, i].max()
        verdict = ("clear" if diag[i] >= CLEAR and largest else
                   "weak" if diag[i] >= WEAK else "none")
        print(f"  {letter} ({f:5.2f} Hz) | {cells}| {verdict}")

    # Which electrodes carry the response: mean diagonal ratio per channel.
    per_channel = np.array([np.diag(rel[..., c]).mean() for c in range(rel.shape[-1])])
    order = np.argsort(per_channel)[::-1]
    ranked = ", ".join(f"{cfg.ELECTRODE_LABELS[c]} {per_channel[c]:.2f}" for c in order)
    print(f"\n  Response by electrode (higher = better): {ranked}")

    # Alpha check: a big ~10 Hz peak in trials where the user WASN'T looking at 10 Hz
    # means resting alpha, which the classifier can confuse with the 10 Hz square.
    ten_hz = [i for i, f in enumerate(fs) if abs(f - 10) < 0.5]
    others = ~np.isin(labels, ten_hz)
    mean_spec = power[..., others].mean(axis=(1, 2))
    in_alpha = (freqs >= 8) & (freqs <= 13)
    flank = ((freqs >= 4) & (freqs < 7)) | ((freqs > 15) & (freqs <= 20))
    alpha_ratio = mean_spec[in_alpha].max() / mean_spec[flank].mean()
    alpha_peak = freqs[in_alpha][np.argmax(mean_spec[in_alpha])]
    strong_alpha = alpha_ratio > 4
    if strong_alpha:
        print(f"  Strong alpha (~{alpha_peak:.1f} Hz, {alpha_ratio:.0f}x background) in non-10 Hz trials: "
              "stay focused on the squares; 10 Hz (C) may be over-predicted.")

    # Overall verdict from the average diagonal, so one noisy letter can't swing it.
    failing = [i for i in range(len(fs)) if diag[i] < CLEAR]
    if not failing:
        print("\n  -> SSVEP visible for every letter. Headset is working; tune software if accuracy is low.")
    elif diag.mean() < WEAK:
        print("\n  -> No SSVEP response. Check electrode contact/placement before changing code.")
    elif strong_alpha and set(failing) <= set(ten_hz):
        print("\n  -> SSVEP visible except at 10 Hz, which is masked by alpha. "
              "Consider replacing 10 Hz with a frequency outside 8-13 Hz.")
    else:
        weak_letters = ", ".join(letters[i] for i in failing)
        print(f"\n  -> Partial response (weak: {weak_letters}). Improve contact on the weakest "
              "electrodes and collect more blocks.")

    if plot_path:
        _plot(freqs, power, labels, rel_mean, session_name, plot_path)
        print(f"\n  Spectrum plot saved to {plot_path}")
    return {"relative_power": rel_mean, "per_channel": per_channel, "alpha_ratio": alpha_ratio}


def _plot(freqs, power, labels, rel_mean, session_name, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = ["#e6194b", "#3cb44b", "#4363d8", "#f58231"]
    shown = (freqs >= 3) & (freqs <= 40)
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True, sharey=True)
    for i, ax in enumerate(axes.flat):
        f = cfg.STIMULUS_FREQUENCIES[i]
        this = power[..., labels == i].mean(axis=(1, 2))
        rest = power[..., labels != i].mean(axis=(1, 2))
        ax.semilogy(freqs[shown], rest[shown], color="0.6", lw=1, label="other letters")
        ax.semilogy(freqs[shown], this[shown], color=colors[i], lw=1.6,
                    label=f"looking at {cfg.TARGET_LETTERS[i]}")
        for j, fj in enumerate(cfg.STIMULUS_FREQUENCIES):
            ax.axvline(fj, color=colors[j], alpha=0.25 if j != i else 0.8, lw=1)
        ax.axvline(2 * f, color=colors[i], ls="--", alpha=0.6, lw=1)
        ax.set_title(f"Looking at {cfg.TARGET_LETTERS[i]} ({f} Hz): {rel_mean[i, i]:.2f}x at f")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(alpha=0.2)
    for ax in axes[1]:
        ax.set_xlabel("Hz  (solid = flicker frequencies, dashed = 2nd harmonic)")
    for ax in axes[:, 0]:
        ax.set_ylabel("power (mean of channels)")
    fig.suptitle(f"SSVEP spectrum check - {session_name}")
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
