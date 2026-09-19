# Did SSVEP actually show up? - spectral check of a calibration session

from __future__ import annotations

import math
import os

import numpy as np

from .session import Trials

NFFT = 1024              # zero-padded FFT -> smooth spectrum
PEAK_HALF_WIDTH = 0.3    # Hz: power at f = max within f +/- this
FILTER_TOP = 40.0        # stream.clean()'s band-pass edge: harmonics at/above it are gone
CLEAR, WEAK = 1.3, 1.1   # thresholds on the diagonal ratio

ALPHA_BAND = (8.0, 13.0)
HARMONICS = (1, 2, 3)    # which multiples of each target count as "its" frequencies
GUARD = 1.5              # Hz kept clear around every target harmonic when picking quiet bands


def trial_spectra(eeg: np.ndarray, rate: int):
    """eeg (samples, channels, trials) -> freqs (F,), power (F, channels, trials)."""
    window = np.hanning(eeg.shape[0])[:, None, None]
    spectrum = np.fft.rfft(eeg * window, n=NFFT, axis=0)
    return np.fft.rfftfreq(NFFT, 1 / rate), np.abs(spectrum) ** 2


def power_at(freqs, power, f):
    """Peak power near f, plus near 2f when 2f survives the 1-40 Hz filter."""
    total = 0
    for h in (1, 2):
        if h * f + PEAK_HALF_WIDTH < min(FILTER_TOP, freqs[-1]):
            band = np.abs(freqs - h * f) <= PEAK_HALF_WIDTH
            total = total + power[band].max(axis=0)
    return total


def relative_power(freqs, power, targets, target_freqs):
    """R[looked-at target, frequency target, channel], normalised per frequency and channel."""
    n = len(target_freqs)
    absolute = np.zeros((n, n, power.shape[1]))
    for looked in range(n):
        for j, f in enumerate(target_freqs):
            absolute[looked, j] = power_at(freqs, power[..., targets == looked], f).mean(axis=-1)
    return absolute / absolute.mean(axis=0, keepdims=True)


def _band(freqs, lo, hi):
    return (freqs >= lo) & (freqs <= hi)


def _harmonics_in_alpha(f: float) -> list[float]:
    return [h * f for h in HARMONICS if ALPHA_BAND[0] <= h * f <= ALPHA_BAND[1]]


def _quiet_mask(freqs, target_freqs):
    """3-40 Hz, minus the alpha band and +/- GUARD around every target harmonic."""
    quiet = _band(freqs, 3.0, FILTER_TOP) & ~_band(freqs, *ALPHA_BAND)
    for f in target_freqs:
        for h in HARMONICS:
            quiet &= np.abs(freqs - h * f) > GUARD
    return quiet


def report(trials: Trials, session_name: str, plot_path: str | None = None) -> dict:
    """Print the SSVEP check for full-flicker trials (load_trials(..., full=True))."""
    freqs, power = trial_spectra(trials.eeg, trials.rate)
    fs, letters = trials.freqs, trials.letters
    rel = relative_power(freqs, power, trials.targets, fs)     # (looked, freq, channel)
    rel_mean = rel.mean(axis=-1)
    diag = np.diag(rel_mean)

    print("SSVEP check: relative power at each flicker frequency (+2nd harmonic below 40 Hz)")
    print("1.00 = average for that frequency; the diagonal (*) should stand out\n")
    print("  looking at    | " + " ".join(f"{f:>7.2f}Hz" for f in fs) + " | verdict")
    for i, (letter, f) in enumerate(zip(letters, fs)):
        cells = " ".join(f"{'*' if i == j else ' '}{rel_mean[i, j]:7.2f}  " for j in range(len(fs)))
        largest = rel_mean[i, i] >= rel_mean[:, i].max()
        verdict = "clear" if diag[i] >= CLEAR and largest else "weak" if diag[i] >= WEAK else "none"
        print(f"  {letter} ({f:5.2f} Hz)  | {cells}| {verdict}")

    per_channel = np.array([np.diag(rel[..., c]).mean() for c in range(rel.shape[-1])])
    ranked = ", ".join(f"{trials.names[c]} {per_channel[c]:.2f}" for c in np.argsort(per_channel)[::-1])
    print(f"\n  Response by electrode (higher = better): {ranked}")

    # Resting alpha, measured on trials whose targets put nothing into 8-13 Hz,
    # against quiet bands clear of every target and harmonic.
    at_risk = [i for i, f in enumerate(fs) if _harmonics_in_alpha(f)]
    clean_trials = ~np.isin(trials.targets, at_risk)
    spec = power[..., clean_trials if clean_trials.any() else slice(None)].mean(axis=(1, 2))
    alpha = _band(freqs, *ALPHA_BAND)
    alpha_ratio = float(spec[alpha].max() / spec[_quiet_mask(freqs, fs)].mean())
    alpha_peak = float(freqs[alpha][np.argmax(spec[alpha])])
    if alpha_ratio > 4:
        risk = ("; " + ", ".join(f"{letters[i]} is most at risk ({'/'.join(f'{h:g}' for h in _harmonics_in_alpha(fs[i]))}"
                                 f" Hz harmonic in the alpha band)" for i in at_risk)) if at_risk else ""
        print(f"  Strong alpha (~{alpha_peak:.1f} Hz, {alpha_ratio:.0f}x background): relax less / keep your "
              f"eyes on the squares{risk}.")

    failing = [i for i in range(len(fs)) if diag[i] < CLEAR]
    if not failing:
        verdict = "working"
        print("\n  -> SSVEP visible for every letter. Headset is working; tune software if accuracy is low.")
    elif diag.mean() < WEAK:
        verdict = "none"
        print("\n  -> No SSVEP response. Check electrode contact/placement before changing code.")
    else:
        verdict = "partial"
        print(f"\n  -> Partial response (weak: {', '.join(letters[i] for i in failing)}). Improve contact on "
              "the weakest electrodes, and check those frequencies are strong enough for you.")

    if plot_path:
        _plot(freqs, power, trials, rel_mean, session_name, plot_path)
        print(f"\n  Spectrum plot saved to {plot_path}")
    return {"verdict": verdict, "relative_power": rel_mean, "per_channel": per_channel,
            "alpha_ratio": alpha_ratio, "alpha_peak": alpha_peak}


def _plot(freqs, power, trials: Trials, rel_mean, session_name, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fs, letters, n = trials.freqs, trials.letters, trials.n_targets
    colors = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#42d4f4", "#f032e6", "#bfef45"]
    shown = _band(freqs, 3, FILTER_TOP)
    cols = min(n, 2)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 3.5 * rows), sharex=True, sharey=True, squeeze=False)
    for i, ax in enumerate(axes.flat):
        if i >= n:
            ax.axis("off")
            continue
        f = fs[i]
        this = power[..., trials.targets == i].mean(axis=(1, 2))
        rest = power[..., trials.targets != i].mean(axis=(1, 2))
        ax.semilogy(freqs[shown], rest[shown], color="0.6", lw=1, label="other letters")
        ax.semilogy(freqs[shown], this[shown], color=colors[i % len(colors)], lw=1.6, label=f"looking at {letters[i]}")
        for j, fj in enumerate(fs):
            ax.axvline(fj, color=colors[j % len(colors)], alpha=0.8 if j == i else 0.25, lw=1)
        if 2 * f < FILTER_TOP:
            ax.axvline(2 * f, color=colors[i % len(colors)], ls="--", alpha=0.6, lw=1)
        ax.axvspan(*ALPHA_BAND, color="0.85", alpha=0.3, lw=0)
        ax.set_title(f"Looking at {letters[i]} ({f:.2f} Hz): {rel_mean[i, i]:.2f}x at f")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(alpha=0.2)
    for ax in axes[-1]:
        ax.set_xlabel("Hz  (solid = flicker frequencies, dashed = 2nd harmonic, shaded = alpha band)")
    for ax in axes[:, 0]:
        ax.set_ylabel("power (mean of channels)")
    fig.suptitle(f"SSVEP spectrum check - {session_name}")
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
