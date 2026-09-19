# How well can the system tell which square you looked at? 

from __future__ import annotations

import argparse
import json
import os
import sys

from . import config as cfg
from .diagnostics import report
from .session import latest_session, load_trials
from .trca_model import cross_validate, print_cv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--session", help="a specific session folder")
    parser.add_argument("--no-plot", action="store_true", help="skip the spectrum plot")
    args = parser.parse_args()

    path = args.session or latest_session()
    if not path:
        sys.exit(f"No sessions in {cfg.TRAINING_DATA_DIR} - run "
                 "python -m ssvep_training.collect_training_data first.")
    name = os.path.basename(os.path.normpath(path))

    try:
        full = load_trials(path, full=True)   # whole flicker, for the spectrum
        trials = load_trials(path)           # analysis window, for TRCA
    except ValueError as exc:
        sys.exit(f"Cannot evaluate session {name}: {exc}")
    blocks = sorted(set(trials.blocks.tolist()))
    targets = ", ".join(f"{l} {f:.2f} Hz" for l, f in zip(trials.letters, trials.freqs))
    accepted = int(trials.eeg.shape[-1])
    counts = {letter: int((trials.targets == t).sum()) for t, letter in enumerate(trials.letters)}
    print(f"Session {name}: {accepted} accepted, {trials.skipped} rejected, {len(blocks)} blocks, "
          f"{trials.rate} Hz, targets {targets}")
    print("Accepted by target: " + ", ".join(f"{letter} {count}" for letter, count in counts.items()))
    reasons = ", ".join(f"{reason}={count}" for reason, count in sorted(trials.rejection_counts.items()))
    print("Rejection flags (can overlap): " + (reasons or "none") + "\n")

    summary = {"session": path, "trials": accepted, "rejected": trials.skipped,
               "rejection_counts": trials.rejection_counts, "accepted_per_target": counts, "blocks": len(blocks)}
    try:
        missing = [f"{letter} ({count})" for letter, count in counts.items() if count == 0]
        if missing:
            raise ValueError("No accepted trials for " + ", ".join(missing) + "; spectra and TRCA skipped.")
        plot = None if args.no_plot else os.path.join(cfg.RESULTS_DIR, f"spectrum_{name}.png")
        spectrum = report(full, name, plot)
        summary.update(spectrum_verdict=spectrum["verdict"], alpha_ratio=spectrum["alpha_ratio"])
        print("\n" + "-" * 72 + "\n")
        if len(blocks) < 3:
            raise ValueError(f"Need at least 3 blocks for the accuracy test, found {len(blocks)}")
        cv = cross_validate(trials)
    except ValueError as exc:
        summary["trca_error"] = str(exc)
        print(f"TRCA unavailable: {exc}")
    else:
        print_cv(cv)
        pair = standout_confusion(cv)
        if pair:
            print("\n  " + explain_confusion(cv, *pair))
        summary.update(trca_accuracy=cv.mean_accuracy, trca_bits_per_min=float(cv.itr.mean()),
                       confusion=cv.confusion.tolist())

    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)
    out = os.path.join(cfg.RESULTS_DIR, f"eval_{name}.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {out}")


def standout_confusion(cv):
    """The pair of letters mixed up far more than any other, if the system otherwise works.
    None on pure noise (every pair confused about equally) and with only 2 targets
    (a single pair can't stand out from others)."""
    m, n = cv.confusion, cv.n_targets
    pairs = {(i, j): m[i, j] + m[j, i] for i in range(n) for j in range(i + 1, n)}
    if len(pairs) < 2:
        return None
    top = max(pairs, key=pairs.get)
    count = pairs.pop(top)
    others = list(pairs.values())
    if (count >= 2 and count > max(others) and count >= 2 * (sum(others) / len(others))
            and cv.mean_accuracy >= 100 / n + 15):
        return top + (count,)
    return None


def explain_confusion(cv, i: int, j: int, count: int) -> str:
    li, lj = cv.letters[i], cv.letters[j]
    fi, fj = cv.freqs[i], cv.freqs[j]
    text = f"{li} and {lj} are confused more than any other pair ({count}x; {fi:.2f} vs {fj:.2f} Hz). "
    for a, b in ((fi, fj), (fj, fi)):
        for k in (2, 3, 4):
            if abs(k * a - b) < 1.0:
                return text + (f"{a:.2f} Hz x {k} = {k * a:.1f} Hz sits next to {b:.2f} Hz - move one of "
                               "them in ssvep_training/config.py.")
    if abs(fi - fj) < 3.0:
        return text + f"They're only {abs(fi - fj):.1f} Hz apart - spread them further in ssvep_training/config.py."
    return text + "Check both squares are clearly visible and the flicker has no late frames."


if __name__ == "__main__":
    main()
