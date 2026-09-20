# How well can the system tell which square you looked at? 

from __future__ import annotations

import argparse
import json
import os
import sys

from . import config as cfg
from .diagnostics import report
from .session import latest_session, load_trials
from .cca_model import evaluate as cca_evaluate, harmonics_for
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

    full = load_trials(path, full=True)       # whole 1.5 s flicker, for the spectrum
    trials = load_trials(path)                # the 1 s analysis window, for TRCA
    blocks = sorted(set(trials.blocks.tolist()))
    targets = ", ".join(f"{l} {f:.2f} Hz" for l, f in zip(trials.letters, trials.freqs))
    print(f"Session {name}: {trials.eeg.shape[-1]} trials, {len(blocks)} blocks, {trials.rate} Hz, targets {targets}"
          + (f", {trials.skipped} skipped" if trials.skipped else "") + "\n")
    if trials.eeg.shape[-1] == 0:
        sys.exit("No usable trials in this session.")

    plot = None if args.no_plot else os.path.join(cfg.RESULTS_DIR, f"spectrum_{name}.png")
    spectrum = report(full, name, plot)
    print("\n" + "-" * 72 + "\n")

    summary = {"session": path, "trials": int(trials.eeg.shape[-1]), "blocks": len(blocks),
               "spectrum_verdict": spectrum["verdict"], "alpha_ratio": spectrum["alpha_ratio"]}
    if len(blocks) >= 2:
        # CCA first: it trains on nothing, so it says whether the signal is
        # separable at all before any model gets a chance to paper over it.
        cca, stats = cca_evaluate(trials)
        bank = f"{cfg.CCA_BANDS}-band filter bank" if cfg.CCA_BANDS > 1 else "single band"
        print_cv(cca, f"CCA ({bank}, {harmonics_for(trials.freqs)} harmonic(s), no training)")
        print(f"\n  mean margin   {stats['margin']:+.3f}   (winner minus runner-up)")
        print(f"  confidence    {stats['sigma']:+.2f} heuristic decoy contrast "
              f"(correct picks {stats['sigma_correct']:+.2f}, wrong {stats['sigma_wrong']:+.2f})")
        print("                 Heuristic contrast only: not a z-score, probability, or proof of SSVEP.")
        summary.update(cca_accuracy=cca.mean_accuracy, cca_bits_per_min=float(cca.itr.mean()),
                       **{f"cca_{k}": v for k, v in stats.items()})
        print("\n" + "-" * 72 + "\n")

        cv = cross_validate(trials)
        print_cv(cv)
        pair = standout_confusion(cv)
        if pair:
            print("\n  " + explain_confusion(cv, *pair))
        summary.update(trca_accuracy=cv.mean_accuracy, trca_bits_per_min=float(cv.itr.mean()),
                       confusion=cv.confusion.tolist())
    else:
        print("Need at least 2 blocks for the accuracy test.")

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
