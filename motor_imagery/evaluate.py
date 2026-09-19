"""Score the decoders on a calibration session, then save the best one.

    python -m motor_imagery.evaluate                    # every model, latest session
    python -m motor_imagery.evaluate --model tangent    # just one
    python -m motor_imagery.evaluate --sweep            # try other analysis windows
    python -m motor_imagery.evaluate --save tangent     # fit on everything and save

Folds hold out whole BLOCKS, not random trials. Trials recorded seconds apart
share drift, electrode state and attention, so shuffling them across a split
inflates the score - which is how the tutorial's own report is produced.

Everything a model learns is fitted inside the fold: spatial filters, the
Riemannian mean, the scalers, and (for the stack) the meta-learner too. The
margin is tuned on the held-out predictions, never on the training trials.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
from sklearn.model_selection import LeaveOneGroupOut

from . import config as cfg
from . import model as models
from .session import latest_session, load_trials, model_path


def cross_validate(trials, name: str, folds: int | None = None):
    """Out-of-fold probabilities for every trial, by held-out block."""
    groups = trials.blocks
    unique = np.unique(groups)
    if len(unique) < 2:
        raise SystemExit("need at least 2 blocks to cross-validate - record more")
    if folds and len(unique) > folds:               # group blocks together into `folds` folds
        groups = np.searchsorted(unique, groups) % folds

    classes = np.unique(trials.labels)
    proba = np.zeros((len(trials), len(classes)))
    for train, test in LeaveOneGroupOut().split(np.zeros(len(trials)), trials.labels, groups):
        if len(np.unique(trials.labels[train])) < len(classes):
            raise SystemExit("a fold is missing a class - every block needs every cue")
        fitted = models.build(name, trials.rate, trials.mode).fit(
            {b: x[train] for b, x in trials.bands.items()}, trials.labels[train])
        columns = [list(fitted.classes_).index(c) for c in classes]
        proba[test] = fitted.predict_proba({b: x[test] for b, x in trials.bands.items()})[:, columns]
    return proba, classes


def report(trials, proba, classes, name: str) -> dict:
    """Accuracy, the confusion matrix, and what the margin costs."""
    labels, class_names = trials.labels, [cfg.CLASS_NAMES[c] for c in classes]
    argmax = classes[np.argmax(proba, axis=1)]
    accuracy = float((argmax == labels).mean())
    chance = 1.0 / len(classes)

    print(f"\n=== {name} ===")
    print(f"  argmax accuracy {100 * accuracy:.1f}%   (chance {100 * chance:.0f}%, "
          f"{len(trials)} trials, {len(np.unique(trials.blocks))} blocks)")

    width = max(len(n) for n in class_names) + 2
    print("\n  cued -> predicted" + "".join(f"{n:>8}" for n in class_names) + "   correct")
    for row, cued in enumerate(classes):
        mask = labels == cued
        counts = [int(((argmax == other) & mask).sum()) for other in classes]
        correct = 100 * counts[row] / max(mask.sum(), 1)
        print(f"  {cfg.CLASS_NAMES[cued]:<{width}}" + "".join(f"{c:>8}" for c in counts)
              + f"{correct:>9.0f}%")

    tuned = models.tune_margin(proba, labels, classes)
    plain = models.score_margin(proba, labels, classes, 0.0)
    print(f"\n  no margin   {plain}")
    print(f"  tuned       {tuned}")
    if cfg.REST not in classes:
        print("  (no rest trials in this session: the margin is a plain confidence floor)")
    return {"model": name, "accuracy": accuracy, "chance": chance, "margin": tuned.margin,
            "margin_accuracy": tuned.accuracy, "false_pick": tuned.false_pick,
            "missed": tuned.missed, "trials": len(trials), "classes": class_names}


def sweep(session: str, name: str, folds: int | None) -> None:
    """The window is a guess until it is measured: try a few placements."""
    print(f"\n=== window sweep ({name}) ===")
    print("  start  length   accuracy")
    for start in (0.0, 0.25, 0.5, 0.75, 1.0):
        for duration in (1.0, 1.5, 2.0):
            if start + duration > cfg.HOLD_DURATION:
                continue
            trials = load_trials(session, start, duration)
            proba, classes = cross_validate(trials, name, folds)
            argmax = classes[np.argmax(proba, axis=1)]
            print(f"  {start:5.2f}  {duration:5.2f}   {100 * (argmax == trials.labels).mean():6.1f}%")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--session", help="calibration folder (default: latest)")
    parser.add_argument("--model", action="append", choices=models.MODELS,
                        help="model to score (repeatable; default: all)")
    parser.add_argument("--folds", type=int, default=None,
                        help="group blocks into this many folds (default: leave one block out)")
    parser.add_argument("--start", type=float, default=None, help="window start after the cue (s)")
    parser.add_argument("--duration", type=float, default=None, help="window length (s)")
    parser.add_argument("--sweep", action="store_true", help="try several analysis windows")
    parser.add_argument("--save", nargs="?", const=cfg.DEFAULT_MODEL, choices=models.MODELS,
                        help="refit this model on every trial and save it beside the session")
    args = parser.parse_args()

    session = args.session or latest_session()
    if not session:
        sys.exit(f"No calibration in {cfg.TRAINING_DATA_DIR} - run "
                 "python -m motor_imagery.collect first.")

    trials = load_trials(session, args.start, args.duration)
    print(f"Session {session}")
    print(cfg.montage_banner(trials.meta.get("eeg_rows")))
    print(f"  mode {trials.mode}, {len(trials)} trials {trials.counts()}, "
          f"{trials.rate} Hz, window {args.start or cfg.ANALYSIS_START:g}-"
          f"{(args.start or cfg.ANALYSIS_START) + (args.duration or cfg.ANALYSIS_DURATION):g} s")
    if trials.skipped:
        print(f"  {trials.skipped} marker(s) skipped (inside the {cfg.PRIME_SECONDS:g} s filter "
              "priming stretch, or the run ended mid-trial)")
    if not len(trials):
        sys.exit("No usable trials in that session.")
    if trials.mode == "clench":
        print("  NOTE: clench trials contain muscle activity. Treat this score as an upper "
              "bound;\n        record an --mode imagine session for the real number.")

    results = []
    for name in (args.model or models.MODELS):
        started = time.time()
        proba, classes = cross_validate(trials, name, args.folds)
        result = report(trials, proba, classes, name)
        result["seconds"] = round(time.time() - started, 1)
        results.append(result)

    if len(results) > 1:
        print("\n=== summary (held-out blocks) ===")
        print(f"  {'model':<12}{'accuracy':>10}{'margin':>9}{'hands ok':>10}{'rest typed':>12}{'fit s':>8}")
        for r in sorted(results, key=lambda r: -r["accuracy"]):
            print(f"  {r['model']:<12}{100 * r['accuracy']:>9.1f}%{r['margin']:>9.2f}"
                  f"{100 * r['margin_accuracy']:>9.0f}%{100 * r['false_pick']:>11.0f}%"
                  f"{r['seconds']:>8.1f}")
        print(f"\n  Pick the winner, then: python -m motor_imagery.evaluate --save <model>")

    if args.sweep:
        sweep(session, (args.model or [cfg.DEFAULT_MODEL])[0], args.folds)

    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)
    out = os.path.join(cfg.RESULTS_DIR, f"eval_{os.path.basename(session)}.json")
    with open(out, "w") as f:
        json.dump({"session": session, "mode": trials.mode, "window":
                   [args.start or cfg.ANALYSIS_START, args.duration or cfg.ANALYSIS_DURATION],
                   "results": results}, f, indent=2)
    print(f"\nSummary saved to {out}")

    if args.save:
        save_model(session, trials, args.save, args.folds,
                   [args.start or cfg.ANALYSIS_START, args.duration or cfg.ANALYSIS_DURATION])


def save_model(session: str, trials, name: str, folds: int | None, window: list[float]) -> None:
    """Refit on every trial and store it with the margin from held-out folds."""
    from joblib import dump

    proba, classes = cross_validate(trials, name, folds)
    tuned = models.tune_margin(proba, trials.labels, classes)
    fitted = models.build(name, trials.rate, trials.mode).fit(trials.bands, trials.labels)
    path = model_path(session)
    dump({"model": fitted, "classes": np.array(fitted.classes_), "margin": tuned.margin,
          "name": name, "rate": trials.rate, "mode": trials.mode,
          "window": window,          # live must cut the same slice this was fitted on
          "names": trials.names, "cv": tuned.__dict__}, path)
    print(f"\nSaved {name} (margin {tuned.margin:.2f}) to {path}")
    print("Next: python -m motor_imagery.speller")


if __name__ == "__main__":
    main()
