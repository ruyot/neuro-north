# How well can the system tell which square you looked at? 

from __future__ import annotations

import argparse
import json
import os
import sys
from contextlib import redirect_stdout
from io import StringIO
from time import perf_counter
from zipfile import BadZipFile

import numpy as np

from . import config as cfg
from .diagnostics import report
from .session import Trials, latest_session, load_session, load_trials
from .trca_model import CVResult, _require_training_trials, bits_per_min, cross_validate, fit, print_cv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--session", help="a specific session folder")
    parser.add_argument("--no-plot", action="store_true", help="skip the spectrum plot")
    parser.add_argument("--compare-fbcca", action="store_true", help="compare TRCA, FBCCA and decoder-local CAR")
    parser.add_argument("--test-session", help="separate held-out session; requires --compare-fbcca and --session")
    args = parser.parse_args()
    if args.test_session is not None and (not args.compare_fbcca or args.session is None):
        parser.error("--test-session requires --compare-fbcca and an explicit --session")

    if not args.compare_fbcca:
        path = args.session or latest_session()
        if not path:
            sys.exit(f"No sessions in {cfg.TRAINING_DATA_DIR} - run "
                     "python -m ssvep_training.collect_training_data first.")
        name = os.path.basename(os.path.normpath(path))
        try:
            full = load_trials(path, full=True)
            trials = load_trials(path)
        except ValueError as exc:
            sys.exit(f"Cannot evaluate session {name}: {exc}")
        summary = _session_summary(path, trials)
        try:
            missing = [f"{letter} ({count})" for letter, count in summary["accepted_per_target"].items() if count == 0]
            if missing:
                raise ValueError("No accepted trials for " + ", ".join(missing) + "; spectra and TRCA skipped.")
            _add_spectrum(summary, full, args.no_plot)
            if summary["blocks"] < 3:
                raise ValueError(f"Need at least 3 blocks for the accuracy test, found {summary['blocks']}")
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
        out = os.path.join(cfg.RESULTS_DIR, f"eval_{name}.json")
    else:
        try:
            path = args.session if args.session is not None else latest_session()
            if not path:
                raise ValueError(f"No sessions in {cfg.TRAINING_DATA_DIR} - run "
                                 "python -m ssvep_training.collect_training_data first.")
            if args.test_session is not None and os.path.realpath(path) == os.path.realpath(args.test_session):
                raise ValueError("Calibration and test must be distinct resolved session paths.")
            full = load_trials(path, full=True)   # whole flicker, for the spectrum
            trials = load_trials(path)           # identical admitted windows for every decoder
            test = test_full = test_meta = None
            raw, meta = load_session(path)
            if args.test_session is not None:
                test_raw, test_meta = load_session(args.test_session)
                if np.array_equal(raw, test_raw, equal_nan=True):
                    raise ValueError("Calibration and test contain the same saved raw-array data.")
                del test_raw
                test_full = load_trials(args.test_session, full=True)
                test = load_trials(args.test_session)
            del raw
            _preflight(trials, test)
        except (OSError, ValueError, KeyError, EOFError, BadZipFile) as exc:
            parser.error(f"Cannot evaluate sessions: {exc}")

        summary = _session_summary(path, trials, meta)
        _add_spectrum(summary, full, args.no_plot)
        inputs = {"calibration": summary}
        suffix = os.path.basename(os.path.normpath(path))
        if test is not None:
            test_summary = _session_summary(args.test_session, test, test_meta)
            _add_spectrum(test_summary, test_full, args.no_plot)
            inputs["test"] = test_summary
            suffix += "_to_" + os.path.basename(os.path.normpath(args.test_session))
        print("ITR is a closed-set estimate, not measured free-spelling throughput.")
        print("Idle/no-control detection is not assessed.\n")
        summary = {
            "session": path, "test_session": args.test_session,
            "mode": "held_out_session" if test is not None else "leave_one_block_out",
            "inputs": inputs, "filterbank": cfg.FILTERBANK,
            "use_ensemble_trca": cfg.USE_ENSEMBLE_TRCA,
            "itr_kind": "closed_set_estimate",
            "comparisons": _compare(trials, test),
        }
        out = os.path.join(cfg.RESULTS_DIR, f"compare_{suffix}.json")

    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)
    with open(out, "w") as f:
        json.dump(summary, f, indent=2, allow_nan=not args.compare_fbcca)
    print(f"\nSummary saved to {out}")
    if args.compare_fbcca and all("error" in entry for entry in summary["comparisons"].values()):
        sys.exit("All decoders are unavailable; the saved comparison contains errors only.")


def _preflight(calibration: Trials, test: Trials | None) -> None:
    for role, trials in (("calibration", calibration), ("test", test)):
        if trials is None:
            continue
        missing = [letter for target, letter in enumerate(trials.letters)
                   if not np.any(trials.targets == target)]
        if missing:
            raise ValueError(f"No accepted {role} trials for " + ", ".join(missing) + "; evaluation skipped.")
    _require_training_trials(calibration)
    if test is not None:
        for field in ("rate", "rows", "names", "freqs", "letters"):
            if getattr(calibration, field) != getattr(test, field):
                raise ValueError(f"Calibration and test must have identical {field} and ordering.")
        if calibration.eeg.shape[:2] != test.eeg.shape[:2]:
            raise ValueError("Calibration and test must have identical sample/channel window shapes.")
    reporting = calibration if test is None else test
    blocks = sorted(set(reporting.blocks.tolist()))
    if len(blocks) < 3:
        raise ValueError(f"Need at least 3 populated reporting blocks for the accuracy test, found {len(blocks)}")
    if test is None:
        # Check every training fold before any decoder runs, not only as CV reaches it.
        for held_out in blocks:
            train = calibration.blocks != held_out
            fold = Trials(calibration.eeg[..., train], calibration.targets[train], calibration.blocks[train],
                          calibration.rate, calibration.names, calibration.freqs, calibration.letters,
                          calibration.rows)
            try:
                _require_training_trials(fold)
            except ValueError as exc:
                raise ValueError(f"Cannot hold out block {held_out}: {exc}") from exc


def _session_summary(path: str, trials: Trials, meta: dict | None = None) -> dict:
    name = os.path.basename(os.path.normpath(path))
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
    if meta is not None:
        summary.update({key: meta[key] for key in (
            "schema_version", "board_id", "rate", "eeg_rows", "names", "frequencies", "letters",
            "preprocessing", "brainflow_version", "meegkit_version", "gain_requested",
            "raw_eeg_unit", "model_eeg_unit", "display_refresh_hz",
        )})
        summary.update(resolved_path=os.path.realpath(path), window_shape=list(trials.eeg.shape[:2]))
        if "evidence_kind" in meta:
            summary["evidence_kind"] = meta["evidence_kind"]
    return summary


def _add_spectrum(summary: dict, full: Trials, no_plot: bool) -> None:
    name = os.path.basename(os.path.normpath(summary["session"]))
    plot = None if no_plot else os.path.join(cfg.RESULTS_DIR, f"spectrum_{name}.png")
    spectrum = report(full, name, plot)
    summary.update(spectrum_verdict=spectrum["verdict"], alpha_ratio=float(spectrum["alpha_ratio"]))
    print("\n" + "-" * 72 + "\n")


def _compare(calibration: Trials, test: Trials | None) -> dict:
    from .fbcca import FBCCA

    evaluated = calibration if test is None else test
    blocks = sorted(set(evaluated.blocks.tolist()))
    comparisons = {}
    if test is not None:
        print("TRCA fits once on calibration; FBCCA variants do not fit templates.\n")
    for decoder in cfg.DECODERS:
        timings = []
        block_order = iter(blocks)
        failed_block = invalid_predictions = None

        def classify(train: Trials, eeg: np.ndarray) -> np.ndarray:
            nonlocal failed_block, invalid_predictions
            failed_block = next(block_order) if test is None else None
            invalid_predictions = None
            if decoder == "trca":
                model = fit(train)
            else:
                model = FBCCA(train.rate, train.freqs, car=decoder == "fbcca-car")
            outputs = []
            # Finish the block/session vector before validating any returned labels.
            for index in range(eeg.shape[-1]):
                epoch = eeg[..., index:index + 1]
                start = perf_counter()
                output = model.predict(epoch)
                elapsed = perf_counter() - start
                timings.append(elapsed * 1000)
                outputs.append(output)
            predictions = []
            for output in outputs:
                try:
                    value = np.asarray(output)
                except ValueError:
                    continue  # A malformed output makes this whole result unavailable below.
                if value.shape != (1,) or not np.issubdtype(value.dtype, np.number) or np.iscomplexobj(value):
                    continue
                label = value[0]
                if np.isfinite(label) and label == np.floor(label) and 0 <= label < train.n_targets:
                    predictions.append(label)
            invalid_predictions = len(outputs) - len(predictions)
            if len(predictions) != eeg.shape[-1]:
                raise ValueError(f"{invalid_predictions} unusable predictions among {eeg.shape[-1]} windows; "
                                 f"expected one finite integer label in [0, {train.n_targets - 1}] per window.")
            return np.asarray(predictions).astype(np.intp, copy=False)

        try:
            if test is None:
                cv = cross_validate(calibration, classify=classify)
            else:
                predicted = classify(calibration, test.eeg)
                confusion = np.zeros((test.n_targets, test.n_targets), dtype=int)
                np.add.at(confusion, (test.targets, predicted), 1)
                accuracy = np.array([100 * np.mean(predicted[test.blocks == block] == test.targets[test.blocks == block])
                                     for block in blocks])
                itr = np.array([bits_per_min(acc / 100, test.n_targets) for acc in accuracy])
                cv = CVResult(blocks, accuracy, itr, confusion, test.freqs, test.letters)
        except (ValueError, np.linalg.LinAlgError, FloatingPointError) as exc:
            comparisons[decoder] = {
                "error": str(exc), "failed_block": failed_block,
                "count_scope": "first_failed_block" if test is None else "test_session",
                "invalid_predictions": invalid_predictions,
            }
            print(f"{decoder} unavailable: {exc}\n")
            continue

        if test is None:
            print_cv(cv, title=decoder)
        else:
            # Reuse the statistics printer without calling reporting groups training folds.
            with redirect_stdout(StringIO()) as printed:
                print_cv(cv, title=decoder)
            print(f"{decoder} - held-out test session ({len(blocks)} reporting blocks, not training folds):")
            print(printed.getvalue().split("\n", 1)[1], end="")
        recall = 100 * cv.confusion.diagonal() / cv.confusion.sum(axis=1)
        balanced = float(recall.mean())
        latency = float(np.median(timings))
        print("  Per-target recall: " + ", ".join(f"{letter} {value:.1f}%"
                                                for letter, value in zip(cv.letters, recall)))
        print(f"  Balanced accuracy {balanced:.1f}%   median single-window prediction {latency:.3f} ms\n")
        comparisons[decoder] = {
            "blocks": cv.blocks, "accuracy": cv.accuracy.tolist(), "confusion": cv.confusion.tolist(),
            "per_target_recall": recall.tolist(), "balanced_accuracy": balanced,
            "median_predict_ms": latency, "itr_bits_per_min": cv.itr.tolist(),
        }
    return comparisons


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
