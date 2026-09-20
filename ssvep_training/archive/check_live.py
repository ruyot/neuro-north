"""
Did the live run actually contain an SSVEP?

Live trials carry no label (marker 99), so accuracy is not computable -- but
"is there a response at all" is, and that is the question worth asking when
typing feels worse than calibration says it should. Everything here is measured
against decoy frequencies that were never on screen, so it needs no labels.

    python -m ssvep_training.check_live                       # newest live run
    python -m ssvep_training.check_live --against <session>   # ...vs a calibration
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np

from . import cca_model
from . import config as cfg
from .session import epoch_at, load_session


def live_epochs(path: str):
    """Every live selection in a saved speller run, cut exactly as it was live."""
    data, meta = load_session(path)
    rate, eeg = meta["rate"], data[meta["eeg_rows"]]
    markers = data[meta["marker_row"]]
    epochs = [epoch_at(eeg, int(o), rate, gaze=min(cfg.GAZE_DURATION, meta["flicker_duration"] - cfg.VISUAL_LATENCY)) for o in np.flatnonzero(markers == cfg.LIVE_MARKER)]
    return [e for e in epochs if e is not None], meta, rate


def describe(name: str, epochs, freqs, rate: int, names) -> None:
    H = cca_model.harmonics_for(freqs)
    sigma = np.array([cca_model.confidence(e, rate, freqs, H) for e in epochs])
    S = np.array([cca_model.scores(e, rate, freqs, H) for e in epochs])
    margin = np.abs(S[:, 0] - S[:, 1]) if len(freqs) == 2 else np.zeros(len(S))
    rms = np.array([np.asarray(e).std(axis=0) for e in epochs]).mean(axis=0)

    print(f"\n{name}: {len(epochs)} epochs")
    print(f"  confidence   mean {sigma.mean():+.2f} contrast   median {np.median(sigma):+.2f}"
          f"   contrast > 2 {int((sigma > 2).sum())}/{len(sigma)}")
    print(f"  margin       mean {margin.mean():+.3f}")
    print("  per-channel rms  " + "  ".join(f"{n} {v:.0f}" for n, v in zip(names, rms)))
    picked = S.argmax(1)
    share = " ".join(f"{cfg.TARGET_LETTERS[i]} {100*np.mean(picked == i):.0f}%" for i in range(len(freqs)))
    print(f"  picks split  {share}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", help="a saved live run (default: the newest live_* folder)")
    ap.add_argument("--against", help="a calibration session to compare against")
    args = ap.parse_args()

    path = args.session
    if not path:
        runs = sorted(glob.glob(os.path.join(cfg.TRAINING_DATA_DIR, "live_*")), reverse=True)
        if not runs:
            sys.exit("No live runs saved. Run the speller with --save first.")
        path = runs[0]

    epochs, meta, rate = live_epochs(path)
    if not epochs:
        sys.exit(f"No usable live trials in {path}.")
    describe(f"LIVE  {os.path.basename(os.path.normpath(path))}",
             epochs, meta["frequencies"], rate, meta["names"])

    if args.against:
        from .session import load_trials
        t = load_trials(args.against)
        cued = [t.eeg[..., k] for k in range(t.eeg.shape[-1])]
        describe(f"CUED  {os.path.basename(os.path.normpath(args.against))}",
                 cued, t.freqs, t.rate, t.names)
        print("\n  These are descriptive scores only. Unlabeled runs cannot establish accuracy,\n"
              "  and decoy contrast cannot prove that a response is neural. Use validate_live.")


if __name__ == "__main__":
    main()
