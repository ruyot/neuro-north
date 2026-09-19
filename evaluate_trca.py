"""
evaluate_trca.py
================
Offline check of a calibration session. Needs no hardware.

1. SSVEP check: did each flicker frequency actually show up in your EEG?
   Prints a relative-power table and saves results/spectrum_<session>.png.
2. TRCA accuracy: leave-one-block-out cross-validation (accuracy + ITR).

    python evaluate_trca.py                                   # latest session
    python evaluate_trca.py --session training_data/session_20260919_020000
    python evaluate_trca.py --synthetic                       # latest fake-board session
"""

from __future__ import annotations

import argparse
import os

from ssvep import config as cfg
from ssvep.diagnostics import report
from ssvep.trca_model import count_blocks, cross_validate, load_training_data, resolve_data_dir, session_rate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--session", help="calibration folder (default: latest)")
    parser.add_argument("--synthetic", action="store_true", help="use the latest fake-board session")
    parser.add_argument("--no-plot", action="store_true", help="skip saving the spectrum plot")
    args = parser.parse_args()

    data_dir = resolve_data_dir(args.session, args.synthetic)
    session = os.path.basename(os.path.normpath(data_dir))
    n_blocks = count_blocks(data_dir)
    rate = session_rate(data_dir)
    print(f"Session: {session} ({n_blocks} complete blocks, {rate} Hz)\n")

    eeg, labels = load_training_data(data_dir, n_blocks, crop=False)  # full 1.5 s trials
    plot_path = None if args.no_plot else os.path.join(cfg.RESULTS_DIR, f"spectrum_{session}.png")
    report(eeg, labels, session, rate, plot_path)

    print("\n" + "-" * 70 + "\n")
    if n_blocks >= 2:
        cross_validate(data_dir, n_blocks)
    else:
        print("Need at least 2 blocks for the accuracy check.")


if __name__ == "__main__":
    main()
