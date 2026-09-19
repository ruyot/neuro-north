"""
evaluate_trca.py
================
Offline check of how well TRCA classifies your calibration data, using
leave-one-block-out cross-validation. Needs no hardware.

    python evaluate_trca.py                                   # latest session
    python evaluate_trca.py --session training_data/session_20260919_020000
"""

from __future__ import annotations

import argparse

from ssvep.trca_model import cross_validate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--session", help="calibration folder (default: latest)")
    args = parser.parse_args()
    cross_validate(args.session)


if __name__ == "__main__":
    main()
