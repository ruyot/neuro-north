"""
evaluate_trca.py
================

Offline sanity check for your recorded training data. Runs leave-one-block-out
cross-validation and prints per-block accuracy plus the mean accuracy and
information transfer rate (ITR). No board or screen required.

Run it with:  python evaluate_trca.py [--data-dir DIR]
(--data-dir defaults to config.TRAINING_DATA_DIR)
"""

import argparse

from ssvep import config as cfg
from ssvep.trca_model import cross_validate

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=cfg.TRAINING_DATA_DIR,
                        help="directory of block_{b}_{t}.csv files")
    cross_validate(parser.parse_args().data_dir)
