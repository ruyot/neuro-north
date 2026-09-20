#!/bin/sh
cd "$(dirname "$0")" || exit 1
exec .venv/bin/python -m ssvep_training.collect_training_data --channels 1,2,3,4 --blocks 12 --seed 20260922 --stimulus motion "$@"
