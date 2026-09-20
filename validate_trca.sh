#!/bin/sh
cd "$(dirname "$0")" || exit 1
exec .venv/bin/python -m ssvep_training.validate_live \
  --channels 1,2,3,4 --decoder trca \
  --session ssvep_training/training_data/session_20260920_012612 \
  --trca-min-peak 0.10 --blocks 10 --idle-trials 6 --seed 20260921 "$@"
