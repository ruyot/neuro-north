#!/bin/sh
cd "$(dirname "$0")" || exit 1
exec .venv/bin/python -m ssvep_training.validate_live \
  --channels 1,2,3,4 --decoder cca --stimulus motion \
  --swap-comparison --blocks 4 --idle-trials 0 --seed 20260924 "$@"
