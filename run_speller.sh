#!/bin/sh
# One short command keeps all options in the same invocation.
cd "$(dirname "$0")" || exit 1
exec .venv/bin/python -m ssvep_training.speller_ui \
  --channels 1,2,3,4 --decoder cca --engine gpt2 --save --imu-debug "$@"
