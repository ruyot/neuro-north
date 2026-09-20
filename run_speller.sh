#!/bin/sh
# One short command keeps all options in the same invocation.
cd "$(dirname "$0")" || exit 1
PYTHON=${PYTHON:-.venv/bin/python}
if ! command -v "$PYTHON" >/dev/null 2>&1; then
  echo "Python environment not found: $PYTHON" >&2
  echo "Create .venv and install ssvep_training/requirements.txt, or set PYTHON." >&2
  exit 1
fi
if [ "${1:-}" = "--simulate" ]; then
  shift
  exec "$PYTHON" -m ssvep_training.speller_ui --keys --engine bigram --agent "$@"
fi
exec "$PYTHON" -m ssvep_training.speller_ui \
  --channels 1,2,3,4 --decoder trca \
  --session ssvep_training/training_data/session_20260920_012612 \
  --evidence-session ssvep_training/training_data/validation_20260920_014107 \
  --trca-min-peak 0.10 --trca-below winner --engine gpt2 --save --imu-debug --agent "$@"
