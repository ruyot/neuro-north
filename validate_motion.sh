#!/bin/sh
cd "$(dirname "$0")" || exit 1
# Resolve once before opening the board/display. Validation never trains on
# its own trials; only this saved motion calibration supplies the templates.
motion_session=$(.venv/bin/python -c '
from ssvep_training import config as cfg
from ssvep_training.session import latest_session
cfg.configure_stimulus("motion")
path = latest_session()
if not path:
    raise SystemExit("No motion calibration found; run sh train_motion.sh first")
print(path)
') || exit 1
echo "Frozen motion calibration: $motion_session"
exec .venv/bin/python -m ssvep_training.validate_live \
  --channels 1,2,3,4 --decoder trca --stimulus motion \
  --session "$motion_session" --blocks 10 --idle-trials 2 \
  --seed 20260923 "$@"
