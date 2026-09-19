"""
collect_training_data.py
========================

Record your own SSVEP training data for the NeuroPawn Knight board.

How it works
------------
The screen shows four flickering squares (one per frequency). For each
*block* you are cued to look at every target once, in a randomised order.
Each 1.5 s flicker window is captured from the board and saved to
`training_data/block_{block}_{trial}.csv`, where `trial` (1..4) encodes which
target you looked at. Collect several blocks so the classifier has enough
repeats to learn a stable spatial filter.

Tips for good data
------------------
* Look ONLY at the cued square and keep your gaze steady for the whole flash.
* Blink between trials (during the rest), not during the flicker.
* Collect data the same way you will use the BCI later -- same posture,
  lighting and electrode placement -- so the training set reflects reality.

Run it with:  python collect_training_data.py
Press Escape at any time to stop early (data already saved is kept).

No headset handy?
-----------------
  --synthetic   use BrainFlow's fake board instead of the Knight board, so the
                whole capture -> save loop runs (and writes real CSVs) with
                nothing plugged in.
  --no-board    stimulus only: show the flicker, record nothing, skip the board
                setup wait. This is the show-it-to-people demo mode.
"""

from __future__ import annotations

import argparse
import random

from ssvep import config as cfg
from ssvep.recording import RecordingProcess, null_recorder

# Number of times to cue every target. More blocks = better classifier,
# but a longer, more tiring session. 6-16 is a sensible range.
N_BLOCKS = 6


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--synthetic", action="store_true",
                    help="use BrainFlow's synthetic board instead of the Knight board")
    ap.add_argument("--no-board", action="store_true",
                    help="stimulus only: no board, nothing recorded, no setup wait")
    args = ap.parse_args()

    # Imported here, not at module scope: psychopy's preferences module parses
    # sys.argv the moment it is imported and swallows --help before argparse
    # ever gets to run.
    from psychopy import core
    from ssvep.stimulus import build_window, build_stimuli, show_message, run_trial

    if args.no_board:
        recorder, setup_wait = null_recorder(), 0
        banner = "Demo mode: flicker only.\nNothing is being recorded."
    else:
        # Background board reader in COLLECT mode. The wait lets the
        # channel-configuration sequence finish before the first trial --
        # the synthetic board has no channels to configure.
        # Synthetic blocks are 250 Hz, real Knight blocks are 125 Hz, and the
        # loader assumes 125 Hz. Keep them in separate directories so a dry run
        # can never poison a real training set.
        data_dir = cfg.TRAINING_DATA_DIR + ("_synthetic" if args.synthetic else "")
        recorder = RecordingProcess(mode="collect", data_dir=data_dir,
                                    variant="synthetic" if args.synthetic else None)
        setup_wait = 2 if args.synthetic else 30
        banner = f"Setting up the board...\nThis takes ~{setup_wait} seconds, please wait."
        print(f"saving blocks to {data_dir}")

    win = build_window()
    show_message(win, banner)
    recorder.start()
    core.wait(setup_wait)

    squares, cues = build_stimuli(win)

    try:
        for block in range(1, N_BLOCKS + 1):
            recorder.block_index.value = block

            show_message(win, f"Block {block} of {N_BLOCKS}\n\nGet ready...")
            core.wait(2)

            # Randomise the order targets are cued in this block. The label is
            # saved by target identity (not presentation order), so the data
            # stays correctly labelled while avoiding an order confound.
            order = list(range(cfg.N_TARGETS))
            random.shuffle(order)

            for target_idx in order:
                if not run_trial(win, squares, cues, target_idx, recorder):
                    raise KeyboardInterrupt

        show_message(win, "Done! Training data saved.\nYou can close this window.")
        core.wait(2)
    except KeyboardInterrupt:
        print("Stopped early by user.")
    finally:
        recorder.stop()
        recorder.join(timeout=5)
        win.close()
        core.quit()


if __name__ == "__main__":
    main()
