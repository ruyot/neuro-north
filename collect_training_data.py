"""
collect_training_data.py
========================
Record labelled SSVEP calibration data for the Knight IMU board.

The screen shows four flickering squares (A/B/C/D, one frequency each). Each
*block* cues every square once, in random order. Each 1.5 s flicker window is
saved to training_data/session_<timestamp>/block_{block}_{trial}.csv, where
`trial` (1..4) encodes which square you were cued to look at.

Tips for good data
------------------
* Look ONLY at the red-outlined square and hold your gaze steady during the flash.
* Blink between trials (during the rest), not during the flicker.
* Sit the way you will when typing: same posture, lighting, screen distance.

Run it with:
    python collect_training_data.py                  # 6 blocks on the default port
    python collect_training_data.py --blocks 8 --port /dev/cu.usbserial-XXXX
Press Escape at any time to stop early (data already saved is kept).
"""

from __future__ import annotations

import argparse
import os
import random
import time

from ssvep import config as cfg
from ssvep.recording import RecordingProcess


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", default=cfg.SERIAL_PORT)
    parser.add_argument("--blocks", type=int, default=6, help="times to cue every target (6-16 is sensible)")
    parser.add_argument("--synthetic", action="store_true", help="fake board, to test the flow without hardware")
    parser.add_argument("--windowed", action="store_true", help="run in a window instead of fullscreen")
    args = parser.parse_args()

    if args.windowed:
        cfg.FULLSCREEN = False
    session_dir = os.path.join(cfg.TRAINING_DATA_DIR, time.strftime("session_%Y%m%d_%H%M%S"))

    # Start the board first: its ~34 s channel setup runs while the window opens.
    recorder = RecordingProcess(mode="collect", serial_port=args.port, data_dir=session_dir,
                                synthetic=args.synthetic)
    recorder.start()

    # Import PsychoPy only in the parent, after the child has been spawned.
    from psychopy import core
    from ssvep.stimulus import build_stimuli, build_window, run_trial, wait_for_board, wait_for_key

    win = build_window()
    completed = 0
    try:
        if not wait_for_board(win, recorder, "Setting up the Knight board (~35 s)...\nPlease wait."):
            raise KeyboardInterrupt

        squares, cues, labels = build_stimuli(win)
        for block in range(1, args.blocks + 1):
            recorder.block_index.value = block
            if not wait_for_key(win, f"Block {block} of {args.blocks}\n\n"
                                     "Look only at the square outlined in red.\n"
                                     "Blink between flashes, not during them.\n\n"
                                     "Press SPACE to start."):
                raise KeyboardInterrupt

            # Randomise the order targets are cued in this block. The label is
            # saved by target identity (not presentation order), so the data
            # stays correctly labelled while avoiding an order confound.
            order = list(range(cfg.N_TARGETS))
            random.shuffle(order)
            for target_idx in order:
                if not run_trial(win, squares, cues, target_idx, recorder, overlay=labels):
                    raise KeyboardInterrupt
            completed = block

        wait_for_key(win, f"Done! {completed} blocks saved.\n\nPress SPACE to close.")
    except KeyboardInterrupt:
        print("Stopped early by user.")
    finally:
        recorder.stop()
        recorder.join(timeout=10)
        win.close()
        if completed:
            print(f"\nSaved {completed} complete block(s) to {session_dir}")
            print("Next: python evaluate_trca.py")
        elif os.path.isdir(session_dir) and not os.listdir(session_dir):
            os.rmdir(session_dir)  # aborted before any trial was saved
        core.quit()


if __name__ == "__main__":
    main()
