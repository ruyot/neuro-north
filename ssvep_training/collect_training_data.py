"""
Record labelled SSVEP trials so TRCA can learn your specific responses.

All squares flash at once, one frequency each (config.py: currently A 15 Hz left, B 20 Hz right)
    python -m ssvep_training.collect_training_data --blocks 8
"""

from __future__ import annotations

import argparse
import os
import random
import time

from . import config as cfg
from .recording import RecordingProcess
from .session import encode_marker


class Quit(Exception):
    """Run ended early: Escape pressed, or the board never started streaming."""

WARNING = (f"This screen FLASHES at {min(cfg.STIMULUS_FREQUENCIES):g}-{max(cfg.STIMULUS_FREQUENCIES):g} Hz.\n\n"
           "Do not use it if you have epilepsy or have ever had a seizure,\n"
           "and stop immediately if you feel unwell.\n\n"
           "Press SPACE to continue, Escape to quit.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", help="override PORT_PATH from .env")
    parser.add_argument("--blocks", type=int, default=8, help="times every square is cued (6-16 sensible)")
    parser.add_argument("--windowed", action="store_true", help="run in a window instead of fullscreen")
    parser.add_argument("--channels", default="",
                        help="board channels to record, e.g. 1,2,3,4 (default: all 8). Dead "
                             "electrodes also sit in the bias loop, so excluding them helps every channel.")
    args = parser.parse_args()
    channels = [int(c) for c in args.channels.split(",")] if args.channels else None

    session_dir = os.path.join(cfg.TRAINING_DATA_DIR, time.strftime("session_%Y%m%d_%H%M%S"))
    # Start the board first: its setup runs while the window opens.
    recorder = RecordingProcess(port=args.port, session_dir=session_dir, channels=channels)
    recorder.start()

    # PsychoPy is imported only in the display process, after the board process exists.
    from psychopy import core
    from .stimulus import build_stimuli, build_window, run_trial, wait_for_board, wait_for_key

    win = build_window(fullscreen=not args.windowed)
    print("Press SPACE (or click) in the flicker window to continue at each screen; Escape quits.")
    completed = 0
    try:
        if not wait_for_key(win, WARNING):
            raise Quit
        if not wait_for_board(win, recorder, "Setting up the Knight board (~8 s)..."):
            raise Quit("the board never started streaming - see [board] above")

        squares, cues, labels = build_stimuli(win)
        for block in range(1, args.blocks + 1):
            if not wait_for_key(win, f"Block {block} of {args.blocks}\n\n"
                                     "Look only at the square outlined in red.\n"
                                     "Blink between flashes, not during them.\n\n"
                                     "Press SPACE to start."):
                raise Quit
            # Random cue order per block, so the model can't learn order/fatigue
            # effects; each trial is labelled by target via its marker.
            order = list(range(cfg.N_TARGETS))
            random.shuffle(order)
            for target in order:
                if not run_trial(win, squares, cues, target, recorder,
                                 encode_marker(block, target), overlay=labels):
                    raise Quit
            recorder.request_save()
            completed = block

        wait_for_key(win, f"Done! {completed} blocks recorded.\n\nPress SPACE to close.")
    except Quit as why:
        print(f"Stopped: {why or 'Escape pressed'}.")
    except KeyboardInterrupt:
        print("Stopped: Ctrl+C in the terminal.")
    finally:
        recorder.stop()
        recorder.join(timeout=15)   # final save happens as the board process exits
        win.close()
        summarize(session_dir, completed)
        core.quit()


def summarize(session_dir: str, completed: int) -> None:
    from .session import load_trials

    if not os.path.exists(os.path.join(session_dir, "raw.npz")):
        print("Nothing was recorded.")
        return
    trials = load_trials(session_dir)
    n = trials.eeg.shape[-1]
    print(f"\nSession saved: {session_dir}")
    print(f"  {completed} complete block(s), {n} usable trial(s)"
          + (f", {trials.skipped} skipped (not enough data around the marker)" if trials.skipped else ""))
    if n:
        print("Next: python -m ssvep_training.evaluate_trca")


if __name__ == "__main__":
    main()
