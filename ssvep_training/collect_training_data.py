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
    """Run ended early: Escape pressed, or the board failed/stopped streaming."""

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
    completed = 0
    win = None
    core = None
    recorder.start()

    try:
        # PsychoPy stays in the display process; setup failures still stop and save the board.
        from psychopy import core, event
        from .stimulus import (build_stimuli, build_window, run_trial, show_message,
                               wait_for_board, wait_for_key)

        win = build_window(fullscreen=not args.windowed)
        print("Press SPACE (or click) in the flicker window to continue at each screen; Escape quits.")
        if not wait_for_key(win, WARNING):
            raise Quit
        if not wait_for_board(win, recorder, "Setting up the Knight board..."):
            raise Quit("the board never started streaming - see [board] above")

        squares, cues, labels = build_stimuli(win)
        for block in range(1, args.blocks + 1):
            if recorder.failed.value or not recorder.is_alive():
                raise Quit("the board stopped streaming - see [board] above")
            if not wait_for_key(win, f"Block {block} of {args.blocks}\n\n"
                                     "Look only at the square outlined in red.\n"
                                     "Blink during block breaks.\n"
                                     "Take breaks or stop whenever you need to.\n\n"
                                     "Press SPACE to start."):
                raise Quit
            history_clock = core.Clock()
            while history_clock.getTime() < cfg.FILTER_HISTORY:
                if recorder.failed.value or not recorder.is_alive():
                    raise Quit("the board stopped streaming - see [board] above")
                if "escape" in event.getKeys(keyList=["escape"]):
                    raise Quit
                show_message(win, "Hold still; preparing clean signal history")
            # Random cue order per block, so the model can't learn order/fatigue
            # effects; each trial is labelled by target via its marker.
            order = list(range(cfg.N_TARGETS))
            random.shuffle(order)
            for target in order:
                if not run_trial(win, squares, cues, target, recorder,
                                 encode_marker(block, target), overlay=labels):
                    raise Quit("the board stopped streaming - see [board] above"
                               if recorder.failed.value or not recorder.is_alive()
                               else "trial interrupted")
            if recorder.failed.value or not recorder.is_alive():
                raise Quit("the board stopped streaming - see [board] above")
            recorder.request_save()
            completed = block

        wait_for_key(win, f"Done! {completed} blocks recorded.\n\nPress SPACE to close.")
    except Quit as why:
        print(f"Stopped: {why or 'Escape pressed'}.")
    except KeyboardInterrupt:
        print("Stopped: Ctrl+C in the terminal.")
    finally:
        recorder.stop()
        recorder.join()   # wait for the final save before leaving the display process
        try:
            if win is not None:
                win.close()
            summarize(session_dir, completed)
        finally:
            if core is not None:
                core.quit()


def summarize(session_dir: str, completed: int) -> None:
    from .session import load_trials

    if not os.path.exists(os.path.join(session_dir, "raw.npz")):
        print("Nothing was recorded.")
        return
    print(f"\nSession saved: {session_dir}")
    try:
        trials = load_trials(session_dir)
    except ValueError as exc:
        print(f"  Calibration trial summary unavailable: {exc}")
        return
    n = trials.eeg.shape[-1]
    print(f"  {completed} complete block(s), {n} accepted trial(s), {trials.skipped} rejected")
    for reason, count in sorted(trials.rejection_counts.items()):
        print(f"    {reason}: {count}")
    if n:
        print("Next: python -m ssvep_training.evaluate_trca")


if __name__ == "__main__":
    main()
