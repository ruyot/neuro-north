"""
realtime_control.py
===================

Use your trained SSVEP-TRCA classifier to control the keyboard in real time.

How it works
------------
1. A background process connects to the Knight board and fits a TRCA model on
   the data you recorded with `collect_training_data.py`.
2. The four squares flicker continuously. You look at whichever target you
   want to trigger.
3. After each 1.5 s flicker window the classifier decides which frequency you
   were looking at and "presses" the mapped key (see cfg.KEY_MAP). Point the
   window you want to control (a game, a robot teleop, etc.) so it has focus.

Default key map (edit ssvep/config.py -> KEY_MAP to change):
    6.67 Hz (top-left)     -> w
    8.57 Hz (top-right)    -> s
    10.0 Hz (bottom-left)  -> a
    12.0 Hz (bottom-right) -> d

Run it with:  python realtime_control.py
Press Escape to stop.

No headset handy?
-----------------
  --synthetic   drive the loop from BrainFlow's fake board instead of the
                Knight board, reading the blocks written by
                `collect_training_data.py --synthetic`. The whole loop runs --
                board, capture, TRCA fit, classify, keypress -- with nothing
                plugged in. The "predictions" are of course meaningless noise;
                this only proves the plumbing.
  --no-board    stimulus only: no board, no TRCA, no key presses. This is the
                show-it-to-people demo mode.
"""

from __future__ import annotations

import argparse
import time

from ssvep import config as cfg
from ssvep.recording import RecordingProcess, null_recorder

# On-screen label for each square. KEY_MAP is the single source of truth, so
# changing it to w/s/a/d for game control relabels the display too.
LETTERS = [cfg.KEY_MAP.get(i, "?").upper() for i in range(cfg.N_TARGETS)]


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--synthetic", action="store_true",
                    help="use BrainFlow's synthetic board instead of the Knight board")
    ap.add_argument("--no-board", action="store_true",
                    help="stimulus only: no board, no TRCA, no key presses")
    args = ap.parse_args()

    # Imported here, not at module scope: psychopy's preferences module parses
    # sys.argv the moment it is imported and swallows --help before argparse
    # ever gets to run.
    from psychopy import core
    from ssvep.stimulus import build_window, build_stimuli, show_message, run_trial

    if args.no_board:
        # Never started, so nothing fits TRCA and nothing presses a key.
        recorder, setup_wait = null_recorder(), 0
        banner = "Demo mode: flicker only.\nNo board, no key presses."
        ready = "Look at a target.\nEscape to quit."
    else:
        # Background board reader in PREDICT mode: fits TRCA on your training
        # data, then classifies + presses a key after every flicker window.
        # --synthetic reads the blocks recorded by collect_training_data.py
        # --synthetic; its predictions are noise, it only proves the plumbing.
        data_dir = cfg.TRAINING_DATA_DIR + ("_synthetic" if args.synthetic else "")
        recorder = RecordingProcess(mode="predict", data_dir=data_dir,
                                    variant="synthetic" if args.synthetic else None)
        setup_wait = 2 if args.synthetic else 30
        banner = f"Setting up board and training TRCA...\nPlease wait (~{setup_wait} s)."
        ready = "Ready! Look at a target to send its key.\nEscape to quit."

    win = build_window()
    show_message(win, banner)
    recorder.start()
    core.wait(setup_wait)

    squares, cues = build_stimuli(win)

    show_message(win, ready)
    core.wait(2)

    typed = ""
    try:
        # Free-running speller loop. There is no "correct" target here, so we
        # do not draw a cue; the user simply looks at the letter they want.
        while True:
            # Pair each prediction with the trial that produced it by watching
            # the child's counter, not the predicted value. Comparing values
            # would silently swallow repeated letters ("aa", "abba"), and a
            # slow classifier could let the previous trial's result be read as
            # this one's.
            seen = recorder.prediction_count.value

            # target_idx only affects the (unused) cue and label in predict
            # mode; pass -1 so no cue frame is highlighted.
            keep_going = run_trial(win, squares, cues, target_idx=-1,
                                   recording_process=recorder)
            if not keep_going:
                break

            # The classifier runs in the child process, so give it a moment to
            # finish the window that just ended. --no-board never predicts, so
            # this simply times out and nothing is appended.
            deadline = time.time() + 1.0
            while recorder.prediction_count.value == seen and time.time() < deadline:
                core.wait(0.01)

            idx = recorder.last_prediction.value
            if recorder.prediction_count.value != seen and idx >= 0:
                typed += LETTERS[idx]
                # Draw the string on the stimulus screen itself. Typing into
                # another app cannot work here: this window is fullscreen and
                # focused, so it would swallow the keystrokes anyway.
                show_message(win, f"{typed}\n\nlast: {LETTERS[idx]}")
                core.wait(0.8)
    except KeyboardInterrupt:
        pass
    finally:
        recorder.stop()
        recorder.join(timeout=5)
        win.close()
        core.quit()


if __name__ == "__main__":
    main()
