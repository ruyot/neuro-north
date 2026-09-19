"""
typer.py
========
Spell a prompt by looking at the squares. The squares flicker CONTINUOUSLY --
nothing goes dark, there is no cue and no window to catch. Look at the square
whose letter you want and it gets typed; look at the next one and so on.

    python -m ssvep_training.typer                    # random 8-letter prompt
    python -m ssvep_training.typer --prompt BAABAABA  # spell this
    python -m ssvep_training.typer --free             # no prompt, just type

Uses the TRCA model from your latest calibration, and streams exactly the
channels that calibration used.

Keys: Escape quits, Backspace deletes the last letter.
Results are saved to results/.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

from . import config as cfg
from .recording import RecordingProcess
from .session import latest_session

# A selection is: mark the EEG -> let the analysis window fill -> classify ->
# show which square won. The next mark only goes in once that has resolved, so
# a slow prediction can never overlap the next selection's marker.
#
# The flicker phase is re-anchored at every marker. Calibration does
# clock.reset() before each trial, so every epoch TRCA learned from starts at
# phase 0; TRCA correlates against a time-domain template, so an epoch taken at
# a different phase correlates poorly and the prediction goes random. The
# squares keep flashing through the re-anchor -- only the phase jumps, which is
# invisible at 15-20 Hz.
SELECTION_WINDOW = 1.8     # s of flicker after the marker before classifying
FEEDBACK_SECONDS = 0.45    # green outline on the chosen square (flicker continues)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prompt", help="letters to spell, e.g. BAABAABA")
    parser.add_argument("--free", action="store_true", help="no prompt, type whatever you look at")
    parser.add_argument("--length", type=int, default=8, help="length of the random prompt")
    parser.add_argument("--port", help="override PORT_PATH from .env")
    parser.add_argument("--session", help="calibration folder to train on (default: latest)")
    parser.add_argument("--windowed", action="store_true", help="run in a window instead of fullscreen")
    args = parser.parse_args()

    letters = cfg.TARGET_LETTERS
    if args.free:
        prompt = None
    elif args.prompt:
        prompt = args.prompt.upper()
        if set(prompt) - set(letters):
            parser.error(f"--prompt may only contain {''.join(letters)}")
    else:
        prompt = "".join(random.choice(letters) for _ in range(args.length))

    session = args.session or latest_session()
    if not session:
        sys.exit(f"No calibration in {cfg.TRAINING_DATA_DIR} - run "
                 "python -m ssvep_training.collect_training_data first.")
    # The model has one weight per channel, so live streaming must use exactly
    # the channels the calibration was recorded with.
    with open(os.path.join(session, "session.json")) as f:
        channels = json.load(f)["eeg_rows"]
    print(f"Training on {session}  (channels {channels})")
    recorder = RecordingProcess(port=args.port, predict_session=session, channels=channels)
    typed, picks, started = "", [], None
    win = core = None
    try:
        recorder.start()
        from psychopy import core, event, visual
        from .stimulus import build_stimuli, build_window, flicker_frame, wait_for_board, wait_for_key

        win = build_window(fullscreen=not args.windowed)
        if not wait_for_key(win, f"This screen FLASHES at {min(cfg.STIMULUS_FREQUENCIES):g}-"
                                 f"{max(cfg.STIMULUS_FREQUENCIES):g} Hz.\n\nDo not use it if you have "
                                 "epilepsy or have ever had a seizure.\n\nPress SPACE to continue."):
            return
        if not wait_for_board(win, recorder, "Setting up the board and training the model..."):
            return

        squares, cues, labels = build_stimuli(win)
        for cue in cues:
            cue.lineColor = "lime"
        line_prompt = visual.TextStim(win, pos=(0, 0.86), height=0.075, color="gray", wrapWidth=1.9)
        line_typed = visual.TextStim(win, pos=(0, -0.86), height=0.1, color="white", wrapWidth=1.9)

        def redraw_text():
            if prompt:
                line_prompt.text = f"Spell:  {' '.join(prompt)}"
                done = " ".join(typed)
                line_typed.text = done + ("  _" if len(typed) < len(prompt) else "")
            else:
                line_prompt.text = "Look at a square to type it - Backspace deletes, Escape quits"
                line_typed.text = " ".join(typed) + " |"

        redraw_text()
        intro = (f"Spell:  {' '.join(prompt)}" if prompt else "Free typing")
        if not wait_for_key(win, intro + "\n\nThe squares flicker the whole time.\nJust look at the "
                                         "letter you want next.\n\nPress SPACE to start."):
            return

        overlay = [*labels, line_prompt, line_typed]
        clock = core.Clock()
        started = time.time()
        state, t_mark, t_fb, choice = "mark", 0.0, 0.0, -1
        trial_id, dropped_at_mark = 0, 0
        phase0 = 0.0                # flicker time origin, re-anchored at each marker

        marking = False
        try:
            win.recordFrameIntervals = True
            while prompt is None or len(typed) < len(prompt):
                if recorder.failed.value or not recorder.is_alive():
                    print("[error] Board worker failed or stopped; ending live selection.")
                    break
                t = clock.getTime()
                # Decide BEFORE drawing, so the phase-0 frame is the one that reaches
                # the screen and the marker goes in immediately after it -- the same
                # order calibration uses (clock.reset -> draw -> flip -> mark_onset).
                if state == "mark":
                    phase0 = t
                    dropped_at_mark = win.nDroppedFrames
                    marking = True
                    t_mark, state = t, "collecting"

                flicker_frame(squares, cfg.STIMULUS_FREQUENCIES, t - phase0)
                for stim in overlay:
                    stim.draw()
                if state == "feedback" and 0 <= choice < len(cues):
                    cues[choice].draw()
                win.flip()

                if recorder.failed.value or not recorder.is_alive():
                    print("[error] Board worker failed or stopped; ending live selection.")
                    break
                if marking:                      # first flicker frame is now on screen
                    trial_id = recorder.mark_onset()
                    marking = False

                keys = event.getKeys(keyList=["escape", "backspace"])
                if "escape" in keys:
                    break
                if "backspace" in keys and prompt is None and typed:
                    typed = typed[:-1]
                    redraw_text()

                if state == "collecting" and t - t_mark >= SELECTION_WINDOW:
                    late = win.nDroppedFrames - dropped_at_mark
                    if late:
                        print(f"[warn] {late} late frame(s) in this selection - flicker timing slipped")
                    recorder.finish_trial(trial_id, late)
                    recorder.request_prediction(trial_id)
                    state = "waiting"
                elif state == "waiting":
                    result = recorder.poll_prediction(trial_id)
                    if result is not None:
                        choice, quality = result
                        if choice is None:
                            print(f"[quality] selection rejected: {quality.name}. Recovery needs "
                                  f"{cfg.FILTER_HISTORY:g} seconds of clean history.")
                            state = "mark"
                        else:
                            typed += letters[choice]
                            picks.append((letters[choice], round(time.time() - started, 2)))
                            print(f"[typed] {letters[choice]}  ->  {typed}")
                            redraw_text()
                            t_fb, state = t, "feedback"
                elif state == "feedback" and t - t_fb >= FEEDBACK_SECONDS:
                    state = "mark"
        finally:
            win.recordFrameIntervals = False

        if prompt and len(typed) == len(prompt):
            summary = report(prompt, typed, time.time() - started, picks)
            wait_for_key(win, summary + "\n\nPress SPACE to close.")
    except KeyboardInterrupt:
        print("Stopped: Ctrl+C in the terminal.")
    finally:
        recorder.stop()
        try:
            if win is not None:
                win.close()
        finally:
            if recorder.pid is not None:
                recorder.join()
        if typed:
            print(f"\nTyped: {typed}")
        if core is not None and sys.exc_info()[0] is None:
            core.quit()


def report(target: str, typed: str, elapsed: float, picks) -> str:
    """Score the run, print it, save it to results/, return the on-screen summary."""
    from .trca_model import bits_per_min

    correct = sum(t == p for t, p in zip(target, typed))
    accuracy = correct / len(target)
    per_pick = elapsed / len(target)
    n = cfg.N_TARGETS
    summary = (f"Prompt: {target}\nTyped:  {typed}\n\n"
               f"Accuracy: {correct}/{len(target)} = {accuracy:.0%}  (chance {100 / n:.0f}%)\n"
               f"Speed: {per_pick:.1f} s per letter")
    print("\n" + summary)
    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)
    path = os.path.join(cfg.RESULTS_DIR, time.strftime("spell_%Y%m%d_%H%M%S.json"))
    with open(path, "w") as f:
        json.dump({"prompt": target, "typed": typed, "accuracy": accuracy,
                   "elapsed_s": round(elapsed, 2), "seconds_per_letter": round(per_pick, 2),
                   "bits_per_min": bits_per_min(accuracy, n), "picks": picks,
                   "frequencies": cfg.STIMULUS_FREQUENCIES}, f, indent=2)
    print(f"Saved to {path}")
    return summary


if __name__ == "__main__":
    main()
