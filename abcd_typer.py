"""
abcd_typer.py
=============
Type with your eyes: look at square A, B, C or D and that letter is typed.

Uses the TRCA model trained on your latest calibration session (run
collect_training_data.py first, in the same headset session).

Two modes:
    python abcd_typer.py                  # free typing until Escape
    python abcd_typer.py --copy ABDCCABD  # copy test: type the given sequence,
                                          # then report accuracy + speed

Each selection: all four squares flash for 1.5 s -> TRCA predicts the square
you looked at -> it's outlined in green and its letter is appended.

Keys: Escape = quit, Backspace = delete last letter (free mode).
Copy-test results are saved to results/.

No headset handy?
    --synthetic   fake board + the latest training_data_synthetic/ session
                  (predictions are meaningless; this only tests the plumbing)
    --no-board    flicker only: no board, no model, nothing typed (demo)
"""

from __future__ import annotations

import argparse
import json
import os
import time

from ssvep import config as cfg
from ssvep.recording import NullRecorder, RecordingProcess

FEEDBACK_SECONDS = 0.5      # green outline on the predicted square
PREDICTION_TIMEOUT = 3.0    # give up waiting for the classifier after this


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", default=cfg.SERIAL_PORT)
    parser.add_argument("--session", help="calibration folder to train on (default: latest)")
    parser.add_argument("--copy", help="copy test: sequence of A/B/C/D to type, e.g. ABDC")
    board_mode = parser.add_mutually_exclusive_group()
    board_mode.add_argument("--synthetic", action="store_true",
                            help="fake board, trained on the latest synthetic session")
    board_mode.add_argument("--no-board", action="store_true",
                            help="flicker only: no board, no model, nothing typed")
    parser.add_argument("--windowed", action="store_true", help="run in a window instead of fullscreen")
    args = parser.parse_args()

    target_text = None
    if args.copy:
        target_text = args.copy.upper()
        bad = set(target_text) - set(cfg.TARGET_LETTERS)
        if bad:
            parser.error(f"--copy may only contain {''.join(cfg.TARGET_LETTERS)}, got {''.join(sorted(bad))}")
    if args.no_board and target_text:
        parser.error("--copy needs a board (it scores predictions); drop --no-board")
    if args.windowed:
        cfg.FULLSCREEN = False

    if args.no_board:
        recorder = NullRecorder()
    else:
        recorder = RecordingProcess(mode="predict", serial_port=args.port, data_dir=args.session,
                                    synthetic=args.synthetic)
        recorder.start()

    from psychopy import core, event, visual
    from ssvep.stimulus import build_stimuli, build_window, run_trial, wait_for_board, wait_for_key

    win = build_window()
    typed = ""
    predictions = []  # (predicted letter, seconds since start)
    start_time = None
    try:
        if not wait_for_board(win, recorder, "Setting up the Knight board and training the model (~35 s)..."):
            raise KeyboardInterrupt

        squares, cues, labels = build_stimuli(win)
        for cue in cues:
            cue.lineColor = "lime"
        prompt = visual.TextStim(win, pos=(0, 0.08), height=0.07, color="gray", wrapWidth=1.8)
        typed_stim = visual.TextStim(win, pos=(0, -0.08), height=0.1, color="white", wrapWidth=1.8)

        def update_text():
            if args.no_board:
                prompt.text = "Demo mode: flicker only, nothing is typed (Esc to quit)"
                typed_stim.text = ""
            elif target_text:
                prompt.text = f"Type: {target_text}"
                typed_stim.text = typed + "_" * (len(target_text) - len(typed))
            else:
                prompt.text = "Free typing (Esc to quit, Backspace to delete)"
                typed_stim.text = typed + "|"

        update_text()
        intro = (f"Copy test: type {target_text}\n\n" if target_text else "Free typing\n\n")
        if not wait_for_key(win, intro + "Look at a letter's square while it flashes.\n\nPress SPACE to start."):
            raise KeyboardInterrupt

        overlay = [*labels, prompt, typed_stim]
        start_time = time.time()
        while target_text is None or len(typed) < len(target_text):
            if "backspace" in event.getKeys(keyList=["backspace"]) and not target_text:
                typed = typed[:-1]
                update_text()

            count_before = recorder.prediction_count.value
            if not run_trial(win, squares, cues, -1, recorder, overlay=overlay):
                break
            if args.no_board:
                continue  # demo: nothing to predict

            # Wait for the classifier's answer, keeping the screen alive.
            waited = core.Clock()
            while recorder.prediction_count.value == count_before:
                if waited.getTime() > PREDICTION_TIMEOUT or not recorder.is_alive():
                    print("[warn] no prediction received for this selection")
                    break
                for sq in squares:
                    sq.draw()
                for stim in overlay:
                    stim.draw()
                win.flip()
            else:
                target = recorder.last_prediction.value
                letter = cfg.TARGET_LETTERS[target]
                typed += letter
                predictions.append((letter, round(time.time() - start_time, 2)))
                update_text()

                # Feedback: outline the chosen square in green.
                feedback = core.Clock()
                while feedback.getTime() < FEEDBACK_SECONDS:
                    for sq in squares:
                        sq.draw()
                    for stim in overlay:
                        stim.draw()
                    cues[target].draw()
                    win.flip()

        if target_text and len(typed) == len(target_text):
            report = copy_test_report(target_text, typed, time.time() - start_time, predictions,
                                      args.session or cfg.latest_session_dir(args.synthetic))
            wait_for_key(win, report["summary"] + "\n\nPress SPACE to close.")
    except KeyboardInterrupt:
        print("Stopped by user.")
    finally:
        recorder.stop()
        recorder.join(timeout=10)
        win.close()
        if typed:
            print(f"\nTyped: {typed}")
        core.quit()


def copy_test_report(target: str, typed: str, elapsed: float, predictions, session) -> dict:
    """Score a copy test, print it, and save it to results/."""
    from meegkit.utils.trca import itr

    correct = sum(t == p for t, p in zip(target, typed))
    accuracy = correct / len(target)
    per_selection = elapsed / len(target)
    bits_per_min = itr(cfg.N_TARGETS, accuracy, per_selection) if accuracy > 1 / cfg.N_TARGETS else 0.0
    summary = (f"Target: {target}\nTyped:  {typed}\n\n"
               f"Accuracy: {correct}/{len(target)} = {accuracy:.0%}  (chance 25%)\n"
               f"Speed: {per_selection:.1f} s per letter, {60 / per_selection:.1f} letters/min\n"
               f"ITR: {bits_per_min:.1f} bits/min")
    print("\n" + summary)

    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)
    path = os.path.join(cfg.RESULTS_DIR, time.strftime("copy_test_%Y%m%d_%H%M%S.json"))
    with open(path, "w") as f:
        json.dump({"target": target, "typed": typed, "accuracy": accuracy,
                   "elapsed_s": round(elapsed, 2), "seconds_per_letter": round(per_selection, 2),
                   "itr_bits_per_min": round(bits_per_min, 2), "predictions": predictions,
                   "session": session}, f, indent=2)
    print(f"Saved to {path}")
    return {"summary": summary, "path": path}


if __name__ == "__main__":
    main()
