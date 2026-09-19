"""The speller, driven by motor imagery.

Same screen, ranges, autocomplete and IMU edges as the SSVEP speller - only the
pick changes: left hand types the left box, right hand the right box, and
"neither" types nothing. There is no flicker, so no epilepsy warning and nothing
to stare at.

    python -m motor_imagery.speller                 # headset + the latest calibration
    python -m motor_imagery.speller --keys          # no headset: 1 / 2 pick a box
    python -m motor_imagery.speller --margin 0.3    # stricter: fewer letters, fewer mistakes

One selection is CUE_DURATION + HOLD_DURATION + feedback, about 4 s against
SSVEP's 2.3 s. That is the cost of a paradigm with no flicker to lock onto;
autocomplete is what makes it bearable.
"""

from __future__ import annotations

import argparse
import sys

from ssvep_training import config as ssvep_cfg
from ssvep_training.speller_ui import FEEDBACK_SECONDS, PREDICTION_TIMEOUT, SpellerUI, handle_keys

from . import config as cfg


class MISpellerUI(SpellerUI):
    """The speller screen with the timing bar relabelled for imagery."""

    PROGRESS_LABELS = {"go": "imagine", "break": "relax"}


def run_imagery(win, ui: MISpellerUI, squares, recorder) -> None:
    """cue -> hold (marker at its first frame) -> classify -> feedback -> cue.

    A UI action mid-selection (wheel, space, a suggestion) means the user was
    doing something else, so that selection is dropped and a fresh one starts -
    the same rule as run_flicker.
    """
    from psychopy import core

    clock = core.Clock()
    state, t_state, t_req, seen, actions_at_cue = "cue", 0.0, 0.0, 0, 0
    marking = False
    win.recordFrameIntervals = True
    try:
        while True:
            t = clock.getTime()
            if state == "cue":                     # relax, then get ready
                seen = recorder.prediction_count.value
                actions_at_cue = ui.action_count
                t_state, state = t, "ready"

            elapsed = t - t_state
            if state == "ready":
                ui.set_progress("break", 1 - elapsed / cfg.CUE_DURATION)
            elif state == "hold":
                ui.set_progress("go", elapsed / cfg.HOLD_DURATION)
            else:
                ui.set_progress("break", 1.0)

            ui.poll()
            for square in squares:
                square.draw()
            ui.draw()
            win.flip()

            if marking:                            # the first hold frame is now on screen
                recorder.mark_onset(ssvep_cfg.LIVE_MARKER)
                marking = False

            if not handle_keys(ui, boxes=False):
                return

            if state == "ready" and elapsed >= cfg.CUE_DURATION:
                if ui.action_count != actions_at_cue:
                    state = "cue"                  # nothing marked yet, safe to restart
                else:
                    t_state, state, marking = t, "hold", True
            elif state == "hold" and elapsed >= cfg.HOLD_DURATION:
                recorder.request_prediction()
                t_req, state = t, "waiting"
            elif state == "waiting":
                # Wait for this prediction even when it will be dropped, so it
                # can't be mistaken for the next selection's.
                if recorder.prediction_count.value != seen:
                    choice = recorder.last_prediction.value
                    if ui.action_count != actions_at_cue:
                        print("[ui] action during the selection - pick dropped")
                        state = "cue"
                    elif choice < 0:
                        print("[ui] neither hand - nothing typed")
                        state = "cue"
                    elif choice == cfg.BOTH:       # only when calibrated with --classes lrrb
                        ui.space()
                        t_state, state = t, "feedback"
                    else:
                        ui.select_box(choice)
                        t_state, state = t, "feedback"
                elif t - t_req > PREDICTION_TIMEOUT or not recorder.is_alive():
                    print("[warn] no prediction for this selection - trying again")
                    state = "cue"
            elif state == "feedback" and elapsed >= FEEDBACK_SECONDS:
                state = "cue"
    finally:
        win.recordFrameIntervals = False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--keys", action="store_true", help="no headset: 1 / 2 pick a box")
    parser.add_argument("--port", help="override PORT_PATH from .env")
    parser.add_argument("--board", default=cfg.DEFAULT_BOARD, choices=sorted(cfg.BOARDS),
                        help="which Knight to ask BrainFlow for (default imu = this headset). "
                             "plain is a DIAGNOSTIC only: it parses packets differently, so do "
                             "not record real data with it")
    parser.add_argument("--session", help="calibration folder to decode with (default: latest)")
    parser.add_argument("--model", choices=["tangent", "fb-tangent", "fbcsp", "stack", "logvar"],
                        help="fit this model if the session has no saved one")
    parser.add_argument("--margin", type=float, default=None,
                        help="how far a hand must beat rest before typing (0 = plain argmax)")
    parser.add_argument("--windowed", action="store_true", help="run in a window instead of fullscreen")
    parser.add_argument("--engine", default="gpt2", choices=["gpt2", "smollm2", "pythia", "bigram", "none"],
                        help="autocomplete model (default gpt2; none = blank suggestions)")
    parser.add_argument("--confidence", type=float, default=None,
                        help="probability given to each picked box (default: autocomplete.PICK_CONFIDENCE)")
    args = parser.parse_args()
    if ssvep_cfg.N_TARGETS != 2:
        parser.error(f"the speller has 2 boxes; ssvep_training/config.py has "
                     f"{ssvep_cfg.N_TARGETS} targets")

    # Started first so the language model loads while the board sets up.
    autocomplete = None
    if args.engine != "none":
        from ssvep_training.autocomplete import PICK_CONFIDENCE, AutocompleteProcess

        autocomplete = AutocompleteProcess(args.engine, args.confidence or PICK_CONFIDENCE)
        autocomplete.start()

    recorder = None
    if not args.keys:
        from .recording import MIRecorder
        from .session import latest_session, session_info

        session = args.session or latest_session()
        if not session:
            sys.exit(f"No calibration in {cfg.TRAINING_DATA_DIR} - run "
                     "python -m motor_imagery.collect first.")
        info = session_info(session)
        print(f"Decoding with {session}  (channels {info['eeg_rows']})")
        print(cfg.montage_banner(info["eeg_rows"]))
        # Live streaming must use exactly the calibration's channels: the spatial
        # filters are per electrode, so a different set means a different model.
        recorder = MIRecorder(port=args.port, predict_session=session,
                              channels=info["eeg_rows"], model=args.model, margin=args.margin, board=args.board)
        recorder.start()

    from psychopy import core

    from ssvep_training.speller_ui import run_keys, wait_for_autocomplete
    from ssvep_training.stimulus import build_stimuli, build_window, wait_for_board, wait_for_key

    win = build_window(fullscreen=not args.windowed)
    ui = None
    try:
        if recorder and not wait_for_board(win, recorder, "Setting up the board and the decoder..."):
            return
        if autocomplete:
            if not wait_for_autocomplete(win, autocomplete):
                return
            if not autocomplete.ready.is_set():
                autocomplete = None

        squares, cues, _ = build_stimuli(win)
        for square in squares:                       # static boxes: nothing flickers here
            square.fillColor = square.lineColor = "#23262e"
        ui = MISpellerUI(win, cues, autocomplete)
        how = "1 / 2 = left / right box" if args.keys else "Imagine squeezing a hand"
        if not wait_for_key(win, f"Motor-imagery speller\n\n{how}: types that box's letters "
                                 "as one item.\nDo nothing to type nothing.\n\n"
                                 "Left arrow = wheel    Right arrow = space\n"
                                 "Up / Down = suggestions    Escape quits\n\n"
                                 "Press SPACE to start.", name="Speller"):
            return
        if recorder:
            run_imagery(win, ui, squares, recorder)
        else:
            run_keys(win, ui, squares)
    except KeyboardInterrupt:
        print("Stopped: Ctrl+C in the terminal.")
    finally:
        if recorder:
            recorder.stop()
            recorder.join(timeout=10)
        if autocomplete:
            autocomplete.stop()
            autocomplete.join(timeout=5)
        win.close()
        if ui and ui.items:
            print(f"\nTyped: {ui.text!r}")
        core.quit()


if __name__ == "__main__":
    main()
