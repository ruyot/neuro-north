"""Record cued motor-imagery trials.

    python -m motor_imagery.collect --mode clench --blocks 12
    python -m motor_imagery.collect --mode imagine --blocks 12
    python -m motor_imagery.collect --classes lr --rest 0     # no "neither" class

Each trial: a cue (LEFT / RIGHT / REST) -> GO, marker stamped at its first frame
-> hold for HOLD_DURATION -> relax. Cues are shuffled within every block, rest
included, so "rest" can never mean "later in the session, when you were tired" -
which is what recording all the rest windows at the end would teach the model.

MODE MATTERS. `clench` physically closes the hand: strong, easy to classify, and
full of muscle activity that will not be there when you imagine. `imagine` is
the real task. They are recorded into separate sessions and never mixed; the
mode is saved in session.json.

The screen only shows cues and records raw EEG; filtering and trial cutting
happen later in session.py, so the analysis window can still be changed.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time

from . import config as cfg
from .recording import MIRecorder
from .session import encode_marker

CUE_TEXT = {cfg.LEFT: "LEFT", cfg.RIGHT: "RIGHT", cfg.REST: "REST", cfg.BOTH: "BOTH"}
CUE_COLOR = {cfg.LEFT: "#4fc3f7", cfg.RIGHT: "#ff8a65", cfg.REST: "#8b93a1", cfg.BOTH: "#ba8cff"}
HOLD_TEXT = {"clench": {cfg.LEFT: "clench LEFT hand", cfg.RIGHT: "clench RIGHT hand",
                        cfg.REST: "stay relaxed", cfg.BOTH: "clench BOTH hands"},
             "imagine": {cfg.LEFT: "imagine LEFT hand", cfg.RIGHT: "imagine RIGHT hand",
                         cfg.REST: "stay relaxed", cfg.BOTH: "imagine BOTH hands"}}


class Quit(Exception):
    """Run ended early: Escape, or the board never streamed."""


def build_cue_stims(win):
    """Cue word, the instruction under it, and the hold bar."""
    from psychopy import visual

    from ssvep_training.speller_ui import FAINT, FONT, MUTED, PANEL, PROGRESS_SIZE, PROGRESS_Y

    return {
        "cue": visual.TextStim(win, text="", pos=(0, 0.15), height=0.18, bold=True, font=FONT),
        "hint": visual.TextStim(win, text="", pos=(0, -0.05), height=0.06, color=MUTED, font=FONT),
        "track": visual.Rect(win, units="norm", width=PROGRESS_SIZE[0], height=PROGRESS_SIZE[1],
                             pos=(0, PROGRESS_Y), fillColor=PANEL, lineColor=None),
        "fill": visual.Rect(win, units="norm", width=0.01, height=PROGRESS_SIZE[1],
                            pos=(-PROGRESS_SIZE[0] / 2, PROGRESS_Y), anchor="left",
                            fillColor=FAINT, lineColor=None),
        "size": PROGRESS_SIZE,
    }


def run_trial(win, stims, class_id: int, mode: str, recorder, marker: int) -> bool:
    """cue -> GO + hold (marker at the first hold frame) -> relax. False on Escape."""
    from psychopy import core, event

    from ssvep_training.speller_ui import ACCENT, FAINT

    def frame(phase: str, fraction: float) -> bool:
        stims["cue"].draw()
        stims["hint"].draw()
        stims["track"].draw()
        if fraction > 0:
            stims["fill"].fillColor = ACCENT if phase == "go" else FAINT
            stims["fill"].width = stims["size"][0] * min(fraction, 1.0)
            stims["fill"].draw()
        win.flip()
        return "escape" not in event.getKeys(keyList=["escape"])

    clock = core.Clock()
    print(f"[trial] cue {CUE_TEXT[class_id].lower()}, marker {marker}")

    # --- CUE: which hand, before the go signal, so the hold isn't a reaction test
    stims["cue"].text = CUE_TEXT[class_id]
    stims["cue"].color = CUE_COLOR[class_id]
    stims["hint"].text = "get ready"
    while clock.getTime() < cfg.CUE_DURATION:
        if not frame("ready", 0.0):
            return False

    # --- HOLD: the marker goes in at the first frame the user sees "GO"
    stims["hint"].text = HOLD_TEXT[mode][class_id]
    marked = False
    clock.reset()
    while clock.getTime() < cfg.HOLD_DURATION:
        if not frame("go", clock.getTime() / cfg.HOLD_DURATION):
            return False
        if not marked:
            recorder.mark_onset(marker)
            marked = True

    # --- BREAK: randomised, so the next cue can't be anticipated
    stims["cue"].text = ""
    stims["hint"].text = "relax"
    pause = random.uniform(*cfg.BREAK_DURATION)
    clock.reset()
    while clock.getTime() < pause:
        if not frame("break", 1 - clock.getTime() / pause):
            return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", default="clench", choices=cfg.MODES,
                        help="clench = physically close the hand; imagine = the real task")
    parser.add_argument("--blocks", type=int, default=12, help="times every cue is shown (8-16 sensible)")
    parser.add_argument("--repeats", type=int, default=2, help="trials per cue per block")
    parser.add_argument("--classes", default=cfg.DEFAULT_CLASSES, choices=sorted(cfg.CLASS_SETS),
                        help="lrr = left/right/rest (default), lr = no rest, lrrb = adds both hands")
    parser.add_argument("--rest", type=int, default=None,
                        help="rest trials per block (default: same as every other cue)")
    parser.add_argument("--port", help="override PORT_PATH from .env")
    parser.add_argument("--board", default=cfg.DEFAULT_BOARD, choices=sorted(cfg.BOARDS),
                        help="which Knight to ask BrainFlow for (default imu = this headset). "
                             "plain is a DIAGNOSTIC only: it parses packets differently, so do "
                             "not record real data with it")
    parser.add_argument("--settle", type=float, default=None,
                        help="seconds to wait for boot chatter before enabling channels "
                             "(default 3; raise to 8-10 right after replugging the USB-C)")
    parser.add_argument("--windowed", action="store_true", help="run in a window instead of fullscreen")
    parser.add_argument("--channels", default="",
                        help="board channels to record, e.g. 1,2,3,4 (default: all 8)")
    args = parser.parse_args()

    channels = [int(c) for c in args.channels.split(",")] if args.channels else None
    classes = cfg.classes_for(args.classes)
    print(cfg.montage_banner())

    session_dir = os.path.join(cfg.TRAINING_DATA_DIR, time.strftime(f"{cfg.SESSION_PREFIX}%Y%m%d_%H%M%S"))
    recorder = MIRecorder(port=args.port, session_dir=session_dir, channels=channels,
                          settle=args.settle, mode=args.mode, classes=classes, board=args.board)
    recorder.start()

    from psychopy import core

    from ssvep_training.stimulus import build_window, wait_for_board, wait_for_key

    win = build_window(fullscreen=not args.windowed)
    completed = 0
    try:
        task = ("CLENCH the hand the cue names - lightly, and the same way every time"
                if args.mode == "clench" else
                "IMAGINE squeezing the hand the cue names, without moving it")
        if not wait_for_key(win, f"Motor imagery calibration ({args.mode})\n\n{task}.\n\n"
                                 "On REST, do nothing and keep looking at the screen.\n"
                                 "Keep your jaw, shoulders and eyes still throughout.\n\n"
                                 "Press SPACE to continue."):
            raise Quit
        # The filter bank needs ~20 s of stream before a trial is usable, and the
        # board setup plus this screen normally covers it - but don't count on it.
        if not wait_for_board(win, recorder, "Setting up the Knight board (~8 s)..."):
            raise Quit("the board never started streaming - see [board] above")

        stims = build_cue_stims(win)
        for block in range(1, args.blocks + 1):
            if not wait_for_key(win, f"Block {block} of {args.blocks}\n\n"
                                     "Follow each cue when the bar starts filling.\n"
                                     "Blink during the breaks, not during the holds.\n\n"
                                     "Press SPACE to start."):
                raise Quit
            order = []
            for class_id in classes:
                repeats = args.repeats if class_id != cfg.REST or args.rest is None else args.rest
                order += [class_id] * repeats
            random.shuffle(order)
            for class_id in order:
                if not run_trial(win, stims, class_id, args.mode, recorder,
                                 encode_marker(block, class_id)):
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
        recorder.join(timeout=15)       # the final save happens as the board process exits
        win.close()
        summarize(session_dir, completed, args.mode)
        core.quit()


def summarize(session_dir: str, completed: int, mode: str) -> None:
    from .session import load_trials, quality_warnings

    if not os.path.exists(os.path.join(session_dir, "raw.npz")):
        print("Nothing was recorded.")
        return
    trials = load_trials(session_dir)
    print(f"\nSession saved: {session_dir}")
    print(f"  {completed} complete block(s), {len(trials)} usable trial(s) {trials.counts()}")
    warnings = quality_warnings(session_dir)
    if warnings:
        print("  RAW EEG WARNING: likely contact/saturation problem:")
        for warning in warnings:
            print(f"    {warning}")
    if trials.skipped:
        print(f"  {trials.skipped} skipped (inside the {cfg.PRIME_SECONDS:g} s filter priming "
              "stretch at the start, or cut short at the end)")
    if len(trials):
        print("Next: python -m motor_imagery.evaluate")
    if mode == "clench":
        print("Remember: clench scores are an upper bound. Record an --mode imagine "
              "session before trusting them.")


if __name__ == "__main__":
    sys.exit(main())
