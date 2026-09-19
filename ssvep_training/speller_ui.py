"""
speller_ui.py
=============
The speller screen: two flicker boxes, four edge zones, and the typed text.

    ┌──────────────── top: suggestion 1 ────────────────┐
    │                 hi [a-f]|                          │
    │ <     a b c d e f            g h i j k l        S │
    │ W     ┌─────────┐            ┌─────────┐        P │
    │ H     │  15 Hz  │            │  20 Hz  │        A │
    │ E     └─────────┘            └─────────┘        C │
    │ E                                               E │
    │ L                                                 │
    └─────────────── bottom: suggestion 2 ──────────────┘

Looking at a box (SSVEP) types its whole range as ONE item, shown as [a-f]
until a real word replaces it. The edges are for the IMU. Every action is a
plain method, so the IMU layer only has to call it:

    ui.select_box(i)            SSVEP pick: 0 = left box, 1 = right box
    ui.next_wheel()             left edge: a-f / g-l  <->  m-r / s-z
    ui.space()                  right edge
    ui.pick_suggestion(slot)    top (0) / bottom (1) edge: replace the current word
    ui.set_suggestions(t, b)    autocomplete fills the top / bottom edges (blank for now)

    python -m ssvep_training.speller_ui           # headset + latest calibration
    python -m ssvep_training.speller_ui --keys    # no headset, no flicker: 1 / 2 pick a box

Keys standing in for the IMU: Left = wheel, Right = space, Up / Down =
suggestions. Escape quits.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

from . import config as cfg

# The four ranges, two per wheel page (one per box).
RANGES = ["abcdef", "ghijkl", "mnopqr", "stuvwxyz"]
RESET_WHEEL_AFTER_PICK = True   # back to a-f / g-l after every pick and space

# Same selection timing as typer.py.
SELECTION_WINDOW = 1.8     # s of flicker after the marker before classifying
FEEDBACK_SECONDS = 0.45    # green outline + picked letters on the chosen box
PREDICTION_TIMEOUT = 4.0   # s to wait for the board process before giving up
FLASH_SECONDS = 0.3        # green outline on an edge zone when its action fires
MAX_SHOWN_CHARS = 40       # typed line shows only the end of long text

# Edge zones, normalised units (screen = -1..1): name -> (centre, size). Kept
# thin; the boxes themselves come from config.TARGET_X / TARGET_SIZE.
ZONES = {
    "top": ((0.0, 0.93), (1.2, 0.14)),
    "bottom": ((0.0, -0.93), (1.2, 0.14)),
    "left": ((-0.93, 0.0), (0.14, 1.0)),
    "right": ((0.93, 0.0), (0.14, 1.0)),
}
LABEL_GAP = 0.14        # box letters sit this far above the box (clear of its outline)
TYPED_Y = 0.77

_DIM = [-0.5, -0.5, -0.5]
_GREEN = "lime"


def range_name(letters: str) -> str:
    """'abcdef' -> 'a-f'"""
    return f"{letters[0]}-{letters[-1]}"


class SpellerUI:
    """Screen state + drawing. Draws everything except the flicker squares,
    so draw() can go on top of any flicker frame."""

    def __init__(self, win, cues):
        """cues: the outline per box from stimulus.build_stimuli(), used for pick feedback."""
        from psychopy import visual

        from .stimulus import layout

        positions, (_, h) = layout(cfg.N_TARGETS)
        self.pages = [RANGES[i:i + len(positions)] for i in range(0, len(RANGES), len(positions))]
        self.page = 0
        self.items: list[str] = []      # typed so far: a range ("abcdef"), a letter, or " "
        self.suggestions = ["", ""]     # top, bottom
        self.action_count = 0           # bumped by every action; lets the loop spot one mid-selection
        self._flash_until: dict[str, float] = {}

        self.cues = cues
        for cue in cues:
            cue.lineColor = _GREEN
        # Letters above the box, not on it, so they don't change its brightness.
        # The green copy shows what was just picked while the page resets underneath.
        label_pos = [(x, y + h / 2 + LABEL_GAP) for x, y in positions]
        self.box_labels = [visual.TextStim(win, pos=p, height=0.07, color="white", wrapWidth=0.6)
                           for p in label_pos]
        self.picked_labels = [visual.TextStim(win, pos=p, height=0.07, color=_GREEN, wrapWidth=0.6)
                              for p in label_pos]
        self.zones = {
            name: (visual.Rect(win, units="norm", width=size[0], height=size[1], pos=pos,
                               fillColor=None, lineColor=_DIM, lineWidth=2),
                   visual.TextStim(win, pos=pos, height=0.05, color="gray", wrapWidth=size[0]))
            for name, (pos, size) in ZONES.items()
        }
        self.typed_text = visual.TextStim(win, pos=(0, TYPED_Y), height=0.08, color="white", wrapWidth=1.6)
        self._refresh()

    # ------------------------------------------------------------------ #
    # Actions: SSVEP / IMU / keyboard call these
    # ------------------------------------------------------------------ #
    def select_box(self, i: int) -> None:
        """Type box i's range as one item."""
        letters = self.pages[self.page][i]
        self.items.append(letters)
        if RESET_WHEEL_AFTER_PICK:
            self.page = 0
        _set_text(self.picked_labels[i], " ".join(letters))
        self._flash(f"box{i}", FEEDBACK_SECONDS)
        self._changed(f"picked [{range_name(letters)}]")

    def next_wheel(self) -> None:
        """Show the next pair of ranges in the boxes."""
        self.page = (self.page + 1) % len(self.pages)
        self._flash("left")
        self._changed("wheel -> " + " / ".join(range_name(r) for r in self.pages[self.page]))

    def space(self) -> None:
        self.items.append(" ")
        if RESET_WHEEL_AFTER_PICK:
            self.page = 0
        self._flash("right")
        self._changed("space")

    def pick_suggestion(self, slot: int) -> None:
        """Replace the current word with suggestion `slot` (0 = top, 1 = bottom)
        plus a space. Does nothing while that suggestion is blank."""
        word = self.suggestions[slot]
        if not word:
            return
        start = len(self.items) - len(self.current_word())
        self.items[start:] = [*word, " "]
        self.page = 0
        self._flash("top" if slot == 0 else "bottom")
        self._changed(f"suggestion {word!r}")

    def set_suggestions(self, top: str = "", bottom: str = "") -> None:
        """Autocomplete hook: fill the top / bottom edges ("" = blank)."""
        self.suggestions = [top, bottom]
        self._refresh()

    # ------------------------------------------------------------------ #
    # State, e.g. for autocomplete
    # ------------------------------------------------------------------ #
    def current_word(self) -> list[str]:
        """Items since the last space, e.g. ["ghijkl", "ghijkl"] for "hi"."""
        word = []
        for item in reversed(self.items):
            if item == " ":
                break
            word.insert(0, item)
        return word

    @property
    def text(self) -> str:
        """Typed text with unresolved ranges as [a-f]."""
        return "".join(f"[{range_name(item)}]" if len(item) > 1 else item for item in self.items)

    # ------------------------------------------------------------------ #
    # Drawing
    # ------------------------------------------------------------------ #
    def draw(self) -> None:
        now = time.perf_counter()
        for name, (rect, label) in self.zones.items():
            _set_line(rect, _GREEN if now < self._flash_until.get(name, 0) else _DIM)
            rect.draw()
            label.draw()
        for i, (label, picked) in enumerate(zip(self.box_labels, self.picked_labels)):
            if now < self._flash_until.get(f"box{i}", 0):
                self.cues[i].draw()
                picked.draw()
            else:
                label.draw()
        self.typed_text.draw()

    def _flash(self, name: str, seconds: float = FLASH_SECONDS) -> None:
        self._flash_until[name] = time.perf_counter() + seconds

    def _changed(self, what: str) -> None:
        self.action_count += 1
        self._refresh()
        print(f"[ui] {what:<24} ->  {self.text!r}")

    def _refresh(self) -> None:
        for label, letters in zip(self.box_labels, self.pages[self.page]):
            _set_text(label, " ".join(letters))
        other = self.pages[(self.page + 1) % len(self.pages)]
        _set_text(self.zones["left"][1], "<\nWHEEL\n\n" + "\n".join(range_name(r) for r in other))
        _set_text(self.zones["right"][1], "SPACE\n>")
        _set_text(self.zones["top"][1], self.suggestions[0])
        _set_text(self.zones["bottom"][1], self.suggestions[1])
        shown = self.text
        if len(shown) > MAX_SHOWN_CHARS:
            shown = "..." + shown[-MAX_SHOWN_CHARS:]
        _set_text(self.typed_text, shown + "|")


def _set_text(stim, text: str) -> None:
    """Re-rendering a TextStim can drop a flicker frame: only do it on a change."""
    if stim.text != text:
        stim.text = text


_line_colors: dict[int, object] = {}


def _set_line(rect, color) -> None:
    if _line_colors.get(id(rect)) != color:
        rect.lineColor = color
        _line_colors[id(rect)] = color


# --------------------------------------------------------------------------- #
# Keyboard stand-ins for the IMU (and for the boxes with --keys)
# --------------------------------------------------------------------------- #
KEY_ACTIONS = {
    "left": ("next_wheel",),
    "right": ("space",),
    "up": ("pick_suggestion", 0),
    "down": ("pick_suggestion", 1),
}
BOX_KEYS = {"1": 0, "2": 1}


def handle_keys(ui: SpellerUI, boxes: bool) -> bool:
    """Apply pending keys to the UI. False on Escape."""
    from psychopy import event
    for key in event.getKeys(keyList=[*KEY_ACTIONS, *(BOX_KEYS if boxes else ()), "escape"]):
        if key == "escape":
            return False
        if key in BOX_KEYS:
            ui.select_box(BOX_KEYS[key])
        else:
            name, *args = KEY_ACTIONS[key]
            getattr(ui, name)(*args)
    return True


# --------------------------------------------------------------------------- #
# Main loops
# --------------------------------------------------------------------------- #
def run_keys(win, ui: SpellerUI, squares) -> None:
    """No board, no flicker: boxes shown dim grey, 1 / 2 pick."""
    for sq in squares:
        sq.fillColor = [-0.7, -0.7, -0.7]
    while handle_keys(ui, boxes=True):
        for sq in squares:
            sq.draw()
        ui.draw()
        win.flip()


def run_flicker(win, ui: SpellerUI, squares, recorder) -> None:
    """typer.py's continuous flicker: mark -> 1.8 s -> classify -> feedback -> mark.
    A UI action (wheel, space, ...) mid-selection means the user looked away:
    that selection is dropped and a fresh one starts."""
    from psychopy import core

    from .stimulus import flicker_frame

    clock = core.Clock()
    state, t_mark, t_req, t_fb, seen, actions_at_mark = "mark", 0.0, 0.0, 0.0, 0, 0
    phase0 = 0.0                # flicker time origin, re-anchored at each marker (see typer.py)
    marking = False
    while True:
        t = clock.getTime()
        if state == "mark":
            phase0 = t
            seen = recorder.prediction_count.value
            actions_at_mark = ui.action_count
            marking = True
            t_mark, state = t, "collecting"

        flicker_frame(squares, cfg.STIMULUS_FREQUENCIES, t - phase0)
        ui.draw()
        win.flip()

        if marking:                      # first flicker frame is now on screen
            recorder.mark_onset(cfg.LIVE_MARKER)
            marking = False

        if not handle_keys(ui, boxes=False):
            return

        if state == "collecting":
            if ui.action_count != actions_at_mark:
                state = "mark"           # nothing requested yet, so restarting is safe
            elif t - t_mark >= SELECTION_WINDOW:
                recorder.request_prediction()
                t_req, state = t, "waiting"
        elif state == "waiting":
            # Wait for this prediction even if it'll be dropped, so it can't be
            # mistaken for the next selection's.
            if recorder.prediction_count.value != seen:
                if ui.action_count != actions_at_mark:
                    print("[ui] action during the selection - pick dropped")
                    state = "mark"
                else:
                    ui.select_box(recorder.last_prediction.value)
                    t_fb, state = t, "feedback"
            elif t - t_req > PREDICTION_TIMEOUT or not recorder.is_alive():
                print("[warn] no prediction for this selection - trying again")
                state = "mark"
        elif state == "feedback" and t - t_fb >= FEEDBACK_SECONDS:
            state = "mark"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--keys", action="store_true", help="no headset, no flicker: 1 / 2 pick a box")
    parser.add_argument("--port", help="override PORT_PATH from .env")
    parser.add_argument("--session", help="calibration folder to train on (default: latest)")
    parser.add_argument("--windowed", action="store_true", help="run in a window instead of fullscreen")
    args = parser.parse_args()
    if cfg.N_TARGETS != 2:
        parser.error(f"the speller has 2 boxes; config.py has {cfg.N_TARGETS} targets")

    recorder = None
    if not args.keys:
        from .recording import RecordingProcess
        from .session import latest_session

        session = args.session or latest_session()
        if not session:
            sys.exit(f"No calibration in {cfg.TRAINING_DATA_DIR} - run "
                     "python -m ssvep_training.collect_training_data first.")
        # Live streaming must use exactly the calibration's channels (see typer.py).
        with open(os.path.join(session, "session.json")) as f:
            channels = json.load(f)["eeg_rows"]
        print(f"Training on {session}  (channels {channels})")
        recorder = RecordingProcess(port=args.port, predict_session=session, channels=channels)
        recorder.start()

    from psychopy import core
    from .stimulus import build_stimuli, build_window, wait_for_board, wait_for_key

    win = build_window(fullscreen=not args.windowed)
    ui = None
    try:
        if recorder:
            if not wait_for_key(win, f"This screen FLASHES at {min(cfg.STIMULUS_FREQUENCIES):g}-"
                                     f"{max(cfg.STIMULUS_FREQUENCIES):g} Hz.\n\nDo not use it if you have "
                                     "epilepsy or have ever had a seizure.\n\nPress SPACE to continue."):
                return
            if not wait_for_board(win, recorder, "Setting up the board and training the model..."):
                return

        squares, cues, _ = build_stimuli(win)
        ui = SpellerUI(win, cues)
        how = "1 / 2 = left / right box" if args.keys else "Look at a box"
        if not wait_for_key(win, f"Speller\n\n{how}: types its letters as one item.\n\n"
                                 "Left arrow = wheel    Right arrow = space\n"
                                 "Up / Down = suggestions    Escape quits\n\n"
                                 "Press SPACE to start.", name="Speller"):
            return
        if recorder:
            run_flicker(win, ui, squares, recorder)
        else:
            run_keys(win, ui, squares)
    except KeyboardInterrupt:
        print("Stopped: Ctrl+C in the terminal.")
    finally:
        if recorder:
            recorder.stop()
            recorder.join(timeout=10)
        win.close()
        if ui and ui.items:
            print(f"\nTyped: {ui.text!r}")
        core.quit()


if __name__ == "__main__":
    main()
