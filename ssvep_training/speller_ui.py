"""
The speller screen: two flicker boxes, four edge zones, and the typed text.

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
SELECTION_WINDOW = 1.8
FEEDBACK_SECONDS = 0.45  
PREDICTION_TIMEOUT = 4.0   
FLASH_SECONDS = 0.3        
MAX_TYPED_CHARS = 26       

FONT = "Helvetica Neue"
TEXT = "#e8eaed"
MUTED = "#8b93a1"
FAINT = "#454b56"
PANEL = "#14161b"
BORDER = "#272b34"
ACCENT = "#4ade80"         # picks and actions
PENDING = "#7cb7ff"        # the word being typed, still ranges

# Panels, normalised units (screen = -1..1): name -> (centre, size)
PANELS = {
    "top": ((0.0, 0.905), (1.3, 0.11)),
    "typed": ((0.0, 0.75), (1.3, 0.12)),
    "bottom": ((0.0, -0.905), (1.3, 0.11)),
    "left": ((-0.925, 0.0), (0.12, 1.0)),
    "right": ((0.925, 0.0), (0.12, 1.0)),
}
CORNER = 0.025             # panel corner radius (fraction of screen height)
LABEL_GAP = 0.125          # box letters sit this far above the box, clear of its outline

PROGRESS_Y = -0.72
PROGRESS_SIZE = (0.8, 0.016)
GO_SECONDS = cfg.VISUAL_LATENCY + cfg.GAZE_DURATION   # marker -> end of the classified window


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
        self._line_colors: dict[int, str] = {}

        aspect = win.size[0] / win.size[1]

        def text(pos, height, color=TEXT, anchor="center", **kw):
            return visual.TextStim(win, pos=pos, height=height, color=color, font=FONT,
                                   anchorHoriz=anchor, alignText=anchor, wrapWidth=2, **kw)

        def panel(pos, size, radius=CORNER):
            return visual.ShapeStim(win, units="norm", pos=pos, fillColor=PANEL, lineColor=BORDER,
                                    lineWidth=2, vertices=_rounded_rect(size, radius, aspect))

        self.cues = cues
        for cue in cues:
            cue.lineColor = ACCENT
            cue.lineWidth = 4

        # Letters above the box, not on it, so they don't change its brightness.
        # The accent copy shows what was just picked while the page resets underneath.
        label_pos = [(x, y + h / 2 + LABEL_GAP) for x, y in positions]
        self.box_labels = [text(p, 0.075) for p in label_pos]
        self.picked_labels = [text(p, 0.075, ACCENT) for p in label_pos]

        self.panels = {name: panel(pos, size) for name, (pos, size) in PANELS.items()}

        top, bottom = PANELS["top"][0], PANELS["bottom"][0]
        self.suggestion_text = [text(top, 0.055), text(bottom, 0.055)]

        # Typed text meets the caret at the centre: finished words grow leftwards,
        # the word in progress (still ranges) grows rightwards in its own colour.
        y = PANELS["typed"][0][1]
        self.done_text = text((-0.005, y), 0.07, anchor="right")
        self.word_text = text((0.015, y), 0.07, PENDING, anchor="left")

        lx, rx = PANELS["left"][0][0], PANELS["right"][0][0]
        dot = 0.012
        self.page_dots = [visual.ShapeStim(win, units="norm", pos=(lx + dx / aspect, 0.03),
                                           vertices=_rounded_rect((2 * dot / aspect, 2 * dot), dot, aspect),
                                           fillColor=FAINT, lineColor=None)
                          for dx in (-0.022, 0.022)]
        self.side_static = [
            *_arrow(win, (lx, 0.19), -1, aspect),
            text((lx, 0.11), 0.036, MUTED, bold=True, text="WHEEL"),
            *_arrow(win, (rx, 0.06), 1, aspect),
            text((rx, -0.02), 0.036, MUTED, bold=True, text="SPACE"),
        ]
        self.next_ranges = text((lx, -0.1), 0.04, FAINT)

        w, bh = PROGRESS_SIZE
        self.bar_track = visual.Rect(win, units="norm", width=w, height=bh, pos=(0, PROGRESS_Y),
                                     fillColor=PANEL, lineColor=None)
        self.bar_fill = visual.Rect(win, units="norm", width=w, height=bh, pos=(-w / 2, PROGRESS_Y),
                                    anchor="left", fillColor=ACCENT, lineColor=None)
        self.bar_label = text((0, PROGRESS_Y - 0.045), 0.035, MUTED)
        self._progress: tuple[str, float] | None = None   # (phase, fraction); None hides the bar
        self._bar_phase = None
        self._refresh()

    def select_box(self, i: int) -> None:
        """Type box i's range as one item."""
        letters = self.pages[self.page][i]
        self.items.append(letters)
        if RESET_WHEEL_AFTER_PICK:
            self.page = 0
        _set_text(self.picked_labels[i], _spaced(letters))
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
        return _render(self.items)

    def draw(self) -> None:
        now = time.perf_counter()
        for name, shape in self.panels.items():
            self._set_line(shape, ACCENT if now < self._flash_until.get(name, 0) else BORDER)
            shape.draw()
        for stim in (*self.suggestion_text, self.done_text, self.word_text,
                     *self.side_static, self.next_ranges, *self.page_dots):
            stim.draw()
        if self._progress:
            self._draw_progress(*self._progress)
        for i, (label, picked) in enumerate(zip(self.box_labels, self.picked_labels)):
            if now < self._flash_until.get(f"box{i}", 0):
                self.cues[i].draw()
                picked.draw()
            else:
                label.draw()

    def set_progress(self, phase: str | None, fraction: float = 0.0) -> None:
        """Selection timing bar: "go" (green, keep looking) or "break" (grey,
        move your eyes), filled to `fraction`. None hides it."""
        self._progress = None if phase is None else (phase, min(max(fraction, 0.0), 1.0))

    def _draw_progress(self, phase: str, fraction: float) -> None:
        if phase != self._bar_phase:        # colour + label only change between phases
            self.bar_fill.fillColor = ACCENT if phase == "go" else FAINT
            _set_text(self.bar_label, "look" if phase == "go" else "next")
            self._bar_phase = phase
        self.bar_track.draw()
        if fraction > 0:
            self.bar_fill.width = PROGRESS_SIZE[0] * fraction
            self.bar_fill.draw()
        self.bar_label.draw()

    def _flash(self, name: str, seconds: float = FLASH_SECONDS) -> None:
        self._flash_until[name] = time.perf_counter() + seconds

    def _changed(self, what: str) -> None:
        self.action_count += 1
        self._refresh()
        print(f"[ui] {what:<24} ->  {self.text!r}")

    def _refresh(self) -> None:
        """Push the state into the stims. Only what changed gets re-rendered."""
        for label, letters in zip(self.box_labels, self.pages[self.page]):
            _set_text(label, _spaced(letters))
        for i, dot in enumerate(self.page_dots):
            color = TEXT if i == self.page else FAINT
            if self._line_colors.get(id(dot)) != color:
                dot.fillColor = color
                self._line_colors[id(dot)] = color
        other = self.pages[(self.page + 1) % len(self.pages)]
        _set_text(self.next_ranges, "\n".join(range_name(r) for r in other))

        for stim, word in zip(self.suggestion_text, self.suggestions):
            _set_text(stim, word or "suggestion")
            stim.color = TEXT if word else FAINT

        word = self.current_word()
        done = _render(self.items[:len(self.items) - len(word)])
        if len(done) > MAX_TYPED_CHARS:
            done = "…" + done[-MAX_TYPED_CHARS:]
        pending = _render(word)
        if len(pending) > MAX_TYPED_CHARS:
            pending = "…" + pending[-MAX_TYPED_CHARS:]
        _set_text(self.done_text, done)
        _set_text(self.word_text, pending + "|")

    def _set_line(self, shape, color: str) -> None:
        if self._line_colors.get(id(shape)) != color:
            shape.lineColor = color
            self._line_colors[id(shape)] = color


def _render(items: list[str]) -> str:
    return "".join(f"[{range_name(item)}]" if len(item) > 1 else item for item in items)


def _spaced(letters: str) -> str:
    return "  ".join(letters)


def _set_text(stim, text: str) -> None:
    """Re-rendering a TextStim can drop a flicker frame: only do it on a change."""
    if stim.text != text:
        stim.text = text


def _arrow(win, pos, direction: int, aspect: float, size: float = 0.028):
    """A thin arrow (shaft + head) pointing left (-1) or right (+1). Drawn as
    lines because the font's arrow glyphs don't all render."""
    from psychopy import visual

    x, y = pos
    tip = x + direction * size / aspect
    tail = x - direction * size / aspect
    back = tip - direction * 0.6 * size / aspect
    style = dict(units="norm", lineColor=MUTED, lineWidth=3, closeShape=False, fillColor=None)
    return [visual.ShapeStim(win, vertices=[(tail, y), (tip, y)], **style),
            visual.ShapeStim(win, vertices=[(back, y + 0.6 * size), (tip, y), (back, y - 0.6 * size)], **style)]


def _rounded_rect(size, radius: float, aspect: float, steps: int = 6):
    """Vertices of a rounded rectangle in norm units. `radius` is in screen
    heights; x is scaled by the aspect ratio so the corners come out round."""
    import numpy as np

    w, h = size
    ry = min(radius, h / 2)
    rx = min(radius / aspect, w / 2)
    verts = []
    for cx, cy, start in ((w / 2 - rx, h / 2 - ry, 0), (-w / 2 + rx, h / 2 - ry, 90),
                          (-w / 2 + rx, -h / 2 + ry, 180), (w / 2 - rx, -h / 2 + ry, 270)):
        for a in np.radians(np.linspace(start, start + 90, steps)):
            verts.append((cx + rx * np.cos(a), cy + ry * np.sin(a)))
    return verts

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

def run_keys(win, ui: SpellerUI, squares) -> None:
    """No board, no flicker: boxes shown dim grey, 1 / 2 pick."""
    for sq in squares:
        sq.fillColor = sq.lineColor = "#23262e"
    win.recordFrameIntervals = True
    try:
        while handle_keys(ui, boxes=True):
            for sq in squares:
                sq.draw()
            ui.draw()
            win.flip()
    finally:
        _report_frames(win)


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
    dropped_at_mark = 0
    win.recordFrameIntervals = True
    try:
        while True:
            t = clock.getTime()
            if state == "mark":
                phase0 = t
                seen = recorder.prediction_count.value
                actions_at_mark = ui.action_count
                dropped_at_mark = win.nDroppedFrames
                marking = True
                t_mark, state = t, "collecting"

            if state == "feedback":          # grey drains away: next selection is about to start
                ui.set_progress("break", 1 - (t - t_fb) / FEEDBACK_SECONDS)
            elif t - t_mark < GO_SECONDS:
                ui.set_progress("go", (t - t_mark) / GO_SECONDS)
            else:
                ui.set_progress("break", 1.0)

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
                    late = win.nDroppedFrames - dropped_at_mark
                    if late:
                        print(f"[warn] {late} late frame(s) in this selection - flicker timing slipped")
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
    finally:
        _report_frames(win)


def _report_frames(win) -> None:
    """How many frames missed the monitor refresh. Non-zero while flickering
    means the UI is too heavy to draw in time."""
    win.recordFrameIntervals = False
    total = len(win.frameIntervals)
    if total:
        print(f"[frames] {win.nDroppedFrames} late of {total} "
              f"({100 * win.nDroppedFrames / total:.1f}%)")


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