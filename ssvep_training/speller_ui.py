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

# Live timing IS calibration's timing (config.py), minus the cue: dark rest ->
# flicker from black -> dark rest. stimulus.run_trial records the same shape.
FLICKER_SECONDS = cfg.FLICKER_DURATION
FEEDBACK_SECONDS = cfg.INTER_TRIAL_INTERVAL   # dark rest after a letter lands
PICK_FLASH_SECONDS = 0.35           # green box highlight, outlives the rest on purpose
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
        self._flash(f"box{i}", PICK_FLASH_SECONDS)
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
            _set_text(self.bar_label, {"go": "look", "ready": "SPACE to select",
                                      "prepare": "get ready"}.get(phase, "next"))
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


def handle_gestures(ui: SpellerUI, recorder, seen: int) -> int:
    """Dispatch the newest gesture through the same mapping as arrow keys.

    A gesture cancels EEG even when its action is a no-op (empty suggestion).
    """
    from .head import GESTURES
    count, code = recorder.read_gesture()
    if count != seen and 0 <= code < len(GESTURES):
        recorder.cancel_prediction()
        name, *args = KEY_ACTIONS[GESTURES[code]]
        getattr(ui, name)(*args)
    return count


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


def run_flicker(win, ui: SpellerUI, squares, recorder, threshold: float = 0.0,
                manual: bool = False, overlay=()) -> None:
    """One live selection, watched until it is confident enough to commit.

        [dark rest] -> [flicker from black, marker on frame 1] -> [dark, feedback]

    The squares keep flickering while the decision is still open. A first pick is
    attempted at GAZE_DURATION; if it is below `threshold` the window grows by
    GAZE_STEP and the same selection is scored again on more data, up to
    MAX_GAZE_DURATION. Nothing on screen changes while this happens, so an easy
    selection lands fast and only a weak one costs extra seconds -- and no data
    is thrown away, unlike restarting.

    Flicker starts from black because unbroken flicker measurably killed the
    response (SSVEP SNR 1.5 -> 1.0), which is also what stimulus.run_trial
    records. A UI action mid-selection means the user looked away: that
    selection is dropped and a fresh one starts.
    """
    import gc

    from psychopy import core

    from .stimulus import draw_blank, flicker_frame
    from .cca_model import accept_prediction

    if manual:
        from psychopy import event
        event.clearEvents(eventType="keyboard")

    clock = core.Clock()
    state, t_mark, t_req, t_fb, seen, actions_at_mark = "mark", 0.0, 0.0, 0.0, 0, 0
    seen_gesture = recorder.read_gesture()[0]
    if manual:
        state = "ready"
    t_prepare = 0.0
    gaze = cfg.GAZE_DURATION
    awaiting = False
    marking = False
    dropped_at_mark = 0
    win.recordFrameIntervals = True
    gc.disable()                     # a collection mid-flicker would drop frames
    try:
        while True:
            t = clock.getTime()
            if manual:
                # Consume SPACE in every state so presses during a selection
                # cannot queue up another selection after feedback.
                start_pressed = bool(event.getKeys(keyList=["space"]))
                if state == "ready" and start_pressed:
                    t_prepare, state = t, "prepare"
                elif state == "prepare" and t - t_prepare >= cfg.CUE_DURATION:
                    state = "mark"
            if state == "mark":
                seen = recorder.prediction_count.value
                actions_at_mark = ui.action_count
                dropped_at_mark = win.nDroppedFrames
                gaze, awaiting, marking = cfg.GAZE_DURATION, False, True
                t_mark, state = t, "flicker"
                frame = 0

            if state == "flicker":
                # Bar fills over the window currently being collected, so it
                # stretches rather than completing and sitting there.
                ui.set_progress("go", min(1.0, (t - t_mark) / (cfg.VISUAL_LATENCY + gaze)))
                # Phase starts at 0 on the first frame after the marker, exactly
                # as run_trial's clock.reset() does before its flicker loop.
                flicker_frame(squares, cfg.STIMULUS_FREQUENCIES, frame, win.ssvep_refresh)
            else:                    # feedback: squares dark, as in the rest period
                fraction = 1 - (t - t_fb) / FEEDBACK_SECONDS if state == "feedback" else 1.0
                phase = state if state in ("ready", "prepare") else "break"
                ui.set_progress(phase, fraction)
                draw_blank(squares)

            ui.draw()
            for stimulus in overlay:
                stimulus.draw()
            if marking:
                win.callOnFlip(recorder.mark_onset, cfg.LIVE_MARKER)
            win.flip()
            if state == "flicker":
                frame += 1

            if marking:              # first flicker frame is now on screen
                marking = False

            if not handle_keys(ui, boxes=False):
                return
            newest_gesture = handle_gestures(ui, recorder, seen_gesture)
            if newest_gesture != seen_gesture:
                seen_gesture = newest_gesture
                awaiting = False
                t_fb, state = t, "feedback"
                continue

            if state == "flicker":
                if ui.action_count != actions_at_mark:
                    recorder.cancel_prediction()
                    awaiting = False
                    t_fb, state = t, "feedback"   # recover on a dark screen
                elif not awaiting and t - t_mark >= cfg.VISUAL_LATENCY + gaze:
                    late = win.nDroppedFrames - dropped_at_mark
                    if late:
                        print(f"[warn] {late} late frame(s) in this selection - flicker timing slipped")
                    recorder.gaze.value = gaze
                    recorder.request_prediction()
                    t_req, awaiting = t, True
                elif awaiting and recorder.prediction_count.value != seen:
                    seen = recorder.prediction_count.value
                    if not recorder.prediction_is_current():
                        continue  # cancelled/older trial: wait for this trial's result
                    awaiting = False
                    sigma = recorder.last_sigma.value
                    choice = recorder.last_prediction.value
                    if win.nDroppedFrames > dropped_at_mark:
                        choice = -1
                        print("[hold] dropped display frame; rejecting selection")
                    if accept_prediction(choice, sigma, threshold):
                        ui.select_box(choice)
                        t_fb, state = t, "feedback"
                    elif gaze >= cfg.MAX_GAZE_DURATION or choice < 0:
                        print(f"[hold] insufficient evidence ({sigma:+.2f} decoy contrast); no selection")
                        t_fb, state = t, "feedback"
                    else:
                        gaze = min(gaze + cfg.GAZE_STEP, cfg.MAX_GAZE_DURATION)
                        print(f"[look] {sigma:+.1f} contrast < {threshold:+.1f} - "
                              f"keep looking, window now {gaze:.1f}s")
                elif awaiting and (t - t_req > PREDICTION_TIMEOUT or not recorder.is_alive()):
                    print("[warn] no prediction for this selection - starting over")
                    recorder.cancel_prediction()
                    awaiting = False
                    t_fb, state = t, "feedback"
            elif state == "feedback" and t - t_fb >= FEEDBACK_SECONDS:
                state = "ready" if manual else "mark"
    finally:
        gc.enable()
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
    parser.add_argument("--decoder", choices=["cca", "trca"], default="cca",
                        help="cca: sine/cosine correlation, needs no calibration (default). "
                             "trca: templates learned from a calibration session.")
    parser.add_argument("--channels", default="",
                        help="board channels when running CCA without a calibration, e.g. 1,2,3,4,6,8")
    parser.add_argument("--threshold", type=float, default=cfg.CONFIDENCE_THRESHOLD,
                        help="minimum heuristic decoy contrast before a letter is typed; "
                             "0 types every selection, higher rejects more selections; accuracy must be measured")
    parser.add_argument("--save", action="store_true",
                        help="record this live run to training_data/ so it can be analysed "
                             "the same way a calibration is (live trials are marked 99, unlabelled)")
    parser.add_argument("--manual", action="store_true",
                        help="SPACE starts one EEG selection after a preparation cue; waits between choices")
    parser.add_argument("--no-imu", action="store_true",
                        help="disable head gestures and use arrow keys only")
    parser.add_argument("--windowed", action="store_true", help="run in a window instead of fullscreen")
    args = parser.parse_args()
    if cfg.N_TARGETS != 2:
        parser.error(f"the speller has 2 boxes; config.py has {cfg.N_TARGETS} targets")

    recorder = None
    if not args.keys:
        from .recording import RecordingProcess
        from .session import latest_session

        # CCA learns nothing from a session, so it only loads one if asked for.
        session = args.session or (latest_session() if args.decoder == "trca" else None)
        if args.decoder == "trca" and not session:
            sys.exit(f"--decoder trca needs a calibration; none in {cfg.TRAINING_DATA_DIR}. "
                     "Record one, or use --decoder cca.")

        # --channels wins. It used to lose to the session's list, which silently
        # streamed unwanted electrodes into every epoch. CCA is scale-invariant,
        # but extra artifact/noise channels can still overfit short windows.
        if args.channels:
            channels = [int(c) for c in args.channels.split(",")]
            source = "--channels"
        elif session:
            with open(os.path.join(session, "session.json")) as f:
                channels = json.load(f)["eeg_rows"]
            source = os.path.basename(os.path.normpath(session))
        else:
            sys.exit("Pass --channels 1,2,3,4 (CCA needs no calibration), or --session to reuse one.")
        if session and args.decoder == "trca":
            # TRCA has one weight per channel, so live must stream exactly the
            # calibration's channels.
            with open(os.path.join(session, "session.json")) as f:
                trained_on = json.load(f)["eeg_rows"]
            if trained_on != channels:
                sys.exit(f"--decoder trca was trained on channels {trained_on} but you asked for "
                         f"{channels}. TRCA has one weight per channel; they must match.")
        print(f"{args.decoder.upper()} | channels {channels} (from {source})")
        live_dir = None
        if args.save:
            live_dir = os.path.join(cfg.TRAINING_DATA_DIR, time.strftime("live_%Y%m%d_%H%M%S"))
            print(f"Recording this run to {live_dir}")
        recorder = RecordingProcess(port=args.port, predict_session=session, session_dir=live_dir,
                                    channels=channels, decoder=args.decoder,
                                    enable_gestures=not args.no_imu)
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
            if not wait_for_board(win, recorder, "Setting up the board..."):
                return

        squares, cues, _ = build_stimuli(win)
        ui = SpellerUI(win, cues)
        how = ("1 / 2 = left / right box" if args.keys else
               "SPACE, then look at a box" if args.manual else "Look at a box")
        if not wait_for_key(win, f"Speller\n\n{how}: types its letters as one item.\n\n"
                                 "Head left / right = wheel / finish word\n"
                                 "Head up / down = accept suggestion\n"
                                 "Arrow keys also work. Escape quits.\n\n"
                                 "Press SPACE to start.", name="Speller"):
            return
        if recorder:
            from psychopy import visual
            center = visual.TextStim(win, text="+", pos=(0, 0), color="gray", height=.06)
            run_flicker(win, ui, squares, recorder, threshold=args.threshold,
                        manual=args.manual, overlay=[center])
        else:
            run_keys(win, ui, squares)
    except KeyboardInterrupt:
        print("Stopped: Ctrl+C in the terminal.")
    finally:
        if recorder:
            if args.save:
                recorder.request_save()
                time.sleep(1.0)          # let the child write raw.npz before it is told to stop
            recorder.stop()
            recorder.join(timeout=10)
        win.close()
        if ui and ui.items:
            print(f"\nTyped: {ui.text!r}")
        core.quit()


if __name__ == "__main__":
    main()
