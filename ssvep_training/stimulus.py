# flicker stimulus and single-trial routine

from __future__ import annotations

import gc

import numpy as np
from psychopy import core, event, visual

from . import config as cfg

_BLACK = [-1, -1, -1]
_WHITE = [1, 1, 1]

FULLSCREEN = True
WINDOW_SIZE = [1280, 800]   # only used when not fullscreen


def is_on(f: float, t: float) -> bool:
    """The flicker rule: a square is white while sin(2*pi*f*t) >= 0."""
    return np.sin(2 * np.pi * f * t) >= 0


def build_window(fullscreen: bool = FULLSCREEN) -> visual.Window:
    win = visual.Window(
        size=WINDOW_SIZE, fullscr=fullscreen, screen=0, winType="pyglet",
        # PsychoPy "rgb" runs -1..1, so [0, 0, 0] would be mid-grey. Black
        # maximises contrast with the white flicker.
        color=_BLACK, colorSpace="rgb", units="norm", allowGUI=False,
    )
    # Frames noticeably longer than one refresh count as late. Recording is only
    # switched on during the flicker, as PsychoPy advises: rests would otherwise
    # count as dropped frames.
    win.refreshThreshold = 1 / cfg.EXPECTED_REFRESH_HZ + 0.004
    win.recordFrameIntervals = False
    _bring_to_front(win)
    return win


def _bring_to_front(win: visual.Window) -> None:
    """macOS: ask for the window to join the current desktop and take keyboard focus.

    Limit (verified on macOS 15.5): if the launching app (e.g. Claude, a terminal)
    is in macOS fullscreen -- its own Space -- the window still opens on another
    desktop, whatever its level. The cursor vanishes and keys arrive, but nothing
    shows. wait_for_key() detects this and tells the user to leave fullscreen or
    swipe to the window.
    """
    try:
        ns = win.winHandle._nswindow
        can_join_all_spaces, fullscreen_auxiliary = 1 << 0, 1 << 8   # NSWindowCollectionBehavior
        ns.setCollectionBehavior_(can_join_all_spaces | fullscreen_auxiliary)
        ns.orderFrontRegardless()
        win.winHandle.activate()
    except Exception:
        pass   # not macOS / not pyglet: nothing to do


def _on_current_desktop(win: visual.Window) -> bool:
    """macOS: is the window on the desktop (Space) the user is looking at? True if unknown."""
    try:
        return bool(win.winHandle._nswindow.isOnActiveSpace())
    except Exception:
        return True


def _app_is_active() -> bool:
    """macOS: is this app the one receiving keystrokes? (True elsewhere / if unknown)"""
    try:
        from pyglet.libs.darwin.cocoapy import ObjCClass
        return bool(ObjCClass("NSApplication").sharedApplication().isActive())
    except Exception:
        return True


def layout(n: int):
    """Square centres and size (normalised units, screen = -1..1) for n targets."""
    if n == 2:                                   # left / right, spacing from config
        return [(-cfg.TARGET_X, 0.0), (cfg.TARGET_X, 0.0)], tuple(cfg.TARGET_SIZE)
    if n == 4:                                   # corners: TL, TR, BL, BR
        return [(-0.5, 0.5), (0.5, 0.5), (-0.5, -0.5), (0.5, -0.5)], (0.5, 0.5)
    width = min(0.5, 1.6 / n)                    # anything else: one row
    xs = np.linspace(-1 + 1 / n, 1 - 1 / n, n)
    return [(float(x), 0.0) for x in xs], (width, width)


def build_stimuli(win: visual.Window):
    """One flicker square per target, its cue outline, and its letter label."""
    positions, size = layout(cfg.N_TARGETS)
    squares, cues, labels = [], [], []
    for pos, letter in zip(positions, cfg.TARGET_LETTERS):
        squares.append(visual.Rect(win, units="norm", width=size[0], height=size[1], pos=pos,
                                   anchor="center", fillColor=_BLACK, lineColor=_BLACK))
        cues.append(visual.Rect(win, units="norm", width=size[0] * 1.15, height=size[1] * 1.15,
                                pos=pos, anchor="center", fillColor=None, lineColor="red", lineWidth=8))
        # Letter outside the square (above it, or below for a bottom row) so it
        # doesn't change the square's brightness.
        side = -1 if pos[1] < 0 else 1
        label_y = pos[1] + side * (size[1] / 2 + 0.17)
        labels.append(visual.TextStim(win, text=letter, pos=(pos[0], label_y),
                                      color="gray", height=0.1, bold=True))
    return squares, cues, labels


_message_stims = {}


def show_message(win: visual.Window, text: str) -> None:
    """Draw a centred message. One TextStim per window, re-rendered only when the
    text changes: building one every frame is slow enough to drop frames."""
    stim = _message_stims.get(id(win))
    if stim is None:
        stim = _message_stims[id(win)] = visual.TextStim(
            win, text=text, pos=(0, 0), color="white", height=0.07, wrapWidth=1.8)
    if stim.text != text:
        stim.text = text
    stim.draw()
    win.flip()


_FOCUS_HINT = "\n\n(Keys not reaching this window - click it, then press SPACE)"
_SPACE_HINT = ("[screen] The flicker window opened on ANOTHER DESKTOP, so you can't see it.\n"
               "         Take the app you launched from out of fullscreen (Ctrl+Cmd+F), or swipe\n"
               "         to the window (3 fingers / Ctrl+Left/Right). Escape quits.")


def wait_for_key(win: visual.Window, text: str, key: str = "space", name: str | None = None) -> bool:
    """Show `text` until `key` is pressed or the window is clicked. False on Escape.

    If macOS isn't sending keystrokes to this app (e.g. the terminal kept focus),
    the screen and terminal say so -- a click both focuses the window and continues.
    """
    name = name or text.splitlines()[0][:40]
    print(f"[screen] {name} - waiting for SPACE / click")
    event.clearEvents()
    mouse = event.Mouse(win=win)
    was_pressed = any(mouse.getPressed())
    clock, warned, warned_space = core.Clock(), False, False
    while True:
        if not warned_space and clock.getTime() > 0.5 and not _on_current_desktop(win):
            print(_SPACE_HINT)
            warned_space = True
        focused = _app_is_active() or clock.getTime() < 1.0   # give focus a moment to arrive
        show_message(win, text if focused else text + _FOCUS_HINT)
        if not focused and not warned:
            print("[focus] the flicker window isn't receiving keys - click it, then press SPACE")
            warned = True
        keys = event.getKeys(keyList=[key, "escape"])
        if "escape" in keys:
            print(f"[screen] {name} - Escape")
            return False
        if key in keys:
            print(f"[screen] {name} - {key}")
            return True
        pressed = any(mouse.getPressed())
        if pressed and not was_pressed:   # a fresh click counts as SPACE
            print(f"[screen] {name} - click")
            return True
        was_pressed = pressed


def wait_for_board(win: visual.Window, recorder, text: str) -> bool:
    """Show a progress message until the board process is ready. False on failure/Escape."""
    clock = core.Clock()
    while not recorder.ready.is_set():
        if recorder.failed.value or not recorder.is_alive():
            show_message(win, "Board setup failed - see the terminal for details.")
            core.wait(3)
            return False
        show_message(win, f"{text}\n\n{clock.getTime():.0f} s")
        if _escape_pressed():
            return False
    return True


def run_trial(win, squares, cues, target_idx: int, recorder, marker_code: int,
              overlay=(), freqs=cfg.STIMULUS_FREQUENCIES) -> bool:
    """One cue -> flicker -> rest trial. True to continue, False if Escape was pressed.

    target_idx  : square to cue (0..3), or -1 for no cue (live typing)
    marker_code : stamped into the EEG at flicker onset (session.encode_marker / LIVE_MARKER)
    overlay     : extra static stimuli (letter labels, typed text) drawn every frame
    """
    clock = core.Clock()
    cue = cfg.TARGET_LETTERS[target_idx] if 0 <= target_idx < len(cues) else "none"
    print(f"[trial] cue {cue}, marker {marker_code}")

    # --- CUE ------------------------------------------------------------- #
    if 0 <= target_idx < len(cues):
        clock.reset()
        while clock.getTime() < cfg.CUE_DURATION:
            _draw_blank(squares, overlay)
            cues[target_idx].draw()
            win.flip()
            if _escape_pressed():
                print("[trial] Escape during cue")
                return False

    # --- FLICKER --------------------------------------------------------- #
    # Garbage collection is paused so a collection can't stall a frame.
    dropped_before = win.nDroppedFrames
    win.recordFrameIntervals = True
    gc.disable()
    try:
        marked = False
        clock.reset()
        while clock.getTime() < cfg.FLICKER_DURATION:
            _flicker_frame(squares, freqs, clock.getTime())
            for stim in overlay:
                stim.draw()
            win.flip()
            if not marked:                    # first flicker frame is now on screen
                recorder.mark_onset(marker_code)
                marked = True
            if _escape_pressed():
                print("[trial] Escape during flicker")
                return False
    finally:
        gc.enable()
        win.recordFrameIntervals = False
    dropped = win.nDroppedFrames - dropped_before
    if dropped:
        print(f"[warn] {dropped} late frame(s) during flicker - timing slipped slightly this trial")

    # --- REST ------------------------------------------------------------ #
    _draw_blank(squares, overlay)
    win.flip()
    core.wait(cfg.INTER_TRIAL_INTERVAL)
    return True


def flicker_frame(squares, freqs, t: float) -> None:
    """Draw one flicker frame at absolute time t. Callers that keep a clock
    running across selections get an unbroken, phase-continuous flicker."""
    for sq, f in zip(squares, freqs):
        sq.fillColor = _WHITE if is_on(f, t) else _BLACK
        sq.draw()


def _flicker_frame(squares, freqs, t: float) -> None:
    for sq, f in zip(squares, freqs):
        sq.fillColor = _WHITE if is_on(f, t) else _BLACK
        sq.draw()


def _draw_blank(squares, overlay=()) -> None:
    for sq in squares:
        sq.fillColor = _BLACK
        sq.draw()
    for stim in overlay:
        stim.draw()


def _escape_pressed() -> bool:
    return "escape" in event.getKeys(keyList=["escape"])
