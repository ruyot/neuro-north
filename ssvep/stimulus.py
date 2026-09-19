"""
PsychoPy flicker stimuli and the single-trial routine.

The four targets are white squares in the corners of the screen, labelled
A (top-left), B (top-right), C (bottom-left), D (bottom-right). Each one is
turned white/black every frame according to the sign of a sine wave at its
assigned frequency:

    colour = white if sin(2*pi*f*t) >= 0 else black

Because the update happens once per `win.flip()` (i.e. locked to the monitor
refresh), the on-screen flicker is frame-accurate. On a 60 Hz display the
chosen frequencies (6.67 / 8.57 / 10 / 12 Hz) land on near-integer frame
counts, so the flicker stays steady. Dropped frames are counted per trial and
reported, because a dropped frame means the real flicker frequency drifted.

A trial has three phases:
    1. CUE      : a coloured frame highlights the target the user must look at.
    2. FLICKER  : all four targets flash for FLICKER_DURATION seconds.
    3. CAPTURE  : the recording flag is raised so the child process grabs the
                  window that just elapsed.
"""

from __future__ import annotations

import gc

import numpy as np
from psychopy import core, event, visual

from . import config as cfg

# Corner positions in normalised units: top-left, top-right, bottom-left, bottom-right.
_POSITIONS = [(-0.5, 0.5), (0.5, 0.5), (-0.5, -0.5), (0.5, -0.5)]
_SIZE = (0.5, 0.5)
_BLACK = [-1, -1, -1]
_WHITE = [1, 1, 1]


def build_window() -> visual.Window:
    """Create the full-screen (or windowed) PsychoPy window."""
    win = visual.Window(
        size=cfg.WINDOW_SIZE,
        fullscr=cfg.FULLSCREEN,
        screen=cfg.STIMULUS_SCREEN,
        winType="pyglet",
        color=[0, 0, 0],
        colorSpace="rgb",
        units="norm",
        allowGUI=False,
    )
    # Count frames that take noticeably longer than one refresh. Recording is
    # switched on only during the flicker (see run_trial), as PsychoPy advises:
    # pauses like the inter-trial rest would otherwise count as dropped frames.
    win.refreshThreshold = 1 / cfg.EXPECTED_REFRESH_HZ + 0.004
    win.recordFrameIntervals = False
    return win


def build_stimuli(win: visual.Window):
    """Create the four flicker squares, their cue frames, and their letter labels."""
    squares, cues, labels = [], [], []
    for pos, letter in zip(_POSITIONS, cfg.TARGET_LETTERS):
        squares.append(visual.Rect(
            win=win, units="norm", width=_SIZE[0], height=_SIZE[1],
            pos=pos, anchor="center", fillColor="white", lineColor="black",
        ))
        # A thick coloured outline used to mark the current target.
        cues.append(visual.Rect(
            win=win, units="norm", width=_SIZE[0] * 1.15, height=_SIZE[1] * 1.15,
            pos=pos, anchor="center", fillColor=None, lineColor="red",
            lineWidth=8,
        ))
        # Letter sits outside the square (above top row, below bottom row) so it
        # doesn't change the square's brightness.
        label_y = pos[1] + np.sign(pos[1]) * (_SIZE[1] / 2 + 0.08)
        labels.append(visual.TextStim(win, text=letter, pos=(pos[0], label_y),
                                      color="gray", height=0.1, bold=True))
    return squares, cues, labels


_message_stims = {}


def show_message(win: visual.Window, text: str) -> None:
    """Draw a centred message (e.g. 'Setting up board...').

    Reuses one TextStim per window and only re-renders when the text changes:
    building a new TextStim every frame is slow enough to drop frames.
    """
    stim = _message_stims.get(id(win))
    if stim is None:
        stim = _message_stims[id(win)] = visual.TextStim(
            win, text=text, pos=(0, 0), color="white", height=0.08, wrapWidth=1.8)
    if stim.text != text:
        stim.text = text
    stim.draw()
    win.flip()


def wait_for_key(win: visual.Window, text: str, key: str = "space") -> bool:
    """Show `text` until `key` is pressed. Returns False if Escape was pressed."""
    event.clearEvents()
    while True:
        show_message(win, text)
        keys = event.getKeys(keyList=[key, "escape"])
        if "escape" in keys:
            return False
        if key in keys:
            return True


def wait_for_board(win: visual.Window, recorder, text: str) -> bool:
    """Show a progress message until the recording process is ready. False on failure/Escape."""
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


def run_trial(win, squares, cues, target_idx, recording_process,
              freqs=cfg.STIMULUS_FREQUENCIES, overlay=()) -> bool:
    """
    Run one cue -> flicker -> capture trial.

    Parameters
    ----------
    target_idx : int
        Which square the user should look at this trial (0..3). Used to draw
        the cue and, in collect mode, to label the saved data. Pass -1 for
        free-running (predict) mode, where no cue is shown.
    recording_process : RecordingProcess
        Shared background acquisition process; we toggle its recording flag.
    overlay : iterable of PsychoPy stimuli
        Extra static stimuli (letter labels, typed text) drawn every frame.

    Returns
    -------
    bool
        True to continue, False if the user pressed Escape.
    """
    clock = core.Clock()
    cued = 0 <= target_idx < len(cues)

    # --- Phase 1: CUE ---------------------------------------------------- #
    # Highlight the target so the user knows where to look. In collect mode we
    # publish the label so the child saves under the right filename. In predict
    # mode (target_idx == -1) we skip the cue entirely.
    recording_process.recording_flag.value = False
    if cued:
        recording_process.label_index.value = target_idx

        clock.reset()
        while clock.getTime() < cfg.CUE_DURATION:
            _draw_blank(squares, overlay)
            cues[target_idx].draw()
            win.flip()
            if _escape_pressed():
                return False

    # --- Phase 2: FLICKER ------------------------------------------------ #
    # Garbage collection is paused so a collection can't stall a frame.
    dropped_before = win.nDroppedFrames
    win.recordFrameIntervals = True
    gc.disable()
    try:
        clock.reset()
        while clock.getTime() < cfg.FLICKER_DURATION:
            t = clock.getTime()
            _flicker_frame(squares, freqs, t)
            for stim in overlay:
                stim.draw()
            win.flip()
            if _escape_pressed():
                return False
    finally:
        gc.enable()
        win.recordFrameIntervals = False
    dropped = win.nDroppedFrames - dropped_before
    if dropped:
        print(f"[warn] {dropped} late frame(s) during flicker - timing slipped slightly this trial")

    # --- Phase 3: CAPTURE ------------------------------------------------ #
    # Blank the squares and raise the flag. The rising edge makes the child
    # grab the last CAPTURE_SAMPLES (the flicker window that just ended).
    _draw_blank(squares, overlay)
    win.flip()
    recording_process.recording_flag.value = True

    # Short rest / gaze-shift; also gives the child time to save or predict.
    core.wait(cfg.INTER_TRIAL_INTERVAL)
    return True


def _flicker_frame(squares, freqs, t: float) -> None:
    """Set each square white/black based on the sign of its sine wave at time t."""
    for sq, f in zip(squares, freqs):
        white = np.sin(2 * np.pi * f * t) >= 0
        sq.fillColor = _WHITE if white else _BLACK
        sq.draw()


def _draw_blank(squares, overlay=()) -> None:
    for sq in squares:
        sq.fillColor = _BLACK
        sq.draw()
    for stim in overlay:
        stim.draw()


def _escape_pressed() -> bool:
    return "escape" in event.getKeys(keyList=["escape"])
