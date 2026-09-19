"""
PsychoPy flicker stimuli and the single-trial routine.

The four targets are white squares in the corners of the screen. Each one is
turned white/black every frame according to the sign of a sine wave at its
assigned frequency:

    colour = white if sin(2*pi*f*t) >= 0 else black

Because the update happens once per `win.flip()` (i.e. locked to the monitor
refresh), the on-screen flicker is frame-accurate. On a 60 Hz display the
chosen frequencies (6.67 / 8.57 / 10 / 12 Hz) land on near-integer frame
counts, so the flicker stays rock-steady -- this is exactly what we verified
with the Arduino + photoresistor rig (see README).

A trial has three phases:
    1. CUE      : a coloured frame highlights the target the user must look at.
    2. FLICKER  : all four targets flash for FLICKER_DURATION seconds.
    3. CAPTURE  : the recording flag is raised so the child process grabs the
                  window that just elapsed.

Run this file directly (`python ssvep/stimulus.py`) for a no-display, no-board
check that each target's square wave really lands on its assigned frequency.
"""

from __future__ import annotations

import numpy as np
from psychopy import visual, core, event

try:
    from . import config as cfg
except ImportError:           # run directly as a script for the self-check
    import config as cfg

# Corner positions in normalised units: top-left, top-right, bottom-left, bottom-right.
_POSITIONS = [(-0.5, 0.5), (0.5, 0.5), (-0.5, -0.5), (0.5, -0.5)]
_SIZE = (0.5, 0.5)


def build_window() -> visual.Window:
    """Create the full-screen (or windowed) PsychoPy window."""
    return visual.Window(
        size=cfg.WINDOW_SIZE,
        fullscr=cfg.FULLSCREEN,
        screen=cfg.STIMULUS_SCREEN,
        winType="pyglet",
        # In PsychoPy's "rgb" space the range is -1..1, so [0,0,0] is MID-GREY,
        # not black. Black maximises surround contrast (and matches the black
        # the squares use), which is what you want for a strong SSVEP.
        color=[-1, -1, -1],
        colorSpace="rgb",
        units="norm",
        allowGUI=False,
    )


def build_stimuli(win: visual.Window):
    """Create the four flicker squares and their cue frames."""
    squares, cues = [], []
    for pos in _POSITIONS:
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
    return squares, cues


def show_message(win: visual.Window, text: str) -> None:
    """Draw a centred one-off message (e.g. 'Setting up board...')."""
    visual.TextStim(win, text=text, pos=(0, 0), color="white", height=0.08).draw()
    win.flip()


def run_trial(win, squares, cues, target_idx, recording_process,
              freqs=cfg.STIMULUS_FREQUENCIES) -> bool:
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
            for sq in squares:
                sq.fillColor = [-1, -1, -1]
                sq.draw()
            cues[target_idx].draw()
            win.flip()
            if _escape_pressed():
                return False

    # --- Phase 2: FLICKER ------------------------------------------------ #
    clock.reset()
    while clock.getTime() < cfg.FLICKER_DURATION:
        t = clock.getTime()
        _flicker_frame(squares, freqs, t)
        win.flip()
        if _escape_pressed():
            return False

    # --- Phase 3: CAPTURE ------------------------------------------------ #
    # Blank the squares and raise the flag. The rising edge makes the child
    # grab the last CAPTURE_SAMPLES (the flicker window that just ended).
    for sq in squares:
        sq.fillColor = [-1, -1, -1]
        sq.draw()
    win.flip()
    recording_process.recording_flag.value = True

    # Short rest / gaze-shift; also gives the child time to save or predict.
    core.wait(cfg.INTER_TRIAL_INTERVAL)
    return True


def _is_on(f: float, t: float) -> bool:
    """The whole flicker design: white while sin(2*pi*f*t) is non-negative."""
    return np.sin(2 * np.pi * f * t) >= 0


def _flicker_frame(squares, freqs, t: float) -> None:
    """Set each square white/black based on the sign of its sine wave at time t."""
    for sq, f in zip(squares, freqs):
        sq.fillColor = [1, 1, 1] if _is_on(f, t) else [-1, -1, -1]
        sq.draw()


def _escape_pressed() -> bool:
    return "escape" in event.getKeys(keyList=["escape"])


if __name__ == "__main__":
    # Needs neither a display nor a board: replay 5 s of 60 Hz frames through
    # the real _is_on() the frame loop uses, FFT the square wave each target
    # actually produces, and check it peaks where we asked it to.
    fps, secs = cfg.EXPECTED_REFRESH_HZ, 5.0
    times = np.arange(int(fps * secs)) / fps
    bins = np.fft.rfftfreq(times.size, 1 / fps)

    peaks = []
    for freq in cfg.STIMULUS_FREQUENCIES:
        wave = np.array([_is_on(freq, t) for t in times], dtype=float)
        peak = bins[np.argmax(np.abs(np.fft.rfft(wave - wave.mean())))]
        peaks.append(peak)
        print(f"  {freq:5.2f} Hz target -> dominant {peak:5.2f} Hz  "
              f"(error {abs(peak - freq):.3f} Hz)")
        assert abs(peak - freq) < 0.2, f"{freq} Hz target flickers at {peak} Hz"

    assert len(set(peaks)) == cfg.N_TARGETS, f"targets share a frequency: {peaks}"
    print(f"OK: {cfg.N_TARGETS} distinct frequencies, each within 0.2 Hz "
          f"on a {fps} Hz monitor.")
