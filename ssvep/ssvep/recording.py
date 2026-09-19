"""
Background EEG acquisition process.

Why a separate process?
------------------------
PsychoPy must flip the screen on every monitor refresh (every ~16.6 ms on a
60 Hz display) to keep the flicker frequencies exact. If we asked it to also
talk to the board, run filters and TRCA, those calls could block a flip and
smear the stimulus timing -- which corrupts the very SSVEP we are trying to
measure. So the board lives in its own `multiprocessing.Process`:

  * The PsychoPy (parent) process only ever sets a few shared flags.
  * This (child) process owns the board, and on the *rising edge* of the
    recording flag it grabs the latest window from the ring buffer and either
    saves it (collect mode) or classifies it and presses a key (predict mode).

Shared state (all `multiprocessing.Value`):
  recording_flag : bool  - parent sets True to request one capture.
  block_index    : int   - which training block we're on (collect mode).
  label_index    : int   - the target the user is cued to look at (collect mode).
  last_prediction: int   - most recent predicted target (predict mode, -1 = none).

`null_recorder()` at the bottom is the `--no-board` stand-in: it exposes the
same flags so the stimulus can run with no hardware attached.
"""

from __future__ import annotations

import os
import sys
import time
from multiprocessing import Process, Value, Event
from types import SimpleNamespace

from . import config as cfg


class RecordingProcess(Process):
    """Owns the board in a child process and reacts to the recording flag."""

    def __init__(self, mode: str, data_dir: str = cfg.TRAINING_DATA_DIR,
                 n_blocks: int | None = None, key_map: dict | None = None,
                 variant: str | None = None):
        super().__init__()
        assert mode in ("collect", "predict")
        self.mode = mode
        self.data_dir = data_dir
        self.n_blocks = n_blocks
        self.key_map = key_map if key_map is not None else cfg.KEY_MAP
        self.variant = variant          # None -> cfg.BOARD_VARIANT

        # Shared, process-safe state read/written by the parent.
        self.recording_flag = Value("b", False)
        self.block_index = Value("i", 1)
        self.label_index = Value("i", 0)
        self.last_prediction = Value("i", -1)
        # Increments once per classified window. The parent pairs a prediction
        # with the trial that produced it by watching this counter change --
        # comparing `last_prediction` to its previous value instead would drop
        # every repeated letter ("aa", "abba"), and a slow classifier could let
        # one trial's result be read as the next trial's.
        self.prediction_count = Value("i", 0)

        self._running = Event()
        self._running.set()

    # --------------------------------------------------------------------- #
    # Child-process entry point
    # --------------------------------------------------------------------- #
    def run(self) -> None:
        # Import inside run() so the heavy libraries load in the CHILD process.
        import numpy as np
        from .board import KnightBoard
        from .preprocessing import filter_eeg, extract_channel_matrix, crop_indices

        board = KnightBoard(cfg.SERIAL_PORT, cfg.NUM_CHANNELS, cfg.CHANNEL_GAIN,
                            variant=self.variant)
        board.start_stream()

        # Window sizes come from the LIVE board, never a constant: the Knight
        # streams at 125 Hz but BrainFlow's synthetic board runs at 250 Hz.
        capture = int(round(cfg.FLICKER_DURATION * board.sr)) + 1
        crop = crop_indices(board.sr)

        # Predict mode needs a trained model up front.
        model = None
        if self.mode == "predict":
            # The speller draws its own letters on the stimulus screen, so a
            # keyboard backend is a bonus, not a requirement -- never kill the
            # session over it. Branch on the platform rather than catching
            # ImportError: a stale non-Windows pydirectinput install blows up
            # inside ctypes on `windll`, which is not an ImportError.
            pdi = None
            try:
                if sys.platform == "win32":
                    import pydirectinput as pdi   # Windows / DirectX games
                else:
                    import pyautogui as pdi       # macOS, Linux (X11 / XWayland)
                pdi.FAILSAFE = False
            except Exception as exc:
                # Compositors like Hyprland start Xwayland with no -auth file,
                # so python-xlib dies on a missing ~/.Xauthority even though
                # XWayland is running fine. `touch ~/.Xauthority` fixes it.
                print(f"[predict] no keyboard backend ({type(exc).__name__}); "
                      "letters still appear on screen. To type into other apps, "
                      "run: touch ~/.Xauthority")
            # ponytail: no Wayland backend; add ydotool if it ever matters.
            from .trca_model import fit_model
            model = fit_model(self.data_dir, self.n_blocks)

        if self.mode == "collect":
            os.makedirs(self.data_dir, exist_ok=True)

        prev_flag = False
        while self._running.is_set():
            flag = self.recording_flag.value

            # Act only on the rising edge (False -> True) so each request
            # triggers exactly one capture.
            if flag and not prev_flag:
                data = board.get_latest(capture)
                if data.shape[1] >= capture:
                    filter_eeg(data, board.eeg_channels, board.sr)
                    window = extract_channel_matrix(data, board.eeg_channels)

                    if self.mode == "collect":
                        self._save(window, np)
                    else:
                        self._predict(window, model, crop, np, pdi)

            prev_flag = flag

            # Without this the loop pegs a whole core, which steals CPU from
            # PsychoPy's render thread and costs you dropped flips -- i.e. it
            # corrupts the very flicker timing this project depends on. 5 ms is
            # far tighter than the 500 ms inter-trial gap, so no capture is missed.
            time.sleep(0.005)

        board.stop_stream()

    # --------------------------------------------------------------------- #
    # Mode-specific handlers
    # --------------------------------------------------------------------- #
    def _save(self, window, np) -> None:
        """Write one captured trial to block_{block}_{label+1}.csv."""
        block = self.block_index.value
        trial = self.label_index.value + 1  # file trial index is 1-based
        path = os.path.join(self.data_dir, f"block_{block}_{trial}.csv")
        header = ",".join(cfg.ELECTRODE_LABELS[:window.shape[1]])
        np.savetxt(path, window, delimiter=",", header=header,
                   comments="", fmt="%.7f")
        print(f"[collect] saved {os.path.basename(path)}")

    def _predict(self, window, model, crop, np, pdi) -> None:
        """Classify one captured trial, publish it, and type the mapped key."""
        # meegkit expects (samples, channels, trials); add a trial axis + crop.
        trial = np.expand_dims(window, axis=2)[crop]
        target = int(model.predict(trial)[0])
        self.last_prediction.value = target
        # Publish the value BEFORE bumping the counter: the parent treats a
        # counter change as "a fresh prediction is readable", so the ordering
        # matters.
        self.prediction_count.value += 1

        freq = cfg.STIMULUS_FREQUENCIES[target]
        key = self.key_map.get(target)
        print(f"[predict] target {target} ({freq} Hz) -> key '{key}'")
        if key is not None and pdi is not None:
            pdi.press(key)

    def stop(self) -> None:
        self._running.clear()


def null_recorder():
    """Stand-in recorder for `--no-board`: run_trial pokes it, nothing listens."""
    noop = lambda *a, **k: None
    return SimpleNamespace(recording_flag=SimpleNamespace(value=False),
                           block_index=SimpleNamespace(value=1),
                           label_index=SimpleNamespace(value=0),
                           last_prediction=SimpleNamespace(value=-1),
                           prediction_count=SimpleNamespace(value=0),
                           start=noop, stop=noop, join=noop)
