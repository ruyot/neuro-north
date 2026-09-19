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
    saves it (collect mode) or classifies it (predict mode).

Shared state (all `multiprocessing` primitives):
  ready           : Event - set once the board is streaming with channels on
                            (and, in predict mode, the model is trained).
  failed          : bool  - set if startup failed; the parent should give up.
  recording_flag  : bool  - parent sets True to request one capture.
  block_index     : int   - which training block we're on (collect mode).
  label_index     : int   - the target the user is cued to look at (collect mode).
  last_prediction : int   - most recent predicted target (predict mode, -1 = none).
  prediction_count: int   - bumped after every prediction so the parent can
                            tell a fresh result from the previous one.
"""

from __future__ import annotations

import os
import time
import traceback
from multiprocessing import Event, Process, Value

from . import config as cfg


class RecordingProcess(Process):
    """Owns the board in a child process and reacts to the recording flag."""

    def __init__(self, mode: str, serial_port: str | None = cfg.SERIAL_PORT,
                 data_dir: str | None = None, n_blocks: int | None = None,
                 synthetic: bool = False):
        """data_dir: collect -> folder to save into; predict -> session to train on
        (None = latest real session, or latest synthetic one if `synthetic`)."""
        super().__init__(daemon=True)
        assert mode in ("collect", "predict")
        if mode == "collect" and data_dir is None:
            raise ValueError("collect mode needs a data_dir to save into")
        # Passed explicitly: on macOS the child is spawned fresh, so any
        # runtime changes to `config` in the parent would not be visible here.
        self.mode = mode
        self.serial_port = serial_port
        self.data_dir = data_dir
        self.n_blocks = n_blocks
        self.synthetic = synthetic

        # Shared, process-safe state read/written by the parent.
        self.ready = Event()
        self.failed = Value("b", False)
        self.recording_flag = Value("b", False)
        self.block_index = Value("i", 1)
        self.label_index = Value("i", 0)
        self.last_prediction = Value("i", -1)
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
        from .preprocessing import crop_indices, extract_channel_matrix, filter_eeg

        board = None
        try:
            board = KnightBoard(self.serial_port, cfg.NUM_CHANNELS, cfg.CHANNEL_GAIN,
                                synthetic=self.synthetic)
            board.start_stream()

            model = None
            if self.mode == "predict":
                from .trca_model import fit_model, resolve_data_dir, session_rate
                data_dir = resolve_data_dir(self.data_dir, self.synthetic)
                rate = session_rate(data_dir)
                if rate != board.sr:
                    raise ValueError(
                        f"calibration {os.path.basename(data_dir)} was recorded at {rate} Hz "
                        f"but this board streams at {board.sr} Hz - recalibrate on this board.")
                model = fit_model(data_dir, self.n_blocks)
            else:
                os.makedirs(self.data_dir, exist_ok=True)
        except Exception:
            traceback.print_exc()
            self.failed.value = True
            if board is not None:
                try:
                    board.stop_stream()
                except Exception:
                    pass
            return

        # Window sizes come from the LIVE board rate: the Knight streams at
        # 125 Hz, BrainFlow's synthetic board at 250 Hz.
        capture = cfg.capture_samples(board.sr)
        crop = crop_indices(board.sr)
        self.ready.set()

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
                        self._predict(window, model, crop, np)
                else:
                    print(f"[warn] only {data.shape[1]} samples in buffer, trial skipped")

            prev_flag = flag
            time.sleep(0.001)  # poll at ~1 kHz instead of spinning a CPU core

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

    def _predict(self, window, model, crop, np) -> None:
        """Classify one captured trial and publish the result."""
        # meegkit expects (samples, channels, trials); add a trial axis + crop.
        trial = np.expand_dims(window, axis=2)[crop]
        target = int(model.predict(trial)[0])
        self.last_prediction.value = target
        with self.prediction_count.get_lock():
            self.prediction_count.value += 1

        print(f"[predict] target {target} ({cfg.STIMULUS_FREQUENCIES[target]} Hz) "
              f"-> {cfg.TARGET_LETTERS[target]}")

    def stop(self) -> None:
        self._running.clear()


class _Shared:
    """Stand-in for a multiprocessing Value / Event."""

    def __init__(self, value=None):
        self.value = value

    def is_set(self) -> bool:
        return True


class NullRecorder:
    """Stand-in recorder for --no-board: the stimulus runs, nothing is recorded or predicted."""

    def __init__(self):
        self.ready = _Shared()
        self.failed = _Shared(False)
        self.recording_flag = _Shared(False)
        self.block_index = _Shared(1)
        self.label_index = _Shared(0)
        self.last_prediction = _Shared(-1)
        self.prediction_count = _Shared(0)

    def start(self): pass
    def stop(self): pass
    def join(self, timeout=None): pass
    def is_alive(self) -> bool: return True
