# Background board process - continuous recording with flicker-onset markers

from __future__ import annotations

import time
import traceback
from multiprocessing import Event, Process, Value

# s of zero samples after startup before declaring the board dead
NO_DATA_TIMEOUT = 6.0

# s between get_board_data() calls
DRAIN_INTERVAL = 0.02

# s between checks of the shared counters
POLL_INTERVAL = 0.001 

# s to wait for a trial's data to arrive before giving up 
PREDICT_TIMEOUT = 3.0   


class RecordingProcess(Process):
    """Owns the board in a child process; records everything, marks flicker onsets."""

    def __init__(self, port: str | None = None, session_dir: str | None = None,
                 settle: float | None = None, predict_session: str | None = None,
                 channels=None, idle: bool = False):
        """session_dir: where to save the recording (calibration); None = don't save.
        predict_session: calibration to train TRCA on for live typing; None = no predictions.
        idle: predictions may be -1 ("neither") if the calibration has rest trials."""
        super().__init__(daemon=True)
        self.port = port
        self.channels = channels          # board channels 1-8; None = all
        self.session_dir = session_dir
        self.settle = settle
        self.predict_session = predict_session
        self.idle = idle
        self._threshold = None            # rest threshold on the winning TRCA score

        self.ready = Event()
        self.failed = Value("b", False)
        self.marker_code = Value("i", 0)
        self.onset_count = Value("i", 0)
        self.save_count = Value("i", 0)
        self.predict_count = Value("i", 0)
        self.last_prediction = Value("i", -1)
        self.prediction_count = Value("i", 0)
        self._running = Event()
        self._running.set()

    def mark_onset(self, code: int) -> None:
        """Call at the first flicker frame: the child stamps `code` into the EEG."""
        self.marker_code.value = int(code)
        with self.onset_count.get_lock():
            self.onset_count.value += 1

    def request_save(self) -> None:
        with self.save_count.get_lock():
            self.save_count.value += 1

    def request_prediction(self) -> None:
        """Call after a live trial's flicker ends; watch prediction_count for the answer."""
        with self.predict_count.get_lock():
            self.predict_count.value += 1

    def stop(self) -> None:
        self._running.clear()

    def run(self) -> None:
        import numpy as np

        from . import config as cfg
        from .board import SETTLE_SECONDS, close_board, open_board
        from .session import save_session

        board = None
        try:
            model = self._train_model() if self.predict_session else None
            kwargs = {"channels": self.channels} if self.channels else {}
            board = self._open_board(SETTLE_SECONDS, open_board, **kwargs)
            if model is not None and board.eeg_rows != self._model_rows:
                raise ValueError(f"calibration used channels {self._model_rows} but this board is "
                                 f"streaming {board.eeg_rows} - they must match.")
            if model is not None and board.rate != self._model_rate:
                raise ValueError(f"calibration was recorded at {self._model_rate} Hz but this board streams "
                                 f"at {board.rate} Hz - recalibrate on this board.")
        except Exception:
            traceback.print_exc()
            self.failed.value = True
            if board is not None:
                close_board(board)
            return

        meta = self._meta(board)
        chunks = []

        recorded = 0

        def drain():
            nonlocal recorded
            chunk = board.shim.get_board_data()
            if chunk.shape[1]:
                chunks.append(chunk)
                recorded += chunk.shape[1]

        def save():
            if self.session_dir and chunks:
                save_session(self.session_dir, np.hstack(chunks), {**meta, "n_markers": seen_onset})

        def predict() -> bool:
            """Classify the latest live trial if its data has all arrived. True when done."""
            choice = self._predict(model, chunks, board)
            if choice is None:
                return False
            self.last_prediction.value = choice
            with self.prediction_count.get_lock():
                self.prediction_count.value += 1
            return True

        seen_onset, seen_save, seen_predict, last_drain = 0, 0, 0, 0.0
        pending_since = None
        # Every trial is filtered with FILTER_HISTORY seconds of EEG before its
        # onset, so don't report ready until that much has actually ARRIVED.
        # Waiting on the clock alone once let a wedged board pass this check and
        # a whole 18-block session recorded zero samples.
        ready_at = time.monotonic() + cfg.FILTER_HISTORY + 0.5
        ready_samples = int(cfg.FILTER_HISTORY * board.rate * 0.5)
        try:
            while self._running.is_set():
                if not self.ready.is_set():
                    now_m = time.monotonic()
                    if recorded >= ready_samples and now_m >= ready_at:
                        self.ready.set()
                    elif now_m > ready_at + NO_DATA_TIMEOUT:
                        print(f"[board] only {recorded} samples in "
                              f"{cfg.FILTER_HISTORY + 0.5 + NO_DATA_TIMEOUT:.0f}s - the board is not "
                              f"streaming.\n[board] Unplug the USB-C for 10s, replug, and try again.")
                        self.failed.value = True
                        break
                if self.onset_count.value != seen_onset:
                    seen_onset = self.onset_count.value
                    board.shim.insert_marker(float(self.marker_code.value))
                if self.save_count.value != seen_save:
                    seen_save = self.save_count.value
                    drain()
                    save()
                if model is not None and self.predict_count.value != seen_predict:
                    seen_predict = self.predict_count.value
                    pending_since = time.monotonic()
                if pending_since is not None:
                    drain()
                    if predict():
                        pending_since = None
                    elif time.monotonic() - pending_since > PREDICT_TIMEOUT:
                        print("[predict] trial data never arrived - no prediction for this selection")
                        pending_since = None
                now = time.monotonic()
                if now - last_drain >= DRAIN_INTERVAL:
                    drain()
                    last_drain = now
                time.sleep(POLL_INTERVAL)
            drain()
            save()
        except Exception:
            traceback.print_exc()
            self.failed.value = True
        finally:
            close_board(board)

    # --- paradigm hooks: a subclass swaps these three to decode something else --- #

    def _open_board(self, default_settle, open_board, **kwargs):
        """Which board to open. A paradigm that does not need the IMU overrides this."""
        return open_board(port=self.port,
                          settle=default_settle if self.settle is None else self.settle,
                          **kwargs)

    def _meta(self, board) -> dict:
        """What goes in session.json beside the recording."""
        from .session import session_meta

        return session_meta(board)

    def _predict(self, model, chunks, board) -> int | None:
        """Classify the latest live trial: target index, -1 for "neither", or
        None while its data is still arriving. `chunks` is every drained chunk
        so far, oldest first."""
        import numpy as np

        from . import config as cfg
        from .session import epoch_at, flicker_samples, history_samples
        from .trca_model import decide, scores

        need = history_samples(board.rate) + flicker_samples(board.rate)
        recent, n = [], 0
        for chunk in reversed(chunks):            # just enough recent data
            recent.append(chunk)
            n += chunk.shape[1]
            if n >= need + 2 * board.rate:
                break
        if not recent:
            return None
        data = np.hstack(recent[::-1])
        onsets = np.flatnonzero(data[board.marker_row] == cfg.LIVE_MARKER)
        if not len(onsets):
            return None
        epoch = epoch_at(data[board.eeg_rows], int(onsets[-1]), board.rate)
        if epoch is None:                          # flicker data not all here yet
            return None
        trial_scores = scores(model, epoch[:, :, None])[0]
        choice = decide(trial_scores, self._threshold)
        print("[predict] scores " + " / ".join(f"{s:.3f}" for s in trial_scores)
              + (" -> neither" if choice < 0 else f" -> {cfg.TARGET_LETTERS[choice]}"))
        return choice

    def _train_model(self):
        """Fit TRCA on the calibration session; its targets must match today's config."""
        from . import config as cfg
        from .session import load_trials
        from .trca_model import fit, fit_idle

        trials = load_trials(self.predict_session)
        if [round(f, 3) for f in trials.freqs] != [round(f, 3) for f in cfg.STIMULUS_FREQUENCIES]:
            raise ValueError(f"calibration used {trials.freqs} Hz but config.py now uses "
                             f"{cfg.STIMULUS_FREQUENCIES} Hz - recalibrate.")
        self._model_rate = trials.rate
        self._model_rows = trials.rows
        model = fit(trials)
        print(f"[predict] TRCA trained on {trials.eeg.shape[-1]} trials from {self.predict_session}")
        if self.idle:
            rest = load_trials(self.predict_session, rest=True)
            if rest.eeg.shape[-1]:
                idle = fit_idle(trials, rest)
                self._threshold = idle.threshold
                print(f"[predict] idle detection on: threshold {idle.threshold:.3f} from "
                      f"{rest.eeg.shape[-1]} rest trials (ignored {100 * idle.rest_ignored:.0f}% of rest, "
                      f"kept {100 * idle.picks_kept:.0f}% of picks)")
            else:
                print("[predict] no rest trials in this calibration - idle detection off "
                      "(recalibrate to add them)")
        return model
