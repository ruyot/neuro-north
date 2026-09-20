# Background board process: one USB owner, raw capture and trial-specific admission.

from __future__ import annotations

import math
import os
import signal
import time
import traceback
from datetime import datetime
from multiprocessing import Event, Process, SimpleQueue, Value

from . import config as cfg
from .signal_quality import Quality

NO_DATA_TIMEOUT = 6.0
DRAIN_INTERVAL = 0.02
POLL_INTERVAL = 0.001
PREDICT_TIMEOUT = 3.0


class RecordingProcess(Process):
    """Own the board in one child; the parent sends small ordered control events."""

    def __init__(self, port: str | None = None, session_dir: str | None = None,
                 settle: float | None = None, predict_session: str | None = None,
                 channels=None, diagnostic: bool = False, decoder: str = "trca"):
        super().__init__()
        if decoder not in cfg.DECODERS:
            raise ValueError(f"Unknown decoder {decoder!r}; choose from {cfg.DECODERS}.")
        if diagnostic and predict_session is not None:
            raise ValueError("Diagnostic capture cannot also predict from a calibration.")
        if channels is not None:
            from .knight import validate_eeg_channels

            channels = validate_eeg_channels(channels)
        if predict_session is not None and session_dir is None:
            session_dir = os.path.join(cfg.TRAINING_DATA_DIR, datetime.now().strftime("live_%Y%m%d_%H%M%S_%f"))
        if session_dir is not None:
            session_dir = os.path.abspath(session_dir)
            if predict_session is not None and os.path.realpath(session_dir) == os.path.realpath(predict_session):
                raise ValueError("Recording output must not be the calibration input directory.")
            if os.path.exists(session_dir) and (not os.path.isdir(session_dir) or os.listdir(session_dir)):
                raise ValueError(f"Recording output must be new or empty: {session_dir}")
        self.port, self.channels = port, channels
        self.session_dir, self.predict_session = session_dir, predict_session
        self.settle, self.diagnostic = settle, diagnostic
        self.decoder = decoder
        self.configured, self.ready = Event(), Event()
        self.failed = Value("b", False)
        self._events = SimpleQueue()
        self._baseline_requested = False
        self._next_trial_id = 0
        self._finished_trials = set()
        self.last_prediction = Value("i", -1, lock=False)
        self.last_quality = Value("i", int(Quality.OK), lock=False)
        self.prediction_trial = Value("q", 0)
        self.gesture = Value("i", -1)
        self.gesture_count = Value("i", 0)
        self._running = Event()
        self._running.set()

    def begin_baseline(self, display_refresh_hz: float | None = None) -> None:
        if not self.configured.is_set() or self._baseline_requested:
            raise RuntimeError("Begin the quiet baseline once, after board configuration.")
        if display_refresh_hz is not None:
            if isinstance(display_refresh_hz, bool) or not math.isfinite(display_refresh_hz) or display_refresh_hz <= 0:
                raise ValueError("Measured display refresh must be finite and positive.")
            display_refresh_hz = float(display_refresh_hz)
        self._events.put(("baseline", display_refresh_hz))
        self._baseline_requested = True

    def mark_onset(self, code: int | None = None) -> int:
        if not self.ready.is_set() or self.failed.value or not self._running.is_set():
            raise RuntimeError("The recorder must be ready before marking a trial.")
        if code is not None and (type(code) is not int or code <= 0):
            raise ValueError("Calibration marker codes must be positive integers.")
        self._next_trial_id += 1
        trial_id = self._next_trial_id
        self._events.put(("onset", trial_id, -trial_id if code is None else code, time.monotonic_ns()))
        return trial_id

    def finish_trial(self, trial_id: int, dropped_frames: int) -> None:
        if type(trial_id) is not int or not 0 < trial_id <= self._next_trial_id:
            raise ValueError("Cannot finish an unknown trial id.")
        if type(dropped_frames) is not int or dropped_frames < 0:
            raise ValueError("Dropped frames must be a nonnegative integer count.")
        if trial_id in self._finished_trials:
            raise ValueError("Trial completion was already recorded.")
        self._events.put(("finish", trial_id, dropped_frames))
        self._finished_trials.add(trial_id)

    def request_prediction(self, trial_id: int) -> None:
        if self.predict_session is None or trial_id not in self._finished_trials:
            raise ValueError("Request prediction for a completed live trial.")
        self._events.put(("predict", trial_id))

    def request_save(self) -> None:
        self._events.put(("save",))

    def poll_prediction(self, trial_id: int) -> tuple[int | None, Quality] | None:
        with self.prediction_trial.get_lock():
            if trial_id <= 0 or self.prediction_trial.value != trial_id:
                return None
            quality = Quality(self.last_quality.value)
            if quality != Quality.OK:
                return None, quality
            target = self.last_prediction.value
            if not 0 <= target < cfg.N_TARGETS:
                return None, Quality.INVALID_RESULT
            return target, Quality.OK

    def stop(self) -> None:
        self._running.clear()

    def run(self) -> None:
        # Ctrl-C belongs to the parent, which requests stop and waits for saving.
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        import numpy as np

        from .board import SETTLE_SECONDS, close_board, open_board
        from .session import epoch_at, flicker_samples, history_samples, save_session, session_meta
        from .signal_quality import fit_baseline

        board, model, meta, gestures = None, None, None, None
        # ponytail: session-sized buffering; use chunked storage if long captures outgrow RAM.
        chunks, marker_offsets, trial_codes = [], {}, {}
        recorded = 0
        baseline_started = None
        last_sample = time.monotonic()
        pending_trial, pending_since, latest_requested = None, None, 0

        def publish(trial_id, target=None, quality=Quality.OK):
            with self.prediction_trial.get_lock():
                self.last_prediction.value = -1 if target is None else target
                self.last_quality.value = int(quality)
                self.prediction_trial.value = trial_id

        def drain():
            nonlocal recorded, last_sample, gestures
            chunk = board.shim.get_board_data()
            if chunk.ndim != 2 or chunk.shape[0] != 22:
                raise ValueError("Knight capture must deliver all 22 SDK rows.")
            if not chunk.shape[1]:
                return
            last_sample = time.monotonic()
            if baseline_started is None:
                return  # Warning/setup time is never retained or fitted as baseline.
            for offset in np.flatnonzero(chunk[board.marker_row]):
                code = chunk[board.marker_row, offset]
                if np.isfinite(code) and code == int(code):
                    marker_offsets.setdefault(int(code), recorded + int(offset))
            chunks.append(chunk)
            recorded += chunk.shape[1]
            if gestures is not None:
                try:
                    name = gestures.feed(chunk[gyro])
                    if name is not None:
                        print(f"[head] {name}")
                        self.gesture.value = GESTURES.index(name)
                        with self.gesture_count.get_lock():
                            self.gesture_count.value += 1
                except Exception:
                    traceback.print_exc()
                    gestures = None

        def interval(start, stop):
            """Copy only the requested context, not the entire retained session."""
            selected, end = [], recorded
            for chunk in reversed(chunks):
                begin = end - chunk.shape[1]
                lo, hi = max(start, begin), min(stop, end)
                if lo < hi:
                    selected.append(chunk[:, lo - begin:hi - begin])
                if begin <= start:
                    break
                end = begin
            return np.hstack(selected[::-1])

        def save():
            if self.session_dir is not None and meta is not None and chunks:
                save_session(self.session_dir, np.hstack(chunks), meta)
                print(f"[recording] saved {recorded} samples to {self.session_dir}")

        def handle_events(final=False):
            nonlocal baseline_started, last_sample, pending_trial, pending_since, latest_requested
            # There is one consumer. A producer arriving after empty() is handled
            # on the next sleep-containing loop; no empty-queue get can block it.
            while not self._events.empty():
                event = self._events.get()
                kind = event[0]
                if kind == "baseline":
                    if final:
                        continue
                    if baseline_started is not None:
                        raise ValueError("Quiet baseline was already started.")
                    discarded = board.shim.get_board_data()  # never retain pre-instruction samples
                    baseline_started = time.monotonic()
                    if discarded.shape[1]:
                        last_sample = baseline_started
                    meta["display_refresh_hz"] = event[1]
                elif kind == "onset":
                    _, trial_id, code, requested = event
                    if trial_id in trial_codes or str(code) in meta["trials"]:
                        raise ValueError(f"Duplicate trial id or calibration marker: {trial_id}, {code}.")
                    if not self.ready.is_set():
                        raise ValueError("Trial onset arrived before the quiet baseline completed.")
                    submitted = time.monotonic_ns()
                    board.shim.insert_marker(float(code))
                    trial_codes[trial_id] = code
                    meta["trials"][str(code)] = {
                        "trial_id": trial_id, "onset_request_monotonic_ns": requested,
                        "marker_submit_monotonic_ns": submitted, "dropped_frames": None,
                    }
                elif kind == "finish":
                    _, trial_id, dropped = event
                    if trial_id not in trial_codes:
                        raise ValueError(f"Completion for unknown trial {trial_id}.")
                    trial = meta["trials"][str(trial_codes[trial_id])]
                    if trial["dropped_frames"] is not None:
                        raise ValueError(f"Duplicate completion for trial {trial_id}.")
                    trial["dropped_frames"] = dropped
                elif kind == "predict":
                    if final:
                        continue
                    trial_id = event[1]
                    if model is None or trial_codes.get(trial_id) != -trial_id:
                        raise ValueError(f"Prediction requested for unknown/non-live trial {trial_id}.")
                    if meta["trials"][str(-trial_id)]["dropped_frames"] is None:
                        raise ValueError(f"Prediction requested before trial {trial_id} completed.")
                    if trial_id > latest_requested:
                        latest_requested = pending_trial = trial_id
                        pending_since = time.monotonic()
                elif kind == "save":
                    if not final:
                        drain()
                        save()
                else:
                    raise ValueError(f"Unknown recorder control event {kind!r}.")

        def finish_baseline(error=None):
            if error is None:
                try:
                    meta["quality_baseline"] = fit_baseline(interval(0, baseline_samples), meta)
                except ValueError as exc:
                    error = str(exc)
            if error is not None:
                meta["quality_baseline"] = None
                meta["quality_baseline_error"] = error
                print(f"[quality] {error}", flush=True)
                if not self.diagnostic:
                    raise ValueError(error)
            self.ready.set()

        def predict(trial_id):
            onset = marker_offsets.get(-trial_id)
            if onset is None:
                return None
            hist, flick = history_samples(board.rate), flicker_samples(board.rate)
            if onset < hist:
                return None, Quality.INCOMPLETE
            if recorded < onset + flick:
                return None
            context = interval(onset - hist, onset + flick)
            epoch, quality = epoch_at(context, hist, meta)
            if quality != Quality.OK:
                return None, quality
            result = model.predict(epoch[:, :, None])[0]
            if not np.isfinite(result) or result != int(result) or not 0 <= result < cfg.N_TARGETS:
                return None, Quality.INVALID_RESULT
            return int(result), Quality.OK

        try:
            model = self._train_model() if self.predict_session is not None else None
            kwargs = {"channels": self.channels} if self.channels is not None else {}
            board = open_board(port=self.port,
                               settle=SETTLE_SECONDS if self.settle is None else self.settle, **kwargs)
            if model is not None and board.eeg_rows != self._model_rows:
                raise ValueError(f"Calibration used channels {self._model_rows}, live board uses {board.eeg_rows}; recalibrate.")
            if model is not None and board.rate != self._model_rate:
                raise ValueError(f"Calibration rate {self._model_rate} differs from live {board.rate}; recalibrate.")
            meta = session_meta(board)
            if self.predict_session is not None:
                meta["decoder"] = self.decoder
            baseline_samples = int(round(cfg.QUALITY_BASELINE_SECONDS * board.rate))
            try:
                from .head import GESTURES, Gestures, gyro_rows

                gestures, gyro = Gestures(board.rate), gyro_rows(board.board_id)
            except Exception:
                traceback.print_exc()
                gestures = None
            last_sample = time.monotonic()
            self.configured.set()
            last_drain = 0.0
            while self._running.is_set():
                handle_events()
                now = time.monotonic()
                if now - last_drain >= DRAIN_INTERVAL:
                    drain()
                    last_drain = now
                if now - last_sample > NO_DATA_TIMEOUT:
                    raise RuntimeError(f"No board samples for {NO_DATA_TIMEOUT:g} seconds; stopping and saving raw evidence.")
                if baseline_started is not None and not self.ready.is_set():
                    elapsed = now - baseline_started
                    if recorded >= baseline_samples and elapsed >= cfg.QUALITY_BASELINE_SECONDS:
                        finish_baseline()
                    elif elapsed > cfg.QUALITY_BASELINE_SECONDS + NO_DATA_TIMEOUT:
                        finish_baseline(f"Quality baseline incomplete: {recorded}/{baseline_samples} samples after {elapsed:.1f} seconds.")
                if pending_trial is not None:
                    trial_id = pending_trial
                    result = predict(trial_id)
                    if result is None and time.monotonic() - pending_since > PREDICT_TIMEOUT:
                        result = (None, Quality.INCOMPLETE)
                    if result is not None:
                        handle_events()  # A newer queued request supersedes a slow computation.
                        if pending_trial == trial_id:
                            publish(trial_id, *result)
                            pending_trial = pending_since = None
                time.sleep(POLL_INTERVAL)
        except Exception:
            traceback.print_exc()
            self.failed.value = True
        finally:
            if pending_trial is not None:
                publish(pending_trial, quality=Quality.INCOMPLETE)
            if board is not None:
                try:
                    if meta is not None:
                        try:
                            handle_events(final=True)
                        except Exception:
                            traceback.print_exc()
                            self.failed.value = True
                    try:
                        drain()
                    except Exception:
                        traceback.print_exc()
                        self.failed.value = True
                    try:
                        save()
                    except Exception:
                        traceback.print_exc()
                        self.failed.value = True
                finally:
                    close_board(board)

    def _train_model(self):
        from .session import load_trials
        from .trca_model import _require_training_trials, fit

        trials = load_trials(self.predict_session)
        if [round(f, 3) for f in trials.freqs] != [round(f, 3) for f in cfg.STIMULUS_FREQUENCIES]:
            raise ValueError(f"Calibration used {trials.freqs} Hz but config now uses {cfg.STIMULUS_FREQUENCIES}; recalibrate.")
        self._model_rate, self._model_rows = trials.rate, trials.rows
        if self.decoder == "trca":
            model = fit(trials)
        else:
            from .fbcca import FBCCA

            _require_training_trials(trials)
            model = FBCCA(trials.rate, trials.freqs, car=self.decoder == "fbcca-car")
        print(f"[predict] {self.decoder}: {trials.eeg.shape[-1]} accepted calibration trials from {self.predict_session}")
        return model


def main() -> None:
    """Capture raw EEG/IMU with a visible terminal baseline instruction, without PsychoPy."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Record native Knight EEG/IMU without flashing stimuli.")
    parser.add_argument("--seconds", type=float, required=True, help="capture duration after the 10-second baseline")
    parser.add_argument("--output", required=True, help="new or empty output directory")
    parser.add_argument("--port", help="override PORT_PATH from .env")
    args = parser.parse_args()
    if not math.isfinite(args.seconds) or args.seconds <= 0:
        parser.error("--seconds must be finite and positive")
    try:
        recorder = RecordingProcess(port=args.port, session_dir=args.output, diagnostic=True)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    error = False
    try:
        recorder.start()
        for ready in (recorder.configured, recorder.ready):
            while not ready.is_set():
                if recorder.failed.value or not recorder.is_alive():
                    raise RuntimeError("Recording worker failed during setup; see the error above.")
                time.sleep(0.05)
            if ready is recorder.configured:
                print("Hold still with eyes open for 10 seconds. Keep your jaw relaxed.", flush=True)
                recorder.begin_baseline(None)
        print(f"[recording] Capturing {args.seconds:g} seconds after baseline. Ctrl-C stops and saves.", flush=True)
        end = time.monotonic() + args.seconds
        while time.monotonic() < end:
            if recorder.failed.value or not recorder.is_alive():
                raise RuntimeError("Recording worker failed; retained raw data is being saved.")
            time.sleep(min(0.05, max(0.0, end - time.monotonic())))
    except KeyboardInterrupt:
        print("\n[recording] Stopping and saving retained samples...", flush=True)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        error = True
    finally:
        recorder.stop()
        if recorder.pid is not None:
            recorder.join()
    if error or recorder.failed.value or recorder.exitcode not in (None, 0):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
