"""The board process, decoding motor imagery instead of SSVEP.

Everything about owning the board - the child process, markers, draining,
saving, the ready/failed flags - is ssvep_training.RecordingProcess. This
subclass only replaces the three paradigm hooks:

    _meta       motor-imagery session.json instead of flicker frequencies
    _train_model    load (or fit) the decoder for this calibration
    _predict    filter bank -> analysis window -> class, or -1 for "neither"

The live filter bank is the same FilterBank class calibration uses, fed every
chunk exactly once in arrival order from the first sample of the stream. That is
what keeps live features identical to training features.
"""

from __future__ import annotations

import numpy as np

from ssvep_training.recording import RecordingProcess

from . import config as cfg


class MIRecorder(RecordingProcess):
    """RecordingProcess with the motor-imagery decoder."""

    def __init__(self, *args, mode: str = "clench", classes: list[int] | None = None,
                 model: str | None = None, margin: float | None = None,
                 board: str = cfg.DEFAULT_BOARD, **kwargs):
        super().__init__(*args, **kwargs)
        self.board = board
        self.mode = mode
        self.classes = classes or cfg.classes_for(cfg.DEFAULT_CLASSES)
        self.model_name = model
        self.margin = margin
        # Live filtering state, built in the child process once the rate is known.
        self._bank = None
        self._buffers: dict[str, np.ndarray] = {}
        self._done = 0              # chunks already filtered
        self._filtered = 0          # samples already filtered (absolute)
        self._onsets: list[int] = []
        self._used = -1             # the last onset already answered
        self._keep = 0              # samples of filtered history to hold on to
        self._cap = 0               # ceiling on that, if an onset is never answered

    # --- hooks ------------------------------------------------------------- #

    def _open_board(self, default_settle, open_board, **kwargs):
        """The plain Knight by default: nothing here reads the IMU."""
        board_id = cfg.BOARDS[self.board]
        print(f"[board] opening the {self.board} Knight (board {board_id})")
        return open_board(port=self.port, board_id=board_id,
                          settle=default_settle if self.settle is None else self.settle,
                          **kwargs)

    def _meta(self, board) -> dict:
        from .session import session_meta

        return session_meta(board, self.mode, self.classes)

    def _train_model(self):
        """Load the model saved beside this calibration, or fit one now."""
        import os

        from joblib import load

        from . import model as models
        from .session import load_trials, model_path, session_info

        info = session_info(self.predict_session)
        path = model_path(self.predict_session)
        if os.path.exists(path):
            saved = load(path)
            print(f"[predict] loaded {saved['name']} from {path} "
                  f"(margin {saved['margin']:.2f}, {saved['mode']} calibration)")
        else:
            trials = load_trials(self.predict_session)
            if not len(trials):
                raise ValueError(f"no usable trials in {self.predict_session}")
            name = self.model_name or cfg.DEFAULT_MODEL
            fitted = models.build(name, trials.rate, trials.mode).fit(trials.bands, trials.labels)
            saved = {"model": fitted, "classes": np.array(fitted.classes_),
                     "margin": cfg.DEFAULT_MARGIN, "name": name, "rate": trials.rate,
                     "mode": trials.mode, "window": [cfg.ANALYSIS_START, cfg.ANALYSIS_DURATION],
                     "names": trials.names}
            print(f"[predict] fitted {name} on {len(trials)} trials {trials.counts()} "
                  f"from {self.predict_session}")
            print(f"[predict] no margin was tuned for it - run evaluate --save for that; "
                  f"using {cfg.DEFAULT_MARGIN:.2f}")

        if self.margin is not None:
            saved["margin"] = self.margin
        if self.model_name and self.model_name != saved["name"]:
            print(f"[predict] NOTE: {path} holds {saved['name']}, not {self.model_name} - "
                  "delete it and rerun evaluate --save to change model")
        if saved["mode"] == "clench":
            print("[predict] this model was trained on CLENCH trials: imagining will score "
                  "lower than the calibration suggested.")
        print(f"[predict] classes {[cfg.CLASS_NAMES[c] for c in saved['classes']]}, "
              f"window {saved['window'][0]:g}-{saved['window'][0] + saved['window'][1]:g} s, "
              f"margin {saved['margin']:.2f}")
        print(cfg.montage_banner())

        # The parent refuses to run if today's board streams different rows or a
        # different rate than the calibration did - spatial filters are per electrode.
        self._model_rate = info["rate"]
        self._model_rows = list(info["eeg_rows"])
        self._window = saved["window"]
        return saved

    def _predict(self, model, chunks, board) -> int | None:
        """Class to type, -1 for "neither", or None while the window is still arriving."""
        from .model import decide

        self._consume(chunks, board)
        if not self._onsets or self._onsets[-1] == self._used:
            return None
        onset = self._onsets[-1]
        start = onset + int(round(self._window[0] * board.rate))
        end = start + int(round(self._window[1] * board.rate))
        if self._filtered < end:
            return None                                   # hold still arriving

        first = self._filtered - next(iter(self._buffers.values())).shape[1]
        if start < first:
            print("[predict] the window scrolled out of the buffer - no pick this selection")
            self._used = onset
            return None
        window = {band: buffer[None, :, start - first:end - first]
                  for band, buffer in self._buffers.items()}

        proba = model["model"].predict_proba(window)[0]
        choice = decide(proba, model["classes"], model["margin"])
        names = [cfg.CLASS_NAMES[c] for c in model["classes"]]
        print("[predict] " + "  ".join(f"{n} {p:.2f}" for n, p in zip(names, proba))
              + (" -> neither" if choice < 0 else f" -> {cfg.CLASS_NAMES[choice]}"))
        self._used = onset
        return cfg.BOX_CLASSES.index(choice) if choice in cfg.BOX_CLASSES else choice

    # --- live filtering ---------------------------------------------------- #

    def _consume(self, chunks, board) -> None:
        """Filter every chunk that hasn't been filtered yet, in arrival order.

        Calibration filters the whole recording in one pass from sample 0; doing
        the same here incrementally gives bit-identical output, which is the
        whole point of keeping the filter state.
        """
        from .filters import FilterBank

        if self._bank is None:
            self._bank = FilterBank(board.rate, len(board.eeg_rows))
            self._keep = int(round(30 * board.rate))
            self._cap = int(round(120 * board.rate))
            print(f"[predict] live filter bank started; the first "
                  f"{cfg.PRIME_SECONDS:g} s of stream are still settling")

        while self._done < len(chunks):
            chunk = chunks[self._done]
            self._done += 1
            if not chunk.shape[1]:
                continue
            filtered = self._bank.process(chunk[board.eeg_rows])
            for marker in np.flatnonzero(chunk[board.marker_row] == cfg.LIVE_MARKER):
                self._onsets.append(self._filtered + int(marker))
            self._filtered += chunk.shape[1]
            for band, values in filtered.items():
                previous = self._buffers.get(band)
                joined = values if previous is None else np.hstack([previous, values])
                self._buffers[band] = joined[:, -self._retain(board.rate):]

    def _retain(self, rate: float) -> int:
        """How much filtered history to keep.

        Normally 30 s is plenty - a prediction is asked for seconds after its
        marker. But if the board process falls behind and drains a long backlog
        in one go, a fixed window would discard the very marker being waited on,
        which silently costs a pick. So never trim past an unanswered onset
        (up to a 120 s ceiling, in case one is never claimed).
        """
        keep = self._keep
        if self._onsets and self._onsets[-1] != self._used:
            needed = self._filtered - self._onsets[-1] + int(round(rate * cfg.HOLD_DURATION + rate))
            keep = max(keep, needed)
        return min(keep, self._cap)
