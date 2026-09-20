"""Offline checks for the motor-imagery package: no board, no screen.

A synthetic recording is built in the shape collect.py saves, with an obvious
left/right ERD (mu power drops on the hemisphere opposite the cued hand). The
point is not the accuracy - a fake signal proves nothing about real EEG - but
that the plumbing holds: trials cut where the markers are, the live filter
matches the offline one sample for sample, and a pick comes out the far end.

    python -m pytest tests/test_motor_imagery.py
"""

import os
import tempfile
import unittest

import numpy as np

from motor_imagery import config as cfg
from motor_imagery import model as models
from motor_imagery.evaluate import cross_validate
from motor_imagery.filters import FilterBank, filter_recording
from motor_imagery.session import encode_marker, load_trials, quality_warnings, save_session

RATE = 125
BLOCKS, REPEATS = 6, 2
CLASSES = [cfg.LEFT, cfg.RIGHT, cfg.REST]
CHANNELS = 8


class FakeBoard:
    rate, eeg_rows, marker_row = RATE, list(range(CHANNELS)), CHANNELS


def synth_session(path, seed=0):
    """A recording with mu/beta, slow drift, and ERD on the opposite hemisphere."""
    rng = np.random.default_rng(seed)
    trial = int((cfg.CUE_DURATION + cfg.HOLD_DURATION + 1.1) * RATE)
    lead = int((cfg.PRIME_SECONDS + 5) * RATE)
    n = lead + BLOCKS * REPEATS * len(CLASSES) * trial + RATE * 5

    t = np.arange(n) / RATE
    eeg = rng.normal(0, 8, (CHANNELS, n))
    for channel in range(CHANNELS):
        eeg[channel] += 12 * np.sin(2 * np.pi * 10 * t + rng.uniform(0, 6))
        eeg[channel] += 5 * np.sin(2 * np.pi * 21 * t + rng.uniform(0, 6))
        eeg[channel] += 30 * np.sin(2 * np.pi * 0.3 * t + rng.uniform(0, 6))

    names = cfg.CHANNEL_NAMES
    left = [i for i, n_ in enumerate(names) if n_ in cfg.LEFT_HEMISPHERE]
    right = [i for i, n_ in enumerate(names) if n_ in cfg.RIGHT_HEMISPHERE]

    markers = np.zeros(n)
    cursor = lead
    for block in range(1, BLOCKS + 1):
        order = [c for c in CLASSES for _ in range(REPEATS)]
        rng.shuffle(order)
        for class_id in order:
            onset = cursor + int(cfg.CUE_DURATION * RATE)
            markers[onset] = encode_marker(block, class_id)
            hold = slice(onset, onset + int(cfg.HOLD_DURATION * RATE))
            if class_id == cfg.LEFT:
                eeg[right, hold] *= 0.55
            elif class_id == cfg.RIGHT:
                eeg[left, hold] *= 0.55
            cursor += trial

    data = np.zeros((CHANNELS + 2, n))
    data[:CHANNELS], data[CHANNELS] = eeg, markers
    save_session(path, data, {
        "paradigm": "motor_imagery", "board_id": 66, "rate": RATE,
        "eeg_rows": list(range(CHANNELS)), "names": names, "marker_row": CHANNELS,
        "mode": "imagine", "classes": CLASSES, "class_names": cfg.names_for(CLASSES),
        "n_markers": BLOCKS * REPEATS * len(CLASSES)})
    return data


class MotorImageryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.session = os.path.join(cls._tmp.name, cfg.SESSION_PREFIX + "test")
        cls.data = synth_session(cls.session)
        cls.trials = load_trials(cls.session)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_chunked_filtering_matches_one_pass(self):
        """The live path filters chunk by chunk, calibration in one pass. If those
        two ever differ, the model is scored on features it was not trained on -
        the SSVEP branch already lost accuracy to a mismatch like this."""
        eeg = self.data[:CHANNELS]
        whole = filter_recording(eeg, RATE)
        bank, pieces, rng, i = FilterBank(RATE, CHANNELS), {b: [] for b in cfg.BANDS}, \
            np.random.default_rng(1), 0
        while i < eeg.shape[1]:
            step = int(rng.integers(1, 40))          # ragged, like the board delivers
            for band, values in bank.process(eeg[:, i:i + step]).items():
                pieces[band].append(values)
            i += step
        for band in cfg.BANDS:
            np.testing.assert_allclose(np.hstack(pieces[band]), whole[band], atol=1e-9)

    def test_trials_are_cut_at_every_marker(self):
        self.assertEqual(len(self.trials), BLOCKS * REPEATS * len(CLASSES))
        self.assertEqual(self.trials.skipped, 0)
        self.assertEqual(set(self.trials.counts()), {"left", "right", "rest"})
        samples = int(round(cfg.ANALYSIS_DURATION * RATE))
        self.assertEqual(self.trials.bands["8-30"].shape, (len(self.trials), CHANNELS, samples))

    def test_markers_inside_the_priming_stretch_are_dropped(self):
        """The filter bank starts from a zero state, so early trials are unusable."""
        data = self.data.copy()
        data[CHANNELS] = 0
        data[CHANNELS, int(2 * RATE)] = encode_marker(1, cfg.LEFT)       # 2 s in: too early
        data[CHANNELS, int((cfg.PRIME_SECONDS + 5) * RATE)] = encode_marker(1, cfg.RIGHT)
        path = os.path.join(self._tmp.name, cfg.SESSION_PREFIX + "priming")
        save_session(path, data, {
            "rate": RATE, "eeg_rows": list(range(CHANNELS)), "names": cfg.CHANNEL_NAMES,
            "marker_row": CHANNELS, "mode": "imagine", "classes": CLASSES, "n_markers": 2})
        trials = load_trials(path)
        self.assertEqual(len(trials), 1)
        self.assertEqual(trials.skipped, 1)

    def test_quality_warnings_flag_saturated_channels(self):
        data = self.data.copy()
        data[0, :2 * RATE] = 333_333.3
        path = os.path.join(self._tmp.name, cfg.SESSION_PREFIX + "saturated")
        save_session(path, data, {
            "rate": RATE, "eeg_rows": list(range(CHANNELS)), "names": cfg.CHANNEL_NAMES,
            "marker_row": CHANNELS, "mode": "imagine", "classes": CLASSES, "n_markers": 0})

        warnings = quality_warnings(path)

        self.assertTrue(any(warning.startswith("FC4:") for warning in warnings))
        self.assertTrue(any("clipped" in warning for warning in warnings))

    def test_models_run_and_separate_a_planted_signal(self):
        for name in ("tangent", "fb-tangent", "fbcsp", "logvar"):
            with self.subTest(model=name):
                proba, classes = cross_validate(self.trials, name)
                self.assertEqual(proba.shape, (len(self.trials), len(classes)))
                argmax = classes[np.argmax(proba, axis=1)]
                self.assertGreater((argmax == self.trials.labels).mean(), 0.5)

    def test_decide_prefers_rest_unless_a_hand_clears_the_margin(self):
        classes = np.array(CLASSES)
        self.assertEqual(models.decide(np.array([0.8, 0.1, 0.1]), classes, 0.15), cfg.LEFT)
        self.assertEqual(models.decide(np.array([0.1, 0.8, 0.1]), classes, 0.15), cfg.RIGHT)
        self.assertEqual(models.decide(np.array([0.4, 0.1, 0.5]), classes, 0.15), -1)
        # A hand ahead of rest, but by less than the margin: type nothing.
        self.assertEqual(models.decide(np.array([0.45, 0.1, 0.44]), classes, 0.15), -1)
        self.assertEqual(models.decide(np.array([0.45, 0.1, 0.44]), classes, 0.0), cfg.LEFT)

    def test_margin_tuning_respects_the_false_pick_budget(self):
        rng = np.random.default_rng(3)
        labels = np.array(([cfg.LEFT] * 20) + ([cfg.RIGHT] * 20) + ([cfg.REST] * 20))
        proba = rng.dirichlet([1, 1, 1], size=60)            # pure noise
        tuned = models.tune_margin(proba, labels, np.array(CLASSES), max_false_pick=0.1)
        self.assertLessEqual(tuned.false_pick, 0.1)

    def test_motor_recorder_waits_for_motor_filter_prime(self):
        from motor_imagery.recording import MIRecorder

        recorder = MIRecorder(predict_session=self.session)

        self.assertEqual(recorder._ready_seconds(), cfg.PRIME_SECONDS)

    def test_live_prediction_matches_the_offline_window(self):
        from joblib import dump, load

        from motor_imagery.recording import MIRecorder

        fitted = models.build("tangent", RATE).fit(self.trials.bands, self.trials.labels)
        saved = {"model": fitted, "classes": np.array(fitted.classes_), "margin": 0.0,
                 "name": "tangent", "rate": RATE, "mode": "imagine",
                 "window": [cfg.ANALYSIS_START, cfg.ANALYSIS_DURATION], "names": cfg.CHANNEL_NAMES}
        dump(saved, os.path.join(self.session, cfg.MODEL_FILE))
        saved = load(os.path.join(self.session, cfg.MODEL_FILE))

        live = self.data.copy()
        live[CHANNELS] = 0
        onset = int((cfg.PRIME_SECONDS + 8) * RATE)
        live[CHANNELS, onset] = cfg.LIVE_MARKER
        hold = slice(onset, onset + int(cfg.HOLD_DURATION * RATE))
        right = [i for i, n_ in enumerate(cfg.CHANNEL_NAMES) if n_ in cfg.RIGHT_HEMISPHERE]
        live[right, hold] *= 0.55                                    # a left-hand trial

        recorder = MIRecorder(predict_session=self.session)
        recorder._window = saved["window"]
        board = FakeBoard()

        ready = onset + int((sum(saved["window"]) + 0.3) * RATE)
        rng = np.random.default_rng(2)
        chunks, i = [], 0
        while i < ready:
            step = int(rng.integers(1, 40))
            chunks.append(live[:, i:i + step])
            i += step

        half = chunks[:len(chunks) // 2]
        self.assertIsNone(recorder._predict(saved, half, board),
                          "predicted before the whole window had arrived")
        self.assertEqual(recorder._predict(saved, chunks, board), 0)   # 0 = left box

        start = onset + int(saved["window"][0] * RATE)
        end = start + int(saved["window"][1] * RATE)
        first = recorder._filtered - recorder._buffers["8-30"].shape[1]
        offline = filter_recording(live[:CHANNELS], RATE)["8-30"][:, start:end]
        np.testing.assert_allclose(recorder._buffers["8-30"][:, start - first:end - first],
                                   offline, atol=1e-9)

    def test_a_backlog_past_the_ceiling_loses_the_pick_quietly(self):
        """If the board process stalls and then drains minutes at once, the marker
        can scroll out of the retained history. That must cost one selection, not
        answer with the wrong window."""
        from motor_imagery.recording import MIRecorder

        live = self.data.copy()
        live[CHANNELS] = 0
        live[CHANNELS, int((cfg.PRIME_SECONDS + 3) * RATE)] = cfg.LIVE_MARKER
        recorder = MIRecorder(predict_session=self.session)
        recorder._window = [cfg.ANALYSIS_START, cfg.ANALYSIS_DURATION]
        chunks = [live[:, i:i + 500] for i in range(0, live.shape[1], 500)]
        self.assertIsNone(recorder._predict({"model": None, "classes": np.array(CLASSES),
                                             "margin": 0.0}, chunks, FakeBoard()))


if __name__ == "__main__":
    unittest.main()
