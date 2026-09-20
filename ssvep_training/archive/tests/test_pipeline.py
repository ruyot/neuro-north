import unittest
import numpy as np
import stream
from ssvep_training import cca_model
from ssvep_training.timing import frame_is_on, frames_per_cycle
from ssvep_training.quality import packet_discontinuities
from ssvep_training.session import epoch_at


class PipelineTests(unittest.TestCase):
    def test_diagnostic_excludes_repeated_filter_startup(self):
        raw = np.full((1, 750), 100000.)
        self.assertIsNone(stream.diagnostic_window(raw[:, :749], 125, 250, True))
        settled = stream.diagnostic_window(raw, 125, 250, True)
        self.assertEqual(settled.shape, (1, 250))
        self.assertLess(settled.std(), 2)
        self.assertGreater(stream.clean(raw, 125).std(), 9000)
        np.testing.assert_array_equal(raw, 100000.)

    def test_flat_channels_have_no_correlation(self):
        ref = stream.cca_reference(15, 125, 150, 1)
        self.assertEqual(stream.cca_score(np.ones((4, 150)), ref), 0)

    def test_duplicates_do_not_invent_evidence(self):
        rng = np.random.default_rng(10)
        x = rng.normal(size=(1, 150))
        ref = stream.cca_reference(20, 125, 150, 1)
        expected = stream.cca_score(x, ref)
        self.assertAlmostEqual(expected, stream.cca_score(np.vstack([x, x, x*1000, x*0]), ref))

    def test_clean_preserves_raw_recording(self):
        raw = np.random.default_rng(1).normal(size=(4, 800))
        copy = raw.copy()
        stream.clean(raw, 125)
        np.testing.assert_array_equal(raw, copy)

    def test_targets_decode_through_raw_epoch_pipeline(self):
        rng = np.random.default_rng(11)
        for frequencies in ([15, 20], [12, 15]):
            for target, frequency in enumerate(frequencies):
                for phase in [0.1, 1.3, 3.2]:
                    raw = rng.normal(scale=.2, size=(4, 800))
                    time = np.arange(300)/125
                    raw[:, 500:] += np.sin(2*np.pi*frequency*time+phase)
                    epoch = epoch_at(raw, 500, 125)
                    result = cca_model.scores(epoch, 125, frequencies, cca_model.harmonics_for(frequencies))
                    self.assertEqual(result.argmax(), target)

    def test_bank_ignores_band_without_reference(self):
        epoch = np.random.default_rng(3).normal(size=(150, 4))
        np.testing.assert_allclose(cca_model.scores(epoch, 125, [15, 20], 1, bands=2),
                                   cca_model.scores(epoch, 125, [15, 20], 1, bands=3))

    def test_weak_evidence_abstains(self):
        self.assertFalse(cca_model.accept_prediction(0, -.2, .5))
        self.assertTrue(cca_model.accept_prediction(1, .8, .5))
        self.assertTrue(cca_model.accept_prediction(0, -.2, 0))
        self.assertFalse(cca_model.accept_prediction(-1, 10, 0))
        self.assertFalse(cca_model.accept_prediction(0, float('nan'), 0))

    def test_frame_schedule(self):
        self.assertEqual([frame_is_on(12, k, 60) for k in range(10)], [True, True, True, False, False]*2)
        self.assertEqual([frame_is_on(15, k, 60) for k in range(8)], [True, True, False, False]*2)
        self.assertEqual([frame_is_on(20, k, 60) for k in range(6)], [True, True, False]*2)
        for refresh in [60, 120]:
            for frequency in [12, 15, 20]:
                self.assertEqual(refresh / frames_per_cycle(frequency, refresh), frequency)
        with self.assertRaises(ValueError):
            frames_per_cycle(20, 75)

    def test_prospective_metrics_include_rejections_and_idle(self):
        from ssvep_training.validate_live import summarize
        rows = [dict(target=0, prediction=0, late_frames=0, accepted=True),
                dict(target=1, prediction=0, late_frames=0, accepted=False),
                dict(target=1, prediction=-1, late_frames=0, accepted=False),
                dict(target=None, prediction=1, late_frames=0, accepted=True)]
        result = summarize(rows)
        self.assertEqual(result['attempted_target_trials'], 3)
        self.assertEqual(result['forced_accuracy'], .5)
        self.assertEqual(result['accepted_target_trials'], 1)
        self.assertEqual(result['idle_false_activations'], 1)

    def test_live_loop_does_not_force_low_confidence_choice(self):
        import sys
        from types import SimpleNamespace as NS
        from unittest.mock import patch
        from ssvep_training import speller_ui
        for manual, start, contrast, expected in [
                (False, False, -.2, 0), (False, False, .8, 1),
                (True, False, .8, 0), (True, True, .8, 1), (True, True, -.2, 0)]:
            ticks = iter(np.arange(0, 10, .1))
            ui = NS(action_count=0, draws=0, picks=0)
            def draw():
                ui.draws += 1
            def select(_):
                ui.picks += 1
            ui.draw, ui.select_box = draw, select
            ui.set_progress = lambda *args: None
            recorder = NS(prediction_count=NS(value=0), last_sigma=NS(value=contrast),
                          last_prediction=NS(value=0), gaze=NS(value=0),
                          read_gesture=lambda: (0, -1), cancel_prediction=lambda: None,
                          prediction_is_current=lambda: True,
                          mark_onset=lambda *args: None, is_alive=lambda: True)
            def request():
                recorder.prediction_count.value += 1
            recorder.request_prediction = request
            win = NS(nDroppedFrames=0, recordFrameIntervals=False, frameIntervals=[],
                     ssvep_refresh=60, flip=lambda: None,
                     callOnFlip=lambda function, *args: function(*args))
            psychopy = NS(core=NS(Clock=lambda: NS(getTime=lambda: next(ticks))),
                         event=NS(clearEvents=lambda **kw: None,
                                  getKeys=lambda **kw: ['space'] if start and ui.draws == 2 else []))
            stimulus = NS(draw_blank=lambda *args: None, flicker_frame=lambda *args: None)
            with patch.dict(sys.modules, {'psychopy': psychopy, 'ssvep_training.stimulus': stimulus}), \
                 patch.object(speller_ui, 'handle_keys',
                              side_effect=lambda *args, **kw: ui.draws < (85 if manual else 35)):
                speller_ui.run_flicker(win, ui, [], recorder, threshold=.5, manual=manual)
            self.assertEqual(ui.picks, expected)
            if manual:
                self.assertEqual(recorder.prediction_count.value, int(start))

    def test_counter_wrap_and_drops(self):
        self.assertEqual(packet_discontinuities([254, 255, 0, 1]), 0)
        self.assertEqual(packet_discontinuities([254, 255, 1, 1, 2]), 2)


if __name__ == '__main__':
    unittest.main()
