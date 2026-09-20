"""IMU integration tests: no physical headset or display required."""
import sys
import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch
import numpy as np
from ssvep_training.head import Gestures, GESTURES
from ssvep_training.recording import RecordingProcess
from ssvep_training.speller_ui import handle_gestures, run_flicker, SpellerUI


class HeadIntegrationTests(unittest.TestCase):
    def test_four_directions_and_rest(self):
        for axis, sign, expected in [(2, -1, 'left'), (2, 1, 'right'),
                                      (1, -1, 'up'), (1, 1, 'down'), (0, 1, None)]:
            data = np.random.default_rng(2).uniform(-40, 40, (3, 500))
            data[axis, 125:165] += sign * 4000 * np.sin(np.linspace(0, np.pi, 40))
            detector = Gestures(125)
            hits = [name for i in range(0, 500, 3)
                    if (name := detector.feed(data[:, i:i+3])) is not None]
            self.assertEqual(hits, [expected] if expected else [])

    def test_dispatch_and_duplicate_suppression(self):
        for code, expected in enumerate([('wheel',), ('space',), ('suggestion', 0), ('suggestion', 1)]):
            calls = []
            ui = NS(next_wheel=lambda: calls.append(('wheel',)),
                    space=lambda: calls.append(('space',)),
                    pick_suggestion=lambda slot: calls.append(('suggestion', slot)))
            recorder = RecordingProcess()
            recorder.gesture.value = code
            recorder.gesture_count.value = 1
            seen = handle_gestures(ui, recorder, 0)
            handle_gestures(ui, recorder, seen)
            self.assertEqual(calls, [expected])
            self.assertEqual(recorder.selection_version.value, 1)

    def test_cancel_invalidates_result_even_after_next_onset(self):
        recorder = RecordingProcess()
        recorder.mark_onset(99)
        recorder.request_prediction()
        old_version = recorder.predict_version.value
        recorder.cancel_prediction()
        recorder.mark_onset(99)
        recorder.prediction_version.value = old_version
        self.assertFalse(recorder.prediction_is_current())
        recorder.request_prediction()
        recorder.prediction_version.value = recorder.predict_version.value
        self.assertTrue(recorder.prediction_is_current())

    def test_empty_suggestion_gesture_preempts_simultaneous_cca(self):
        recorder = RecordingProcess()
        ticks = iter(np.arange(0, 10, .1))
        ui = NS(draws=0, action_count=0, picks=[], suggestions=[])
        ui.draw = lambda: setattr(ui, 'draws', ui.draws + 1)
        ui.set_progress = lambda *args: None
        ui.select_box = lambda choice: ui.picks.append(choice)
        ui.pick_suggestion = lambda slot: ui.suggestions.append(slot)  # no action_count change
        def request():
            recorder.prediction_version.value = recorder.selection_version.value
            recorder.last_prediction.value = 0
            recorder.last_sigma.value = 10
            recorder.prediction_count.value += 1
            recorder.gesture.value = GESTURES.index('up')
            recorder.gesture_count.value += 1
        recorder.request_prediction = request
        win = NS(nDroppedFrames=0, frameIntervals=[], ssvep_refresh=60,
                 flip=lambda: None, callOnFlip=lambda fn, *args: fn(*args))
        psychopy = NS(core=NS(Clock=lambda: NS(getTime=lambda: next(ticks))))
        stimulus = NS(draw_blank=lambda *args: None, flicker_frame=lambda *args: None)
        with patch.dict(sys.modules, {'psychopy': psychopy, 'ssvep_training.stimulus': stimulus}), \
             patch('ssvep_training.speller_ui.handle_keys', side_effect=lambda *a, **kw: ui.draws < 35):
            run_flicker(win, ui, [], recorder, threshold=.5)
        self.assertEqual(ui.suggestions, [0])
        self.assertEqual(ui.picks, [])
        self.assertFalse(recorder.prediction_is_current())

    def test_accepting_suggestion_finishes_word_and_resets_page(self):
        ui = SpellerUI.__new__(SpellerUI)
        ui.items = ['h', 'i', ' ', 'abcdef', 'ghijkl']
        ui.suggestions = ['cat', 'dog']
        ui.page = 1
        ui._flash = lambda *a: None
        ui._changed = lambda *a: None
        ui.pick_suggestion(1)
        self.assertEqual(ui.items, ['h', 'i', ' ', 'd', 'o', 'g', ' '])
        self.assertEqual(ui.page, 0)


if __name__ == '__main__':
    unittest.main()
