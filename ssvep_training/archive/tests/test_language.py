"""Range evidence, word events and asynchronous UI contracts; no headset/model needed."""
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
from queue import Queue
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

from linguistic_model import ContextModel, DecoderError, LetterRanges, Lexicon, LinguisticDecoder, PipelineSimulator
from ssvep_training import config as cfg
from ssvep_training.language import LanguageCore, LanguageService, RangeEvidence
from ssvep_training.speller_ui import RANGES, SpellerUI, run_flicker
from ssvep_training.recording import RecordingProcess


class LanguageTests(unittest.TestCase):
    def service(self):
        service = LanguageService.__new__(LanguageService)
        service.evidence = None
        service.sequence = 0
        service.pending = service.failed = service.queued_boundary = False
        service.current_action = 'startup'
        service.started = time.monotonic()
        service.requests, service.replies = Queue(), Queue()
        service.process = NS(is_alive=lambda: True)
        return service

    def test_right_swipe_during_inference_finishes_word_after_latest_range(self):
        core, service = self.core(), self.service()
        self.select(core, 0, 1)  # first letter of "hi"
        ui = self.ui()
        ui.language = service
        ui.page = 0
        ui.items = ['ghijkl']
        ui.select_box(1)
        # Exactly the observed race: gesture arrives before the select reply.
        ui.space()
        ui.space()  # duplicate swipes should not finish another word
        self.assertTrue(service.queued_boundary)
        self.assertEqual(service.requests.qsize(), 1)
        number, action, payload = service.requests.get_nowait()
        core.apply(action, payload)
        service.replies.put({'id': number, 'state': core.snapshot(), 'error': None})
        ui.poll_language()
        self.assertTrue(service.pending)
        self.assertEqual(ui.suggestions, ['', ''])
        number, action, payload = service.requests.get_nowait()
        self.assertEqual(action, 'boundary')
        core.apply(action, payload)
        service.replies.put({'id': number, 'state': core.snapshot(), 'error': None})
        ui.poll_language()
        self.assertFalse(service.pending)
        self.assertEqual(ui.text, 'hi ')
        self.assertTrue(service.requests.empty())

    def test_delayed_poll_does_not_drop_finish_word(self):
        service = self.service()
        self.assertTrue(service.select(('A-F', 'G-L'), 0, 0))
        service.replies.put({'id': service.sequence, 'state': {}, 'error': None})
        self.assertTrue(service.submit('boundary', {}))
        service.poll()
        self.assertEqual([service.requests.get_nowait()[1] for _ in range(2)], ['select', 'boundary'])

    def test_busy_worker_still_rejects_new_ranges_and_stale_acceptance(self):
        service = self.service()
        service.select(('A-F', 'G-L'), 0, 0)
        self.assertFalse(service.select(('A-F', 'G-L'), 1, 0))
        self.assertFalse(service.submit('accept', {'word': 'stale'}))
        self.assertTrue(service.submit('boundary', {}))
        service.replies.put({'id': service.sequence, 'state': {}, 'error': 'invalid range'})
        reply = service.poll()
        self.assertIn('queued finish-word action cancelled', reply['error'])
        self.assertFalse(service.pending)
        self.assertFalse(service.queued_boundary)
        self.assertEqual(service.requests.qsize(), 1)

    def test_duplicate_finish_while_accepting_cannot_add_another_word(self):
        for action, payload in [('accept', {'word':'apple'}), ('boundary', {})]:
            service = self.service()
            service.submit(action, payload)
            self.assertTrue(service.submit('boundary', {}))
            self.assertFalse(service.queued_boundary)
            service.replies.put({'id': service.sequence, 'state': {}, 'error': None})
            service.poll()
            self.assertFalse(service.pending)
            self.assertEqual(service.requests.qsize(), 1)

    def core(self):
        ranges = LetterRanges.alphabet_quarters()
        lexicon = Lexicon({'apple': 10, 'hello': 5, 'hi': 2}, ranges)
        decoder = LinguisticDecoder(ranges, lexicon, ContextModel(lexicon))
        return LanguageCore('bigram', simulator=PipelineSimulator(decoder))

    def select(self, core, page, choice):
        offered = core.simulator.pages[page]
        core.apply('select', {'page': page, 'selected': offered[choice],
                             'probabilities': dict(zip(offered, [.8, .2] if choice == 0 else [.2, .8])),
                             'selection_priors': dict.fromkeys(offered, .5)})

    def test_apple_acceptance_and_next_word_boundary(self):
        core = self.core()
        for page, choice in [(0,0), (1,0), (1,0), (0,1), (0,0)]:
            self.select(core, page, choice)
            self.assertEqual(core.simulator.page_index, 0)
        self.assertEqual(core.snapshot()['candidates'][0]['word'], 'apple')
        core.apply('accept', {'word': 'apple'})
        self.assertEqual(core.snapshot()['confirmed_words'], ['apple'])
        self.assertFalse(core.snapshot()['current_ranges'])
        self.select(core, 0, 1)
        self.select(core, 0, 1)
        core.apply('boundary', {})
        self.assertEqual(core.snapshot()['confirmed_words'], ['apple', 'hi'])
        core.apply('boundary', {})
        self.assertEqual(core.snapshot()['confirmed_words'], ['apple', 'hi'])

    def test_failed_boundary_keeps_observation(self):
        core = self.core()
        self.select(core, 0, 0)
        with self.assertRaises(DecoderError):
            core.apply('boundary', {})
        self.assertEqual(core.snapshot()['current_ranges'], ['A-F'])
        self.assertEqual(core.snapshot()['confirmed_words'], [])

    def test_wrong_page_is_rejected_without_mutation(self):
        core = self.core()
        with self.assertRaises(ValueError):
            core.apply('select', {'page': 1, 'selected': 'A-F', 'probabilities': {'A-F': .8, 'G-L': .2}})
        self.assertFalse(core.simulator.decoder.observations)

    def test_evidence_requires_matching_settings_and_maps_actual_page(self):
        with tempfile.TemporaryDirectory() as path:
            path = Path(path)
            meta = {'frequencies': list(cfg.STIMULUS_FREQUENCIES), 'eeg_rows': [1,2,3,4],
                    'processing': {key: getattr(cfg, key.upper()) for key in
                                   ['gaze_duration', 'visual_latency', 'filter_band', 'cca_bands',
                                    'confidence_threshold', 'filter_history', 'decoy_frequencies']}}
            report = {'decoder': 'cca', 'threshold': cfg.CONFIDENCE_THRESHOLD,
                      'trials': [{'target': i%2, 'prediction': i%2, 'accepted': True, 'late_frames': 0}
                                 for i in range(8)]}
            report['trials'] += [{'target': None, 'prediction': 0, 'accepted': True, 'late_frames': 0},
                                 {'target': 1, 'prediction': 0, 'accepted': False, 'late_frames': 0},
                                 {'target': 1, 'prediction': 0, 'accepted': True, 'late_frames': 1}]
            (path/'session.json').write_text(json.dumps(meta))
            (path/'validation.json').write_text(json.dumps(report))
            evidence = RangeEvidence.from_validation(path, [1,2,3,4])
            self.assertEqual(evidence.samples, 8)
            self.assertEqual(evidence.priors, [.5,.5])
            observation = evidence.observation(['M-R','S-Z'], 1)
            self.assertAlmostEqual(observation['probabilities']['S-Z'], 5/6)
            meta['frequencies'].reverse()
            (path/'session.json').write_text(json.dumps(meta))
            with self.assertRaises(ValueError):
                RangeEvidence.from_validation(path, [1,2,3,4])

    def ui(self):
        ui = SpellerUI.__new__(SpellerUI)
        ui.page, ui.pages = 1, [RANGES[:2], RANGES[2:]]
        ui.items, ui.suggestions = ['abcdef'], ['apple', 'hello']
        ui.picked_labels = [NS(text=''), NS(text='')]
        ui._flash, ui._changed, ui._refresh = Mock(), Mock(), Mock()
        ui.language = NS(select=Mock(return_value=True), submit=Mock(return_value=True), poll=Mock())
        return ui

    def test_selection_sends_displayed_page_and_clears_stale_suggestions(self):
        ui = self.ui()
        ui.select_box(1)
        ui.language.select.assert_called_once_with(['M-R','S-Z'], 1, 1)
        self.assertEqual(ui.page, 0)
        self.assertEqual(ui.items, ['abcdef', 'stuvwxyz'])
        self.assertEqual(ui.suggestions, ['', ''])

    def test_busy_worker_cannot_duplicate_choices_or_accept_stale_word(self):
        ui = self.ui()
        ui.language.select.return_value = False
        ui.language.submit.return_value = False
        ui.select_box(0)
        ui.space()
        ui.pick_suggestion(0)
        self.assertEqual(ui.items, ['abcdef'])
        ui._changed.assert_not_called()

    def test_worker_result_replaces_ranges_with_word_and_preserves_user_page(self):
        ui = self.ui()
        ui.pick_suggestion(0)
        ui.language.submit.assert_called_once_with('accept', {'word': 'apple'})
        self.assertEqual(ui.page, 0)
        ui.page = 1  # user cycles while model is working
        ui.language.poll.return_value = {'state': {'confirmed_words': ['apple'], 'current_ranges': [],
                                                   'candidates': [{'word':'is'}, {'word':'and'}]}, 'error': None}
        ui.poll_language()
        self.assertEqual(ui.text, 'apple ')
        self.assertEqual(ui.page, 1)
        self.assertEqual(ui.suggestions, ['is', 'and'])

    def test_flicker_waits_for_model_and_does_not_poll_during_selection(self):
        recorder = RecordingProcess()
        times = iter(i*.1 for i in range(100))
        ui = NS(action_count=0, draws=0, language=NS(pending=True),
                set_progress=lambda *args: None)
        ui.draw = lambda: setattr(ui, 'draws', ui.draws + 1)
        phases, polls, frames = [], [], []
        def poll():
            polls.append(ui.draws)
            if ui.draws == 10:
                ui.language.pending = False
        ui.poll_language = poll
        stimulus = NS(draw_blank=lambda *args: phases.append('dark'),
                      flicker_frame=lambda *args: frames.append(ui.draws))
        win = NS(nDroppedFrames=0, frameIntervals=[], ssvep_refresh=60,
                 flip=lambda: None, callOnFlip=lambda fn, *args: fn(*args))
        psychopy = NS(core=NS(Clock=lambda: NS(getTime=lambda: next(times))))
        with patch.dict(sys.modules, {'psychopy': psychopy, 'ssvep_training.stimulus': stimulus}), \
             patch('ssvep_training.speller_ui.handle_keys', side_effect=lambda *args, **kwargs: ui.draws < 20):
            run_flicker(win, ui, [], recorder)
        self.assertEqual(frames[0], 10)
        self.assertEqual(polls, list(range(11)))


if __name__ == '__main__':
    unittest.main()
