"""Asynchronous adapter from committed range choices to Mika's local decoder.

CCA correlations are NOT probabilities. For EEG input we estimate P(true target |
accepted pick) from a matching labeled validation, with add-one smoothing. These
small-sample estimates are provisional, not a claim of calibrated certainty.
"""
from __future__ import annotations
import json
import multiprocessing as mp
from pathlib import Path
from queue import Empty
import time
from . import config as cfg


class RangeEvidence:
    def __init__(self, matrix, priors, source, samples):
        self.matrix, self.priors = matrix, priors
        self.source, self.samples = source, samples

    @classmethod
    def from_validation(cls, path, channels):
        path = Path(path)
        metadata = json.loads((path / 'session.json').read_text())
        validation = json.loads((path / 'validation.json').read_text())
        processing = metadata['processing']
        expected = {'gaze_duration': cfg.GAZE_DURATION, 'visual_latency': cfg.VISUAL_LATENCY,
                    'filter_band': list(cfg.FILTER_BAND), 'cca_bands': cfg.CCA_BANDS,
                    'confidence_threshold': cfg.CONFIDENCE_THRESHOLD,
                    'filter_history': cfg.FILTER_HISTORY,
                    'decoy_frequencies': list(cfg.DECOY_FREQUENCIES)}
        if (metadata['frequencies'] != list(cfg.STIMULUS_FREQUENCIES)
                or metadata['eeg_rows'] != channels
                or any(processing.get(k) != value for k, value in expected.items())
                or validation.get('decoder') != 'cca'
                or validation.get('threshold') != cfg.CONFIDENCE_THRESHOLD):
            raise ValueError('Validation frequencies/channels/processing do not match live CCA')
        counts = [[1., 1.], [1., 1.]]  # predicted target -> true target, add-one smoothing
        truths = [1., 1.]
        samples = 0
        for row in validation['trials']:
            if (row['target'] in (0, 1) and row['prediction'] in (0, 1)
                    and row['accepted'] and not row['late_frames']):
                counts[row['prediction']][row['target']] += 1
                truths[row['target']] += 1
                samples += 1
        if min(truths) <= 1 or samples < 6:
            raise ValueError('Need accepted labeled examples of both targets for language evidence')
        return cls([[v/sum(row) for v in row] for row in counts],
                   [sum(row[i] for row in counts)/(samples+4) for i in (0, 1)], str(path), samples)

    @classmethod
    def latest(cls, channels):
        for path in sorted(Path(cfg.TRAINING_DATA_DIR).glob('validation_*'), reverse=True):
            try:
                return cls.from_validation(path, channels)
            except (OSError, KeyError, ValueError):
                continue
        raise ValueError('No matching labeled validation for range uncertainty; use --engine off '
                         'or record validation with the current frequencies/channels')

    def observation(self, offered, choice):
        if len(offered) != 2 or choice not in (0, 1):
            raise ValueError('Language integration needs two offered ranges and a valid pick')
        return {'probabilities': dict(zip(offered, self.matrix[choice])),
                'selection_priors': dict(zip(offered, self.priors))}


class LanguageCore:
    """Owns model state in its worker process; UI sends explicit events only."""
    def __init__(self, engine='gpt2', context='', simulator=None):
        from linguistic_model.simulator import PipelineSimulator, ENGINE_MODELS
        self.simulator = simulator or PipelineSimulator()
        if engine == 'bigram':
            from linguistic_model.simulator import _DATA
            if not (_DATA / 'bigrams.tsv').exists():
                print('[language] No bigram corpus supplied; lightweight engine uses word frequencies only.', flush=True)
        self.simulator.frequencies = tuple(cfg.STIMULUS_FREQUENCIES)
        if engine != 'bigram':
            from linguistic_model.causal_scorer import CausalCandidateScorer
            # CPU avoids competing with PsychoPy's rendering for the GPU.
            self.simulator._scorers[engine] = CausalCandidateScorer(ENGINE_MODELS[engine][1], device='cpu')
        self.simulator.set_engine(engine)
        self.simulator.set_decode_mode('sentence-beam')
        self.simulator.set_context_prefix(context)

    def apply(self, action, payload):
        simulator = self.simulator
        if action == 'select':
            page = payload['page']
            offered = simulator.pages[page]
            if set(payload['probabilities']) != set(offered) or payload['selected'] not in offered:
                raise ValueError('Range observation does not match the displayed page')
            simulator.decoder.observe(payload['probabilities'], payload['selection_priors'], f'page-{page+1}')
            simulator.history.append({'selected': payload['selected'], 'page': page,
                                      'probabilities': payload['probabilities']})
            simulator.page_index = 0
        elif action == 'accept':
            simulator.accept(payload['word'])
            simulator.page_index = 0
        elif action == 'boundary':
            if simulator.decoder.observations:
                simulator.boundary()
            elif simulator.tentative_words:
                simulator.finish_sentence()
            simulator.page_index = 0
        elif action == 'undo':
            simulator.backspace()
        else:
            raise ValueError(f'Unknown language action: {action}')

    def snapshot(self):
        state = self.simulator.state()
        return {key: state[key] for key in ('confirmed_words', 'tentative_words',
                                             'current_ranges', 'candidates', 'engine',
                                             'decode_mode', 'can_finish', 'last_event')}


def _worker(requests, replies, engine, context):
    try:
        if engine != 'bigram':
            import torch
            torch.set_num_threads(2)
        core = LanguageCore(engine, context)
        replies.put({'id': 0, 'state': core.snapshot(), 'error': None})
    except Exception as exc:
        replies.put({'id': 0, 'state': None, 'error': f'Model startup failed: {exc}'})
        return
    while True:
        request = requests.get()
        if request is None:
            return
        number, action, payload = request
        error = None
        try:
            core.apply(action, payload)
        except Exception as exc:
            error = str(exc)
        try:
            replies.put({'id': number, 'state': core.snapshot(), 'error': error})
        except Exception as exc:
            replies.put({'id': number, 'state': None, 'error': str(exc)})
            return


class LanguageService:
    def __init__(self, engine='gpt2', context='', evidence=None):
        self.evidence = evidence
        self.sequence = 0
        self.pending = True
        self.failed = False
        self.current_action = 'startup'
        self.queued_boundary = False
        self.started = time.monotonic()
        self.requests, self.replies = mp.Queue(), mp.Queue()
        self.process = mp.Process(target=_worker, args=(self.requests, self.replies, engine, context), daemon=True)
        self.process.start()

    def submit(self, action, payload):
        if self.failed:
            return False
        if self.pending:
            if action != 'boundary':
                return False
            # A right swipe often arrives while the just-selected range is
            # being ranked. Preserve that word boundary until the range has
            # reached the worker. Repeated right swipes refer to the same word.
            if self.current_action not in ('boundary', 'accept'):
                self.queued_boundary = True
            return True
        self.sequence += 1
        self.current_action = action
        self.pending = True
        self.started = time.monotonic()
        self.requests.put((self.sequence, action, payload))
        return True

    def select(self, offered, choice, page):
        if self.evidence is None:  # keyboard-only selection is explicit, not an EEG estimate
            posterior = {label: float(i == choice) for i, label in enumerate(offered)}
            observation = {'probabilities': posterior, 'selection_priors': {label: .5 for label in offered}}
        else:
            observation = self.evidence.observation(offered, choice)
        return self.submit('select', {**observation, 'selected': offered[choice], 'page': page})

    def poll(self):
        latest = None
        while True:
            try:
                reply = self.replies.get_nowait()
            except Empty:
                break
            if reply['id'] == self.sequence:
                latest = reply
        if latest is not None:
            self.pending = False
            if latest['state'] is None:
                self.failed = True
                self.queued_boundary = False
            elif self.queued_boundary:
                self.queued_boundary = False
                if latest['error']:
                    latest['error'] += '; queued finish-word action cancelled'
                else:
                    self.submit('boundary', {})
            return latest
        if self.pending and (not self.process.is_alive() or time.monotonic()-self.started > 120):
            self.failed, self.pending = True, False
            return {'id': self.sequence, 'state': None, 'error': 'Language worker stopped or timed out'}
        return None

    def close(self):
        self.requests.put(None)
        self.process.join(timeout=3)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=2)
