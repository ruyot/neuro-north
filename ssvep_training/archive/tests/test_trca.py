import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch
import numpy as np
from ssvep_training import config as cfg
from ssvep_training.session import Trials, save_session, session_meta
from ssvep_training.trca_model import fit, load_calibration
from ssvep_training.evaluate_trca import compare


class TRCATests(unittest.TestCase):
    def test_peak_gate_boundaries_and_invalid_trials(self):
        from ssvep_training.validate_live import accept_trca
        self.assertTrue(accept_trca(1, .10, .10))
        self.assertFalse(accept_trca(0, .099, .10))
        self.assertFalse(accept_trca(-1, .5, .10))
        self.assertFalse(accept_trca(0, .5, .10, late_frames=1))
        for peak in (float('nan'), float('inf'), float('-inf')):
            self.assertFalse(accept_trca(0, peak, .10))

    def test_exposed_scores_match_library_predictions(self):
        from ssvep_training.trca_model import scores
        trials = self.trials()
        model = fit(trials)
        predicted = np.array([scores(model, trials.eeg[...,i]).argmax() for i in range(len(trials.targets))])
        np.testing.assert_array_equal(predicted, model.predict(trials.eeg))

    def test_unfitted_idle_gate_is_not_reported_as_zero_false_activations(self):
        from ssvep_training.validate_live import summarize
        result = summarize([{'target': None, 'prediction':0, 'accepted':None, 'late_frames':0}])
        self.assertFalse(result['rejection_gate_evaluated'])
        self.assertNotIn('idle_false_activations', result)

    def trials(self):
        rng = np.random.default_rng(19)
        t = np.arange(250)/125
        signals, targets, blocks = [], [], []
        for block in range(1,7):
            for target, frequency in enumerate(cfg.STIMULUS_FREQUENCIES):
                wave = np.sin(2*np.pi*frequency*t+.2) + .3*np.sin(4*np.pi*frequency*t+.1)
                signals.append(wave[:,None]*np.array([1.,.6,-.5,.8]) + rng.normal(0,.2,(250,4)))
                targets.append(target); blocks.append(block)
        return Trials(np.stack(signals,axis=2),np.array(targets),np.array(blocks),125,
                      ['Oz','O1','O2','PO7'],list(cfg.STIMULUS_FREQUENCIES),['A','B'],[1,2,3,4])

    def test_trca_separates_synthetic_trials_without_training_on_test_block(self):
        trials = self.trials()
        original = trials.eeg.copy()
        trained_blocks = []
        def train(part):
            trained_blocks.append(set(part.blocks.tolist()))
            return fit(part)
        with patch('ssvep_training.evaluate_trca.fit', side_effect=train):
            report = compare(trials)
        for block, training in enumerate(trained_blocks[:6],1):
            self.assertNotIn(block, training)
            self.assertEqual(len(training),5)
        self.assertEqual(trained_blocks[-1],{1,2,3,4})
        self.assertEqual(report['leave_one_block_out']['trca']['accuracy'],1.)
        self.assertEqual(report['later_blocks_holdout']['trca']['accuracy'],1.)
        np.testing.assert_array_equal(trials.eeg,original)

    def test_one_trial_per_class_is_not_valid_trca_training(self):
        from ssvep_training.evaluate_trca import subset
        with self.assertRaisesRegex(ValueError, 'at least two'):
            fit(subset(self.trials(), np.arange(2)))

    def test_bad_display_and_transport_are_excluded_and_metadata_is_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            data = np.random.default_rng(7).normal(size=(22,2300))
            data[0] = np.arange(2300)%256
            data[21] = 0
            onsets, codes = [600,1250,1900], [11,12,21]
            data[21,onsets] = codes
            data[0,1300] += 2
            meta = session_meta(NS(board_id=66,rate=125,eeg_rows=[1,2,3,4],
                                   names=['Oz','O1','O2','PO7'],marker_row=21))
            save_session(str(path),data,meta)
            (path/'calibration.json').write_text(json.dumps({'trials':[
                {'marker': code, 'completed':True, 'late_frames':int(i==0)} for i,code in enumerate(codes)]}))
            trials = load_calibration(str(path))
            self.assertEqual(trials.skipped,2)
            self.assertEqual(trials.targets.tolist(),[0])
            data[3:5] = 0
            save_session(str(path), data, meta)
            self.assertEqual(len(load_calibration(str(path)).targets), 0)
            subset = load_calibration(str(path), channels=[1,2])
            self.assertEqual(subset.rows, [1,2])
            self.assertEqual(subset.names, ['Oz','O1'])
            self.assertEqual(len(subset.targets), 1)
            with self.assertRaises(ValueError):
                load_calibration(str(path), channels=[1,8])
            meta['frequencies'] = [10,15]
            (path/'session.json').write_text(json.dumps(meta))
            with self.assertRaisesRegex(ValueError,'differ'):
                load_calibration(str(path))


if __name__ == '__main__':
    unittest.main()
