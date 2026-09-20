"""Motion timing and calibration isolation without a display or headset."""
import importlib
import sys
import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch
import numpy as np
from ssvep_training import config as cfg
from ssvep_training.timing import motion_phase
from ssvep_training.trca_model import load_calibration


class MotionTests(unittest.TestCase):
    def test_motion_cycles_and_reversal_rates(self):
        for frequency, period in [(12, 10), (15, 8)]:
            phases = np.array([motion_phase(frequency, i, 60) for i in range(120)])
            np.testing.assert_allclose(phases[period:], phases[:-period], atol=1e-12)
            self.assertAlmostEqual(phases.max(), .25)
            self.assertAlmostEqual(phases.min(), -.25)
            velocity = np.roll(phases, -1) - phases
            reversals = np.count_nonzero(np.sign(velocity) != np.roll(np.sign(velocity), 1))
            self.assertEqual(reversals, frequency * 2)  # two seconds

    def test_motion_cannot_load_flicker_calibration(self):
        mode = cfg.STIMULUS_MODE
        try:
            cfg.configure_stimulus('motion')
            with patch('ssvep_training.session.load_session', return_value=(None, {
                    'frequencies': list(cfg.STIMULUS_FREQUENCIES),
                    'stimulus_method': 'integer_frame_cycles_v1'})):
                with self.assertRaises(ValueError):
                    load_calibration('unused')
        finally:
            cfg.configure_stimulus(mode)

    def test_motion_draws_grating_and_rest_draws_black(self):
        from unittest.mock import Mock
        previous = sys.modules.pop('ssvep_training.stimulus', None)
        try:
            with patch.dict(sys.modules, {'psychopy': NS(core=NS(), event=NS(), visual=NS())}):
                stimulus = importlib.import_module('ssvep_training.stimulus')
                grating = NS(draw=Mock())
                square = NS(motion_grating=grating, draw=Mock(), fillColor=None)
                stimulus.flicker_frame([square], [12], 2, 60)
                self.assertEqual(grating.phase, (motion_phase(12, 2, 60), 0))
                grating.draw.assert_called_once()
                square.draw.assert_not_called()
                stimulus.draw_blank([square])
                self.assertEqual(square.fillColor, [-1, -1, -1])
                square.draw.assert_called_once()
        finally:
            sys.modules.pop('ssvep_training.stimulus', None)
            if previous is not None:
                sys.modules['ssvep_training.stimulus'] = previous
