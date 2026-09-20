import unittest
import numpy as np
from ssvep_training.head import Gestures, GESTURES
from ssvep_training.imu_setup import fit_profile


def pulse(direction, amplitude=.3, samples=50):
    return np.asarray(direction)[:, None] * amplitude * np.sin(np.linspace(0, np.pi, samples))


def feed_all(detector, data, chunk=3):
    return [hit for i in range(0, data.shape[1], chunk)
            if (hit := detector.feed(data[:, i:i+chunk])) is not None]


class SetupTests(unittest.TestCase):
    def profile(self):
        return {'version': 1, 'gestures': list(GESTURES),
                'directions': [[0,0,-1], [0,0,1], [0,-1,0], [0,1,0]],
                'minimum_speed': [.6]*4, 'rest_baseline': [.01]*3, 'rest_noise': [.01]*3}

    def test_constant_startup_motion_cannot_replace_measured_rest(self):
        profile = self.profile()
        detector = Gestures(125, profile=profile)
        startup = np.tile(np.array([.063, .259, -.117])[:, None], (1, 75))
        self.assertEqual(feed_all(detector, startup), [])
        np.testing.assert_allclose(detector.baseline, profile['rest_baseline'])
        rest = np.full((3, 125), .01)
        out = pulse([0, 0, -1], amplitude=2)
        signal = np.hstack([rest, out+.01, rest, -out+.01, rest, out+.01, rest])
        self.assertEqual(feed_all(detector, signal), ['left', 'left'])

    def test_normal_jitter_does_not_prevent_rearming(self):
        profile = self.profile()
        rest = np.full((3, 125), .01)
        # Every sample exceeds the .04 baseline-learning limit on one axis,
        # but is well below calibrated gesture strength (.6).
        jitter = rest.copy(); jitter[0] += np.tile([.08, -.08], 63)[:125]
        out = pulse([0, 0, -1], amplitude=2)
        signal = np.hstack([rest, out+.01, jitter, -out+.01, jitter, out+.01, jitter])
        for chunk in (1, 3, 25):
            self.assertEqual(feed_all(Gestures(125, profile=profile), signal, chunk), ['left', 'left'])
        self.assertEqual(feed_all(Gestures(125, profile=profile), np.hstack([rest, np.tile(jitter, 10)])), [])

    def test_subunit_motion_detected_and_one_sample_spike_ignored(self):
        rest = np.zeros((3, 125))
        signal = np.hstack([rest, pulse([0, 0, -1]), rest])
        self.assertEqual(feed_all(Gestures(125), signal), ['left'])
        signal = np.zeros((3, 500)); signal[2, 160] = .3
        self.assertEqual(feed_all(Gestures(125), signal), [])

    def test_return_motion_is_not_an_opposite_action(self):
        rest = np.zeros((3, 125))
        out = pulse([0, 0, -1])
        # Hold pose for a full second, then return: a plain cooldown would
        # already have expired and would incorrectly select right.
        signal = np.hstack([rest, out, rest, -out, rest, out, rest, -out, rest])
        for chunk in (1, 3, 25):
            self.assertEqual(feed_all(Gestures(125), signal, chunk), ['left', 'left'])

    def test_sustained_turn_does_not_repeat(self):
        signal = np.zeros((3, 750)); signal[2, 125:625] = .3
        self.assertEqual(feed_all(Gestures(125), signal), ['right'])

    def test_partial_return_does_not_lock_out_subsequent_swipes(self):
        rest = np.zeros((3, 125))
        # Return only half the estimated angle. The former integration-only
        # gate stays disarmed forever, even through later deliberate swipes.
        for direction, name in [([0, 0, -1], 'left'), ([0, 0, 1], 'right'),
                                ([0, -1, 0], 'up'), ([0, 1, 0], 'down')]:
            out = pulse(direction)
            signal = np.hstack([rest, out, rest, -.5*out, rest,
                                out, rest, -.5*out, rest, out, rest])
            for chunk in (1, 3, 25):
                with self.subTest(direction=name, chunk=chunk):
                    self.assertEqual(feed_all(Gestures(125), signal, chunk), [name]*3)

    def test_holding_pose_and_reverse_spike_do_not_rearm(self):
        rest = np.zeros((3, 125))
        out = pulse([0, 0, -1])
        spike = np.zeros((3, 125)); spike[2, 10] = .3
        signal = np.hstack([rest, out, np.tile(rest, 10), spike, rest, out, rest])
        self.assertEqual(feed_all(Gestures(125), signal), ['left'])

    def test_integrated_return_survives_correction_before_stillness(self):
        rest = np.full((3, 125), .01)
        out = pulse([0, 0, 1], amplitude=2)
        # Returning on a mixed axis passes the integrated-angle check but not
        # the strict reverse-direction check. A correction then leaves >35%
        # residual travel before the user finally settles (recorded regression).
        returning = -out + pulse([1, 0, 0], amplitude=4)
        correction = .6 * out
        for chunk in (1, 3, 25):
            detector = Gestures(125, profile=self.profile())
            signal = np.hstack([rest, out+.01, returning+.01, correction+.01, rest])
            self.assertEqual(feed_all(detector, signal, chunk), ['right'])
            self.assertGreater(detector.travel, .35 * detector.peak_travel)
            self.assertTrue(detector.armed)
            self.assertEqual(feed_all(detector, out+.01, chunk), ['right'])

    def test_diagnostics_identify_reset_gate_without_changing_detection(self):
        import json
        detector = Gestures(125)
        rest = np.zeros((3, 125))
        out = pulse([0, 0, -1])
        feed_all(detector, np.hstack([rest, out, rest]))
        state = detector.diagnostics(reset_peak=True)
        self.assertEqual(state['waiting'], ['return'])
        self.assertGreater(state['peak_ratio'], 1)
        self.assertFalse(detector.armed)
        json.dumps(state)  # trace must remain serializable in session.json
        feed_all(detector, -out)
        self.assertEqual(detector.diagnostics()['waiting'], ['stillness'])
        feed_all(detector, rest)
        self.assertEqual(detector.diagnostics()['waiting'], [])
        self.assertEqual(feed_all(detector, out), ['left'])

    def test_learns_rotated_mount_and_native_scale(self):
        yaw = np.array([1., 0, 1.]) / np.sqrt(2)
        pitch = np.array([0., 1., 0.])
        vectors = [-yaw, yaw, pitch, -pitch]
        rest = np.random.default_rng(4).normal(0, .001, (3, 250))
        movements = {name: [pulse(vec, amplitude=.3), pulse(vec, amplitude=.35)]
                     for name, vec in zip(GESTURES, vectors)}
        profile = fit_profile(rest, movements, 125)
        for name, vec in zip(GESTURES, vectors):
            signal = np.hstack([rest, pulse(vec), np.zeros((3, 125))])
            self.assertEqual(feed_all(Gestures(125, profile=profile), signal), [name])
        self.assertTrue(all(.05 < x < .15 for x in profile['minimum_speed']))

    def test_rejects_wrong_directions_and_no_movement(self):
        rest = np.zeros((3, 250))
        movements = {name: [pulse([0, 0, 1]), pulse([0, 0, 1])] for name in GESTURES}
        with self.assertRaises(ValueError): fit_profile(rest, movements, 125)
        movements = {name: [rest, rest] for name in GESTURES}
        with self.assertRaises(ValueError): fit_profile(rest, movements, 125)


if __name__ == '__main__':
    unittest.main()
