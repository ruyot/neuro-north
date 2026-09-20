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
