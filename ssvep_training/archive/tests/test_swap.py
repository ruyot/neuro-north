import random
import unittest
from ssvep_training.validate_live import position_choice, swap_schedule


class SwapTests(unittest.TestCase):
    def test_each_block_balances_side_and_frequency(self):
        schedule = swap_schedule(4, random.Random(24))
        self.assertEqual(len(schedule), 16)
        for start in range(0, len(schedule), 4):
            self.assertEqual(set(schedule[start:start+4]),
                             {(0, False), (1, False), (0, True), (1, True)})

    def test_detected_frequency_maps_back_to_cued_position(self):
        for swapped in (False, True):
            displayed = [15, 12] if swapped else [12, 15]
            for side, frequency in enumerate(displayed):
                decoder_choice = [12, 15].index(frequency)
                self.assertEqual(position_choice(decoder_choice, swapped), side)
            self.assertEqual(position_choice(-1, swapped), -1)
