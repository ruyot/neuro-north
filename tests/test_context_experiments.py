import unittest

from linguistic_model.experiments.benchmark import exact_range_candidates, rank
from linguistic_model.experiments.cases import CASES


class ContextExperimentTests(unittest.TestCase):
    def test_every_case_has_an_ambiguous_exact_range_candidate_set(self):
        for case in CASES:
            with self.subTest(case=case.text):
                candidates, decoder = exact_range_candidates(case.target)
                self.assertIn(case.target, candidates)
                self.assertGreater(len(candidates), 1)
                expected = [decoder.ranges.label_for(char) for char in case.target]
                for candidate in candidates:
                    actual = [decoder.ranges.label_for(char) for char in candidate]
                    self.assertEqual(expected, actual)

    def test_rank_is_probability_ordered_and_deterministic_on_ties(self):
        scores = {"beta": -2.0, "alpha": -2.0, "target": -1.0}

        self.assertEqual(1, rank(scores, "target"))
        self.assertEqual(2, rank(scores, "alpha"))
        self.assertEqual(3, rank(scores, "beta"))


if __name__ == "__main__":
    unittest.main()
