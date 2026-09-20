import unittest

from linguistic_model import ContextModel, DecoderError, LetterRanges, Lexicon, LinguisticDecoder


class LinguisticDecoderTests(unittest.TestCase):
    def make_decoder(self, counts, bigrams=None, confirmed=None):
        ranges = LetterRanges.alphabet_quarters()
        lexicon = Lexicon(counts, ranges)
        return LinguisticDecoder(
            ranges,
            lexicon,
            ContextModel(lexicon, bigrams),
            confirmed_words=list(confirmed or ()),
        )

    def test_context_can_correct_a_marginal_range_classification(self):
        decoder = self.make_decoder(
            {"eat": 10, "apple": 10, "hello": 10},
            {("eat", "apple"): 100, ("eat", "hello"): 1},
            confirmed=["eat"],
        )

        decoder.observe({"A-F": 0.4, "G-L": 0.6}, page="first pair")

        result = decoder.candidates(limit=2)
        self.assertEqual("apple", result.candidates[0].word)
        self.assertGreater(result.candidates[0].probability, result.candidates[1].probability)

    def test_observation_only_matches_ranges_offered_on_its_page(self):
        decoder = self.make_decoder({"apple": 1, "hello": 1, "mouse": 1})

        decoder.observe({"A-F": 0.5, "G-L": 0.5}, page="first pair")

        words = [candidate.word for candidate in decoder.candidates(limit=10).candidates]
        self.assertEqual(["apple", "hello"], words)
        self.assertNotIn("mouse", words)

    def test_boundary_requires_exact_length_but_completion_is_explicit(self):
        decoder = self.make_decoder({"an": 2, "ant": 1})
        decoder.observe({"A-F": 1.0})
        decoder.observe({"M-R": 1.0})

        result = decoder.candidates(limit=10)
        self.assertEqual({"an", "ant"}, {candidate.word for candidate in result.candidates})
        self.assertEqual("an", decoder.confirm_boundary())
        self.assertEqual(["an"], decoder.confirmed_words)
        self.assertEqual([], decoder.observations)

        decoder.observe({"A-F": 1.0})
        decoder.observe({"M-R": 1.0})
        self.assertEqual("ant", decoder.accept("ant", allow_completion=True))

    def test_classifier_prior_is_removed_before_language_fusion(self):
        decoder = self.make_decoder({"apple": 1, "hello": 1})
        decoder.observe(
            {"A-F": 0.8, "G-L": 0.2},
            selection_priors={"A-F": 0.8, "G-L": 0.2},
        )

        result = decoder.candidates(limit=2)
        self.assertAlmostEqual(0.5, result.candidates[0].probability)
        self.assertAlmostEqual(0.5, result.candidates[1].probability)

    def test_invalid_confirmation_preserves_current_state(self):
        decoder = self.make_decoder({"apple": 1, "hello": 1})
        decoder.observe({"A-F": 1.0})

        with self.assertRaises(DecoderError):
            decoder.accept("hello")

        self.assertEqual(1, len(decoder.observations))
        self.assertEqual([], decoder.confirmed_words)

    def test_undo_removes_only_the_last_confirmed_word(self):
        decoder = self.make_decoder({"an": 1, "ant": 1}, confirmed=["an", "ant"])

        self.assertEqual("ant", decoder.undo_last_word())
        self.assertEqual(["an"], decoder.confirmed_words)


if __name__ == "__main__":
    unittest.main()
