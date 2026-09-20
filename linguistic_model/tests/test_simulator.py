import unittest

from linguistic_model import ContextModel, DecoderError, PipelineSimulator

class PreferredWordScorer:
    def __init__(self, preferred):
        self.preferred = preferred

    def score(self, context, candidates):
        return {
            candidate: (100.0 if candidate == self.preferred else 0.0)
            for candidate in candidates
        }

class ContextualBeamScorer:
    def score(self, context, candidates):
        prefix = tuple(context)
        scores = {candidate: -100.0 for candidate in candidates}
        if prefix == ("hi",):
            if "int" in scores:
                scores["int"] = 10.0
            if "how" in scores:
                scores["how"] = 9.0
        elif prefix == ("hi", "how") and "are" in scores:
            scores["are"] = 10.0
        elif prefix == ("hi", "int") and "are" in scores:
            scores["are"] = 0.0
        else:
            for candidate in candidates:
                scores[candidate] = 0.0
        return scores
class RecordingScorer:
    def __init__(self):
        self.contexts = []

    def score(self, context, candidates):
        self.contexts.append(tuple(context))
        return {candidate: 0.0 for candidate in candidates}








class PipelineSimulatorTests(unittest.TestCase):
    def setUp(self):
        self.simulator = PipelineSimulator()

    def test_initial_state_exposes_first_page_and_next_word_predictions(self):
        state = self.simulator.state()
        self.assertEqual("", state["context_prefix"])

        self.assertEqual(["A-F", "G-L"], [target["range"] for target in state["targets"]])
        self.assertEqual([15, 20], [target["frequency"] for target in state["targets"]])
        self.assertEqual(0, state["observation_count"])
        self.assertGreater(state["candidate_count"], 0)
        self.assertEqual(50_000, state["candidate_count"])
        words = self.simulator.decoder.lexicon.words
        self.assertTrue(all(word.isascii() and word.isalpha() for word in words))
        self.assertTrue(all(word == word.lower() for word in words))
        self.assertEqual({"a", "i"}, {word for word in words if len(word) == 1})
        self.assertEqual("bigram", state["engine"])
        self.assertEqual(
            ["bigram", "gpt2", "smollm2", "pythia"],
            [engine["id"] for engine in state["engines"]],
        )
        self.assertEqual(
            [["a", "b", "c", "d", "e", "f"], ["g", "h", "i", "j", "k", "l"]],
            [target["letters"] for target in state["targets"]],
        )

    def test_page_selection_records_only_the_displayed_pair(self):
        self.simulator.next_page()
        self.simulator.select_range("M-R", 0.8)

        state = self.simulator.state()
        self.assertEqual(1, state["observation_count"])
        self.assertEqual("M-R", state["history"][0]["selected"])
        self.assertAlmostEqual(0.8, state["history"][0]["probabilities"]["M-R"])
        self.assertAlmostEqual(0.2, state["history"][0]["probabilities"]["S-Z"])

        with self.assertRaises(DecoderError):
            self.simulator.select_range("A-F", 0.8)

    def test_complete_mouse_sequence_surfaces_apple_for_explicit_acceptance(self):
        self.simulator.select_range("A-F", 0.9)
        self.simulator.next_page()
        self.simulator.select_range("M-R", 0.9)
        self.simulator.select_range("M-R", 0.9)
        self.simulator.next_page()
        self.simulator.select_range("G-L", 0.9)
        self.simulator.select_range("A-F", 0.9)

        words = [candidate["word"] for candidate in self.simulator.state()["candidates"]]
        self.assertIn("apple", words)
        self.simulator.accept("apple")

        state = self.simulator.state()
        self.assertEqual("apple", state["sentence"])
        self.assertEqual(0, state["observation_count"])
        self.assertEqual([], state["history"])

    def test_accepting_prediction_clears_current_word_and_updates_context(self):
        self.simulator.select_range("G-L", 0.9)
        candidate = self.simulator.state()["candidates"][0]["word"]

        self.simulator.accept(candidate)

        state = self.simulator.state()
        self.assertEqual(candidate, state["sentence"])
        self.assertEqual(0, state["observation_count"])
        self.assertEqual([], state["history"])

    def test_full_range_sequence_decodes_name_after_my(self):
        self.simulator.accept("hello")
        self.simulator.accept("my")
        self.simulator.next_page()
        self.simulator.select_range("M-R", 0.9)
        self.simulator.next_page()
        self.simulator.select_range("A-F", 0.9)
        self.simulator.next_page()
        self.simulator.select_range("M-R", 0.9)
        self.simulator.next_page()
        self.simulator.select_range("A-F", 0.9)

        state = self.simulator.state()
        self.assertEqual("name", state["candidates"][0]["word"])
        self.simulator.boundary()
        self.assertEqual("hello my name", self.simulator.state()["sentence"])

    def test_backspace_removes_observation_before_confirmed_context(self):
        self.simulator.select_range("A-F", 0.8)
        self.simulator.backspace()
        self.assertEqual(0, self.simulator.state()["observation_count"])

        self.simulator.accept("the")
        self.simulator.backspace()
        self.assertEqual("", self.simulator.state()["sentence"])

    def test_causal_engine_reranks_the_decoder_shortlist(self):
        shortlist = self.simulator.decoder.candidates(limit=256).candidates
        preferred = shortlist[-1].word
        self.simulator._scorers["gpt2"] = PreferredWordScorer(preferred)

        self.simulator.set_engine("gpt2")
        state = self.simulator.state()

        self.assertEqual("gpt2", state["engine"])
        self.assertEqual("GPT-2", state["engine_label"])
        self.assertEqual(preferred, state["candidates"][0]["word"])

    def test_boundary_uses_the_selected_completion_engine(self):
        self.simulator.select_range("A-F", 0.9)
        self.simulator.select_range("A-F", 0.9)
        exact = self.simulator.decoder.candidates(
            limit=256,
            completed_only=True,
        ).candidates
        preferred = exact[-1].word
        self.simulator._scorers["gpt2"] = PreferredWordScorer(preferred)
        self.simulator.set_engine("gpt2")

        self.simulator.boundary()

        self.assertEqual(preferred, self.simulator.state()["sentence"])

    def test_unknown_completion_engine_is_rejected(self):
        with self.assertRaisesRegex(DecoderError, "unknown completion engine"):
            self.simulator.set_engine("unknown")
        self.assertEqual("bigram", self.simulator.state()["engine"])

    def _accept_hi(self):
        self.simulator.select_range("G-L", 0.9)
        self.simulator.select_range("G-L", 0.9)
        self.simulator.accept("hi")

    def _enter_how_ranges(self):
        self.simulator.select_range("G-L", 0.9)
        self.simulator.next_page()
        self.simulator.select_range("M-R", 0.9)
        self.simulator.select_range("S-Z", 0.9)

    def _configure_contextual_beam(self, mode):
        self.simulator._scorers["gpt2"] = ContextualBeamScorer()
        self.simulator.set_engine("gpt2")
        self.simulator.set_decode_mode(mode)

    def test_fixed_lag_beam_revises_previous_word_from_later_context(self):
        self._configure_contextual_beam("fixed-lag")
        self._accept_hi()
        self._enter_how_ranges()
        self.simulator.boundary()

        state = self.simulator.state()
        self.assertEqual(["hi"], state["confirmed_words"])
        self.assertEqual(["int"], state["tentative_words"])
        self.assertGreater(state["beam_count"], 1)

        self.simulator.next_page()
        self.simulator.select_range("A-F", 0.9)
        self.simulator.accept("are")

        state = self.simulator.state()
        self.assertEqual(["hi", "how"], state["confirmed_words"])
        self.assertEqual(["are"], state["tentative_words"])

    def test_sentence_beam_defers_all_words_until_finish(self):
        self._configure_contextual_beam("sentence-beam")
        self._accept_hi()
        self._enter_how_ranges()
        self.simulator.boundary()
        self.simulator.next_page()
        self.simulator.select_range("A-F", 0.9)
        self.simulator.accept("are")

        state = self.simulator.state()
        self.assertEqual([], state["confirmed_words"])
        self.assertEqual(["hi", "how", "are"], state["tentative_words"])
        self.assertTrue(state["can_finish"])

        self.simulator.finish_sentence()
        state = self.simulator.state()
        self.assertEqual(["hi", "how", "are"], state["confirmed_words"])
        self.assertEqual([], state["tentative_words"])

    def test_decode_mode_change_finalizes_tentative_words(self):
        self.simulator.set_decode_mode("fixed-lag")
        self.simulator.accept("the")

        self.simulator.set_decode_mode("greedy")

        state = self.simulator.state()
        self.assertEqual(["the"], state["confirmed_words"])
        self.assertEqual([], state["tentative_words"])
        self.assertEqual("greedy", state["decode_mode"])
    def test_conversation_context_changes_bigram_prediction(self):
        # Context behavior uses a fixture: the branch ships no bigram corpus.
        decoder = self.simulator.decoder
        decoder.context_model = ContextModel(decoder.lexicon, {("my", "name"): 100})
        self.assertEqual("the", self.simulator.state()["candidates"][0]["word"])

        self.simulator.set_context_prefix("Hello, my")
        state = self.simulator.state()

        self.assertEqual("Hello, my", state["context_prefix"])
        self.assertEqual("name", state["candidates"][0]["word"])

    def test_causal_scorer_receives_question_before_decoded_words(self):
        scorer = RecordingScorer()
        self.simulator._scorers["gpt2"] = scorer
        self.simulator.set_engine("gpt2")
        self.simulator.set_context_prefix("What is your name?")

        self.simulator.state()

        self.assertEqual(("What is your name?",), scorer.contexts[-1])

    def test_context_change_finalizes_pending_beam(self):
        self.simulator.set_decode_mode("fixed-lag")
        self.simulator.accept("the")

        self.simulator.set_context_prefix("What comes next?")

        state = self.simulator.state()
        self.assertEqual(["the"], state["confirmed_words"])
        self.assertEqual([], state["tentative_words"])

    def test_conversation_context_length_is_bounded(self):
        with self.assertRaisesRegex(DecoderError, "at most 500 characters"):
            self.simulator.set_context_prefix("x" * 501)


    def test_unknown_decode_mode_is_rejected(self):
        with self.assertRaisesRegex(DecoderError, "unknown decode mode"):
            self.simulator.set_decode_mode("unknown")

    def test_reset_restores_the_first_page_and_empty_session(self):
        self.simulator.next_page()
        self.simulator.select_range("M-R", 0.8)
        self.simulator.reset()

        state = self.simulator.state()
        self.assertEqual(0, state["page"])
        self.assertEqual("", state["sentence"])
        self.assertEqual(0, state["observation_count"])


if __name__ == "__main__":
    unittest.main()
