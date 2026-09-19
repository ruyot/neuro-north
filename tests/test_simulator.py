import unittest

from linguistic_model import DecoderError, PipelineSimulator


class PipelineSimulatorTests(unittest.TestCase):
    def setUp(self):
        self.simulator = PipelineSimulator()

    def test_initial_state_exposes_first_page_and_next_word_predictions(self):
        state = self.simulator.state()

        self.assertEqual(["A-F", "G-L"], [target["range"] for target in state["targets"]])
        self.assertEqual([15, 20], [target["frequency"] for target in state["targets"]])
        self.assertEqual(0, state["observation_count"])
        self.assertGreater(state["candidate_count"], 0)
        self.assertEqual(50_000, state["candidate_count"])
        words = self.simulator.decoder.lexicon.words
        self.assertTrue(all(word.isascii() and word.isalpha() for word in words))
        self.assertTrue(all(word == word.lower() for word in words))
        self.assertEqual({"a", "i"}, {word for word in words if len(word) == 1})

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
