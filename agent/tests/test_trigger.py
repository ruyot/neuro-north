"""The double-right-edge send trigger, exercised without psychopy or a headset.

SpellerUI needs a window, so these bind the real methods to a stub carrying the
same attributes. That keeps the state machine under test even though the widget
around it cannot be built in CI.
"""
from __future__ import annotations

import time
import unittest

from ssvep_training.speller_ui import SEND_WINDOW, SpellerUI


class StubAgent:
    def __init__(self, accept=True):
        self.accept, self.sent = accept, []

    def submit(self, text):
        self.sent.append(text)
        return self.accept


class StubUI:
    """Just the attributes the trigger methods touch."""

    def __init__(self, agent=None, language=object(), confirmed="", tentative=""):
        self.agent, self.language = agent, language
        self._confirmed_text, self._tentative_text = confirmed, tentative
        self._send_armed_at = 0.0
        self.action_count = 0
        self.agent_status = type("S", (), {"text": "", "color": ""})()
        self._flashed = []

    _send_armed = SpellerUI._send_armed
    message_text = SpellerUI.message_text
    send_to_agent = SpellerUI.send_to_agent

    def _flash(self, name, seconds=0.3):
        self._flashed.append(name)


class MessageText(unittest.TestCase):
    def test_joins_confirmed_and_tentative(self):
        ui = StubUI(confirmed="book dentist", tentative="tomorrow")
        self.assertEqual(ui.message_text(), "book dentist tomorrow")

    def test_confirmed_only(self):
        self.assertEqual(StubUI(confirmed="hello").message_text(), "hello")

    def test_empty_when_nothing_decoded(self):
        self.assertEqual(StubUI().message_text(), "")


class Arming(unittest.TestCase):
    def test_not_armed_without_agent(self):
        ui = StubUI(agent=None)
        ui._send_armed_at = time.perf_counter()
        self.assertFalse(ui._send_armed())

    def test_not_armed_without_language(self):
        """--engine off keeps a real double space; the trigger must stay away."""
        ui = StubUI(agent=StubAgent(), language=None)
        ui._send_armed_at = time.perf_counter()
        self.assertFalse(ui._send_armed())

    def test_armed_after_a_space(self):
        ui = StubUI(agent=StubAgent())
        ui._send_armed_at = time.perf_counter()
        self.assertTrue(ui._send_armed())

    def test_expires(self):
        ui = StubUI(agent=StubAgent())
        ui._send_armed_at = time.perf_counter() - (SEND_WINDOW + 0.1)
        self.assertFalse(ui._send_armed())

    def test_cold_start_is_not_armed(self):
        self.assertFalse(StubUI(agent=StubAgent())._send_armed())


class Sending(unittest.TestCase):
    def test_sends_the_message(self):
        agent = StubAgent()
        ui = StubUI(agent=agent, confirmed="dentist 3pm")
        ui.send_to_agent()
        self.assertEqual(agent.sent, ["dentist 3pm"])

    def test_disarms_after_sending(self):
        ui = StubUI(agent=StubAgent(), confirmed="hi")
        ui._send_armed_at = time.perf_counter()
        ui.send_to_agent()
        self.assertFalse(ui._send_armed())

    def test_nothing_typed_is_a_no_op(self):
        agent = StubAgent()
        StubUI(agent=agent).send_to_agent()
        self.assertEqual(agent.sent, [])

    def test_busy_agent_does_not_queue(self):
        agent = StubAgent(accept=False)
        ui = StubUI(agent=agent, confirmed="hi")
        ui.send_to_agent()
        self.assertEqual(ui.agent_status.text, "")   # status untouched on refusal

    def test_send_does_not_bump_action_count(self):
        """An in-flight EEG selection must survive a send: run_flicker drops the
        selection when action_count moves, and a send changes no text."""
        ui = StubUI(agent=StubAgent(), confirmed="hi")
        ui.send_to_agent()
        self.assertEqual(ui.action_count, 0)


if __name__ == "__main__":
    unittest.main()
