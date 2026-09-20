"""What the agent remembers between commands, with no model and no browser.

The bug these pin: `run` used to build its message list from scratch, so the
second command never learned a page was already open and reopened the web
instead of continuing on it. What matters is therefore the message list the
NEXT command is sent, not the reply to this one.
"""
from __future__ import annotations

import json
import unittest
from types import SimpleNamespace
from unittest import mock

from agent import runner


def _reply(content=None, calls=()):
    """One assistant message, optionally asking for tool calls."""
    return SimpleNamespace(role="assistant", content=content, tool_calls=[
        SimpleNamespace(id=f"call_{i}",
                        function=SimpleNamespace(name=name, arguments=json.dumps(args)))
        for i, (name, args) in enumerate(calls)])


class Client:
    """Replays scripted replies and keeps every request it was sent."""

    def __init__(self, *replies):
        self.replies, self.sent = list(replies), []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, messages, **_):
        self.sent.append(list(messages))
        return SimpleNamespace(choices=[SimpleNamespace(message=self.replies.pop(0))])


def _roles(messages):
    return [runner._role(m) for m in messages]


def _drive(client, text, live=False):
    """One run() against a scripted client. Returns the tools actually executed."""
    executed = []

    def execute(name, args, **_):
        executed.append(name)
        return f"{name} | https://example.com\npage text"

    with mock.patch.object(runner, "_client", lambda: client), \
         mock.patch.object(runner.toolkit, "execute", execute):
        runner.run(text, live=live)
    return executed


class Continuity(unittest.TestCase):
    def setUp(self):
        runner._history.clear()
        self.addCleanup(runner._history.clear)

    def test_the_next_command_is_shown_the_open_page(self):
        opened = Client(_reply(calls=[("open_web_page", {"url": "https://google.com"})]),
                        _reply(content="Opened Google."))
        _drive(opened, "open google")

        following = Client(_reply(content="Searched."))
        _drive(following, "search for neuralink")

        self.assertEqual(_roles(following.sent[0]),
                         ["system", "user", "assistant", "tool", "assistant", "user"])
        self.assertIn("https://example.com", str(following.sent[0]))

    def test_a_trimmed_history_never_starts_mid_turn(self):
        # A tool result carried over without the assistant message that asked
        # for it makes the next request invalid: whole turns or nothing.
        with mock.patch.object(runner, "HISTORY", 3):
            _drive(Client(_reply(calls=[("open_web_page", {"url": "https://x.com"})]),
                          _reply(content="Opened.")), "open x")
        self.assertNotIn(_roles(runner._history)[:1], (["tool"], ["assistant"]))


class DryRun(unittest.TestCase):
    def setUp(self):
        runner._history.clear()
        self.addCleanup(runner._history.clear)

    def test_the_browser_runs_but_the_calendar_does_not(self):
        executed = _drive(
            Client(_reply(calls=[("open_web_page", {"url": "https://x.com"}),
                                 ("create_calendar_event", {"summary": "s", "start": "a", "end": "b"})]),
                   _reply(content="Done.")),
            "open x and book it")
        self.assertEqual(executed, ["open_web_page"])

    def test_live_runs_everything(self):
        executed = _drive(
            Client(_reply(calls=[("create_calendar_event", {"summary": "s", "start": "a", "end": "b"})]),
                   _reply(content="Done.")),
            "book it", live=True)
        self.assertEqual(executed, ["create_calendar_event"])

    def test_status_separates_what_ran_from_what_was_only_planned(self):
        # The panel used to label the whole list "would create", which now reads
        # as a lie about the browsing the user just watched happen.
        status = runner.Result(ok=True, summary="", live=False, calls=[
            ("open_web_page", {"url": "https://google.com"}),
            ("create_calendar_event", {"summary": "Dentist", "start": "2026-09-21T15:00", "end": ""}),
        ]).status()
        self.assertEqual(status,
                         "did: open https://google.com | would create: Dentist @ 2026-09-21T15:00")


if __name__ == "__main__":
    unittest.main()
