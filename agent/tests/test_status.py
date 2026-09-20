"""User-facing agent status strings."""
from __future__ import annotations

import unittest

from agent.runner import Result


class StatusText(unittest.TestCase):
    def test_email_draft_without_recipient_does_not_mention_recipient(self):
        status = Result(
            ok=True,
            summary="",
            live=True,
            calls=[("create_email_draft", {"subject": "Running late"})],
        ).status()
        self.assertEqual(status, "Agent done: Gmail draft: Running late")
        self.assertNotIn("recipient", status.lower())

    def test_contact_lookup_is_hidden_when_draft_is_created(self):
        status = Result(
            ok=True,
            summary="",
            live=True,
            calls=[
                ("search_email_contacts", {"query": "sam"}),
                ("create_email_draft", {"subject": "Running late"}),
            ],
        ).status()
        self.assertEqual(status, "Agent done: Gmail draft: Running late")
        self.assertNotIn("lookup", status.lower())

    def test_dry_run_uses_plan_prefix(self):
        status = Result(
            ok=True,
            summary="",
            live=False,
            calls=[("open_web_page", {"url": "https://news.ycombinator.com"})],
        ).status()
        self.assertEqual(status, "Agent plan: Browser open: https://news.ycombinator.com")


if __name__ == "__main__":
    unittest.main()
