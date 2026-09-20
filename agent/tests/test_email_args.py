"""Mapping our small email schema onto GMAIL_CREATE_EMAIL_DRAFT."""
from __future__ import annotations

import unittest

from agent.tools import (
    _compose_contact_search,
    _compose_email_draft,
    describe,
    execute,
    local_contact_email,
    local_contact_hint,
)


class ComposeContactSearch(unittest.TestCase):
    def test_search_is_small_and_email_focused(self):
        out = _compose_contact_search({"query": "sam"})
        self.assertEqual(out["query"], "sam")
        self.assertEqual(out["page_size"], 5)
        self.assertEqual(out["person_fields"], "names,emailAddresses")
        self.assertIs(out["other_contacts"], True)

    def test_local_contact_search_is_narrowed_to_email(self):
        out = _compose_contact_search({"query": "aaron"})
        self.assertEqual(out["query"], "aaronvrgs6561@gmail.com")
        self.assertEqual(out["page_size"], 1)
        self.assertIs(out["other_contacts"], False)


class ComposeEmailDraft(unittest.TestCase):
    def test_subject_body_and_plain_text_mode(self):
        out = _compose_email_draft({"subject": "Running late", "body": "I will be 10 minutes late."})
        self.assertEqual(out["subject"], "Running late")
        self.assertEqual(out["body"], "I will be 10 minutes late.")
        self.assertIs(out["is_html"], False)
        self.assertEqual(out["user_id"], "me")

    def test_first_to_recipient_becomes_primary(self):
        out = _compose_email_draft({
            "to": ["sam@example.com", "alex@example.com"],
            "subject": "Hello",
            "body": "Hi",
        })
        self.assertEqual(out["recipient_email"], "sam@example.com")
        self.assertEqual(out["extra_recipients"], ["alex@example.com"])

    def test_draft_can_have_no_recipient(self):
        out = _compose_email_draft({"to": [], "subject": "Hello", "body": "Hi"})
        self.assertNotIn("recipient_email", out)
        self.assertNotIn("extra_recipients", out)

    def test_cc_and_bcc_are_kept_when_given(self):
        out = _compose_email_draft({
            "cc": ["copy@example.com"],
            "bcc": ["secret@example.com"],
            "subject": "Hello",
            "body": "Hi",
        })
        self.assertEqual(out["cc"], ["copy@example.com"])
        self.assertEqual(out["bcc"], ["secret@example.com"])

    def test_empty_recipient_strings_are_dropped(self):
        out = _compose_email_draft({
            "to": ["", "  ", "sam@example.com"],
            "cc": [" "],
            "subject": "Hello",
            "body": "Hi",
        })
        self.assertEqual(out["recipient_email"], "sam@example.com")
        self.assertNotIn("cc", out)

    def test_local_contact_name_becomes_email_recipient(self):
        out = _compose_email_draft({
            "to": ["aaron"],
            "subject": "Hello",
            "body": "Hi",
        })
        self.assertEqual(out["recipient_email"], "aaronvrgs6561@gmail.com")


class DescribeEmailDraft(unittest.TestCase):
    def test_describes_recipient_and_subject(self):
        self.assertEqual(
            describe("create_email_draft", {"to": ["sam@example.com"], "subject": "Running late"}),
            "Gmail draft to sam@example.com: Running late",
        )

    def test_describes_missing_recipient(self):
        self.assertEqual(
            describe("create_email_draft", {"subject": "Running late"}),
            "Gmail draft: Running late",
        )

    def test_describes_contact_search(self):
        self.assertEqual(
            describe("search_email_contacts", {"query": "sam"}),
            "Contact lookup: sam",
        )

    def test_describes_local_contact_email(self):
        self.assertEqual(
            describe("create_email_draft", {"to": ["aaron"], "subject": "Hello"}),
            "Gmail draft to aaronvrgs6561@gmail.com: Hello",
        )


class LocalContacts(unittest.TestCase):
    def test_aaron_is_available_to_the_agent(self):
        self.assertEqual(local_contact_email("Aaron"), "aaronvrgs6561@gmail.com")
        self.assertIn("aaron <aaronvrgs6561@gmail.com>", local_contact_hint())

    def test_local_contact_search_returns_without_composio_key(self):
        self.assertEqual(
            execute("search_email_contacts", {"query": "aaron"}),
            "LOCAL CONTACT: aaron <aaronvrgs6561@gmail.com>",
        )


if __name__ == "__main__":
    unittest.main()
