"""Mapping our tool fields onto GOOGLECALENDAR_CREATE_EVENT.

Composio strips any offset from start_datetime/end_datetime and then assumes UTC
when no timezone field is sent. Sending '15:00:00-04:00' therefore books 15:00
UTC -- 11am local. Every case here exists to keep that from coming back.
"""
from __future__ import annotations

import unittest

from agent.tools import _compose_calendar, _naive, local_zone


class Naive(unittest.TestCase):
    def test_strips_negative_offset(self):
        self.assertEqual(_naive("2026-09-21T15:00:00-04:00"), "2026-09-21T15:00:00")

    def test_strips_positive_offset(self):
        self.assertEqual(_naive("2026-09-21T15:00:00+05:30"), "2026-09-21T15:00:00")

    def test_strips_zulu(self):
        self.assertEqual(_naive("2026-09-21T15:00:00Z"), "2026-09-21T15:00:00")

    def test_leaves_naive_alone(self):
        self.assertEqual(_naive("2026-09-21T15:00:00"), "2026-09-21T15:00:00")

    def test_leaves_minute_precision_alone(self):
        self.assertEqual(_naive("2026-09-21T15:00"), "2026-09-21T15:00")

    def test_date_dashes_are_not_mistaken_for_an_offset(self):
        self.assertEqual(_naive("2026-09-21T09:30:00"), "2026-09-21T09:30:00")

    def test_tolerates_whitespace(self):
        self.assertEqual(_naive("  2026-09-21T15:00:00-04:00 "), "2026-09-21T15:00:00")


class Compose(unittest.TestCase):
    def setUp(self):
        self.args = {"summary": "Dentist appointment",
                     "start": "2026-09-21T15:00:00-04:00",
                     "end": "2026-09-21T16:00:00-04:00"}

    def test_times_lose_their_offset(self):
        out = _compose_calendar(self.args, "America/Toronto")
        self.assertEqual(out["start_datetime"], "2026-09-21T15:00:00")
        self.assertEqual(out["end_datetime"], "2026-09-21T16:00:00")

    def test_zone_travels_separately(self):
        out = _compose_calendar(self.args, "America/Toronto")
        self.assertEqual(out["timezone"], "America/Toronto")

    def test_no_meet_link(self):
        self.assertIs(_compose_calendar(self.args, "UTC")["create_meeting_room"], False)

    def test_primary_calendar(self):
        self.assertEqual(_compose_calendar(self.args, "UTC")["calendar_id"], "primary")

    def test_description_defaults_to_provenance(self):
        self.assertIn("speller", _compose_calendar(self.args, "UTC")["description"])

    def test_description_is_kept_when_given(self):
        out = _compose_calendar({**self.args, "description": "back molar"}, "UTC")
        self.assertEqual(out["description"], "back molar")


class Zone(unittest.TestCase):
    def test_is_an_iana_name_not_an_abbreviation(self):
        import zoneinfo

        zone = local_zone()
        self.assertIn("/", zone, f"{zone!r} is not IANA; Composio rejects abbreviations")
        zoneinfo.ZoneInfo(zone)          # raises if not a real zone


if __name__ == "__main__":
    unittest.main()
