"""Tool schemas the model may call, and the Composio calls that execute them.

The schemas here are deliberately OURS rather than Composio's own. Composio
exposes every optional field an action supports; a speller gives the model four
noisy words to fill them from, and a wide schema invites invented arguments.
These are narrow on purpose: required fields only, absolute timestamps only.

Translating our fields to Composio's argument names happens in ONE place per
tool (`_compose_*`), because those names are the part most likely to be wrong
until someone runs `python -m agent.tools --schema` against a real API key.
"""
from __future__ import annotations

import os

# Composio action slugs. Verify with --schema; slugs are stable, arguments drift.
CALENDAR_CREATE = "GOOGLECALENDAR_CREATE_EVENT"

# Direct execution refuses to run without a pinned version ("latest" is rejected).
# Pinned rather than version-skipped on purpose: the argument mapping below was
# read off THIS version, and a toolkit update the night before a demo should not
# be able to change what the speller sends. `--versions` lists newer ones.
TOOLKIT = "googlecalendar"
TOOLKIT_VERSION = os.environ.get("COMPOSIO_GOOGLECALENDAR_VERSION", "20260915_00")

SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "create_calendar_event",
            "description": (
                "Create an event on the user's primary Google Calendar. "
                "Use for any request to schedule, book, remind, or set up something "
                "at a time. Resolve relative times ('tomorrow', '3pm') against the "
                "current time given in the system message."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": "Event title, expanded into natural English. "
                                       "'dentist' -> 'Dentist appointment'.",
                    },
                    "start": {
                        "type": "string",
                        "description": "Start as a NAIVE local ISO-8601 timestamp with NO "
                                       "timezone offset, e.g. 2026-09-21T15:00:00. The user's "
                                       "timezone is applied downstream. Never relative.",
                    },
                    "end": {
                        "type": "string",
                        "description": "End as a naive local ISO-8601 timestamp, no offset. "
                                       "Default to one hour after start.",
                    },
                    "description": {
                        "type": "string",
                        "description": "Optional detail. Say it came from the speller.",
                    },
                },
                "required": ["summary", "start", "end"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "open_web_page",
            "description": (
                "Point the user's visible cloud browser at a URL. Use it to START "
                "somewhere: a named site, or https://duckduckgo.com/?q=<url-encoded+query> "
                "for a search from nothing. The SAME browser stays open between "
                "messages, so once a page is up prefer browser_act -- typing a new "
                "search into the page the user is looking at is what they asked for; "
                "reopening the web from scratch throws their page away."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "Absolute URL including the scheme, "
                                       "e.g. https://news.ycombinator.com.",
                    },
                },
                "required": ["url"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "browser_act",
            "description": (
                "Do one thing on the page the user's browser is already showing: "
                "type into a box (a search field, a form), click something, scroll, "
                "go back, or re-read it. This is how a request continues from the "
                "current page -- 'search for x' with Google already open, 'click the "
                "first story', 'scroll down'. The page text in the most recent tool "
                "result is what is on screen right now; work from it."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["type", "click", "scroll_down", "scroll_up", "back", "read"],
                        "description": "What to do on the current page.",
                    },
                    "target": {
                        "type": "string",
                        "description": "For click: the visible text of the link or "
                                       "button, copied from the page text you were "
                                       "given. For type: which box to type in, named "
                                       "by its placeholder or label (e.g. 'Search'); "
                                       "empty picks the page's main text box. Empty "
                                       "for every other action.",
                    },
                    "text": {
                        "type": "string",
                        "description": "For type only: what to enter. It is submitted "
                                       "with Enter, so a search box searches straight "
                                       "away. Empty for every other action.",
                    },
                },
                "required": ["action", "target", "text"],
                "additionalProperties": False,
            },
        },
    },
]


def local_zone() -> str:
    """This machine's IANA zone name, e.g. America/Toronto.

    Composio rejects abbreviations like EDT, and time.tzname only gives those,
    so read the zoneinfo path /etc/localtime points at. UTC if that fails --
    wrong, but wrong loudly rather than four hours off in silence.
    """
    import os

    path = os.path.realpath("/etc/localtime")
    marker = "/zoneinfo/"
    return path.split(marker, 1)[1] if marker in path else "UTC"


def _naive(stamp: str) -> str:
    """Drop any offset the model emitted anyway.

    Composio strips 'Z' and '+/-hh:mm' itself and then assumes UTC when no
    timezone field is sent -- which silently moved a 3pm event to 11am in
    testing. We always send the zone separately, so the offset must not travel.
    """
    stamp = stamp.strip()
    if stamp.endswith("Z"):
        return stamp[:-1]
    head, sep, tail = stamp.rpartition("+")
    if sep and ":" in tail:
        return head
    # A '-' offset only ever follows the time, so look after 'T'.
    date, _, time_part = stamp.partition("T")
    head, sep, tail = time_part.rpartition("-")
    return f"{date}T{head}" if sep and ":" in tail else stamp


def _compose_calendar(args: dict, timezone: str) -> dict:
    """Our fields -> GOOGLECALENDAR_CREATE_EVENT's arguments."""
    return {
        "summary": args["summary"],
        "start_datetime": _naive(args["start"]),
        "end_datetime": _naive(args["end"]),
        "timezone": timezone,
        "description": args.get("description", "Created from the SSVEP speller."),
        "calendar_id": "primary",
        # Defaults to True, which hangs a Google Meet link off a dentist
        # appointment. Noise on screen; turn it off.
        "create_meeting_room": False,
    }


COMPOSIO_CALLS = {
    "create_calendar_event": (CALENDAR_CREATE, _compose_calendar),
}


def _browse(args: dict) -> str:
    from .browser import open_page
    return open_page(args["url"])


def _act(args: dict) -> str:
    from .browser import act
    return act(args["action"], args.get("target", ""), args.get("text", ""))


# Tools this machine runs itself. Composio is for accounts we act on behalf of;
# a browser session needs no third-party authorisation, so it stays local.
LOCAL_CALLS = {
    "open_web_page": _browse,
    "browser_act": _act,
}


def describe(name: str, args: dict) -> str:
    """One human line for the console and the speller's status panel."""
    if name == "create_calendar_event":
        return f"{args.get('summary', '?')} @ {args.get('start', '?')}"
    if name == "open_web_page":
        return f"open {args.get('url', '?')}"
    if name == "browser_act":
        return " ".join(p for p in (args.get("action", "?"), args.get("target", ""),
                                    args.get("text", "")) if p)
    return f"{name}({args})"


def execute(name: str, args: dict, user_id: str | None = None,
            timezone: str | None = None) -> str:
    """Run the tool for real. Raises if its backend is not configured."""
    if name in LOCAL_CALLS:
        return LOCAL_CALLS[name](args)
    if name not in COMPOSIO_CALLS:
        raise ValueError(f"unknown tool {name!r}")
    if not os.environ.get("COMPOSIO_API_KEY"):
        raise RuntimeError("COMPOSIO_API_KEY is not set; run without --live to dry-run")

    from composio import Composio

    slug, compose = COMPOSIO_CALLS[name]
    composio = Composio(toolkit_versions={TOOLKIT: TOOLKIT_VERSION})
    result = composio.tools.execute(
        slug,
        user_id=user_id or os.environ.get("COMPOSIO_USER_ID", "speller"),
        arguments=compose(args, timezone or local_zone()),
    )
    return str(result)


def main() -> None:
    """--schema prints Composio's real argument names, so _compose_* can be fixed."""
    import argparse
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schema", action="store_true",
                        help="print each action's live argument names (needs COMPOSIO_API_KEY)")
    parser.add_argument("--versions", action="store_true",
                        help="list toolkit versions; the pinned one is marked")
    args = parser.parse_args()
    if not (args.schema or args.versions):
        print(json.dumps(SCHEMAS, indent=2))
        return

    from composio import Composio

    composio = Composio()
    for slug, _ in COMPOSIO_CALLS.values():
        raw = composio.tools.get_raw_composio_tool_by_slug(slug)
        if args.versions:
            print(f"\n=== {slug} ===")
            print(f"pinned:  {TOOLKIT_VERSION}")
            print(f"current: {getattr(raw, 'version', '?')}")
            for version in (getattr(raw, "available_versions", None) or [])[:8]:
                print(f"  {version}{'   <- pinned' if version == TOOLKIT_VERSION else ''}")
            continue
        print(f"\n=== {slug} (version {getattr(raw, 'version', '?')}) ===")
        params = json.loads(json.dumps(getattr(raw, "input_parameters", {}), default=str))
        print("required:", params.get("required"))
        for field, spec in (params.get("properties") or {}).items():
            print(f"  {field:26} {spec.get('type','?')}")


if __name__ == "__main__":
    main()
