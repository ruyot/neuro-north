"""The agent loop: spelled words in, executed tool calls out.

Deliberately the ONLY file that knows which model provider is in use. Swapping
OpenAI for another vendor should not touch tools.py, the speller, or the adapter.

Dry-run is the default. A demo gets iterated on dozens of times, and live runs
write to real accounts, so executing is opt-in (`live=True`).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime

from . import tools as toolkit

DEFAULT_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o")
MAX_STEPS = 4          # one action is typical; the rest is room for a chain

SYSTEM = """You act on messages typed through a brain-computer interface.

The user selects letters by staring at flickering squares, so the text reaching \
you is SHORT, lowercase, and may be missing words or slightly misspelled \
("dentst 3pm", "meet sara tmrw"). Read through the noise and act on the evident \
intent. Do not ask clarifying questions -- the user cannot easily answer.

The current time is {now} in {tz}. Resolve every relative time against it and \
emit absolute ISO-8601 timestamps in the user's LOCAL time with NO offset \
(2026-09-21T15:00:00, never ...T15:00:00-04:00). The timezone is attached for you.

Choose sensible defaults rather than refusing: one hour is a good default \
duration, and a bare time like "3pm" means the next occurrence of it.

For email requests, create a Gmail draft only. Never send email. If the user \
names someone without giving a real email address, search email contacts first \
and use a likely returned email address. If contact search is unavailable or no \
reasonable match is returned, omit the recipient rather than inventing one.

When you have made the tool calls the message calls for, reply with one short \
sentence confirming what you did."""


@dataclass
class Result:
    ok: bool
    summary: str
    calls: list[tuple[str, dict]] = field(default_factory=list)
    live: bool = False
    error: str | None = None

    def status(self) -> str:
        """One short line for the speller's status panel."""
        if not self.ok:
            return f"Agent error: {self.error}"
        if not self.calls:
            return "Agent idle: nothing to do"
        visible = [(name, args) for name, args in self.calls if name != "search_email_contacts"]
        if not visible:
            visible = self.calls
        prefix = "Agent done" if self.live else "Agent plan"
        return f"{prefix}: " + "; ".join(toolkit.describe(n, a) for n, a in visible)


def _client():
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set (put it in .env next to PORT_PATH)")
    from openai import OpenAI

    return OpenAI()


def run(text: str, live: bool = False, timezone: str | None = None,
        model: str = DEFAULT_MODEL, now: datetime | None = None) -> Result:
    """Send `text` through the model and run whatever tools it asks for."""
    text = (text or "").strip()
    if not text:
        return Result(ok=False, summary="", error="nothing typed yet")

    now = now or datetime.now().astimezone()
    zone = timezone or toolkit.local_zone()
    system = SYSTEM.format(now=now.strftime("%A %Y-%m-%dT%H:%M"), tz=zone)

    try:
        client = _client()
    except RuntimeError as exc:
        return Result(ok=False, summary="", error=str(exc))

    messages = [{"role": "system", "content": system},
                {"role": "user", "content": text}]
    calls: list[tuple[str, dict]] = []

    try:
        for _ in range(MAX_STEPS):
            response = client.chat.completions.create(
                model=model, messages=messages,
                tools=toolkit.SCHEMAS, tool_choice="auto",
            )
            message = response.choices[0].message
            messages.append(message)
            if not message.tool_calls:
                return Result(ok=True, summary=message.content or "", calls=calls, live=live)

            for call in message.tool_calls:
                import json

                args = json.loads(call.function.arguments)
                calls.append((call.function.name, args))
                if live:
                    output = toolkit.execute(call.function.name, args, timezone=zone)
                else:
                    # Stop short of the side effect but keep the loop honest: the
                    # model is told the call succeeded so it produces its summary.
                    output = "DRY RUN: not executed."
                messages.append({"role": "tool", "tool_call_id": call.id, "content": output})

            if not live:
                # One round is enough to see the plan; further steps would chain
                # off a fabricated result.
                break

        summary = next((m.content for m in reversed(messages)
                        if getattr(m, "content", None) and getattr(m, "role", "") == "assistant"), "")
        return Result(ok=True, summary=summary or "", calls=calls, live=live)
    except Exception as exc:                      # noqa: BLE001 - surfaced to the UI
        return Result(ok=False, summary="", calls=calls, live=live,
                      error=f"{type(exc).__name__}: {exc}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run one message through the agent.")
    parser.add_argument("text", help="what the speller produced, e.g. 'dentist 3pm' or 'email sara running late'")
    parser.add_argument("--live", action="store_true",
                        help="actually execute tool calls (writes real calendar events or Gmail drafts)")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--tz", default=None, help="timezone name shown to the model")
    args = parser.parse_args()

    result = run(args.text, live=args.live, timezone=args.tz, model=args.model)
    print(result.status())
    if result.summary:
        print(f"model: {result.summary}")
    raise SystemExit(0 if result.ok else 1)


if __name__ == "__main__":
    main()
