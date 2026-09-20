"""The agent loop: spelled words in, executed tool calls out.

Deliberately the ONLY file that knows which model provider is in use. Swapping
OpenAI for another vendor should not touch tools.py, the speller, or the adapter.

Dry-run is the default, because every live run writes a real event to a real
calendar and a demo gets iterated on dozens of times. It gates the tools that
touch the user's accounts; the browser tools run either way, since a cloud
browser session is ours and the model cannot steer a page it was never shown.

The conversation is kept between calls (`_history`), so "search for x" continues
on the page "open google" left open instead of starting the web again.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime

from . import tools as toolkit

DEFAULT_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o")
MAX_STEPS = 8          # a calendar event is one call; a browse chain is several
HISTORY = 40           # messages carried to the next command, oldest turns dropped

# The browser session outlives one message, so the conversation has to as well.
# Rebuilt from scratch each call, the model never learns a page is already open:
# it reopens the web on every command instead of continuing on the page in front
# of the user. This is that memory.
_history: list = []

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

A cloud browser stays open between messages, and the tool results already in \
this conversation show the page it is on right now. Continue on that page -- \
type into it, click it, scroll it -- rather than opening a fresh one. Only open \
a new URL when nothing is open yet or the user names a different site.

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
            return f"agent failed: {self.error}"
        if not self.calls:
            return "agent: nothing to do"
        # The browser runs in a dry run, the calendar does not, so one label for
        # the whole list would lie about half of it.
        executed = "did: ", [toolkit.describe(n, a) for n, a in self.calls
                             if self.live or n in toolkit.LOCAL_CALLS]
        held = "would create: ", [toolkit.describe(n, a) for n, a in self.calls
                                  if not (self.live or n in toolkit.LOCAL_CALLS)]
        return " | ".join(label + "; ".join(lines) for label, lines in (executed, held) if lines)


def _client():
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set (put it in .env next to PORT_PATH)")
    from openai import OpenAI

    return OpenAI()


def _role(message) -> str:
    """Messages are dicts going out and model objects coming back."""
    return message["role"] if isinstance(message, dict) else getattr(message, "role", "")


def _remember(messages: list) -> None:
    """Carry this exchange to the next command, minus the system message whose
    clock is rebuilt every call. Trimmed to whole turns: a tool result separated
    from the assistant message that asked for it makes the next request invalid.
    """
    kept = messages[1:][-HISTORY:]
    while kept and _role(kept[0]) != "user":
        kept.pop(0)
    _history[:] = kept


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

    messages = [{"role": "system", "content": system}, *_history,
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
                _remember(messages)
                return Result(ok=True, summary=message.content or "", calls=calls, live=live)

            faked = False
            for call in message.tool_calls:
                args = json.loads(call.function.arguments)
                calls.append((call.function.name, args))
                # Dry run exists to protect the user's ACCOUNTS. Driving a browser
                # session touches nothing of theirs, and faking its result would
                # leave the model acting on a page it was never shown -- so the
                # local tools run either way.
                if live or call.function.name in toolkit.LOCAL_CALLS:
                    output = toolkit.execute(call.function.name, args, timezone=zone)
                else:
                    # Stop short of the side effect but keep the loop honest: the
                    # model is told the call succeeded so it produces its summary.
                    output = "DRY RUN: not executed."
                    faked = True
                messages.append({"role": "tool", "tool_call_id": call.id, "content": output})

            if faked:
                # One round is enough to see the plan; further steps would chain
                # off a fabricated result.
                break

        _remember(messages)
        summary = next((m.content for m in reversed(messages)
                        if getattr(m, "content", None) and getattr(m, "role", "") == "assistant"), "")
        return Result(ok=True, summary=summary or "", calls=calls, live=live)
    except Exception as exc:                      # noqa: BLE001 - surfaced to the UI
        return Result(ok=False, summary="", calls=calls, live=live,
                      error=f"{type(exc).__name__}: {exc}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run one message through the agent.")
    parser.add_argument("text", help="what the speller produced, e.g. 'dentist 3pm tomorrow'")
    parser.add_argument("--live", action="store_true",
                        help="actually execute the tool calls (writes to your real calendar)")
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
