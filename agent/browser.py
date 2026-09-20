"""A visible cloud browser the speller keeps driving.

One Browserbase session is created on first use and held open, so "open hacker
news" and then "click the first story" land on the same page. The live view is
opened in the user's own browser once: the session runs in Browserbase's cloud,
and this is the only way to watch it happen.

    python -m agent.browser https://news.ycombinator.com

Needs BROWSERBASE_API_KEY in .env.

Playwright's sync API is bound to the thread that started it. Every tool call
runs on AgentService's single worker thread, so the session is created and used
there; nothing else may touch it.
"""
from __future__ import annotations

import atexit
import os
import webbrowser

TIMEOUT = 30_000        # ms; AgentService allows 60s for the whole tool call
EXCERPT = 600           # characters of page text handed back to the model
SCROLL = 700            # pixels per scroll action

_LIVE = None            # the one open session; None until the first command


class _Live:
    """One Browserbase session plus the local tab showing it."""

    def __init__(self) -> None:
        key = os.environ.get("BROWSERBASE_API_KEY")
        if not key:
            raise RuntimeError("BROWSERBASE_API_KEY is not set; add it to .env")

        from browserbase import Browserbase
        from playwright.sync_api import sync_playwright

        client = Browserbase(api_key=key)
        project = os.environ.get("BROWSERBASE_PROJECT_ID")
        self.session = client.sessions.create(**({"project_id": project} if project else {}))
        self.view = client.sessions.debug(self.session.id).debugger_fullscreen_url

        # .start() rather than a with-block: the session must outlive this call.
        self.playwright = sync_playwright().start()
        self.browser = self.playwright.chromium.connect_over_cdp(self.session.connect_url)
        self.page = self.browser.contexts[0].pages[0]

        print(f"[browser] live view: {self.view}", flush=True)
        print(f"[browser] replay:    https://browserbase.com/sessions/{self.session.id}", flush=True)
        webbrowser.open(self.view)
        atexit.register(self.close)

    def close(self) -> None:
        for shutdown in (self.browser.close, self.playwright.stop):
            try:
                shutdown()
            except Exception:                     # noqa: BLE001 - teardown must not raise
                pass


def _live() -> _Live:
    global _LIVE
    if _LIVE is None:
        _LIVE = _Live()
    return _LIVE


def _state(note: str = "") -> str:
    """What the model gets back: where we are, and what is on screen."""
    page = _live().page
    try:
        body = page.inner_text("body")
    except Exception as exc:                      # noqa: BLE001 - a blank page is not fatal
        body = f"(could not read page: {exc})"
    excerpt = " ".join(body.split())[:EXCERPT]
    head = f"{note}\n" if note else ""
    return f"{head}{page.title()} | {page.url}\n{excerpt}"


def open_page(url: str) -> str:
    """Navigate the visible browser to `url`."""
    page = _live().page
    page.goto(url, wait_until="domcontentloaded", timeout=TIMEOUT)
    return _state()


def act(action: str, target: str = "") -> str:
    """One step on the page already open. Errors come back as text: a failed
    click is information the model can act on, not a reason to kill the run."""
    page = _live().page
    try:
        if action == "click":
            if not target:
                return "click needs the visible text of what to click"
            page.get_by_text(target, exact=False).first.click(timeout=TIMEOUT)
            page.wait_for_load_state("domcontentloaded", timeout=TIMEOUT)
        elif action == "scroll_down":
            page.mouse.wheel(0, SCROLL)
        elif action == "scroll_up":
            page.mouse.wheel(0, -SCROLL)
        elif action == "back":
            page.go_back(wait_until="domcontentloaded", timeout=TIMEOUT)
        elif action == "read":
            pass
        else:
            return f"unknown action {action!r}"
    except Exception as exc:                      # noqa: BLE001 - reported to the model
        return _state(f"{action} failed: {type(exc).__name__}: {exc}")
    return _state()


def main() -> None:
    """The package __init__ has already loaded .env by the time this runs."""
    import sys
    import time

    if len(sys.argv) != 2:
        raise SystemExit("usage: python -m agent.browser <url>")
    print(open_page(sys.argv[1]))
    print("\nlive view stays up for 60s; Ctrl+C to stop sooner")
    time.sleep(60)


if __name__ == "__main__":
    main()
