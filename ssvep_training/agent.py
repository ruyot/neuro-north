"""Asynchronous adapter from a finished message to the agent.

Same contract as LanguageService (submit / poll / close) so the UI drives both
the same way. A thread rather than a process: this waits on sockets, it does not
compute, and a thread blocked on I/O costs the flicker loop nothing. Nothing here
is allowed to block -- a network round trip inside run_flicker would drop frames
and corrupt the very SSVEP timing the selection depends on.
"""
from __future__ import annotations

import threading
import time
from queue import Empty, Queue

TIMEOUT = 60.0          # a model call plus a tool call; past this something hung


def _worker(requests: Queue, replies: Queue, live: bool, timezone: str | None) -> None:
    from agent.runner import run

    while True:
        request = requests.get()
        if request is None:
            return
        number, text = request
        try:
            result = run(text, live=live, timezone=timezone)
            replies.put({'id': number, 'status': result.status(), 'ok': result.ok,
                         'summary': result.summary})
        except Exception as exc:                  # noqa: BLE001 - reported, never raised at the UI
            replies.put({'id': number, 'status': f'agent failed: {exc}', 'ok': False,
                         'summary': ''})


class AgentService:
    def __init__(self, live: bool = False, timezone: str | None = None):
        self.live = live
        self.sequence = 0
        self.pending = False
        self.started = 0.0
        self.requests: Queue = Queue()
        self.replies: Queue = Queue()
        self.thread = threading.Thread(target=_worker,
                                       args=(self.requests, self.replies, live, timezone),
                                       daemon=True)
        self.thread.start()

    def submit(self, text: str) -> bool:
        """Queue one message. False while an earlier one is still in flight."""
        if self.pending or not text.strip():
            return False
        self.sequence += 1
        self.pending = True
        self.started = time.monotonic()
        self.requests.put((self.sequence, text))
        return True

    def poll(self) -> dict | None:
        """Newest reply, or None. Never blocks."""
        latest = None
        while True:
            try:
                reply = self.replies.get_nowait()
            except Empty:
                break
            if reply['id'] == self.sequence:
                latest = reply
        if latest is not None:
            self.pending = False
            return latest
        if self.pending and time.monotonic() - self.started > TIMEOUT:
            self.pending = False
            return {'id': self.sequence, 'status': 'agent timed out', 'ok': False, 'summary': ''}
        return None

    def close(self) -> None:
        self.requests.put(None)
        self.thread.join(timeout=3)
