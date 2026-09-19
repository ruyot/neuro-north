# Background autocomplete process - Mika's linguistic_model turns range picks into words

from __future__ import annotations

import queue
import traceback
from multiprocessing import Event, Process, Queue, Value

# P(picked box) for every SSVEP pick, until recording.py reports per-box probabilities
PICK_CONFIDENCE = 0.85

# top + bottom edges
N_SUGGESTIONS = 2


class AutocompleteProcess(Process):
    """Owns the decoder (+ GPT-2) in a child process, so scoring never stalls a flicker frame.

    The UI sends every action it takes and gets a reply per action:
    {"id", "committed": word a space decoded or None, "suggestions": list or None}.
    Suggestions are None when a newer action was already queued (skipped as stale)."""

    def __init__(self, engine: str = "gpt2", confidence: float = PICK_CONFIDENCE):
        """engine: "gpt2" / "smollm2" / "pythia" (causal re-ranking) or "bigram"."""
        super().__init__(daemon=True)
        self.engine = engine
        self.confidence = confidence

        self.ready = Event()
        self.failed = Value("b", False)
        self._requests = Queue()
        self._replies = Queue()
        self._next_id = 0

    def pick(self, letters: str) -> int:
        """An SSVEP pick of the box showing `letters` (e.g. "abcdef")."""
        return self._send("pick", letters)

    def space(self) -> int:
        """Word boundary: decode the picks so far into the best full-length word."""
        return self._send("space")

    def accept(self, word: str) -> int:
        """The user picked suggestion `word`."""
        return self._send("accept", word)

    def replies(self) -> list[dict]:
        """Replies that arrived since the last call. Never blocks."""
        out = []
        while True:
            try:
                out.append(self._replies.get_nowait())
            except queue.Empty:
                return out

    def stop(self) -> None:
        self._requests.put(None)

    def _send(self, *request) -> int:
        self._next_id += 1
        self._requests.put((self._next_id, *request))
        return self._next_id

    def run(self) -> None:
        try:
            from linguistic_model.simulator import PipelineSimulator

            sim = PipelineSimulator()
            if self.engine != "bigram":
                sim.set_engine(self.engine)
            sim.state()     # warm-up: the first scoring call is ~1 s slower than the rest
            print(f"[autocomplete] {sim.engine_label} ready")
        except Exception:
            traceback.print_exc()
            self.failed.value = True
            return
        self.ready.set()

        while True:
            request = self._requests.get()
            if request is None:
                return
            rid, action, *args = request
            committed, suggestions = None, None
            try:
                committed = self._apply(sim, action, *args)
                if self._requests.empty():
                    suggestions = [c["word"] for c in sim.state()["candidates"][:N_SUGGESTIONS]]
            except Exception:
                traceback.print_exc()
            self._replies.put({"id": rid, "committed": committed, "suggestions": suggestions})

    def _apply(self, sim, action: str, *args) -> str | None:
        from linguistic_model import DecoderError

        if action == "pick":
            label = sim.decoder.ranges.label_for(args[0][0])
            sim.page_index = next(i for i, page in enumerate(sim.pages) if label in page)
            sim.select_range(label, self.confidence)
        elif action == "space":
            if not sim.decoder.observations:
                return None
            try:
                sim.boundary()
            except DecoderError:
                sim.clear_word()    # no dictionary word fits: the ranges stay as typed
                print("[autocomplete] no word fits those ranges - left as typed")
                return None
            return sim.decoder.confirmed_words[-1]
        elif action == "accept":
            word = args[0]
            try:
                sim.accept(word)
            except DecoderError:
                # Picks moved on since this suggestion was shown: keep the word the UI typed.
                sim.clear_word()
                sim.decoder.confirmed_words.append(word)
        return None
