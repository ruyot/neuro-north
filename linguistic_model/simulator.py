"""State machine used by the mouse-driven SSVEP pipeline simulator."""

from dataclasses import dataclass
from functools import lru_cache
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from wordfreq import iter_wordlist, word_frequency

from .decoder import ContextModel, DecoderError, LetterRanges, Lexicon, LinguisticDecoder


_DATA = Path(__file__).with_name("data")
DEFAULT_LEXICON_SIZE = 50_000
DEFAULT_LANGUAGE_WEIGHT = 0.35
CAUSAL_LANGUAGE_WEIGHT = 0.35
MAX_CONTEXT_LENGTH = 500
_CONTEXT_WORD = re.compile(r"[a-z]+")
CAUSAL_SHORTLIST_SIZE = 256
ENGINE_MODELS = {
    "gpt2": ("GPT-2", "openai-community/gpt2"),
    "smollm2": ("SmolLM2", "HuggingFaceTB/SmolLM2-135M"),
    "pythia": ("Pythia", "EleutherAI/pythia-160m"),
}
BEAM_WIDTH = 8
DECODE_MODES = {
    "greedy": ("Greedy", 0),
    "fixed-lag": ("One-word fixed-lag beam", 1),
    "sentence-beam": ("Sentence beam", None),
}


@dataclass(frozen=True)
class BeamHypothesis:
    words: Tuple[str, ...]
    word_scores: Tuple[float, ...]

    @property
    def score(self) -> float:
        return sum(self.word_scores)


def _load_wordfreq_counts(limit: int) -> Dict[str, float]:
    if limit <= 0:
        raise DecoderError("lexicon size must be positive")

    counts: Dict[str, float] = {}
    for word in iter_wordlist("en", wordlist="best"):
        if (
            word.isascii()
            and word.isalpha()
            and word == word.lower()
            and (len(word) > 1 or word in {"a", "i"})
        ):
            frequency = word_frequency(word, "en", wordlist="best")
            if frequency > 0.0:
                counts[word] = frequency
                if len(counts) == limit:
                    break
    if len(counts) != limit:
        raise DecoderError(
            "wordfreq provided %d usable English words; expected %d"
            % (len(counts), limit)
        )
    return counts


def _load_bigram_counts(path: Path) -> Dict[Tuple[str, str], float]:
    counts: Dict[Tuple[str, str], float] = {}
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) != 3:
                raise DecoderError("invalid bigram row at %s:%d" % (path, line_number))
            counts[(parts[0], parts[1])] = float(parts[2])
    return counts


@lru_cache(maxsize=None)
def _decoder_components(
    lexicon_size: int,
) -> Tuple[LetterRanges, Lexicon, ContextModel]:
    ranges = LetterRanges.alphabet_quarters()
    lexicon = Lexicon(_load_wordfreq_counts(lexicon_size), ranges)
    context = ContextModel(lexicon, _load_bigram_counts(_DATA / "bigrams.tsv"))
    return ranges, lexicon, context


def build_demo_decoder(lexicon_size: int = DEFAULT_LEXICON_SIZE) -> LinguisticDecoder:
    ranges, lexicon, context = _decoder_components(lexicon_size)
    return LinguisticDecoder(
        ranges,
        lexicon,
        context,
        language_weight=DEFAULT_LANGUAGE_WEIGHT,
    )


class PipelineSimulator:
    """UI-facing state around the decoder's explicit interaction events."""

    pages = (
        ("A-F", "G-L"),
        ("M-R", "S-Z"),
    )
    frequencies = (15, 20)

    def __init__(self, decoder: Optional[LinguisticDecoder] = None) -> None:
        self.decoder = decoder or build_demo_decoder()
        self.page_index = 0
        self.history: List[Dict[str, Any]] = []
        self.engine = "bigram"
        self.decode_mode = "greedy"
        self.context_prefix = ""
        self._scorers: Dict[str, Any] = {}
        self._beams: List[BeamHypothesis] = []
        self.last_event = "Ready for a simulated selection."

    def select_range(self, label: str, confidence: float) -> None:
        offered = self.pages[self.page_index]
        if label not in offered:
            raise DecoderError("range %r is not displayed on the current page" % label)
        confidence = float(confidence)
        if confidence < 0.01 or confidence > 0.99:
            raise DecoderError("simulated confidence must be between 0.01 and 0.99")

        other = offered[1] if label == offered[0] else offered[0]
        probabilities = {label: confidence, other: 1.0 - confidence}
        page = "page-%d" % (self.page_index + 1)
        self.decoder.observe(probabilities, page=page)
        self.history.append(
            {
                "position": len(self.decoder.observations),
                "page": self.page_index,
                "selected": label,
                "probabilities": probabilities,
            }
        )
        self.last_event = "Added position %d with %d%% evidence for %s." % (
            len(self.decoder.observations),
            round(confidence * 100),
            label,
        )

    def next_page(self) -> None:
        self.page_index = (self.page_index + 1) % len(self.pages)
        self.last_event = "Showing range page %d." % (self.page_index + 1)

    @property
    def tentative_words(self) -> Tuple[str, ...]:
        return self._beams[0].words if self._beams else ()

    @property
    def active_context(self) -> List[str]:
        return list(self.decoder.confirmed_words) + list(self.tentative_words)


    def _bigram_context(self, words: Sequence[str]) -> List[str]:
        prefix = _CONTEXT_WORD.findall(self.context_prefix.lower())
        return prefix + list(words)

    def _causal_context(self, words: Sequence[str]) -> List[str]:
        prefix = [self.context_prefix] if self.context_prefix else []
        return prefix + list(words)
    def _candidate_increments(
        self,
        context: Sequence[str],
        candidates: Sequence[Any],
    ) -> Dict[str, float]:
        if self.engine == "bigram":
            bigram_context = self._bigram_context(context)
            return {
                candidate.word: (
                    self.decoder.eeg_weight * candidate.eeg_log_likelihood
                    + self.decoder.language_weight
                    * self.decoder.context_model.log_probability(
                        self.decoder.lexicon.word_to_id[candidate.word],
                        bigram_context,
                    )
                )
                for candidate in candidates
            }

        language_scores = self._scorers[self.engine].score(
            self._causal_context(context),
            [candidate.word for candidate in candidates],
        )
        return {
            candidate.word: (
                self.decoder.eeg_weight * candidate.eeg_log_likelihood
                + CAUSAL_LANGUAGE_WEIGHT * language_scores[candidate.word]
            )
            for candidate in candidates
        }

    def _advance_beam(self, candidates: Sequence[Any]) -> str:
        if not candidates:
            raise DecoderError("no word matches the current range evidence")

        bases = self._beams or [BeamHypothesis((), ())]
        expanded: List[BeamHypothesis] = []
        for hypothesis in bases:
            context = list(self.decoder.confirmed_words) + list(hypothesis.words)
            increments = self._candidate_increments(context, candidates)
            expanded.extend(
                BeamHypothesis(
                    hypothesis.words + (candidate.word,),
                    hypothesis.word_scores + (increments[candidate.word],),
                )
                for candidate in candidates
            )
        self._beams = sorted(
            expanded,
            key=lambda hypothesis: (-hypothesis.score, hypothesis.words),
        )[:BEAM_WIDTH]

        lag = DECODE_MODES[self.decode_mode][1]
        if lag is not None and len(self._beams[0].words) > lag:
            committed = self._beams[0].words[0]
            matching = [
                hypothesis
                for hypothesis in self._beams
                if hypothesis.words[0] == committed
            ]
            self.decoder.confirmed_words.append(committed)
            self._beams = [
                BeamHypothesis(
                    hypothesis.words[1:],
                    hypothesis.word_scores[1:],
                )
                for hypothesis in matching
            ]

        self.decoder.observations.clear()
        self.history.clear()
        return self.tentative_words[-1]

    def accept(self, word: str) -> None:
        if self.decode_mode == "greedy":
            accepted = self.decoder.accept(word, allow_completion=True)
        else:
            normalized = word.strip().lower()
            result = self.decoder.candidates(
                limit=CAUSAL_SHORTLIST_SIZE,
                context=self.active_context,
            )
            candidate = next(
                (item for item in result.candidates if item.word == normalized),
                None,
            )
            if candidate is None:
                raise DecoderError(
                    "%r is not compatible with the current range evidence" % word
                )
            accepted = self._advance_beam([candidate])
        self.history.clear()
        self.last_event = "Accepted suggestion %r." % accepted

    def boundary(self) -> None:
        if self.decode_mode == "greedy":
            candidates, _, _ = self._candidate_state(completed_only=True, limit=1)
            if not candidates:
                raise DecoderError(
                    "no complete dictionary word matches the current observations"
                )
            accepted = self.decoder.accept(
                candidates[0]["word"],
                allow_completion=False,
            )
        else:
            result = self.decoder.candidates(
                limit=CAUSAL_SHORTLIST_SIZE,
                completed_only=True,
                context=self.active_context,
            )
            accepted = self._advance_beam(result.candidates)
        self.history.clear()
        self.last_event = "Decoded %r at the word boundary." % accepted

    def finish_sentence(self) -> None:
        if not self._beams:
            raise DecoderError("there are no tentative words to finish")
        words = list(self._beams[0].words)
        self.decoder.confirmed_words.extend(words)
        self._beams.clear()
        self.decoder.observations.clear()
        self.history.clear()
        self.last_event = "Confirmed %d tentative word%s." % (
            len(words),
            "" if len(words) == 1 else "s",
        )

    def backspace(self) -> None:
        if self.decoder.observations:
            self.decoder.observations.pop()
            if self.history:
                self.history.pop()
            self.last_event = "Removed the latest range observation."
        elif self._beams:
            shortened = [
                BeamHypothesis(hypothesis.words[:-1], hypothesis.word_scores[:-1])
                for hypothesis in self._beams
            ]
            best_by_words: Dict[Tuple[str, ...], BeamHypothesis] = {}
            for hypothesis in shortened:
                previous = best_by_words.get(hypothesis.words)
                if previous is None or hypothesis.score > previous.score:
                    best_by_words[hypothesis.words] = hypothesis
            self._beams = sorted(
                (
                    hypothesis
                    for hypothesis in best_by_words.values()
                    if hypothesis.words
                ),
                key=lambda hypothesis: (-hypothesis.score, hypothesis.words),
            )[:BEAM_WIDTH]
            self.last_event = "Removed the latest tentative word."
        else:
            removed = self.decoder.undo_last_word()
            self.last_event = "Removed confirmed word %r." % removed

    def clear_word(self) -> None:
        self.decoder.reject_current_word()
        self.history.clear()
        self.last_event = "Cleared the current word."

    def reset(self) -> None:
        self.decoder.observations.clear()
        self.decoder.confirmed_words.clear()
        self.history.clear()
        self._beams.clear()
        self.page_index = 0
        self.last_event = "Reset the simulated session."

    def set_context_prefix(self, text: str) -> None:
        normalized = " ".join(text.strip().split())
        if len(normalized) > MAX_CONTEXT_LENGTH:
            raise DecoderError(
                "conversation context must be at most %d characters"
                % MAX_CONTEXT_LENGTH
            )
        if self._beams:
            self.finish_sentence()
        self.context_prefix = normalized
        self.last_event = (
            "Conversation context applied."
            if normalized
            else "Conversation context cleared."
        )

    def set_decode_mode(self, mode: str) -> None:
        if mode not in DECODE_MODES:
            raise DecoderError("unknown decode mode %r" % mode)
        if self._beams:
            self.finish_sentence()
        self.decode_mode = mode
        self.last_event = "Decode mode set to %s." % self.decode_mode_label

    @property
    def decode_mode_label(self) -> str:
        return DECODE_MODES[self.decode_mode][0]

    def set_engine(self, engine: str) -> None:
        if engine != "bigram" and engine not in ENGINE_MODELS:
            raise DecoderError("unknown completion engine %r" % engine)
        if engine in ENGINE_MODELS and engine not in self._scorers:
            try:
                from .causal_scorer import CausalCandidateScorer

                self._scorers[engine] = CausalCandidateScorer(ENGINE_MODELS[engine][1])
            except ImportError as exc:
                raise DecoderError(
                    "causal engines require Python 3.10 and the dependencies in "
                    "linguistic_model/experiments/requirements.txt"
                ) from exc
            except Exception as exc:
                raise DecoderError(
                    "could not load %s: %s" % (ENGINE_MODELS[engine][0], exc)
                ) from exc
        if self._beams:
            self.finish_sentence()
        self.engine = engine
        self.last_event = "Completion engine set to %s." % self.engine_label

    @property
    def engine_label(self) -> str:
        return "Bigram" if self.engine == "bigram" else ENGINE_MODELS[self.engine][0]

    def _candidate_state(
        self,
        completed_only: bool = False,
        limit: int = 8,
    ) -> Tuple[List[Dict[str, Any]], int, Dict[str, float]]:
        source_limit = limit if self.engine == "bigram" else CAUSAL_SHORTLIST_SIZE
        result = self.decoder.candidates(
            limit=source_limit,
            completed_only=completed_only,
            context=self._bigram_context(self.active_context),
        )
        if self.engine == "bigram":
            candidates = [
                {
                    "word": candidate.word,
                    "probability": candidate.probability,
                    "complete": candidate.complete,
                    "eeg_log_likelihood": candidate.eeg_log_likelihood,
                    "language_log_probability": candidate.language_log_probability,
                }
                for candidate in result.candidates
            ]
            ambiguity = {
                "top_probability": result.ambiguity.top_probability,
                "top_two_margin": result.ambiguity.top_two_margin,
                "normalized_entropy": result.ambiguity.normalized_entropy,
            }
            return candidates[:limit], result.total_candidate_count, ambiguity

        scorer = self._scorers[self.engine]
        words = [candidate.word for candidate in result.candidates]
        language_scores = scorer.score(
            self._causal_context(self.active_context),
            words,
        )
        scored = [
            (
                candidate,
                self.decoder.eeg_weight * candidate.eeg_log_likelihood
                + CAUSAL_LANGUAGE_WEIGHT * language_scores[candidate.word],
                language_scores[candidate.word],
            )
            for candidate in result.candidates
        ]
        if not scored:
            return [], 0, {
                "top_probability": 0.0,
                "top_two_margin": 0.0,
                "normalized_entropy": 0.0,
            }

        maximum = max(item[1] for item in scored)
        weights = [math.exp(item[1] - maximum) for item in scored]
        normalizer = sum(weights)
        ranked = sorted(
            (
                {
                    "word": item[0].word,
                    "probability": weight / normalizer,
                    "complete": item[0].complete,
                    "eeg_log_likelihood": item[0].eeg_log_likelihood,
                    "language_log_probability": item[2],
                }
                for item, weight in zip(scored, weights)
            ),
            key=lambda candidate: (-candidate["probability"], candidate["word"]),
        )
        probabilities = [candidate["probability"] for candidate in ranked]
        entropy = -sum(value * math.log(value) for value in probabilities if value > 0.0)
        normalized_entropy = entropy / math.log(len(ranked)) if len(ranked) > 1 else 0.0
        runner_up = probabilities[1] if len(probabilities) > 1 else 0.0
        ambiguity = {
            "top_probability": probabilities[0],
            "top_two_margin": probabilities[0] - runner_up,
            "normalized_entropy": normalized_entropy,
        }
        return ranked[:limit], result.total_candidate_count, ambiguity

    def dispatch(self, action: str, payload: Mapping[str, Any]) -> None:
        if action == "select":
            self.select_range(str(payload.get("range", "")), float(payload.get("confidence", 0.8)))
        elif action == "page":
            self.next_page()
        elif action == "accept":
            self.accept(str(payload.get("word", "")))
        elif action == "space":
            self.boundary()
        elif action == "backspace":
            self.backspace()
        elif action == "clear":
            self.clear_word()
        elif action == "reset":
            self.reset()
        elif action == "engine":
            self.set_engine(str(payload.get("engine", "")))
        elif action == "decode":
            self.set_decode_mode(str(payload.get("mode", "")))
        elif action == "finish":
            self.finish_sentence()
        elif action == "context":
            self.set_context_prefix(str(payload.get("context", "")))
        else:
            raise DecoderError("unknown simulator action %r" % action)

    def state(self) -> Dict[str, Any]:
        candidates, candidate_count, ambiguity = self._candidate_state()
        sentence = " ".join(self.active_context)
        current_ranges = [item["selected"] for item in self.history]
        offered = self.pages[self.page_index]
        return {
            "sentence": sentence,
            "confirmed_words": list(self.decoder.confirmed_words),
            "tentative_words": list(self.tentative_words),
            "context_prefix": self.context_prefix,
            "current_ranges": current_ranges,
            "observation_count": len(self.decoder.observations),
            "page": self.page_index,
            "page_number": self.page_index + 1,
            "page_count": len(self.pages),
            "targets": [
                {
                    "range": label,
                    "letters": list(self.decoder.ranges.letters_for(label)),
                    "frequency": frequency,
                }
                for label, frequency in zip(offered, self.frequencies)
            ],
            "history": list(self.history),
            "candidates": candidates,
            "candidate_count": candidate_count,
            "ambiguity": ambiguity,
            "engine": self.engine,
            "engine_label": self.engine_label,
            "engines": [
                {"id": "bigram", "label": "Bigram"},
                *[
                    {"id": engine, "label": values[0]}
                    for engine, values in ENGINE_MODELS.items()
                ],
            ],
            "decode_mode": self.decode_mode,
            "decode_mode_label": self.decode_mode_label,
            "decode_modes": [
                {"id": mode, "label": values[0]}
                for mode, values in DECODE_MODES.items()
            ],
            "beam_count": len(self._beams),
            "can_finish": bool(self._beams),
            "last_event": self.last_event,
        }
