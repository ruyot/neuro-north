"""Probabilistic range-to-word decoding for the Neuro North interface.

The decoder deliberately knows nothing about EEG acquisition. Its input is a
committed selection event containing calibrated posterior probabilities for the
letter ranges that were visible for that event.
"""

from dataclasses import dataclass, field
import math
import re
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


_WORD = re.compile(r"^[a-z]+$")


class DecoderError(ValueError):
    """Raised when an observation or decoder transition is invalid."""


class LetterRanges:
    """A disjoint mapping from range labels to lowercase ASCII letters."""

    def __init__(self, ranges: Mapping[str, Iterable[str]]) -> None:
        if not ranges:
            raise DecoderError("at least one letter range is required")

        normalized: Dict[str, frozenset] = {}
        by_letter: Dict[str, str] = {}
        for label, letters in ranges.items():
            if not label:
                raise DecoderError("range labels cannot be empty")
            chars = frozenset(str(letter).lower() for letter in letters)
            if not chars or any(len(char) != 1 or not char.isascii() or not char.isalpha() for char in chars):
                raise DecoderError("range %r must contain ASCII letters" % label)
            for char in chars:
                if char in by_letter:
                    raise DecoderError(
                        "letter %r appears in both %r and %r" % (char, by_letter[char], label)
                    )
                by_letter[char] = label
            normalized[label] = chars

        self._ranges = normalized
        self._by_letter = by_letter

    @classmethod
    def alphabet_quarters(cls) -> "LetterRanges":
        return cls(
            {
                "A-F": "abcdef",
                "G-L": "ghijkl",
                "M-R": "mnopqr",
                "S-Z": "stuvwxyz",
            }
        )

    @property
    def labels(self) -> Tuple[str, ...]:
        return tuple(self._ranges)

    def letters_for(self, label: str) -> Tuple[str, ...]:
        try:
            return tuple(sorted(self._ranges[label]))
        except KeyError as exc:
            raise DecoderError("unknown range label %r" % label) from exc

    def label_for(self, letter: str) -> str:
        try:
            return self._by_letter[letter.lower()]
        except KeyError as exc:
            raise DecoderError("letter %r is not covered by a configured range" % letter) from exc

    def supports(self, word: str) -> bool:
        return bool(_WORD.fullmatch(word)) and all(char in self._by_letter for char in word)


@dataclass(frozen=True)
class RangeObservation:
    """One selection event over the ranges offered on a particular UI page.

    ``probabilities`` are P(range | EEG observation). ``selection_priors`` are
    the priors under which that classifier was calibrated. Dividing the two
    yields a quantity proportional to P(observation | range), avoiding a second
    application of the classifier prior during language-model fusion.
    """

    probabilities: Mapping[str, float]
    selection_priors: Mapping[str, float]
    page: Optional[str] = None

    @classmethod
    def create(
        cls,
        probabilities: Mapping[str, float],
        selection_priors: Optional[Mapping[str, float]] = None,
        page: Optional[str] = None,
    ) -> "RangeObservation":
        if not probabilities:
            raise DecoderError("an observation must offer at least one range")
        if any(not math.isfinite(value) or value < 0.0 for value in probabilities.values()):
            raise DecoderError("range probabilities must be finite and non-negative")
        total = sum(probabilities.values())
        if total <= 0.0:
            raise DecoderError("range probabilities must have positive mass")
        posterior = {label: value / total for label, value in probabilities.items()}

        if selection_priors is None:
            prior = {label: 1.0 / len(posterior) for label in posterior}
        else:
            if set(selection_priors) != set(posterior):
                raise DecoderError("selection priors must cover exactly the offered ranges")
            if any(not math.isfinite(value) or value <= 0.0 for value in selection_priors.values()):
                raise DecoderError("selection priors must be finite and positive")
            prior_total = sum(selection_priors.values())
            prior = {label: value / prior_total for label, value in selection_priors.items()}

        return cls(posterior, prior, page)

    def log_likelihood(self, label: str) -> Optional[float]:
        posterior = self.probabilities.get(label, 0.0)
        if posterior <= 0.0:
            return None
        return math.log(posterior) - math.log(self.selection_priors[label])


class _TrieNode:
    __slots__ = ("children", "word_ids")

    def __init__(self) -> None:
        self.children: Dict[str, "_TrieNode"] = {}
        self.word_ids: List[int] = []


class Lexicon:
    """Normalized vocabulary, unigram counts, and a prefix trie."""

    def __init__(self, word_counts: Mapping[str, float], ranges: LetterRanges) -> None:
        merged: Dict[str, float] = {}
        for raw_word, raw_count in word_counts.items():
            word = raw_word.strip().lower()
            count = float(raw_count)
            if not math.isfinite(count) or count <= 0.0:
                raise DecoderError("word counts must be finite and positive")
            if not ranges.supports(word):
                raise DecoderError("word %r contains unsupported characters" % raw_word)
            merged[word] = merged.get(word, 0.0) + count
        if not merged:
            raise DecoderError("the lexicon cannot be empty")

        self.words = tuple(sorted(merged))
        self.counts = tuple(merged[word] for word in self.words)
        self.total_count = sum(self.counts)
        self.word_to_id = {word: index for index, word in enumerate(self.words)}
        self.root = _TrieNode()

        for word_id, word in enumerate(self.words):
            node = self.root
            node.word_ids.append(word_id)
            for char in word:
                node = node.children.setdefault(char, _TrieNode())
                node.word_ids.append(word_id)

    @classmethod
    def from_words(cls, words: Iterable[str], ranges: LetterRanges) -> "Lexicon":
        counts: Dict[str, float] = {}
        for word in words:
            normalized = word.strip().lower()
            counts[normalized] = counts.get(normalized, 0.0) + 1.0
        return cls(counts, ranges)

    def unigram_probability(self, word_id: int) -> float:
        return self.counts[word_id] / self.total_count


class ContextModel:
    """Interpolated unigram/bigram prior over a fixed lexicon."""

    def __init__(
        self,
        lexicon: Lexicon,
        bigram_counts: Optional[Mapping[Tuple[str, str], float]] = None,
        smoothing: float = 20.0,
    ) -> None:
        if not math.isfinite(smoothing) or smoothing <= 0.0:
            raise DecoderError("smoothing must be finite and positive")
        self.lexicon = lexicon
        self.smoothing = smoothing
        self._bigrams: Dict[Tuple[str, str], float] = {}
        self._context_totals: Dict[str, float] = {}

        for (raw_previous, raw_word), raw_count in (bigram_counts or {}).items():
            previous = raw_previous.strip().lower()
            word = raw_word.strip().lower()
            count = float(raw_count)
            if word not in lexicon.word_to_id:
                raise DecoderError("bigram target %r is not in the lexicon" % raw_word)
            if not math.isfinite(count) or count <= 0.0:
                raise DecoderError("bigram counts must be finite and positive")
            self._bigrams[(previous, word)] = self._bigrams.get((previous, word), 0.0) + count
            self._context_totals[previous] = self._context_totals.get(previous, 0.0) + count

    def log_probability(self, word_id: int, context: Sequence[str]) -> float:
        unigram = self.lexicon.unigram_probability(word_id)
        if not context:
            return math.log(unigram)

        previous = context[-1].lower()
        total = self._context_totals.get(previous)
        if total is None:
            return math.log(unigram)
        word = self.lexicon.words[word_id]
        probability = (
            self._bigrams.get((previous, word), 0.0) + self.smoothing * unigram
        ) / (total + self.smoothing)
        return math.log(probability)


@dataclass(frozen=True)
class Candidate:
    word: str
    probability: float
    eeg_log_likelihood: float
    language_log_probability: float
    complete: bool


@dataclass(frozen=True)
class Ambiguity:
    top_probability: float
    top_two_margin: float
    normalized_entropy: float
    candidate_count: int


@dataclass(frozen=True)
class CandidateSet:
    candidates: Tuple[Candidate, ...]
    ambiguity: Ambiguity
    total_candidate_count: int


@dataclass
class LinguisticDecoder:
    """Stateful current-word decoder with explicit confirmation transitions."""

    ranges: LetterRanges
    lexicon: Lexicon
    context_model: ContextModel
    eeg_weight: float = 1.0
    language_weight: float = 1.0
    confirmed_words: List[str] = field(default_factory=list)
    observations: List[RangeObservation] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.lexicon is not self.context_model.lexicon:
            raise DecoderError("context model and decoder must share the same lexicon")
        if not math.isfinite(self.eeg_weight) or self.eeg_weight < 0.0:
            raise DecoderError("eeg_weight must be finite and non-negative")
        if not math.isfinite(self.language_weight) or self.language_weight < 0.0:
            raise DecoderError("language_weight must be finite and non-negative")

    def observe(
        self,
        probabilities: Mapping[str, float],
        selection_priors: Optional[Mapping[str, float]] = None,
        page: Optional[str] = None,
    ) -> RangeObservation:
        unknown = set(probabilities) - set(self.ranges.labels)
        if unknown:
            raise DecoderError("unknown range labels: %s" % ", ".join(sorted(unknown)))
        observation = RangeObservation.create(probabilities, selection_priors, page)
        self.observations.append(observation)
        return observation

    def candidates(self, limit: int = 5, completed_only: bool = False) -> CandidateSet:
        if limit <= 0:
            raise DecoderError("candidate limit must be positive")

        states: Dict[_TrieNode, float] = {self.lexicon.root: 0.0}
        for observation in self.observations:
            next_states: Dict[_TrieNode, float] = {}
            for node, evidence in states.items():
                for char, child in node.children.items():
                    increment = observation.log_likelihood(self.ranges.label_for(char))
                    if increment is not None:
                        next_states[child] = evidence + increment
            states = next_states
            if not states:
                break

        scored: List[Tuple[int, float, float, float]] = []
        observed_length = len(self.observations)
        for node, evidence in states.items():
            for word_id in node.word_ids:
                word = self.lexicon.words[word_id]
                if completed_only and len(word) != observed_length:
                    continue
                language = self.context_model.log_probability(word_id, self.confirmed_words)
                score = self.eeg_weight * evidence + self.language_weight * language
                scored.append((word_id, score, evidence, language))

        if not scored:
            return CandidateSet((), Ambiguity(0.0, 0.0, 0.0, 0), 0)

        maximum = max(item[1] for item in scored)
        weights = [math.exp(item[1] - maximum) for item in scored]
        normalizer = sum(weights)
        ranked = sorted(
            (
                Candidate(
                    word=self.lexicon.words[item[0]],
                    probability=weight / normalizer,
                    eeg_log_likelihood=item[2],
                    language_log_probability=item[3],
                    complete=len(self.lexicon.words[item[0]]) == observed_length,
                )
                for item, weight in zip(scored, weights)
            ),
            key=lambda candidate: (-candidate.probability, candidate.word),
        )

        probabilities = [candidate.probability for candidate in ranked]
        entropy = -sum(value * math.log(value) for value in probabilities if value > 0.0)
        normalized_entropy = entropy / math.log(len(ranked)) if len(ranked) > 1 else 0.0
        runner_up = probabilities[1] if len(probabilities) > 1 else 0.0
        ambiguity = Ambiguity(
            top_probability=probabilities[0],
            top_two_margin=probabilities[0] - runner_up,
            normalized_entropy=normalized_entropy,
            candidate_count=len(ranked),
        )
        return CandidateSet(tuple(ranked[:limit]), ambiguity, len(ranked))

    def accept(self, word: str, allow_completion: bool = True) -> str:
        normalized = word.strip().lower()
        candidates = self.candidates(limit=len(self.lexicon.words), completed_only=not allow_completion)
        if normalized not in {candidate.word for candidate in candidates.candidates}:
            qualifier = "current range evidence" if allow_completion else "current completed word"
            raise DecoderError("%r is not compatible with the %s" % (word, qualifier))
        self.confirmed_words.append(normalized)
        self.observations.clear()
        return normalized

    def confirm_boundary(self) -> str:
        candidates = self.candidates(limit=1, completed_only=True)
        if not candidates.candidates:
            raise DecoderError("no complete dictionary word matches the current observations")
        return self.accept(candidates.candidates[0].word, allow_completion=False)

    def reject_current_word(self) -> None:
        self.observations.clear()

    def undo_last_word(self) -> str:
        if not self.confirmed_words:
            raise DecoderError("there is no confirmed word to undo")
        return self.confirmed_words.pop()
