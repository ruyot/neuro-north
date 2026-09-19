"""Conversational cases for comparing context models under range ambiguity."""

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class ContextCase:
    context: Tuple[str, ...]
    target: str

    @property
    def text(self) -> str:
        return " ".join(self.context + (self.target,))


CASES = (
    ContextCase(("hello", "my"), "name"),
    ContextCase(("please", "tell", "me", "your"), "name"),
    ContextCase(("i", "am", "very"), "tired"),
    ContextCase(("please", "take", "my"), "medication"),
    ContextCase(("i", "am", "in"), "pain"),
    ContextCase(("please", "close", "the"), "door"),
    ContextCase(("it", "is", "time", "to"), "sleep"),
    ContextCase(("please", "call", "my"), "mom"),
    ContextCase(("i", "am", "feeling"), "hot"),
    ContextCase(("i", "would", "like"), "food"),
    ContextCase(("the", "answer", "is"), "yes"),
    ContextCase(("we", "need", "to", "go"), "home"),
)
