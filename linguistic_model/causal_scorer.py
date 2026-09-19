"""Batched conditional word scoring with pretrained causal language models."""

from dataclasses import dataclass
from time import perf_counter
from typing import Dict, List, Optional, Sequence


@dataclass(frozen=True)
class ScorerStats:
    model_name: str
    device: str
    load_seconds: float


class CausalCandidateScorer:
    """Scores complete candidate words as continuations of confirmed text.

    The candidate includes a leading and trailing space. The trailing space is
    intentional: it scores the word boundary instead of treating a shorter word
    as an unfinished prefix of a longer token sequence.
    """

    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
        batch_size: int = 64,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        started = perf_counter()
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if device is None:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        if device not in {"cpu", "mps", "cuda"}:
            raise ValueError("device must be cpu, mps, or cuda")

        self.torch = torch
        self.device = device
        self.batch_size = batch_size
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token_id is None:
                raise ValueError("tokenizer must define either a pad or EOS token")
            self.tokenizer.pad_token = self.tokenizer.eos_token

        dtype = torch.float16 if device in {"mps", "cuda"} else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
        self.model.to(device)
        self.model.eval()
        self.stats = ScorerStats(model_name, device, perf_counter() - started)

    def _context_tokens(self, context: Sequence[str]) -> List[int]:
        text = " ".join(context).strip()
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        if tokens:
            return tokens
        fallback = self.tokenizer.bos_token_id
        if fallback is None:
            fallback = self.tokenizer.eos_token_id
        if fallback is None:
            raise ValueError("tokenizer has no token available for empty context")
        return [fallback]

    def score(self, context: Sequence[str], candidates: Sequence[str]) -> Dict[str, float]:
        """Return log P(" candidate " | context) for each unique candidate."""
        unique = list(dict.fromkeys(candidate.strip().lower() for candidate in candidates))
        if not unique:
            return {}
        if any(not candidate for candidate in unique):
            raise ValueError("candidates cannot be empty")

        torch = self.torch
        context_ids = self._context_tokens(context)
        encoded = []
        for word in unique:
            continuation = self.tokenizer.encode(" " + word + " ", add_special_tokens=False)
            if not continuation:
                raise ValueError("candidate %r produced no tokens" % word)
            encoded.append((word, context_ids + continuation, len(context_ids)))

        scores: Dict[str, float] = {}
        for offset in range(0, len(encoded), self.batch_size):
            batch = encoded[offset:offset + self.batch_size]
            max_length = max(len(item[1]) for item in batch)
            input_ids = []
            attention_masks = []
            for _, sequence, _ in batch:
                padding = max_length - len(sequence)
                input_ids.append(sequence + [self.tokenizer.pad_token_id] * padding)
                attention_masks.append([1] * len(sequence) + [0] * padding)

            ids = torch.tensor(input_ids, dtype=torch.long, device=self.device)
            mask = torch.tensor(attention_masks, dtype=torch.long, device=self.device)
            with torch.inference_mode():
                logits = self.model(input_ids=ids, attention_mask=mask).logits
                log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)

            labels = ids[:, 1:]
            selected = torch.gather(log_probs, 2, labels.unsqueeze(-1)).squeeze(-1)
            for row, (word, sequence, context_length) in enumerate(batch):
                # selected[k] scores sequence token k+1. The first continuation
                # token is at sequence index context_length.
                start = context_length - 1
                stop = len(sequence) - 1
                scores[word] = float(selected[row, start:stop].sum().item())

        return scores
