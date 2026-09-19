"""Compare causal base models with current contextual ranking.

Run from the repository root, for example:
    python -m linguistic_model.experiments.benchmark --model HuggingFaceTB/SmolLM2-135M
"""

import argparse
import json
import math
from time import perf_counter
from typing import Dict, Iterable, List, Sequence, Tuple

from linguistic_model.simulator import build_demo_decoder

from .cases import CASES, ContextCase
from linguistic_model.causal_scorer import CausalCandidateScorer


MODEL_ALIASES = {
    "gpt2": "openai-community/gpt2",
    "smollm2": "HuggingFaceTB/SmolLM2-135M",
    "pythia": "EleutherAI/pythia-160m",
}


def exact_range_candidates(target: str) -> Tuple[List[str], object]:
    decoder = build_demo_decoder()
    for char in target:
        label = decoder.ranges.label_for(char)
        pair = ("A-F", "G-L") if label in {"A-F", "G-L"} else ("M-R", "S-Z")
        decoder.observe({pair[0]: float(label == pair[0]), pair[1]: float(label == pair[1])})
    result = decoder.candidates(limit=len(decoder.lexicon.words), completed_only=True)
    return [candidate.word for candidate in result.candidates], decoder


def rank(scores: Dict[str, float], target: str) -> int:
    ordered = sorted(scores, key=lambda word: (-scores[word], word))
    try:
        return ordered.index(target) + 1
    except ValueError:
        return len(ordered) + 1


def reciprocal_rank(value: int) -> float:
    return 1.0 / value


def baseline_scores(case: ContextCase, candidates: Sequence[str], decoder: object) -> Dict[str, Dict[str, float]]:
    lexicon = decoder.lexicon
    unigram = {
        word: math.log(lexicon.unigram_probability(lexicon.word_to_id[word]))
        for word in candidates
    }
    bigram = {
        word: decoder.context_model.log_probability(lexicon.word_to_id[word], case.context)
        for word in candidates
    }
    return {"unigram": unigram, "bigram": bigram}


def summarize(rows: Sequence[Dict[str, object]], scorer_name: str) -> Dict[str, object]:
    model_ranks = [int(row[scorer_name + "_rank"]) for row in rows]
    return {
        "cases": len(rows),
        "top1": sum(value == 1 for value in model_ranks),
        "top5": sum(value <= 5 for value in model_ranks),
        "mean_reciprocal_rank": sum(reciprocal_rank(value) for value in model_ranks) / len(model_ranks),
        "median_latency_ms": sorted(float(row["latency_ms"]) for row in rows)[len(rows) // 2],
    }


def run(model_name: str, device: str, batch_size: int) -> Dict[str, object]:
    scorer = CausalCandidateScorer(model_name, device=device or None, batch_size=batch_size)
    rows = []
    for case in CASES:
        candidates, decoder = exact_range_candidates(case.target)
        if case.target not in candidates:
            raise RuntimeError("target %r is absent from its exact range candidates" % case.target)
        baselines = baseline_scores(case, candidates, decoder)
        cold_started = perf_counter()
        scorer.score(case.context, candidates)
        cold_elapsed_ms = (perf_counter() - cold_started) * 1000.0
        started = perf_counter()
        causal = scorer.score(case.context, candidates)
        elapsed_ms = (perf_counter() - started) * 1000.0
        causal_order = sorted(causal, key=lambda word: (-causal[word], word))
        row = {
            "text": case.text,
            "candidate_count": len(candidates),
            "unigram_rank": rank(baselines["unigram"], case.target),
            "bigram_rank": rank(baselines["bigram"], case.target),
            "causal_rank": rank(causal, case.target),
            "causal_top": causal_order[0],
            "cold_latency_ms": cold_elapsed_ms,
            "latency_ms": elapsed_ms,
        }
        rows.append(row)
        print(
            "%-34s candidates=%4d unigram=%3d bigram=%3d causal=%3d "
            "top=%-12s warm=%6.1fms"
            % (
                row["text"], row["candidate_count"], row["unigram_rank"],
                row["bigram_rank"], row["causal_rank"], row["causal_top"],
                row["latency_ms"],
            )
        )

    output = {
        "model": model_name,
        "device": scorer.stats.device,
        "load_seconds": scorer.stats.load_seconds,
        "unigram": summarize(rows, "unigram"),
        "bigram": summarize(rows, "bigram"),
        "causal": summarize(rows, "causal"),
        "rows": rows,
    }
    print(json.dumps(output, indent=2))
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=sorted(MODEL_ALIASES), required=True)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output")
    args = parser.parse_args()

    output = run(MODEL_ALIASES[args.model], args.device, args.batch_size)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as destination:
            json.dump(output, destination, indent=2)
            destination.write("\n")


if __name__ == "__main__":
    main()
