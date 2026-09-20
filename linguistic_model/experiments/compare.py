"""Summarize persisted contextual model benchmark results."""

import json
from pathlib import Path
from typing import Dict, List


RESULTS = Path(__file__).with_name("results")


def load_results() -> List[Dict[str, object]]:
    return [json.loads(path.read_text(encoding="utf-8")) for path in sorted(RESULTS.glob("*.json"))]


def main() -> None:
    results = load_results()
    if not results:
        raise SystemExit("no benchmark results found")

    print("model                              top1  top5    MRR  median-ms  load-s")
    print("-" * 76)
    for result in results:
        causal = result["causal"]
        print(
            "%-34s %4d  %4d  %5.3f  %9.1f  %6.2f"
            % (
                result["model"], causal["top1"], causal["top5"],
                causal["mean_reciprocal_rank"], causal["median_latency_ms"],
                result["load_seconds"],
            )
        )

    baseline = results[0]
    print("\nbaselines                           top1  top5    MRR")
    print("-" * 60)
    for name in ("unigram", "bigram"):
        values = baseline[name]
        print(
            "%-34s %4d  %4d  %5.3f"
            % (name, values["top1"], values["top5"], values["mean_reciprocal_rank"])
        )


if __name__ == "__main__":
    main()
