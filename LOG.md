# Development Log

This log records accuracy, usability, and performance issues observed while building and exercising the decoder, together with the current response. Items marked **Open** remain prototype limitations.

## Decoder accuracy and interaction

| Issue | Evidence and impact | Response | Status |
|---|---|---|---|
| Coarse ranges alias unrelated words | `how` and `int` both map to `G-L, M-R, S-Z`; `tacos` and `years` also collide. EEG evidence cannot distinguish words with the same pattern. | Added contextual completion engines, explicit candidate selection, and beam-based decode modes. | Mitigated; exact alternatives still need a boundary-selection UI. |
| Greedy decoding commits too early | GPT-2 could commit `hi int …` before seeing the context in `are you`. Once committed, discarded alternatives could not recover. | Added **One-word fixed-lag beam** and **Sentence beam** modes with eight retained hypotheses and visibly tentative words. | Addressed by selectable decode strategy. |
| One future word can be insufficient | In the boundary-only `hi how are you` walkthrough, fixed lag produced `hi int and try`. | Added Sentence beam as the higher-latency option; it recovered the suffix to `how are you`. | Open for fixed lag; this is its intended latency/accuracy trade-off. |
| Full-sentence scoring can revise a correct word incorrectly | Sentence beam recovered `how are you` but changed `hi` to `ii`; both map to `G-L, G-L`, have similar unigram frequency, and GPT-2 preferred the `ii` hypothesis. | Added tentative rendering and explicit **Finish sentence** rather than silently presenting beam output as final. | Open; exact boundary alternatives should be shown and lockable. |
| Word boundary previously forced one top candidate | If the intended word was absent from the eight visible candidates, boundary immediately accepted another compatible word, such as `ride` for `nice`. | Beam modes now retain alternatives instead of immediately committing. Candidate clicks remain hard user choices. | Mitigated; Greedy still intentionally commits immediately. |
| Missing conversational context weakened predictions | GPT-2 saw only decoded text such as `hi`, not the question that prompted the response. | Added a 500-character **Question or conversation context** field. Causal models receive the full text with punctuation; Bigram uses its final words. | Addressed. |
| Search-mode state was not visible | Delayed decoding would be confusing if tentative and confirmed words looked identical. | Tentative words are amber and underlined; **Finish sentence** commits the best beam. Switching engine or mode also finalizes the current beam. | Addressed. |

## Model quality and evaluation

| Issue | Evidence and impact | Response | Status |
|---|---|---|---|
| Small offline benchmarks overstated general performance | GPT-2 and SmolLM2 both scored 12/12 Top-1 on the exact-range benchmark, but live phrase walkthroughs were more difficult; SmolLM2 completed only 3/6 phrases in the Greedy UI evaluation. | Added browser-driven phrase walkthroughs and documented their protocol separately from the exact-range benchmark in `EXPERIMENTS.md`. | Addressed for current experiments; the dataset remains small. |
| Results were sensitive to hidden protocol choices | Selecting a visible intended candidate gives different results from always pressing Word boundary. Decode mode, confidence, and prompt context also change outcomes. | Experiment documentation now states Greedy mode, 82% confidence, no conversation prompt, and the candidate-selection policy. | Addressed. |
| Model choice had a clear quality/latency trade-off | GPT-2 completed 6/6 Greedy walkthrough phrases at about 153 ms per warm action. Bigram completed 5/6 at about 51 ms. SmolLM2 and Pythia completed 3/6. | Added a completion-engine selector and preserved Bigram as the low-latency, dependency-light option. | Addressed through user-selectable policy. |

## Performance and implementation

| Issue | Evidence and impact | Response | Status |
|---|---|---|---|
| Scoring the full 50,000-word lexicon with a causal model on every action is impractical | Candidate scoring occurs after range, page, and acceptance actions; full-vocabulary transformer evaluation would dominate interaction latency. | Use the existing decoder to retrieve a 256-word shortlist, then rerank it with the causal model. Models are loaded lazily and cached for the server process. | Intentional trade-off; shortlist pruning can remove a recoverable word. |
| Beam search multiplies model work | A width-eight beam may score candidates under up to eight different sentence histories. Sentence beam also retains hypotheses longer than fixed lag. | Beam width is bounded at eight; Greedy remains available when latency matters more than revision accuracy. | Addressed with bounded search. |
| Current Torch and Transformers releases do not support the project's Python 3.9 environment | Causal engines could not run in the base environment. | Added a separate Python 3.10 `.venv-lm` environment and pinned experiment dependencies. Bigram continues to work without those dependencies. | Addressed operationally. |
| Range labels were harder to scan than their contents | Targets displayed `A-F`, `G-L`, and similar labels, forcing users to mentally expand them. | Targets now display every contained letter while retaining stable range identifiers internally. | Addressed. |

## Remaining priorities

1. Show exact completed-word alternatives at Word boundary and let the user lock one.
2. Evaluate prompt context together with fixed-lag and sentence beam modes on a larger conversational set.
3. Measure whether wider beams or a two-word/adaptive lag recover more ambiguities without unacceptable latency.
4. Investigate shortlist recall so future context can recover words that were not in the initial 256 candidates.
5. Add hypothesis confidence/stability criteria rather than committing solely at a fixed word count.
