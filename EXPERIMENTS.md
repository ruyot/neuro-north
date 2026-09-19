# Completion Engine Experiments

Browser-driven walkthroughs used the simulator's default 82% confidence. For each word, the intended candidate was accepted as soon as it appeared in the eight visible candidates; otherwise, its letter ranges were entered and **Word boundary** was used after the final letter. A phrase failed when the boundary selected a different range-compatible word.

Actions count range selections, page cycles, and candidate/boundary acceptance. Session resets and engine selection are excluded.

## Phrase results

| Phrase | Bigram | GPT-2 | SmolLM2 | Pythia |
|---|---:|---:|---:|---:|
| Hello nice to meet you what is your name | Pass · 37 actions | Pass · 26 actions | Fail at `nice` → `ride` · 10 actions | Pass · 24 actions |
| my name is mike | Pass · 11 actions | Pass · 12 actions | Pass · 15 actions | Fail at `my` → `ot` · 4 actions |
| where are you going | Pass · 16 actions | Pass · 8 actions | Pass · 17 actions | Pass · 17 actions |
| i am craving tacos | Fail at `tacos` → `years` · 22 actions | Pass · 25 actions | Fail at `am` → `en` · 6 actions | Fail at `tacos` → `years` · 25 actions |
| what time is it | Pass · 16 actions | Pass · 14 actions | Pass · 15 actions | Fail at `what` → `that` · 8 actions |
| i want to save the world by solving climate change | Pass · 42 actions | Pass · 32 actions | Fail at `want` → `sent` · 10 actions | Pass · 26 actions |

## Aggregate performance

| Engine | Phrases completed | Words completed / attempted | Mean actions per completed phrase | Warm UI latency per action | Exact-range benchmark Top-1 |
|---|---:|---:|---:|---:|---:|
| Bigram | 5/6 | 34/35 | 24.4 | 51 ms | 3/12 |
| GPT-2 | **6/6** | **35/35** | **19.5** | 153 ms | **12/12** |
| SmolLM2 | 3/6 | 15/18 | 15.7* | 187 ms | **12/12** |
| Pythia | 3/6 | 26/29 | 22.3* | 158 ms | 9/12 |

\* Mean actions only covers completed phrases and therefore favors engines that failed before harder phrases were completed.

**Result:** GPT-2 was the only engine to complete every phrase and used about 25% fewer actions than Bigram across the five phrases both completed. Bigram remained roughly three times faster per action. SmolLM2 and Pythia were limited by incorrect boundary choices between words sharing the same range sequence.
