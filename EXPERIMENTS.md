# Completion Engine Experiments

Browser-driven walkthroughs used **Greedy** decoding, no conversation-context prompt, and the simulator's default 82% confidence. For each word, the intended candidate was accepted as soon as it appeared in the eight visible candidates; otherwise, its letter ranges were entered and **Word boundary** was used after the final letter. A phrase failed when the boundary selected a different range-compatible word.

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

## Conversation-context paired test

GPT-2 was rerun in Greedy mode on the same six phrases with semantic question
prompts that avoided the target wording (for example, `Check the clock.` for
`what time is it`). The model was warm on Apple MPS; applying the prompt and
resetting the session were excluded from timing.

| Phrase | Actions without context | Actions with context |
|---|---:|---:|
| Hello nice to meet you what is your name | 26 | 23 |
| my name is mike | 12 | 10 |
| where are you going | 8 | 4 |
| i am craving tacos | 25 | 19 |
| what time is it | 14 | 11 |
| i want to save the world by solving climate change | 32 | 27 |

| Metric | No context | Question context | Change |
|---|---:|---:|---:|
| Target-aware phrase completion | 6/6 | 6/6 | No change |
| User actions | 117 | 94 | **-19.7%** |
| Range selections | 51 | 39 | **-23.5%** |
| Page cycles | 31 | 20 | **-35.5%** |
| Mean accepted-candidate rank | 3.30 | 2.41 | **-27.0%** |
| Words accepted without a range | 13 | 19 | **+46.2%** |
| Warm decoding time | 17.62 s | 17.33 s | -1.7% |
| Mean latency per action | 151 ms | 184 ms | +21.9% |

An automatic boundary-only accuracy probe showed no improvement: both
conditions completed 2/6 phrases and decoded 20/24 attempted words correctly.
Context reduced the number of interactions, but the longer transformer prompt
made each action slower and did not change Top-1 boundary accuracy in this
small sample.

## Decode-mode paired test

GPT-2 was also compared across all three decode modes with no conversation
prompt, 82% confidence, and a warm model on Apple MPS.

When the user selected the intended candidate as soon as it appeared, all modes
completed all six phrases:

| Mode | Phrases | Correct word positions | Actions | Warm decoding time |
|---|---:|---:|---:|---:|
| Greedy | 6/6 | 35/35 | 117 | 17.61 s |
| One-word fixed lag | 6/6 | 35/35 | 123 | 19.03 s |
| Sentence beam | 6/6 | 35/35 | 123 | 18.80 s |

The six additional beam actions are **Finish sentence**. With explicit
candidate selection, alternatives collapse to the selected word, so beam search
provided no accuracy gain.

An autonomous probe entered every complete range sequence and always used Word
boundary, without candidate corrections:

| Mode | Exact phrases | Correct word positions | Actions | Warm decoding time |
|---|---:|---:|---:|---:|
| Greedy | 2/6 | 28/35 (80.0%) | 235 | 31.76 s |
| One-word fixed lag | 2/6 | 25/35 (71.4%) | 241 | 40.33 s |
| Sentence beam | **5/6** | **32/35 (91.4%)** | 241 | 44.92 s |

| Phrase | Greedy | Fixed lag | Sentence beam |
|---|---:|---:|---:|
| Hello nice to meet you what is your name | Fail | Fail | Pass |
| my name is mike | Pass | Pass | Pass |
| where are you going | Pass | Fail | Pass |
| i am craving tacos | Fail | Fail | Fail |
| what time is it | Fail | Fail | Pass |
| i want to save the world by solving climate change | Fail | Pass | Pass |

Sentence beam materially improved autonomous accuracy, at 41% higher wall-clock
decoding time than Greedy. The current one-word lag was slower than Greedy and
less accurate on this sample because it revised words with too little future
context and then committed those revisions.
