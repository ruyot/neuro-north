# Neuro North

## Mouse-driven pipeline simulator

Install the pinned dependency and run the local simulator with Python 3:

```bash
python3 -m pip install -r linguistic_model/requirements.txt
python3 -m linguistic_model.server
```

Open <http://127.0.0.1:8000>. Click the two simulated SSVEP targets to submit
soft range evidence, use **Cycle page** to switch between alphabet ranges, and
use **Word boundary** or a displayed prediction to confirm a word. The
**Completion engine** selector switches between the built-in bigram ranker and
the available causal models. The bigram engine only requires the base
dependency above.

The confidence slider controls how much classifier probability is assigned to
the clicked target. Values below 50% intentionally simulate a classifier error.
The server uses `wordfreq` to load 50,000 frequency-ranked, lowercase ASCII
English words. It remains independent of the EEG collection process.

Enter the other speaker's question in **Question or conversation context** and
select **Apply context**. The text is prepended to the language-model prompt but
is not added to the decoded sentence. It remains active across session resets;
clearing the field and applying it removes the context. Applying new context
commits any tentative beam first.

### Search and decode modes

The **Search / decode** selector controls when decoded words become final:

- **Greedy** commits each selected word immediately.
- **One-word fixed-lag beam** keeps eight alternatives and commits a word after
  one later word supplies additional context.
- **Sentence beam** keeps the full sentence tentative until **Finish sentence**
  is selected.

Tentative words appear in amber. **Finish sentence** commits the highest-scoring
beam; changing the engine or decode mode also commits that beam first.

## Context model experiments

The causal-model bakeoff uses a separate Python 3.10 environment because the
current PyTorch and Transformers releases no longer support Python 3.9:

```bash
uv venv --python 3.10 .venv-lm
uv pip install --python .venv-lm/bin/python \
  -r linguistic_model/experiments/requirements.txt
```

Launch the simulator from that environment to enable GPT-2, SmolLM2, and
Pythia in the completion-engine selector:

```bash
.venv-lm/bin/python -m linguistic_model.server
```

Models load on first selection and remain cached for the server process.

Run an individual base-model benchmark:

```bash
.venv-lm/bin/python -m linguistic_model.experiments.benchmark \
  --model gpt2 --device mps
```

Available aliases are `gpt2`, `smollm2`, and `pythia`. Each model ranks complete
words that exactly match the target range sequence. The checked-in results can
be summarized with:

```bash
python3 -m linguistic_model.experiments.compare
```