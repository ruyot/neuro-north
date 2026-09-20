# Neuro North speller

Run with the Knight EEG headset:

```sh
./run_speller.sh
```

Run without a headset, using the keyboard and mouse:

```sh
./run_speller.sh --simulate
```

`run_speller.sh` is a shell script, so execute it directly; do not pass it to
`python`. To bypass the shell launcher with `uv`, run the Python module:

```sh
uv run python -m ssvep_training.speller_ui --keys --engine bigram --windowed
```

Simulation defaults to the lightweight bigram engine, which does not require
PyTorch. To use GPT-2, install `linguistic_model/experiments/requirements.txt`,
prepare the local weights, and pass `--engine gpt2`.

Add `--windowed` for a development window. In simulation mode, press `1` / `2`
or click a box to select a range. Use the left arrow or panel to change the
wheel. Use the right arrow or panel to end a word; use it again with no pending
ranges to commit the sentence. Up/down or the top/bottom panels accept
suggestions. Tentative sentence-beam words appear blue and may be revised as
later words add context. This exercises the same speller and language-model
path, but does not start BrainFlow, flash SSVEP stimuli, or create EEG
recordings.

The launcher uses `.venv/bin/python` by default. Set `PYTHON` to use another
environment, for example `PYTHON=python3 ./run_speller.sh --simulate`.

