# Neuro North

## Mouse-driven pipeline simulator

Install the pinned dependency and run the local simulator with Python 3:

```bash
python3 -m pip install -r linguistic_model/requirements.txt
python3 -m linguistic_model.server
```

Open <http://127.0.0.1:8000>. Click the two simulated SSVEP targets to submit
soft range evidence, use **Next page** to switch between alphabet ranges, and
use **Word boundary** or a displayed prediction to confirm a word.

The confidence slider controls how much classifier probability is assigned to
the clicked target. Values below 50% intentionally simulate a classifier error.
The server uses `wordfreq` to load 50,000 frequency-ranked, lowercase ASCII
English words. It remains independent of the EEG collection process.