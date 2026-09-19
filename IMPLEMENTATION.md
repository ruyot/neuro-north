# Implementation Progress

Tracks what's built, what's next, and what we've learned. Design rationale lives in [APPROACH.md](APPROACH.md).

Status key: ✅ done · 🚧 in progress · ⬜ not started

## Phase 0 — Board connection ✅

- ✅ BrainFlow installed in `.venv` (Python 3.9), deps in `requirements.txt`
- ✅ `stream_test.py` — connects to Knight IMU, enables channels, prints EEG + accel/gyro (`--synthetic` for no hardware)
- ✅ `live_plot.py` — live pyqtgraph view: 8 EEG traces (1–40 Hz bandpass + 60 Hz notch, `F` toggles raw) + accel/gyro
- ✅ Verified on hardware: 125 Hz stream, all 8 channels enable, IMU reads gravity correctly
- ✅ Channel labels for the NeuroPawn motor-imagery montage

## Phase 1 — Prove the SSVEP signal exists ⬜

Goal: a clear spectral peak at the stimulus frequency (and harmonics) over occipital electrodes.

- ⬜ Move electrodes to occipital montage; update `CHANNEL_NAMES`
- ⬜ Single-target flicker script (sampled sinusoidal, vsync'd), fixed frequency, ~10 s
- ⬜ Record EEG during flicker vs. rest; save to CSV
- ⬜ Plot PSD — confirm peak at *f*, 2*f*, 3*f* vs. rest
- ⬜ Check display: actual refresh rate, dropped frames

## Phase 2 — Offline FBCCA ⬜

- ⬜ Data collection script: 4 targets, cued trials (~5 per target), labeled + saved
- ⬜ FBCCA implementation (filter bank capped ~50 Hz, 3 harmonics, weights n^-1.25 + 0.25)
- ⬜ Accuracy vs. window length: 4 / 3 / 2 / 1 / 0.5 s
- ⬜ Pick the 4 best-separated frequencies

## Phase 3 — Real-time selection ⬜

- ⬜ 4-target flicker UI
- ⬜ Live FBCCA on a sliding window
- ⬜ Confidence threshold (margin over runner-up)
- ⬜ IMU motion gate — reject predictions during head movement
- ⬜ Measure live accuracy and selections/minute

## Phase 4 — Hierarchical speller ⬜

- ⬜ Group → subgroup → character navigation
- ⬜ Space, backspace, back/undo
- ⬜ Expand from 4 to 8 targets if accuracy holds

## Phase 5 — Word & sentence prediction ⬜

- ⬜ Word-suggestion mode (local dictionary first)
- ⬜ LLM completion (Claude API) for words/sentences
- ⬜ Output: display + text-to-speech

## Later / stretch

- ⬜ FBCCA + KNN or TRCA (per-user trained)
- ⬜ Clench as confirm/undo signal
- ⬜ Split into streamer / DSP / GUI processes if needed

## Log

Newest first. Record findings, numbers, and gotchas here.

### 2026-09-19
- Created branch `abish_test`; set up BrainFlow; Knight IMU = board 66, 8 EEG @ 125 Hz, IMU in rows 11–19.
- EEG read all zeros at first — channels are off by default; fixed with `chon_{ch}_12` + `rldadd_{ch}` sent **after** `start_stream()` with NeuroPawn's 1–2 s pauses. Sending before streaming causes UTF-8 decode errors (board streams binary nonstop).
- Hit corrupted frames / ~1–13 samples/s — cause was two processes reading the port at once. Only run one BrainFlow session at a time.
- Board once dropped off USB mid-setup; didn't recur.
- Decided on SSVEP + FBCCA + hierarchical keyboard (see APPROACH.md), inspired by WATOLINK's mind-speech-interface.
