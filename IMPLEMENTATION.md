# Implementation Progress

Tracks what's built, what's next, and what we've learned. Design rationale lives in [APPROACH.md](APPROACH.md).

Status key: ✅ done · 🚧 in progress · ⬜ not started

## Phase 0 — Board connection ✅

- ✅ BrainFlow set up; `.venv` on Python 3.12, deps in `requirements.txt`
- ✅ `stream_test.py` — connects to Knight IMU, enables channels, prints EEG + accel/gyro (`--synthetic` for no hardware)
- ✅ `live_plot.py` — live pyqtgraph view: 8 EEG traces (1–40 Hz bandpass + 60 Hz notch, `F` toggles raw) + accel/gyro
- ✅ Verified on hardware: 125 Hz stream, all 8 channels enable, IMU reads gravity correctly

## Phase 1 — SSVEP pipeline (A/B/C/D typer) 🚧

Based on NeuroPawn's SSVEP + TRCA pipeline, adapted for the Knight IMU on macOS.

- ✅ `ssvep/` package: config, board, preprocessing, stimulus (PsychoPy), recording process, TRCA model
- ✅ PsychoPy verified frame-locked at 60.0 Hz on the MacBook Air display (pygame rejected: no vsync on macOS)
- ✅ `collect_training_data.py` — cued calibration blocks → `training_data/session_*/`
- ✅ `evaluate_trca.py` — SSVEP spectrum check (relative power table + plot, electrode ranking, alpha warning) + leave-one-block-out accuracy & ITR
- ✅ `abcd_typer.py` — free typing + `--copy` test with accuracy / letters-per-min / ITR saved to `results/`
- ✅ Tested without hardware: TRCA on simulated SSVEP (100% clear / 62% weak), recording process collect/predict/failure paths on synthetic board
- ⬜ Electrodes moved to occipital montage (Oz, O1, O2, PO7, PO8, PO3, PO4, POz)
- ✅ Visual check of the flicker window (`--synthetic --windowed`); frame timing fixed
- ⬜ First real calibration session (8 blocks) + evaluate
- ⬜ First copy test on the real headset

## Phase 2 — Tune the A/B/C/D typer ⬜

- ⬜ Accuracy vs. flicker length (1.5 / 1.0 / 0.5 s)
- ⬜ Check 10 Hz vs. alpha confusion; try alternative frequencies if needed
- ⬜ Confidence threshold (reject low-score predictions)
- ⬜ IMU motion gate — reject predictions during head movement
- ⬜ Does a model from one day work the next day?

## Phase 3 — Hierarchical speller ⬜

- ⬜ Input-agnostic speller core (keyboard input for testing)
- ⬜ Group → subgroup → character navigation with 4 targets
- ⬜ Space, backspace, back/undo
- ⬜ Expand from 4 to 8 targets if accuracy holds

## Phase 4 — Word & sentence prediction ⬜

- ⬜ Word-suggestion mode (local dictionary first)
- ⬜ LLM completion (Claude API) for words/sentences
- ⬜ Output: display + text-to-speech

## Later / stretch

- ⬜ FBCCA zero-calibration fallback
- ⬜ Clench as confirm/undo signal
- ⬜ Arduino + photoresistor check of the real flicker frequencies (NeuroPawn's freq-checker)

## Log

Newest first. Record findings, numbers, and gotchas here.

### 2026-09-19 (bug fixes from `ssvep-knight-validation`)
- Ported three fixes found by Abdullah on his parallel branch: `normfit` was given alpha instead of the confidence level (every "95% CI" was a 5% CI); ITR used the 1 s analysis window instead of the real 2 s per selection (overstated bits/min); window background `[0,0,0]` is mid-grey in PsychoPy rgb → now black.
- Letter labels moved out so they no longer touch the cue outline.

### 2026-09-19 (later)
- Switched base design to NeuroPawn's SSVEP + TRCA pipeline (built for the Knight board): 4 corner targets at 6.67 / 8.57 / 10 / 12 Hz, 1.5 s flicker, occipital montage. WATOLINK kept for later speller ideas only.
- Moved to Python 3.12 (NeuroPawn's deps need numpy ≥ 2.x, pandas 3).
- pygame vsync doesn't work on macOS (measured 122 Hz – 10 kHz). PsychoPy flips at a steady 60.0 Hz (16.4–17.0 ms) → using PsychoPy.
- NeuroPawn code needed: board 57 → 66, longer channel-setup pauses, no Windows-only `pydirectinput`, ready/failed handshake instead of a fixed 30 s wait.

### 2026-09-19
- Created branch `abish_test`; set up BrainFlow; Knight IMU = board 66, 8 EEG @ 125 Hz, IMU in rows 11–19.
- EEG read all zeros at first — channels are off by default; fixed with `chon_{ch}_12` + `rldadd_{ch}` sent **after** `start_stream()` with NeuroPawn's 1–2 s pauses. Sending before streaming causes UTF-8 decode errors (board streams binary nonstop).
- Hit corrupted frames / ~1–13 samples/s — cause was two processes reading the port at once. Only run one BrainFlow session at a time.
- Board once dropped off USB mid-setup; didn't recur.
- Decided on SSVEP (see APPROACH.md) after comparing with motor imagery.
