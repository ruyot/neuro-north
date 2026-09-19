# Approach

How we're building the neuro-north EEG speller, and why. Update this when a design decision changes; track day-to-day progress in [IMPLEMENTATION.md](IMPLEMENTATION.md).

## Goal

Type with an EEG headset: select letters → build words → build sentences, without a keyboard or mouse.

## Hardware

| | |
|---|---|
| Board | NeuroPawn Knight IMU (BrainFlow `BoardIds.NEUROPAWN_KNIGHT_BOARD_IMU` = 66) |
| EEG | 8 channels, 16-bit, gain 12, **125 Hz** (Nyquist 62.5 Hz) |
| IMU | accel x/y/z (m/s²), gyro x/y/z, magnetometer x/y/z |
| Connection | USB serial, 115200 baud (macOS: `/dev/cu.usbserial-*`) |
| Reference | ear clip on `COMM`, optionally `RLD` |

BrainFlow row layout: 1–8 EEG, 9–10 lead-off status, 11–13 accel, 14–16 gyro, 17–19 magnetometer, 20 timestamp.

**Board quirks**
- EEG channels are **off by default**. Enable after `start_stream()` with `chon_{ch}_12` then `rldadd_{ch}`, with ~1–2 s pauses between commands (NeuroPawn's documented sequence, ~34 s for 8 channels). Faster sends get dropped.
- Only one process can open the serial port. Two readers → corrupted frames ("Wrong end byte").
- Data starts ~3 s after opening the port.

## Paradigm: SSVEP (decided)

User looks at one of several targets flickering at different frequencies; the visual cortex oscillates at that frequency and its harmonics; we detect which one.

**Why SSVEP over motor imagery for typing**
- 4–8 reliable commands vs ~2 for motor imagery → far fewer selections per letter.
- Only ~2 minutes of calibration (TRCA), or none at all with FBCCA; MI needs calibration plus user practice, and some people can't produce usable MI.
- Typically much higher accuracy on 8-channel consumer hardware.

**Trade-offs we accept**
- It's gaze-dependent ("looking", not "thinking").
- Visual fatigue over long sessions.
- ⚠️ **Photosensitive epilepsy risk** — flicker in the 3–30 Hz range. Anyone with a seizure history should not use it; warn testers.

**Possible later addition:** a clench (physical or imagined) as a "confirm/undo" signal — SSVEP picks, clench confirms. The motor montage in NeuroPawn's MI tutorial already captures this.

**Motor imagery — considered and set aside (2026-09-19)**
- Binary left/right spelling works in the literature (e.g. Hex-o-Spell) but needs ~5 selections per letter; at ~80% per-selection accuracy only ~33% of letters come out right (0.8⁵), vs ~81% for 8-target SSVEP (0.9²).
- 4-class MI (left hand / right hand / feet / tongue) typically reaches only ~60–75% even with 22 channels; feet needs Cz and is weak at the scalp.
- Practical 4-command MI would be hybrid (L/R imagery + jaw clench + double blink), where half the commands aren't brain signals.
- Decision: **SSVEP is the primary input.** The speller stays input-agnostic (see Architecture), so MI could be added later as a comparison mode.

## Base design: NeuroPawn's SSVEP pipeline

We follow NeuroPawn's own SSVEP + TRCA pipeline for the Knight board ([github.com/NeuroPawn/ssvep](https://github.com/NeuroPawn/ssvep)), adapted in `ssvep/` for the Knight **IMU** board on macOS and for A/B/C/D typing. WATOLINK's speller is a source of **later** ideas only (hierarchical keyboard, word prediction).

Our adaptations vs. NeuroPawn's repo:
- Board ID **66** (Knight IMU) instead of 57 — different packet format.
- Channel-setup pauses of 1 s / 2 s / 1 s per channel (the timing verified on our board).
- Board process signals **ready / failed** instead of a fixed 30 s wait; polls at 1 kHz instead of spinning a core.
- No `pydirectinput` (Windows-only); predictions are published to the display process instead.
- Dropped-frame counter per trial; letter labels; calibration saved per session (`training_data/session_*`).

## Electrode montage

Board channel 1–8 → `Oz, O1, O2, PO7, PO8, PO3, PO4, POz` (`ssvep/config.py` → `ELECTRODE_LABELS`). Reference on one earlobe/mastoid, ground/bias on the other. SSVEP is strongest over the occipital cortex; electrodes elsewhere just add noise.

## Stimulus

- 4 squares in the screen corners: **A** top-left 6.67 Hz, **B** top-right 8.57 Hz, **C** bottom-left 10 Hz, **D** bottom-right 12 Hz.
- On/off square wave: each frame, a square is white if `sin(2π·f·t) ≥ 0`, else black. The frequencies land on near-integer frame counts at 60 Hz (9 / 7 / 6 / 5 frames).
- Drawn with **PsychoPy** (OpenGL, locked to the display refresh). Measured on the MacBook Air M4: **60.0 Hz, frame intervals 16.4–17.0 ms**. pygame was tried first and rejected: on macOS its vsync was not honoured (122 Hz to ~10 kHz frame rates).
- ⚠️ 10 Hz sits in the alpha band — watch for it being over-predicted when the user relaxes.

## Trial timing

```
|<-- 1.0 s cue -->|<----- 1.5 s flicker ----->| capture + 0.5 s rest |
   (calibration)      eyes on target            grab last 188 samples
```

TRCA uses samples [19:144] of the 188-sample capture: skip the first 150 ms (visual latency), keep 1.0 s.

## Signal processing

- **Pre-processing** (identical in calibration and live use): detrend → 3–48 Hz band-pass (2nd order) → 49–51 and 59–61 Hz band-stops (4th order), all zero-phase.
- **Classifier: ensemble TRCA** (meegkit). Learns a per-target spatial filter + template from your calibration trials; 6-band filter bank (6/14/22/30/38/46–48 Hz) to use harmonics. Supervised → needs ~6 calibration blocks per headset session.
- **Evaluation:** leave-one-block-out cross-validation, reporting accuracy and ITR (bits/min).
- **Later options:** FBCCA as a zero-calibration fallback; IMU motion gate; confidence threshold on the TRCA score.

## Architecture: two processes

```
Parent (PsychoPy)                         Child (RecordingProcess)
draw every frame, cue, flicker   ──flag──▶ grab last 188 samples → filter
                                                    ├─ collect: save labelled CSV
show result  ◀──prediction/count──                  └─ predict: TRCA → target
```

The display loop must flip every 16.7 ms, so the board, filtering and TRCA live in a separate process and never block it. Shared state is a few `multiprocessing` values (ready, failed, recording flag, label, prediction, prediction count).

## Speller UI

Hierarchical keyboard, so 4–8 commands reach the whole alphabet:

```
group (e.g. "abc | def | ghi")  →  subgroup ("abc")  →  character ("b")
```

Plus dedicated targets for **space, backspace, word-mode toggle, enter/speak**.

**Word prediction:** type a few letters, then pick a suggested word with a single selection. Use a modern LLM (e.g. the Claude API) for word/sentence completion, with a local fallback (frequency dictionary) for when offline. Do not reuse WATOLINK's `text-davinci-002` code.

## Speller architecture (later)

**Input-agnostic speller:** the speller consumes abstract commands (`select(i)`, `back`, …), not frequencies. Input sources plug in behind that interface:

```
SSVEP (TRCA) ───────┐
Keyboard (testing) ─┼──▶ commands ──▶ speller UI ──▶ word prediction ──▶ text
(later: MI / clench)┘
```

This lets us build and test the whole speller with keyboard keys before the EEG side is ready, and compare input methods on the same task.

Always query channels, sampling rate and timestamp row from BrainFlow (`get_eeg_channels`, `get_sampling_rate`, `get_timestamp_channel`) rather than hardcoding board-specific numbers.

## Key open questions

- Accuracy on our headset after 6 calibration blocks? Does it hold across days, or recalibrate each session?
- How short can the flicker get (1.5 → 1.0 → 0.5 s) before accuracy drops?
- Is 10 Hz confused with resting alpha? Would a different 4th frequency do better?
- What IMU motion threshold rejects artifacts without rejecting normal sitting?

## References

- WATOLINK mind-speech-interface (SSVEP speller we're borrowing ideas from): https://github.com/WATOLINK/mind-speech-interface-ssvep
- NeuroPawn Knight docs: https://docs.neuropawn.tech/knight-board/ (command set, specs, IMU)
- NeuroPawn motor-imagery tutorial: https://docs.neuropawn.tech/tutorials/motor-imagery/
- BrainFlow Knight IMU driver: https://github.com/brainflow-dev/brainflow/blob/master/src/board_controller/neuropawn/knight_imu.cpp
- NeuroPawn SSVEP pipeline (our base design): https://github.com/NeuroPawn/ssvep
- FBCCA: Chen et al., "Filter bank canonical correlation analysis for implementing a high-speed SSVEP-based brain–computer interface", J. Neural Eng. 2015
- TRCA: Nakanishi et al., "Enhancing detection of SSVEPs for a high-speed brain speller using task-related component analysis", IEEE TBME 2018
