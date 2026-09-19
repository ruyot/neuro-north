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
- Little/no per-user training (FBCCA works zero-shot); MI needs calibration and user practice, and some people can't produce usable MI.
- Typically much higher accuracy on 8-channel consumer hardware.

**Trade-offs we accept**
- It's gaze-dependent ("looking", not "thinking").
- Visual fatigue over long sessions.
- ⚠️ **Photosensitive epilepsy risk** — flicker in the 3–30 Hz range. Anyone with a seizure history should not use it; warn testers.

**Possible later addition:** a clench (physical or imagined) as a "confirm/undo" signal — SSVEP picks, clench confirms. The motor montage in NeuroPawn's MI tutorial already captures this.

## Electrode montage

SSVEP is strongest over the **occipital / parieto-occipital** cortex. Target 8-channel layout (to finalize):

```
PO3, POz, PO4, O1, Oz, O2, + two of {PO7, PO8, Pz}
```

The motor-imagery montage currently in `CHANNEL_NAMES` (FC4, C4, CP4, C2, C1, CP3, C3, FC3) is **not** suitable for SSVEP.

## Signal processing: FBCCA

Filter Bank Canonical Correlation Analysis — no training data required.

1. **Preprocess:** 60 Hz notch (North America).
2. **Filter bank:** N sub-bands, each bandpassed from `lowₙ` up to a shared upper edge. Because Nyquist is 62.5 Hz, cap the upper edge around **~50 Hz** and use **~4–5 banks** (WATOLINK's 10 banks up to 90 Hz assumed 250 Hz sampling and don't transfer).
3. **References:** for each target frequency *f*, build `sin/cos(2π·h·f·t)` for harmonics *h* = 1..3. Harmonic 4 of our higher frequencies lands near 57 Hz, next to the notch — skip it.
4. **CCA** between the EEG window and each reference set, per bank → correlation ρₙ(f).
5. **Combine:** score(f) = Σₙ wₙ·ρₙ(f)², with wₙ = n^-1.25 + 0.25.
6. **Decide:** argmax score, accepted only if it clears a **confidence threshold** (e.g. margin over runner-up) **and** the IMU reports no significant head motion.

**Later:** FBCCA + KNN (per-user trained), TRCA, or other trained SSVEP classifiers — only after plain FBCCA works.

## Stimulus presentation

- Candidate frequency set (from WATOLINK): 8.25, 8.75, 9.75, 10.75, 11.75, 12.75, 13.75, 14.25 Hz. Start with **4 targets**, expand to 8.
- **Sampled sinusoidal stimulation:** set each target's luminance every frame to `0.5·(1 + sin(2π·f·t_frame))`. This gives accurate frequencies even when *f* doesn't divide the monitor refresh rate — unlike toggling on a Qt timer, which drifts with OS scheduling.
- Needs a vsync'd renderer (PsychoPy or pygame with vsync). Verify the actual refresh rate and log dropped frames.

## Speller UI

Hierarchical keyboard, so 4–8 commands reach the whole alphabet:

```
group (e.g. "abc | def | ghi")  →  subgroup ("abc")  →  character ("b")
```

Plus dedicated targets for **space, backspace, word-mode toggle, enter/speak**.

**Word prediction:** type a few letters, then pick a suggested word with a single selection. Use a modern LLM (e.g. the Claude API) for word/sentence completion, with a local fallback (frequency dictionary) for when offline. Do not reuse WATOLINK's `text-davinci-002` code.

## Architecture

Start with **one Python process**: a stimulus window + an acquisition/detection thread. Split into separate processes (streamer / DSP / GUI, as WATOLINK does) only if timing or performance requires it.

```
NeuroPawn Knight IMU → BrainFlow ─┬─ EEG → notch → filter bank → CCA → FBCCA ─┐
                                   └─ IMU → motion detection ─────────────────┤
                                                                              ↓
                                                       confidence gate → target
                                                                              ↓
                                                 hierarchical keyboard → text
                                                                              ↑
                                                              word prediction (LLM)
```

Always query channels, sampling rate and timestamp row from BrainFlow (`get_eeg_channels`, `get_sampling_rate`, `get_timestamp_channel`) rather than hardcoding board-specific numbers.

## Key open questions

- What window length gives acceptable accuracy on this hardware? Measure 4 / 3 / 2 / 1 / 0.5 s.
- Which 4 (then 8) frequencies separate best for our users and monitor?
- What IMU motion threshold rejects artifacts without rejecting normal sitting?
- Does the ear-clip reference + RLD give clean enough occipital signal, or do we need a different reference?

## References

- WATOLINK mind-speech-interface (SSVEP speller we're borrowing ideas from): https://github.com/WATOLINK/mind-speech-interface-ssvep
- NeuroPawn Knight docs: https://docs.neuropawn.tech/knight-board/ (command set, specs, IMU)
- NeuroPawn motor-imagery tutorial: https://docs.neuropawn.tech/tutorials/motor-imagery/
- BrainFlow Knight IMU driver: https://github.com/brainflow-dev/brainflow/blob/master/src/board_controller/neuropawn/knight_imu.cpp
- FBCCA: Chen et al., "Filter bank canonical correlation analysis for implementing a high-speed SSVEP-based brain–computer interface", J. Neural Eng. 2015
