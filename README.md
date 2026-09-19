# neuro-north

SSVEP EEG acquisition, quality gating, and TRCA-based target classification for an 8-channel NeuroPawn Knight board (BrainFlow board id 66, 22 raw rows, nominal 125 Hz, gain 12 requested). Target/letter selection, not a free-form thought/intent decoder: TRCA always classifies to one configured flicker target; the quality gate rejects contaminated epochs before they ever reach it.

## Setup

Create the environment only if `.venv` is absent; otherwise reuse it with the pinned requirements.

```
uv venv --python /home/abuudiii/.local/bin/python3.12 .venv
uv pip install --python .venv/bin/python -r ssvep_training/requirements.txt
```

`brainflow==5.23.0` / `meegkit==0.2.0` are pinned; do not upgrade in place. No ASR/ICA/MNE runtime dependency is added — those stay optional offline tools. Real-board commands read the port from `PORT_PATH` in `.env`, or `--port /dev/ttyACM0` overrides it per run.

## Hardware-free checks (no board)

```
.venv/bin/python -m ssvep_training.signal_quality --self-check
.venv/bin/python -m ssvep_training.head
.venv/bin/python stream.py --seconds 3 --filter --fft
```

Synthetic/in-memory math only — no port, no real EEG, no timing evidence. A separate non-flashing native-window smoke check (still no headset, but it does open a display):

```
.venv/bin/python -m ssvep_training.speller_ui --keys --windowed
```

## Safe real-board commands

Record-only, no flashing, no PsychoPy window:

```
.venv/bin/python -m ssvep_training.recording --seconds 60 --output ssvep_training/training_data/quality_check
```

Offline inspection of any saved session (raw or legacy):

```
.venv/bin/python -m ssvep_training.signal_quality ssvep_training/training_data/quality_check
```

**Timing:** `--seconds` counts *after* the explicit "hold still for 10 seconds" baseline handshake, not from process start. Ctrl-C stops cleanly and saves whatever data was retained, closing the board — graceful shutdown, not crash/SIGKILL-proof durability; a killed process loses unsaved data.

## Signal contract (schema v2)

- Raw rows are saved untouched: all 22 SDK rows, SDK order (`raw.npz`). Filtering, quality checks, and unit conversion never mutate raw data.
- Sessions record board id 66, nominal rate 125 Hz, `gain_requested: 12`, the pinned `brainflow`/`meegkit` versions, and the confirmed row map (EEG 1–8, packet counter 0, lead-off 9/10, IMU 11–19, host timestamp 20, marker 21). TRCA consumes BrainFlow-native units, not µV.
- Selecting a channel subset (`--channels`) also removes unselected channels from the electrode bias loop (`choff_*` + `rldremove_*`), not just adds the selected ones — opening the port does not reset firmware state.
- **Legacy sessions** stay readable by `signal_quality`/raw diagnostics, but `load_trials` refuses to train on them: no schema/baseline contract, so a new calibration recording is required, not a metadata patch.

### Scale, mains, and contact bits (conservative-by-default)

- `MAINS_HZ` — `50` or `60`, default `60`; only one notch is ever applied.
- `EEG_UV_PER_SDK_UNIT` (finite, positive) + `EEG_SCALE_EVIDENCE` (nonempty provenance string) are **paired** — set both or neither. Default null: data stays in native SDK units. The disputed 79.57× driver-vs-vendor-doc discrepancy is **not** auto-applied.
- `LEADOFF_OFF_BIT` (`0`/`1`) stays unset until a controlled connected/disconnected firmware check confirms polarity; unset means contact status is displayed but not used to reject windows.

### Admission gating

Every epoch requires a complete 4-second filter history plus the full 1.5-second flicker window. Missing history, flicker samples, or completion annotation is `INCOMPLETE`; a positive dropped-frame count is `TIMING`. Onset/completion are tracked by exact trial id (calibration `block*10+target+1`, live `-trial_id`) through one ordered queue, so a stale/superseded id is never mistaken for a fresh selection. Rejections leave text/ranges unchanged and start a fresh selection; recovery requires four seconds of clean history. No samples or channels are interpolated.

The PsychoPy warning screen is acknowledged first; only then does a blank (non-flashing) window measure the real refresh rate, and the run requires that measurement within **60 ± 1 Hz** or refuses to flicker. Only after that check passes does the "hold still" quiet instruction appear and the 10-second baseline start. This is a monitor-refresh check only — **physical optical onset timing and true EEG-to-stimulus delay are not verified** without a photodiode/common timing reference.

## Calibration and evaluation

Complete the physical/control checks below first. Use flashing modes only with participant consent and the existing seizure warnings; blink during block breaks, take breaks as needed, and stop immediately if uncomfortable.

Pilot first, small and disposable:

```
.venv/bin/python -m ssvep_training.collect_training_data --blocks 3 --windowed
```

Before progressing from the pilot to a fresh calibration, require at least 4 of its 6 trials to survive admission, with at least one accepted per target; otherwise inspect rejection reasons/contact/timing first. This is a bring-up check, not a final accuracy result:

```
.venv/bin/python -m ssvep_training.collect_training_data --blocks 20 --windowed
.venv/bin/python -m ssvep_training.evaluate_trca --session SESSION --no-plot
```

Replace `SESSION` with the fresh 20-block folder printed by the recorder; keep pilot data separate.

TRCA training requires **at least 2 accepted trials per target in every cross-validation fold**; cross-validation requires **at least 3 blocks total** — both enforced with an explicit error, never silently loosened. `evaluate_trca` reports filtered target-frequency responses, rejection coverage, and held-out accuracy vs. chance (a raw-vs-filtered spectral view lives in `signal_quality`, not here) — never a plausible-looking claim without that coverage/accuracy report attached.

Live UIs, each requiring a prior calibration session (`--port`, `--session PATH` default latest calibration, `--windowed` optional):

```
.venv/bin/python -m ssvep_training.speller_ui --windowed
.venv/bin/python -m ssvep_training.typer --prompt BAABAABA --windowed
```

`speller_ui --keys` (above) is the only variant that skips the headset and refresh-timing gate entirely.

## Physical bring-up checklist (not yet done)

1. Confirm the physical common-reference switch, earclip contact, montage/channel mapping and gain 12; verify sustained host-frame rate within ±5% of 125 Hz and the 22-row layout. RLD commands control bias feedback, not the physical reference. Stop decoding on a 250/500 Hz mismatch rather than relabeling/resampling it.
2. Check all IMU axes and lead-off polarity with controlled movement/contact checks before setting `LEADOFF_OFF_BIT`. Resolve EEG scale via matched manufacturer confirmation or an approved unworn/isolated known-amplitude test before setting the paired scale variables. Never attach an unisolated bench source to a worn headset.
3. Collect at least 20 independently annotated quiet and 20 gross-artifact contexts: 688 samples each, advancing by 125 samples; a quiet label covers the entire context. Require ≤5% quiet rejection and ≥95% gross-artifact rejection. If the initial multiplier 6 fails, evaluate fixed candidates 4, 6, 8, 10, 12 on these controls and choose the smallest that meets both targets before a new calibration. If none does, retain diagnostic-only use. Never tune against held-out target labels.

Implementation verification so far is software/fixture evidence, not headset, participant, physical-scale or photodiode validation; nothing here claims real-world denoising or decoding performance. Separately, meegkit 0.2.0's TRCA filter bank differs between `fit` (cascaded across bands) and `predict` (each band from the original epoch) — an upstream decoder asymmetry left unchanged by this pipeline.
