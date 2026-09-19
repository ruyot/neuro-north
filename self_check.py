"""
self_check.py
=============
Checks the SSVEP pipeline with NO headset and NO display. Run it after
changing code or setting up a new machine:

    python self_check.py            # all checks (~10 s)
    python self_check.py --quick    # skip the synthetic-board check

Everything runs the project's real code:
  1. config    board id, EEG rows and rate; filters below Nyquist for every board rate
  2. flicker   60 Hz frames through stimulus.is_on(): each square's dominant
               frequency is its target (frame-to-frame evenness isn't measured;
               the per-trial late-frame warnings cover real timing)
  3. pipeline  simulated SSVEP -> real filters -> saved like a calibration ->
               spectrum verdict + TRCA accuracy; pure noise must NOT pass
  4. board     the recording process on BrainFlow's synthetic board saves
               correctly sized, labelled trials

Exits non-zero if any check fails.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import os
import sys
import tempfile
import time

import numpy as np

from ssvep import config as cfg

results = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append(ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" - {detail}" if detail else ""))


def quiet(fn, *args, **kwargs):
    """Run fn with its printing suppressed."""
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


# --------------------------------------------------------------------------- #
def check_config() -> None:
    from brainflow.board_shim import BoardIds, BoardShim

    print("1. config")
    board = int(cfg.BOARD_ID)
    check("board is the Knight IMU (66)", board == BoardIds.NEUROPAWN_KNIGHT_BOARD_IMU.value == 66)
    eeg = BoardShim.get_eeg_channels(board)
    check("8 EEG rows (1-8)", eeg[:cfg.NUM_CHANNELS] == list(range(1, 9)), str(eeg))
    rate = BoardShim.get_sampling_rate(board)
    check("board streams at the configured rate", rate == cfg.SAMPLING_RATE, f"{rate} Hz")
    check("one electrode label per channel", len(cfg.ELECTRODE_LABELS) == cfg.NUM_CHANNELS)
    check("one letter per frequency", len(cfg.TARGET_LETTERS) == cfg.N_TARGETS)

    for rate in cfg.BOARD_RATES:
        nyquist = rate / 2
        top = max(max(band[0][1], band[1][1]) for band in cfg.FILTERBANK)
        check(f"filter bank below Nyquist at {rate} Hz", top < nyquist, f"top {top} Hz < {nyquist} Hz")
        check(f"band-pass below Nyquist at {rate} Hz", cfg.BANDPASS_HIGH < nyquist)

    from ssvep.preprocessing import crop_indices
    crop = crop_indices(cfg.SAMPLING_RATE)
    capture = cfg.capture_samples(cfg.SAMPLING_RATE)
    check("analysis window fits inside the capture", crop[-1] < capture,
          f"samples {crop[0]}-{crop[-1]} of {capture}")


# --------------------------------------------------------------------------- #
def check_flicker() -> None:
    from ssvep.stimulus import is_on

    print("2. flicker")
    fps, seconds = cfg.EXPECTED_REFRESH_HZ, 10.0
    frames = np.arange(int(fps * seconds)) / fps
    freqs = np.fft.rfftfreq(frames.size, 1 / fps)
    peaks = []
    for f in cfg.STIMULUS_FREQUENCIES:
        wave = np.array([is_on(f, t) for t in frames], dtype=float)
        peak = freqs[np.argmax(np.abs(np.fft.rfft(wave - wave.mean())))]
        peaks.append(peak)
        check(f"{f:5.2f} Hz square's dominant frequency on {fps} Hz", abs(peak - f) < 0.2,
              f"measured {peak:.2f} Hz")
    check("all targets distinct", len(set(np.round(peaks, 1))) == cfg.N_TARGETS)


# --------------------------------------------------------------------------- #
def write_session(path: str, amplitude: float, rng: np.random.Generator,
                  n_blocks: int = 6, rate: int = cfg.SAMPLING_RATE) -> None:
    """Simulated calibration: each trial = SSVEP at the cued frequency (+2nd harmonic)
    with a stable per-channel phase, plus fresh noise, run through the real filters."""
    from ssvep.preprocessing import extract_channel_matrix, filter_eeg

    os.makedirs(path)
    n = cfg.capture_samples(rate)
    t = np.arange(n) / rate
    phases = rng.uniform(0, 2 * np.pi, cfg.NUM_CHANNELS)
    gains = np.linspace(1.0, 0.4, cfg.NUM_CHANNELS)  # Oz strongest ... POz weakest
    for block in range(1, n_blocks + 1):
        for target, f in enumerate(cfg.STIMULUS_FREQUENCIES):
            data = np.zeros((cfg.NUM_CHANNELS + 1, n))
            for ch in range(cfg.NUM_CHANNELS):
                ssvep = np.sin(2 * np.pi * f * t + phases[ch]) + 0.5 * np.sin(4 * np.pi * f * t + phases[ch])
                data[ch + 1] = amplitude * gains[ch] * ssvep + rng.normal(0, 5, n)
            rows = range(1, cfg.NUM_CHANNELS + 1)
            filter_eeg(data, rows, rate)
            np.savetxt(os.path.join(path, f"block_{block}_{target + 1}.csv"),
                       extract_channel_matrix(data, rows), delimiter=",",
                       header=",".join(cfg.ELECTRODE_LABELS), comments="", fmt="%.7f")


def check_pipeline(tmp: str) -> None:
    from ssvep.diagnostics import report
    from ssvep.trca_model import count_blocks, cross_validate, load_training_data, session_rate

    print("3. pipeline")
    rng = np.random.default_rng(0)
    chance = 100 / cfg.N_TARGETS
    for name, amplitude in [("clear SSVEP", 3.0), ("pure noise", 0.0)]:
        path = os.path.join(tmp, name.replace(" ", "_"))
        write_session(path, amplitude, rng)
        rate = session_rate(path)
        eeg, labels = load_training_data(path, count_blocks(path), crop=False)
        verdict = quiet(report, eeg, labels, name, rate)["verdict"]
        accuracy = quiet(cross_validate, path).mean()
        if amplitude:
            check("session rate recovered from file length", rate == cfg.SAMPLING_RATE, f"{rate} Hz")
            check("spectrum check sees the clear SSVEP", verdict == "working", verdict)
            check("TRCA classifies the clear SSVEP", accuracy >= 90, f"{accuracy:.0f}%")
        else:
            check("spectrum check rejects pure noise", verdict == "none", verdict)
            check("TRCA stays near chance on pure noise", accuracy <= chance + 30,
                  f"{accuracy:.0f}% (chance {chance:.0f}%)")


# --------------------------------------------------------------------------- #
def check_board(tmp: str) -> None:
    from ssvep.recording import RecordingProcess
    from ssvep.trca_model import session_rate

    print("4. synthetic board")
    path = os.path.join(tmp, "synthetic_session")
    rec = RecordingProcess("collect", serial_port=None, data_dir=path, synthetic=True)
    quiet(rec.start)
    try:
        ready = rec.ready.wait(15)
        check("recording process starts", ready and not rec.failed.value)
        if not ready:
            return
        time.sleep(2)  # let the ring buffer fill past one capture
        for target in range(cfg.N_TARGETS):
            rec.label_index.value = target
            rec.recording_flag.value = False
            time.sleep(0.05)
            rec.recording_flag.value = True
            time.sleep(0.3)
    finally:
        rec.stop()
        rec.join(10)
    saved = sorted(os.listdir(path)) if os.path.isdir(path) else []
    expected = [f"block_1_{t}.csv" for t in range(1, cfg.N_TARGETS + 1)]
    check("one labelled file per cued target", saved == expected, ", ".join(saved) or "none")
    if saved == expected:
        rate = session_rate(path)
        check("trials sized for the board's real rate", rate == 250, f"{rate} Hz synthetic board")


# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick", action="store_true", help="skip the synthetic-board check")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        for step in (check_config, check_flicker, lambda: check_pipeline(tmp),
                     None if args.quick else lambda: check_board(tmp)):
            if step is None:
                continue
            try:
                step()
            except Exception as exc:  # a crash is a failure, not a traceback wall
                check("ran without crashing", False, f"{type(exc).__name__}: {exc}")

    failed = results.count(False)
    print(f"\n{len(results) - failed}/{len(results)} checks passed" + (f", {failed} FAILED" if failed else ""))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
