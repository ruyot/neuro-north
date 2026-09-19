"""Conservative admission of unmodified, all-row Knight SDK samples.

EEG amplitudes stay in BrainFlow-native units. The driver step below is not a
verified electrode-level microvolt conversion; magnetometer values are not an
admission condition. No transport, filter, decoder or UI is imported here.
"""

from __future__ import annotations

from enum import IntFlag
from numbers import Integral, Real

import numpy as np

from . import config as cfg


class Quality(IntFlag):
    OK = 0
    INCOMPLETE = 1
    NONFINITE = 2
    COUNTER = 4
    TIMEBASE = 8
    CLIPPED = 16
    FLAT = 32
    EEG_BURST = 64
    MOTION = 128
    IMU_UNAVAILABLE = 256
    CONTACT = 512
    TIMING = 1024
    INVALID_RESULT = 2048


SDK_STEP = 4 / 32767 / 12 * 1e6
_BASELINE_VERSION = "relative-qc-v1"
_BASELINE_ERROR = "Quality baseline unavailable; record a quiet baseline."
_BASELINE_KEYS = {
    "version", "samples", "window_samples", "hop_samples", "multiplier",
    "sdk_step", "eeg_ptp_ref", "eeg_step_ref", "gyro_center",
    "gyro_noise_ref", "accel_step_ref",
}
_IMU_ROWS = {"accel": [11, 12, 13], "gyro": [14, 15, 16], "mag": [17, 18, 19]}


def _integer(value) -> bool:
    return isinstance(value, Integral) and not isinstance(value, bool)


def _finite(value) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool) and bool(np.isfinite(value))


def _positive(value) -> bool:
    return _finite(value) and value > 0


def _layout(meta: dict) -> int:
    """Reject unknown layouts before applying Knight-specific rails/status bits."""
    if not isinstance(meta, dict):
        raise ValueError("Quality metadata must be a dictionary.")
    if not _integer(meta.get("board_id")) or meta["board_id"] != 66:
        raise ValueError("Signal quality requires Knight IMU board 66.")
    if not _positive(meta.get("rate")) or meta["rate"] != 125:
        raise ValueError("Signal quality requires the nominal 125 Hz Knight contract.")
    rows = meta.get("eeg_rows")
    if (not isinstance(rows, list) or not rows
            or any(not _integer(row) or not 1 <= row <= 8 for row in rows)
            or rows != sorted(set(rows))):
        raise ValueError("EEG rows must be a nonempty ascending unique subset of 1-8.")
    for key, expected in (("packet_row", 0), ("timestamp_row", 20)):
        if not _integer(meta.get(key)) or meta[key] != expected:
            raise ValueError(f"Knight {key} must be {expected}.")
    imu = meta.get("imu_rows")
    if (not isinstance(imu, dict) or imu.keys() != _IMU_ROWS.keys()
            or any(not isinstance(imu[name], list) or imu[name] != rows
                   or any(not _integer(row) for row in imu[name])
                   for name, rows in _IMU_ROWS.items())):
        raise ValueError("Knight IMU rows must be accel 11-13, gyro 14-16, mag 17-19.")
    off = meta.get("leadoff_off_bit")
    if off is not None and (not _integer(off) or off not in (0, 1)):
        raise ValueError("leadoff_off_bit must be null, 0 or 1.")
    return 125


def validate_baseline(meta: dict) -> dict:
    """Return the stored baseline or raise the shared unavailable-baseline error.

    Counts, row order, driver step and reference floors are part of its versioned
    contract. A positive declared multiplier is allowed for offline control
    sweeps; model loading must also compare it with cfg.QUALITY_MULTIPLIER.
    """
    try:
        rate = _layout(meta)
        baseline = meta["quality_baseline"]
        if not isinstance(baseline, dict) or baseline.keys() != _BASELINE_KEYS:
            raise ValueError
        if baseline["version"] != _BASELINE_VERSION:
            raise ValueError
        counts = {
            "samples": int(round(cfg.QUALITY_BASELINE_SECONDS * rate)),
            "window_samples": int(round(1.0 * rate)),
            "hop_samples": int(round(0.5 * rate)),
        }
        for key, expected in counts.items():
            if not _integer(baseline[key]) or baseline[key] != expected or expected <= 0:
                raise ValueError
        for key in ("multiplier", "sdk_step", "gyro_noise_ref", "accel_step_ref"):
            if not _positive(baseline[key]):
                raise ValueError
        if baseline["sdk_step"] != SDK_STEP:
            raise ValueError
        for key, size, floor in (("eeg_ptp_ref", len(meta["eeg_rows"]), 2 * SDK_STEP),
                                 ("eeg_step_ref", len(meta["eeg_rows"]), SDK_STEP),
                                 ("gyro_center", 3, None)):
            values = baseline[key]
            if (not isinstance(values, list) or len(values) != size
                    or any(not _finite(value) or (floor is not None and value < floor)
                           for value in values)):
                raise ValueError
    except (KeyError, TypeError, ValueError, OverflowError):
        raise ValueError(_BASELINE_ERROR) from None
    return baseline


def _raw_data(data: np.ndarray) -> np.ndarray:
    data = np.asarray(data)
    if data.ndim != 2 or data.shape[0] != 22 or data.dtype.kind not in "fiu":
        raise ValueError("Quality data must be a real numeric all-row array with shape (22, samples).")
    return data


def _raw_features(data: np.ndarray, meta: dict, window_samples: int):
    """Basic flags and reusable features, before any relative thresholds."""
    eeg = np.asarray(data[meta["eeg_rows"]], dtype=np.float64)
    accel = np.asarray(data[meta["imu_rows"]["accel"]], dtype=np.float64)
    gyro = np.asarray(data[meta["imu_rows"]["gyro"]], dtype=np.float64)
    counter = np.asarray(data[meta["packet_row"]], dtype=np.float64)
    timestamps = np.asarray(data[meta["timestamp_row"]], dtype=np.float64)
    eeg_finite = bool(np.isfinite(eeg).all())
    accel_finite = bool(np.isfinite(accel).all())
    gyro_finite = bool(np.isfinite(gyro).all())
    counter_finite = bool(np.isfinite(counter).all())
    time_finite = bool(np.isfinite(timestamps).all())
    flags = Quality.OK
    if not all((eeg_finite, accel_finite, gyro_finite, counter_finite, time_finite)):
        flags |= Quality.NONFINITE

    if (not counter_finite or np.any(counter < 0) or np.any(counter > 255)
            or np.any(counter != np.floor(counter))):
        flags |= Quality.COUNTER
    elif np.any(np.remainder(np.diff(counter), 256) != 1):
        flags |= Quality.COUNTER

    if not time_finite:
        flags |= Quality.TIMEBASE
    else:
        gaps = np.diff(timestamps)
        if np.any(timestamps <= 0) or np.any(timestamps[1:] <= timestamps[:-1]) or np.any(gaps > 0.1):
            flags |= Quality.TIMEBASE
        if data.shape[1] >= int(round(5.0 * meta["rate"])):
            span = float(timestamps[-1]) - float(timestamps[0])
            if span <= 0 or not 0.95 * meta["rate"] <= (data.shape[1] - 1) / span <= 1.05 * meta["rate"]:
                flags |= Quality.TIMEBASE

    if np.any(eeg <= -32768 * SDK_STEP * 0.995) or np.any(eeg >= 32767 * SDK_STEP * 0.995):
        flags |= Quality.CLIPPED
    # Nonfinite inputs are already rejected; their derived NaNs cannot admit data.
    with np.errstate(invalid="ignore", over="ignore"):
        windows = np.lib.stride_tricks.sliding_window_view(eeg, window_samples, axis=1)
        ptp = np.ptp(windows, axis=-1)
        steps = np.abs(np.diff(eeg, axis=1))
    if np.any(ptp < 2 * SDK_STEP):
        flags |= Quality.FLAT

    # Six jointly zero accel/gyro values in any frame indicate missing motion
    # information even if a magnetometer happens to remain nonzero.
    if (not accel_finite or not gyro_finite
            or np.any(np.all(accel == 0, axis=0) & np.all(gyro == 0, axis=0))):
        flags |= Quality.IMU_UNAVAILABLE

    off = meta.get("leadoff_off_bit")
    if off is not None:
        status = data[[9, 10]]
        status_finite = bool(np.isfinite(status).all())
        if not status_finite:
            flags |= Quality.NONFINITE
        if (not status_finite or np.any(status < 0) or np.any(status > 255)
                or np.any(status != np.floor(status))):
            flags |= Quality.CONTACT
        else:
            selected_bits = sum(1 << (row - 1) for row in meta["eeg_rows"])
            selected_status = status.astype(np.uint8) & selected_bits
            if np.any(selected_status != (selected_bits if off == 0 else 0)):
                flags |= Quality.CONTACT
    return flags, ptp, steps, accel, gyro


def _motion_levels(accel: np.ndarray, gyro: np.ndarray, center: np.ndarray):
    with np.errstate(invalid="ignore", over="ignore"):
        gyro_level = np.linalg.norm(gyro - center[:, None], axis=0)
        accel_step = np.linalg.norm(np.diff(accel, axis=1), axis=0)
    return gyro_level, accel_step


def fit_baseline(data: np.ndarray, meta: dict) -> dict:
    """Fit exactly the first configured ten seconds after the baseline handshake.

    Automatic checks reject gross faults, not an otherwise noisy calibration;
    quiet/movement controls must still establish useful relative thresholds.
    """
    rate = _layout(meta)
    data = _raw_data(data)
    samples = int(round(cfg.QUALITY_BASELINE_SECONDS * rate))
    window_samples = int(round(1.0 * rate))
    hop_samples = int(round(0.5 * rate))
    if data.shape[1] < samples:
        raise ValueError(f"Quality baseline needs {samples} samples; received {data.shape[1]} (INCOMPLETE).")
    if not _positive(cfg.QUALITY_MULTIPLIER):
        raise ValueError("QUALITY_MULTIPLIER must be finite and positive.")
    data = data[:, :samples]
    flags, ptp, steps, accel, gyro = _raw_features(data, meta, window_samples)
    if flags != Quality.OK:
        raise ValueError(f"Quality baseline rejected: {flags.name}.")

    gyro_center = np.median(gyro, axis=1)
    gyro_level, accel_step = _motion_levels(accel, gyro, gyro_center)
    gyro_noise_ref = float(np.percentile(gyro_level, 95))
    accel_step_ref = float(np.percentile(accel_step, 95))
    if not _positive(gyro_noise_ref) or not _positive(accel_step_ref):
        raise ValueError("Quality baseline rejected: IMU_UNAVAILABLE; positive finite gyro and accel noise required.")
    step_windows = np.lib.stride_tricks.sliding_window_view(steps, window_samples - 1, axis=1)
    step_maxima = np.max(step_windows[:, ::hop_samples], axis=-1)
    return {
        "version": _BASELINE_VERSION,
        "samples": samples,
        "window_samples": window_samples,
        "hop_samples": hop_samples,
        "multiplier": float(cfg.QUALITY_MULTIPLIER),
        "sdk_step": SDK_STEP,
        "eeg_ptp_ref": np.maximum(np.median(ptp[:, ::hop_samples], axis=1), 2 * SDK_STEP).tolist(),
        "eeg_step_ref": np.maximum(np.median(step_maxima, axis=1), SDK_STEP).tolist(),
        "gyro_center": gyro_center.tolist(),
        "gyro_noise_ref": gyro_noise_ref,
        "accel_step_ref": accel_step_ref,
    }


def assess_window(data: np.ndarray, meta: dict) -> Quality:
    """Assess all contiguous one-second EEG windows and all raw motion frames.

    A valid stored baseline is required even for incomplete data. No sample is
    repaired, converted to physical units, filtered or changed in place.
    """
    baseline = validate_baseline(meta)
    data = _raw_data(data)
    if data.shape[1] < baseline["window_samples"]:
        return Quality.INCOMPLETE
    flags, ptp, steps, accel, gyro = _raw_features(data, meta, baseline["window_samples"])
    multiplier = baseline["multiplier"]
    ptp_limit = multiplier * np.asarray(baseline["eeg_ptp_ref"])
    step_limit = multiplier * np.asarray(baseline["eeg_step_ref"])
    # Every adjacent step belongs to a full one-second window, so its global
    # maximum is enough; no second overlapping-window reduction is needed.
    if np.any(ptp > ptp_limit[:, None]) or np.any(np.max(steps, axis=1) > step_limit):
        flags |= Quality.EEG_BURST
    gyro_level, accel_step = _motion_levels(accel, gyro, np.asarray(baseline["gyro_center"]))
    if (np.any(gyro_level > multiplier * baseline["gyro_noise_ref"])
            or np.any(accel_step > multiplier * baseline["accel_step_ref"])):
        flags |= Quality.MOTION
    return flags


def self_check() -> None:
    """Exercise the real signal contract with temporary, deterministic SDK fixtures."""
    import os
    from collections import Counter
    from copy import deepcopy
    from pathlib import Path
    from tempfile import TemporaryDirectory
    from types import SimpleNamespace

    from . import knight as stream, session
    from .recording import RecordingProcess
    from .trca_model import cross_validate, fit

    def requires_error(call, *details):
        try:
            call()
        except ValueError as exc:
            assert all(detail in str(exc) for detail in details), str(exc)
        else:
            raise AssertionError(f"Expected ValueError containing {details}")

    rate, rows = 125, list(range(1, 9))
    hist, flick = session.history_samples(rate), session.flicker_samples(rate)
    baseline_n = int(round(cfg.QUALITY_BASELINE_SECONDS * rate))
    fixture_env = {key: os.environ.pop(key, None) for key in
                   ("EEG_UV_PER_SDK_UNIT", "EEG_SCALE_EVIDENCE", "LEADOFF_OFF_BIT")}
    try:
        meta = session.session_meta(SimpleNamespace(
            board_id=66, rate=rate, eeg_rows=rows,
            names=[stream.NAMES[row - 1] for row in rows], marker_row=21,
            packet_row=0, timestamp_row=20, imu_rows=deepcopy(_IMU_ROWS),
        ))
    finally:
        for key, value in fixture_env.items():
            if value is not None:
                os.environ[key] = value
    # These are fixture annotations, not measurements of firmware or a display.
    meta.update(eeg_uv_per_sdk_unit=None, eeg_scale_evidence=None,
                leadoff_off_bit=1, display_refresh_hz=float(cfg.EXPECTED_REFRESH_HZ))

    def samples(n):
        t = np.arange(n) / rate
        phase = np.arange(8)[:, None] * 0.37
        axes = np.arange(3)[:, None]
        data = np.zeros((22, n), dtype=np.float64)
        data[0] = (np.arange(n) + 254) % 256  # Includes 254 -> 255 -> 0 -> 1.
        data[1:9] = SDK_STEP * (12 * np.sin(2 * np.pi * 15 * t + phase)
                               + 4 * np.sin(2 * np.pi * 20 * t + phase / 2))
        data[11:14] = np.array([[0], [0], [9.81]]) + 0.03 * np.sin(2 * np.pi * 0.7 * t + axes)
        data[14:17] = 0.01 * np.cos(2 * np.pi * 1.1 * t + axes)
        data[17:20] = 20 + 10 * axes + 0.02 * np.sin(2 * np.pi * 0.3 * t + axes)
        data[20] = 1_700_000_000 + t
        return data

    def annotation(trial_id, dropped=0):
        return {"trial_id": trial_id, "onset_request_monotonic_ns": trial_id * 1000,
                "marker_submit_monotonic_ns": trial_id * 1000 + 1, "dropped_frames": dropped}

    baseline_raw = samples(baseline_n)
    baseline_bytes = baseline_raw.tobytes()
    meta["quality_baseline"] = fit_baseline(baseline_raw, meta)
    frozen = deepcopy(meta["quality_baseline"])
    assert baseline_raw.tobytes() == baseline_bytes, "Baseline fitting mutated raw data"
    later_bad = np.concatenate((baseline_raw, np.full((22, rate), np.nan)), axis=1)
    assert fit_baseline(later_bad, meta) == frozen, "Baseline included later task samples"
    bad_base = baseline_raw.copy()
    bad_base[1, rate] = np.nan
    requires_error(lambda: fit_baseline(np.concatenate((bad_base, baseline_raw), axis=1), meta), "NONFINITE")
    for sensor_rows in (slice(11, 14), slice(14, 17)):
        bad_base = baseline_raw.copy()
        bad_base[sensor_rows] = 0
        requires_error(lambda: fit_baseline(bad_base, meta), "IMU_UNAVAILABLE")

    quiet = samples(hist + flick)
    assert assess_window(quiet, meta) == Quality.OK, "Quiet data, including counter wrap, rejected"
    assert assess_window(quiet[:, :rate - 1], meta) == Quality.INCOMPLETE
    requires_error(lambda: assess_window(quiet, {**meta, "quality_baseline": None}), "baseline")

    # The original ascontiguousarray implementation aliases both of these inputs.
    eight = samples(8 * rate)[1:9].copy(order="C")
    eight_bytes = eight.tobytes()
    for source in (eight, eight[3:4]):
        first = stream.clean(source, rate)
        second = stream.clean(source, rate)
        assert eight.tobytes() == eight_bytes, "Filtering mutated the eight-channel source or its view"
        assert np.array_equal(first, second), "Repeated filtering changed its result"
        second_bytes = second.tobytes()
        first[0, 0] = np.nan
        assert second.tobytes() == second_bytes and eight.tobytes() == eight_bytes, "Filter outputs alias"

    frequencies = (15, 20, 30, 40, 50, 60)
    angles = 2 * np.pi * np.array(frequencies)[:, None] * np.arange(8 * rate) / rate
    tones, cosines = np.sin(angles), np.cos(angles[:, 4 * rate:])
    gains = {}
    for mains in (50, 60):
        filtered = stream.clean(tones, rate, mains_hz=mains)[:, 4 * rate:]
        gains[mains] = {
            frequency: float(2 * np.hypot(signal @ sine, signal @ cosine) / signal.size)
            for frequency, signal, sine, cosine in zip(frequencies, filtered, tones[:, 4 * rate:], cosines)
        }
        for frequency, tolerance in ((15, 0.01), (20, 0.01), (30, 0.02)):
            assert abs(gains[mains][frequency] - 1) <= tolerance, (mains, frequency, gains[mains])
        assert abs(gains[mains][40] - 2 ** -0.5) <= 0.01, gains[mains]
    assert gains[50][50] <= gains[60][50] / 10, "50 Hz notch selection has no expected effect"
    assert gains[60][60] <= gains[50][60] / 2, "60 Hz notch selection has no expected effect"

    cases = [("quiet", quiet, Quality.OK, 0)]

    def fault(name, reason, dropped=0):
        data = quiet.copy()
        cases.append((name, data, reason, dropped))
        return data

    for row, reason in ((1, Quality.NONFINITE), (0, Quality.NONFINITE | Quality.COUNTER),
                        (20, Quality.NONFINITE | Quality.TIMEBASE),
                        (11, Quality.NONFINITE | Quality.IMU_UNAVAILABLE),
                        (14, Quality.NONFINITE | Quality.IMU_UNAVAILABLE)):
        fault(f"nonfinite row {row}", reason)[row, rate] = np.nan
    for name, delta in (("dropped counter", 1), ("duplicate counter", -1)):
        data = fault(name, Quality.COUNTER)
        data[0, rate:] = (data[0, rate:] + delta) % 256
    data = fault("host arrival gap", Quality.TIMEBASE)
    data[20, rate:] += 0.2
    data = fault("gross received rate", Quality.TIMEBASE)
    data[20] = data[20, 0] + (data[20] - data[20, 0]) * 1.2
    for rail in (-32768, 32767):
        fault(f"rail {rail}", Quality.CLIPPED)[1, rate:rate + 5] = rail * SDK_STEP
    fault("one-second flat channel", Quality.FLAT)[1, rate:2 * rate] = 0
    fault("EEG burst", Quality.EEG_BURST)[1, rate] += 2 * frozen["multiplier"] * max(frozen["eeg_ptp_ref"])
    for group, reference in (("gyro", "gyro_noise_ref"), ("accel", "accel_step_ref")):
        for row in meta["imu_rows"][group]:
            fault(f"{group} row {row}", Quality.MOTION)[row, rate] += 2 * frozen["multiplier"] * frozen[reference]
    fault("zero IMU", Quality.IMU_UNAVAILABLE)[11:20] = 0
    fault("missing accel/gyro frame", Quality.IMU_UNAVAILABLE)[11:17, rate] = 0
    for row in (9, 10):
        fault(f"contact row {row}", Quality.CONTACT)[row, rate] = 1 << (rows[-1] - 1)
    fault("unfinished flicker", Quality.INCOMPLETE, dropped=None)
    fault("late display frame", Quality.TIMING, dropped=1)
    gravity = quiet.copy()
    gravity[11:14] += np.array([[20], [-10], [40]])
    cases.append(("stationary gravity offset", gravity, Quality.OK, 0))
    cases.append(("clean recovery", quiet.copy(), Quality.OK, 0))

    # Contact polarity is explicit; unknown polarity cannot become a contact gate.
    for off in (0, 1):
        contact = quiet.copy()
        contact[9:11] = 255 if off == 0 else 0
        contact_meta = {**meta, "leadoff_off_bit": off}
        assert assess_window(contact, contact_meta) == Quality.OK
        contact[9, rate] = 254 if off == 0 else 1
        assert assess_window(contact, contact_meta) & Quality.CONTACT
        assert assess_window(contact, {**meta, "leadoff_off_bit": None}) == Quality.OK

    # Each disjoint context has its own calibration marker and a matching live replay.
    capture_meta, chunks = deepcopy(meta), [baseline_raw]
    expected_epochs = {False: [], True: []}
    expected_targets, expected_blocks, reasons = [], [], []
    for trial_id, (name, context, expected, dropped) in enumerate(cases, 1):
        block, target = divmod(trial_id - 1, len(meta["letters"]))
        block += 1
        code = session.encode_marker(block, target)
        start = baseline_n + (trial_id - 1) * (hist + flick)
        calibration = context.copy()
        calibration[0] = (calibration[0] + start) % 256
        calibration[20] += start / rate
        calibration[21, hist] = code
        capture_meta["trials"][str(code)] = annotation(trial_id, dropped)
        chunks.append(calibration)
        live = calibration.copy()
        live[21, hist] = -trial_id
        live_meta = {**meta, "trials": {str(-trial_id): annotation(trial_id, dropped)}}
        before, flags = live.tobytes(), []
        for full in (False, True):
            epoch, quality = session.epoch_at(live, hist, live_meta, full=full)
            assert quality & expected == expected if expected else quality == Quality.OK, (name, quality)
            assert (epoch is None) == bool(quality), (name, quality)
            assert live.tobytes() == before, f"Epoch admission mutated {name} raw rows"
            flags.append(quality)
            if quality == Quality.OK:
                assert epoch.shape == ((188 if full else 125), 8), (name, epoch.shape)
                expected_epochs[full].append(epoch)
        assert flags[0] == flags[1], f"full=True changed admission for {name}"
        reasons.append(flags[0])
        if flags[0] == Quality.OK:
            assert np.array_equal(expected_epochs[False][-1], expected_epochs[True][-1][19:144])
            expected_targets.append(target)
            expected_blocks.append(block)
    assert meta["quality_baseline"] == frozen, "Admission retuned the frozen baseline"
    capture = np.concatenate(chunks, axis=1)
    capture_bytes = capture.tobytes()
    expected_counts = dict(Counter(flag.name for quality in reasons for flag in Quality if quality & flag))
    with TemporaryDirectory(prefix="neuro_quality_") as tmp:
        path = str(Path(tmp) / "calibration")
        session.save_session(path, capture, capture_meta)
        restored, _ = session.load_session(path)
        assert restored.tobytes() == capture_bytes, "Raw/status/IMU/marker rows changed in roundtrip"
        for full in (False, True):
            trials = session.load_trials(path, full=full)
            assert trials.targets.tolist() == expected_targets and trials.blocks.tolist() == expected_blocks
            assert np.array_equal(trials.eeg, np.stack(expected_epochs[full], axis=2)), "Calibration/live epochs differ"
            assert trials.skipped == sum(bool(quality) for quality in reasons)
            assert trials.rejection_counts == expected_counts, (trials.rejection_counts, expected_counts)
        assert capture.tobytes() == capture_bytes, "Saving changed the caller's raw array"
        legacy = str(Path(tmp) / "legacy")
        legacy_meta = {key: meta[key] for key in ("rate", "eeg_rows", "names", "marker_row", "frequencies", "letters")}
        session.save_session(legacy, capture, legacy_meta)
        legacy_raw, _ = session.load_session(legacy)
        assert legacy_raw.tobytes() == capture_bytes, "Legacy raw inspection no longer works"
        requires_error(lambda: session.load_trials(legacy), "validated signal contract")

    # Reassemble the same raw stream; refit only its first ten seconds, never a chunk.
    counter_fault = next(data for name, data, _, _ in cases if name == "dropped counter")
    for context, expected in ((quiet, Quality.OK), (counter_fault, Quality.COUNTER)):
        replay = np.concatenate((baseline_raw, context), axis=1)
        replay[0, baseline_n:] = (replay[0, baseline_n:] + baseline_n) % 256
        replay[20, baseline_n:] += baseline_n / rate
        onset = baseline_n + hist
        replay[21, onset] = 11
        replay_bytes = replay.tobytes()
        replay_meta = {**meta, "trials": {"11": annotation(1)}}
        reference, reference_quality = session.epoch_at(replay, onset, replay_meta)
        assert reference_quality == expected
        for size in (1, 3, 37, replay.shape[1]):
            assembled = np.empty_like(replay)
            chunk_meta = {**replay_meta, "quality_baseline": None}
            for start in range(0, replay.shape[1], size):
                end = min(start + size, replay.shape[1])
                assembled[:, start:end] = replay[:, start:end]
                if chunk_meta["quality_baseline"] is None and end >= baseline_n:
                    chunk_meta["quality_baseline"] = fit_baseline(assembled[:, :end], chunk_meta)
                if end > onset:
                    epoch, quality = session.epoch_at(assembled[:, :end], onset, chunk_meta)
                    if end < onset + flick:
                        assert epoch is None and quality == Quality.INCOMPLETE
            assert quality == reference_quality, (size, quality, reference_quality)
            assert epoch is None if reference is None else np.array_equal(epoch, reference), size
            assert assembled.tobytes() == replay_bytes and replay.tobytes() == replay_bytes

    marked = quiet.copy()
    marked[21, hist] = -1
    marked_meta = {**meta, "trials": {"-1": annotation(1)}}
    for data, onset in ((marked[:, 1:], hist - 1), (marked[:, :-1], hist)):
        epoch, quality = session.epoch_at(data, onset, marked_meta)
        assert epoch is None and quality == Quality.INCOMPLETE
    epoch, quality = session.epoch_at(marked, hist, {**meta, "trials": {}})
    assert epoch is None and quality == Quality.INCOMPLETE, "Missing completion metadata was admitted"

    # A blink in discarded history still rejects, including its final boundary sample.
    recovery = samples(baseline_n + hist + flick + 1)
    recovery[1, baseline_n] = np.nan
    recovery_meta = {**meta, "trials": {}}
    for trial_id, onset in enumerate((baseline_n + rate, baseline_n + hist, baseline_n + hist + 1), 1):
        recovery[21, onset] = -trial_id
        recovery_meta["trials"][str(-trial_id)] = annotation(trial_id)
        epoch, quality = session.epoch_at(recovery, onset, recovery_meta)
        if trial_id < 3:
            assert epoch is None and quality & Quality.NONFINITE
        else:
            assert quality == Quality.OK and epoch is not None, "A fully clean history did not recover"
            future = np.concatenate((recovery, np.full((22, rate), np.nan)), axis=1)
            again, quality = session.epoch_at(future, onset, recovery_meta)
            assert quality == Quality.OK and np.array_equal(again, epoch), "Samples after flicker affected the epoch"

    def trial_set(targets, blocks):
        return session.Trials(np.repeat(expected_epochs[False][0][:, :, None], len(targets), axis=2),
                              np.array(targets), np.array(blocks), rate, meta["names"],
                              meta["frequencies"], meta["letters"], rows)

    requires_error(lambda: fit(trial_set([0, 1, 1], [1, 1, 2])), meta["letters"][0], "1")
    requires_error(lambda: fit(trial_set([0, 0], [1, 2])), meta["letters"][1], "0")
    requires_error(lambda: cross_validate(trial_set([0, 1, 0, 1], [1, 1, 2, 2])), "3", "2")

    def no_classification(train, test):
        raise AssertionError("A deficient CV training fold reached the classifier")

    curated = trial_set([0, 1, 0, 1, 0], [1, 1, 2, 2, 3])
    for classify in (None, no_classification):
        requires_error(lambda: cross_validate(curated, classify=classify), "block 1", meta["letters"][1], "1")

    # Populate the worker's real locked result channel without starting a process.
    recorder = RecordingProcess()

    def publish(trial_id, target, quality):
        with recorder.prediction_trial.get_lock():
            recorder.last_prediction.value = target
            recorder.last_quality.value = int(quality)
            recorder.prediction_trial.value = trial_id

    try:
        assert recorder.poll_prediction(1) is None
        text = ""
        for wanted, published, target, quality in ((1, 1, 0, Quality.OK), (2, 2, -1, Quality.MOTION),
                                                    (3, 2, 1, Quality.OK), (4, 4, 0, Quality.OK)):
            publish(published, target, quality)
            result = recorder.poll_prediction(wanted)
            if result is not None and result[0] is not None:
                text += meta["letters"][result[0]]
            if wanted == 2:
                assert result == (None, Quality.MOTION), "Expected rejection sentinel became INVALID_RESULT"
            if wanted == 3:
                assert result is None, "Stale B completed a different selection"
        assert text == "AA", f"Accepted/rejected/stale/repeated sequence produced {text!r}"
        for quality in (*Quality, Quality.NONFINITE | Quality.MOTION):
            publish(5, 0, quality)
            assert recorder.poll_prediction(5) == (None, quality), quality
        for target in (-1, cfg.N_TARGETS):
            publish(6, target, Quality.OK)
            assert recorder.poll_prediction(6) == (None, Quality.INVALID_RESULT), target
        publish(7, 1, Quality.OK)
        assert recorder.poll_prediction(7) == (1, Quality.OK)
        assert recorder.poll_prediction(0) is None
    finally:
        recorder.stop()
        recorder._events.close()
        recorder.close()  # Unstarted: no board, child, or join is involved.

    print(f"signal_quality self-check OK: {len(cases)} admission fixtures, chunk replay, raw roundtrip, TRCA guards, results AA")
    for mains in (50, 60):
        print(f"  {mains} Hz notch gains: " + ", ".join(f"{frequency} Hz={gains[mains][frequency]:.6f}" for frequency in frequencies))
    print("  Numerical/fixture evidence only; no headset, optical timing, or human decoding verified.")


def _report_session(path: str) -> None:
    """Inspect raw samples without repairing them or inventing a signal contract."""
    from collections import Counter

    from . import session
    from .diagnostics import NFFT, trial_spectra
    from .knight import stream

    data, meta = session.load_session(path)
    rate, rows = meta.get("rate"), meta.get("eeg_rows")
    if not _positive(rate) or rate != int(rate):
        raise ValueError("Session rate must be a positive integer nominal sampling rate.")
    rate = int(rate)
    if (not isinstance(rows, list) or not rows
            or any(not _integer(row) or not 0 <= row < data.shape[0] for row in rows)
            or len(set(rows)) != len(rows)):
        raise ValueError("Session eeg_rows must select unique rows present in raw.npz.")
    names = meta.get("names", [f"row {row}" for row in rows])
    if (not isinstance(names, list) or len(names) != len(rows)
            or any(not isinstance(name, str) or not name.strip() for name in names)):
        raise ValueError("Session names must label every selected EEG row.")
    factor, evidence = meta.get("eeg_uv_per_sdk_unit"), meta.get("eeg_scale_evidence")
    if factor is not None or evidence is not None:
        if not _positive(factor) or not isinstance(evidence, str) or not evidence.strip():
            raise ValueError("EEG scale needs a positive finite factor and nonempty evidence together.")
    off = meta.get("leadoff_off_bit")
    if off is not None and (not _integer(off) or off not in (0, 1)):
        raise ValueError("Session leadoff_off_bit must be null, 0 or 1.")

    n = data.shape[1]
    knight = meta.get("board_id") == 66 and data.shape[0] == 22 and all(1 <= row <= 8 for row in rows)
    layout = {key: meta.get(key) for key in ("packet_row", "timestamp_row", "marker_row", "imu_rows")}
    inferred = []
    if knight:
        for key, value in (("packet_row", 0), ("timestamp_row", 20), ("marker_row", 21), ("imu_rows", _IMU_ROWS)):
            if layout[key] is None:
                layout[key] = value
                inferred.append(f"{key}={value}")
    for key in ("packet_row", "timestamp_row", "marker_row"):
        row = layout[key]
        if row is not None and (not _integer(row) or not 0 <= row < data.shape[0]):
            raise ValueError(f"Session {key} must identify a row present in raw.npz.")
    imu_rows = layout["imu_rows"] if layout["imu_rows"] is not None else {}
    if not isinstance(imu_rows, dict):
        raise ValueError("Session imu_rows must map IMU groups to row lists.")
    for group in _IMU_ROWS:
        group_rows = imu_rows.get(group)
        if group_rows is not None and (not isinstance(group_rows, list) or len(group_rows) != 3
                or any(not _integer(row) or not 0 <= row < data.shape[0] for row in group_rows)
                or len(set(group_rows)) != 3):
            raise ValueError(f"Session {group} rows must identify three distinct recorded axes.")

    def numbers(values):
        return "[" + ", ".join(f"{value:.5g}" if np.isfinite(value) else "n/a" for value in values) + "]"

    def finite_range(values):
        finite = values[np.isfinite(values)]
        return numbers((finite.min(), finite.max())) if finite.size else "n/a"

    def rejection_report(label, total, accepted, reasons):
        coverage = f" ({100 * accepted / total:.1f}% accepted)" if total else ""
        print(f"  {label}: {accepted} accepted, {total - accepted} rejected / {total}{coverage}")
        print("    rejection counts: " + (", ".join(f"{key}={value}" for key, value in sorted(reasons.items())) or "none"))

    print(f"Session: {path}\n  Raw: {data.shape[0]} rows x {n} samples; {n / rate:.3f} s at nominal {rate} Hz")
    print(f"  Schema: {meta.get('schema_version', 'legacy/unrecorded')}; board: {meta.get('board_id', 'unrecorded')}")
    print(f"  EEG order: " + ", ".join(f"{name}(row {row})" for name, row in zip(names, rows)))
    print(f"  Recorded BrainFlow: {meta.get('brainflow_version', 'unrecorded')}; gain requested: {meta.get('gain_requested', 'unrecorded')}")
    if inferred:
        print("  Diagnostic-only Knight SDK layout inference: " + "; ".join(inferred))
        print("  Inferred rows are not written to metadata and do not establish the validated signal contract.")
    print(f"\nScale: raw unit={meta.get('raw_eeg_unit', 'unrecorded')}; model unit={meta.get('model_eeg_unit', 'unrecorded')}")
    print("  Native amplitudes below are SDK units; physical EEG scale is not inferred from amplitude.")
    if factor is None:
        print("  No evidenced SDK-to-µV conversion recorded; µV reporting disabled.")
    else:
        print(f"  Optional reporting factor: {factor:.9g} µV/SDK unit; supplied evidence: {evidence}")
        print("  This is recorded provenance, not an independent physical verification; raw/model values stay unchanged.")

    baseline, baseline_error = None, "no stored baseline"
    if meta.get("quality_baseline") is not None:
        try:
            baseline = validate_baseline(meta)
            _raw_data(data)
        except ValueError as error:
            baseline, baseline_error = None, str(error)
    print(f"\nBaseline recorded error: {meta.get('quality_baseline_error', 'unrecorded')!r}")
    if baseline is None:
        print(f"  Admission unavailable: {baseline_error}; raw measurements only.")
    else:
        print(f"  {baseline['version']}: {baseline['samples']} samples ({baseline['samples'] / rate:.3f} s), "
              f"window={baseline['window_samples']} samples, hop={baseline['hop_samples']} samples "
              f"({baseline['hop_samples'] / rate:.3f} s), multiplier={baseline['multiplier']:g}")

    print("\nCounter / host timebase (not a hardware sampling-clock measurement):")
    if layout["packet_row"] is None:
        print("  Counter: row unavailable; continuity not assessed.")
    else:
        counter = np.asarray(data[layout["packet_row"]], dtype=float)
        if knight:
            valid = np.isfinite(counter) & (counter >= 0) & (counter <= 255) & (counter == np.floor(counter))
            pairs = valid[:-1] & valid[1:]
            with np.errstate(invalid="ignore"):
                delta = np.remainder(np.diff(counter), 256)
            print(f"  Counter row {layout['packet_row']}: invalid={np.count_nonzero(~valid)}, "
                  f"discontinuities={np.count_nonzero(pairs & (delta != 1))}/{max(0, n - 1)}, "
                  f"duplicates={np.count_nonzero(pairs & (delta == 0))}; modulo-256, 255→0 is valid.")
        else:
            print(f"  Counter row {layout['packet_row']}: range={finite_range(counter)}; "
                  "counter modulus unverified for this board, continuity not assessed.")
    if layout["timestamp_row"] is None:
        print("  Observed host-frame rate unavailable: timestamp row unrecorded.")
    else:
        times = np.asarray(data[layout["timestamp_row"]], dtype=float)
        with np.errstate(invalid="ignore", over="ignore"):
            gaps = np.diff(times)
        span = float(times[-1] - times[0]) if n >= 2 and np.isfinite(times[[0, -1]]).all() else float("nan")
        observed = (n - 1) / span if np.isfinite(span) and span > 0 else float("nan")
        print(f"  Timestamp kind: {meta.get('timestamp_kind', 'unrecorded; host-parse interpretation is diagnostic-only')}")
        print(f"  Nominal={rate} Hz; observed endpoint host-frame rate={numbers([observed])} Hz; span={numbers([span])} s")
        print(f"  Timestamp faults: nonfinite={np.count_nonzero(~np.isfinite(times))}, "
              f"nonpositive={np.count_nonzero(np.isfinite(times) & (times <= 0))}, "
              f"nonincreasing pairs={np.count_nonzero(np.isfinite(gaps) & (gaps <= 0))}, "
              f"arrival gaps >0.1 s={np.count_nonzero(np.isfinite(gaps) & (gaps > 0.1))}; "
              f"gap range={finite_range(gaps)} s")
        coarse = ("outside ±5%" if not np.isfinite(observed) or not 0.95 * rate <= observed <= 1.05 * rate else "within ±5%")
        print(f"  Whole-record coarse rate check: {coarse if n >= 5 * rate else 'not assessed: fewer than five nominal seconds'}.")

    eeg = np.asarray(data[rows], dtype=float)
    driver_rails = knight and meta.get("brainflow_version", "5.23.0") == "5.23.0" and meta.get("raw_eeg_unit", "brainflow_native") == "brainflow_native"
    print(f"\nEEG: finite-value ranges and every contiguous {rate}-sample (1.000 s) window; counts overlap.")
    if driver_rails:
        print(f"  Driver-reference step={SDK_STEP:.9g} SDK units; rails at 99.5% of [-32768,32767] steps; flat <{2 * SDK_STEP:.9g} SDK units.")
        if meta.get("brainflow_version") is None or meta.get("raw_eeg_unit") is None:
            print("  Rail/flat thresholds assume the 5.23.0 native Knight numeric scale for diagnostics only; legacy scaling is unrecorded.")
    else:
        print("  Driver numeric scale unavailable: rail/flat thresholds not applied.")
    for index, (name, row) in enumerate(zip(names, eeg)):
        with np.errstate(invalid="ignore", over="ignore"):
            ptp = np.ptp(np.lib.stride_tricks.sliding_window_view(row, rate), axis=-1) if n >= rate else np.empty(0)
            steps = np.abs(np.diff(row))
        finite_ptp = ptp[np.isfinite(ptp)]
        rail_count = np.count_nonzero(np.isfinite(row) & ((row <= -32768 * SDK_STEP * 0.995) | (row >= 32767 * SDK_STEP * 0.995))) if driver_rails else "n/a"
        flat_count = np.count_nonzero(ptp < 2 * SDK_STEP) if driver_rails else "n/a"
        print(f"  {name}: nonfinite={np.count_nonzero(~np.isfinite(row))}/{n}, range={finite_range(row)} SDK units, "
              f"rail samples={rail_count}; flat windows={flat_count}/{ptp.size}, finite windows={finite_ptp.size}/{ptp.size}")
        print(f"    one-second ptp range={finite_range(ptp)}; adjacent |step| range={finite_range(steps)} SDK units")
        if factor is not None and finite_ptp.size:
            print(f"    optional one-second ptp range={numbers([finite_ptp.min() * factor, finite_ptp.max() * factor])} µV (supplied scale)")
        if baseline is not None:
            ptp_limit = baseline["multiplier"] * baseline["eeg_ptp_ref"][index]
            step_limit = baseline["multiplier"] * baseline["eeg_step_ref"][index]
            print(f"    burst limits ptp={ptp_limit:.5g}, |step|={step_limit:.5g} SDK units; "
                  f"above-limit windows={np.count_nonzero(ptp > ptp_limit)}/{ptp.size}, "
                  f"steps={np.count_nonzero(steps > step_limit)}/{steps.size}")
    if baseline is None:
        print("  Relative EEG burst thresholds unavailable without a validated stored baseline.")

    print("\nRaw P/N status: " + ("polarity unverified; no contact inference/gating" if off is None else f"recorded off_bit={off}; configured polarity, not re-verified here"))
    if not knight:
        print("  P/N row layout unavailable for this board.")
    else:
        for label, row_index in (("P", 9), ("N", 10)):
            status = np.asarray(data[row_index], dtype=float)
            valid = np.isfinite(status) & (status >= 0) & (status <= 255) & (status == np.floor(status))
            raw = status[valid].astype(np.uint8)
            values, counts = np.unique(raw, return_counts=True)
            distribution = ", ".join(f"0x{value:02x}:{count}" for value, count in zip(values[:16], counts[:16])) or "none"
            if values.size > 16:
                distribution += f", ... ({values.size} distinct bytes)"
            bits = (raw[None, :] >> (np.asarray(rows)[:, None] - 1)) & 1
            print(f"  {label} row {row_index}: raw byte:count {{{distribution}}}; invalid={n - raw.size}")
            print(f"    bit=1 counts in EEG row order: {numbers(bits.sum(axis=1))}/{raw.size} valid status samples")
            if off is not None:
                print(f"    any selected bit matching off={off}: {np.count_nonzero(np.any(bits == off, axis=0))}/{raw.size}")

    print("\nIMU: firmware values; documented units are not physically verified. Zero/constant data do not prove sensor freshness.")
    imu = {}
    for group, unit in (("accel", "m/s^2"), ("gyro", "rad/s"), ("mag", "µT")):
        group_rows = imu_rows.get(group)
        if group_rows is None:
            print(f"  {group}: row layout unavailable.")
            continue
        values = imu[group] = np.asarray(data[group_rows], dtype=float)
        finite_frames = np.isfinite(values).all(axis=0)
        finite_axes = [axis[np.isfinite(axis)] for axis in values]
        medians = [np.median(axis) if axis.size else float("nan") for axis in finite_axes]
        ranges = [np.ptp(axis) if axis.size else float("nan") for axis in finite_axes]
        with np.errstate(invalid="ignore", over="ignore"):
            movement = np.linalg.norm(np.diff(values, axis=1), axis=0)
        documented = unit if knight else "unrecorded for this board"
        print(f"  {group} rows={group_rows}; documented unit={documented} (not verified); "
              f"finite frames={np.count_nonzero(finite_frames)}/{n}, all-zero frames={np.count_nonzero(np.all(values == 0, axis=0))}/{n}")
        print(f"    axis median={numbers(medians)}, ptp={numbers(ranges)}; ||adjacent delta|| range={finite_range(movement)} firmware units")
    if "accel" in imu and "gyro" in imu:
        unavailable = ~np.isfinite(imu["accel"]).all(axis=0) | ~np.isfinite(imu["gyro"]).all(axis=0)
        unavailable |= np.all(imu["accel"] == 0, axis=0) & np.all(imu["gyro"] == 0, axis=0)
        print(f"  Motion-information unavailable frames (nonfinite or jointly zero accel+gyro): {np.count_nonzero(unavailable)}/{n}")
        if baseline is not None:
            gyro_level, accel_step = _motion_levels(imu["accel"], imu["gyro"], np.asarray(baseline["gyro_center"]))
            print(f"  Stored gyro center={numbers(baseline['gyro_center'])}")
            for label, levels, reference in (("gyro deviation", gyro_level, baseline["gyro_noise_ref"]), ("accel delta", accel_step, baseline["accel_step_ref"])):
                limit = baseline["multiplier"] * reference
                print(f"  {label}: range={finite_range(levels)}, reference={reference:.5g}, limit={limit:.5g}; above limit={np.count_nonzero(levels > limit)}/{levels.size}")
    print("  Magnetometer is reported only; it is excluded from admission and the EEG model.")

    preprocessing = meta.get("preprocessing")
    if preprocessing is None:
        hist, flick = session.history_samples(rate), session.flicker_samples(rate)
        mains = stream.MAINS_HZ
        print("\nPreprocessing unrecorded: lengths/filter below use current diagnostic-only settings, not an invented historical contract.")
    else:
        if (not isinstance(preprocessing, dict) or not _positive(preprocessing.get("history_s"))
                or not _positive(preprocessing.get("flicker_s")) or preprocessing.get("mains_hz") not in (50, 60)):
            raise ValueError("Session preprocessing needs positive history_s/flicker_s and mains_hz=50 or 60.")
        hist = int(round(preprocessing["history_s"] * rate))
        flick = int(round(preprocessing["flicker_s"] * rate))
        mains = preprocessing["mains_hz"]
    context = hist + flick
    if hist < 1 or flick < 1:
        raise ValueError("History and flicker lengths must each span at least one sample.")
    print(f"\nRaw admission contexts: history={hist} samples ({hist / rate:.3f} s) + flicker={flick} samples ({flick / rate:.3f} s) "
          f"= {context} samples ({context / rate:.3f} s); stride={rate} samples (1 s).")
    starts = range(0, max(0, n - context + 1), rate)
    covered = context + (len(starts) - 1) * min(context, rate) if starts else 0
    print(f"  Available full positions={len(starts)}; candidate raw-sample union={covered}/{n} "
          f"({100 * covered / n if n else 0:.1f}%); uncovered tail={n - (starts[-1] + context) if starts else n} samples.")
    if starts:
        print(f"  Context starts at samples {starts[0]}..{starts[-1]}; corresponding onset positions {hist}..{starts[-1] + hist}. Baseline samples are not excluded.")
    if baseline is None:
        print("  Contexts not assessed: no validated stored baseline; no baseline is fitted from task data.")
    else:
        accepted, reasons = 0, Counter()
        for start in starts:
            quality = assess_window(data[:, start:start + context], meta)
            accepted += quality == Quality.OK
            reasons.update(reason.name for reason in Quality if quality & reason)
        rejection_report("Raw contexts (no stimulus-timing claim)", len(starts), accepted, reasons)

    if layout["marker_row"] is None:
        print("\nMarked trials unavailable: marker row unrecorded.")
    else:
        markers = data[layout["marker_row"]]
        onsets = np.flatnonzero(markers)
        print(f"\nMarked trials: {onsets.size} nonzero raw marker frames (including invalid values).")
        if onsets.size and baseline is not None and preprocessing == session.preprocessing_meta() and isinstance(meta.get("trials"), dict):
            accepted, reasons = 0, Counter()
            targets = {label: [0, 0] for label in meta.get("letters", [])}
            epoch_lengths = set()
            for onset in onsets:
                epoch, quality = session.epoch_at(data, int(onset), meta)
                admitted = epoch is not None and quality == Quality.OK
                accepted += admitted
                if admitted:
                    epoch_lengths.add(epoch.shape[0])
                reasons.update(reason.name for reason in Quality if quality & reason)
                decoded = session.decode_marker(markers[onset], len(meta.get("letters", [])))
                label = meta["letters"][decoded[1]] if decoded is not None else "live/unlabelled"
                counts = targets.setdefault(label, [0, 0])
                counts[0 if admitted else 1] += 1
            rejection_report("epoch_at admission", onsets.size, accepted, reasons)
            for label, (good, rejected) in targets.items():
                print(f"    {label}: {good} accepted, {rejected} rejected")
            print("  Admitted analysis lengths: " + (", ".join(f"{count} samples ({count / rate:.3f} s)" for count in sorted(epoch_lengths)) or "none"))
            print("  These are epoch admission counts, not model readiness, target accuracy, or intention detection.")
        elif onsets.size:
            print("  Trial admission unavailable: requires a valid stored baseline, trial completion annotations, and matching current preprocessing.")
        annotations = meta.get("trials")
        if isinstance(annotations, dict):
            observed_codes = {str(int(code)) for code in markers[onsets] if np.isfinite(code) and code == int(code)}
            print(f"  Requested annotations without a raw marker: {len(annotations.keys() - observed_codes)} (not counted as observed trials).")
        print("  Rejection reasons can overlap; reason totals need not equal rejected trials/contexts.")

    length = min(NFFT, n - hist)
    print("\nRaw versus filtered spectral view (first post-history segment, not a whole-record or admission summary):")
    if length < 2 or rate <= 80 or not np.isfinite(eeg[:, :hist + max(0, length)]).all():
        print("  Unavailable: need finite EEG, full filter history, at least two remaining samples, and Nyquist above 40 Hz.")
    else:
        print(f"  Matching raw/filtered samples [{hist}:{hist + length}): {length} samples, T={length / rate:.3f} s; "
              f"1/T={rate / length:.4g} Hz; Hann window, FFT grid={rate / NFFT:.4g} Hz ({NFFT} points).")
        print(f"  Zero-padding adds no frequency resolution. Filter: notch {mains} Hz, causal fourth-order 1–40 Hz; "
              f"discarded history={hist} samples. Band shares are relative power, not calibrated PSD.")
        filtered = stream.clean(eeg[:, :hist + length], rate, mains_hz=mains)[:, hist:]
        for label, values in (("raw", eeg[:, hist:hist + length]), ("filtered", filtered)):
            centered = values - values.mean(axis=1, keepdims=True)
            freqs, power = trial_spectra(centered.T[:, :, None], rate)
            power = power[:, :, 0]
            usable = freqs >= 0.5
            total = power[usable].sum(axis=0)
            peak = freqs[usable][np.argmax(power[usable], axis=0)]
            peak[total == 0] = np.nan
            mains_power = power[np.abs(freqs - mains) <= 2].sum(axis=0)
            share = np.divide(100 * mains_power, total, out=np.full(total.shape, np.nan), where=total > 0)
            rms = np.sqrt(np.mean(centered ** 2, axis=1))
            print(f"  {label}: AC RMS={numbers(rms)} SDK units; peak={numbers(peak)} Hz; {mains}±2 Hz share={numbers(share)}% (EEG row order)")
            if factor is not None:
                print(f"    optional AC RMS={numbers(rms * factor)} µV (supplied scale)")
    print("\nThis report does not verify physical EEG scale, sensor freshness, optical timing, denoising efficacy, or human decoding.")


def main() -> None:
    import argparse
    from zipfile import BadZipFile

    parser = argparse.ArgumentParser(description="Read-only raw-session diagnostics; no board or display is opened.")
    parser.add_argument("session", nargs="?", metavar="SESSION", help="directory containing session.json and raw.npz")
    parser.add_argument("--self-check", action="store_true", help="run deterministic numerical admission checks")
    args = parser.parse_args()
    if bool(args.session) == args.self_check:
        parser.error("pass SESSION or --self-check, not both")
    if args.self_check:
        self_check()
        return
    try:
        _report_session(args.session)
    except (OSError, ValueError, TypeError, KeyError, IndexError, EOFError, BadZipFile, ImportError) as error:
        parser.error(f"Cannot inspect {args.session!r}: {error}")


if __name__ == "__main__":
    main()
