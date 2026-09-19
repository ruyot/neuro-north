# Minimal BrainFlow read from the NeuroPawn Knight IMU board.

#    python stream.py               synthetic
#    python stream.py --real        actual
import argparse
import os
import sys
import time

import numpy as np
from brainflow.board_shim import BoardShim, BoardIds, BrainFlowInputParams
from brainflow.data_filter import DataFilter, FilterTypes, NoiseTypes
from dotenv import load_dotenv

load_dotenv()                     # reads .env from the project directory
PORT = os.getenv("PORT_PATH")     # e.g. /dev/cu.usbserial-XXXX


def _load_mains_hz() -> int:
    raw = os.getenv("MAINS_HZ", "60")
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(f"MAINS_HZ must be 50 or 60, got {raw!r}") from None
    if value not in (50, 60):
        raise ValueError(f"MAINS_HZ must be 50 or 60, got {value!r}")
    return value


MAINS_HZ = _load_mains_hz()

# BrainFlow 5.23.0 does not tag this board's IMU rows, so we take them from the
# driver source (knight_imu.cpp): other_channels[0..1] are lead-off, then
# other_channels[2+i] for i=0..8 is ax,ay,az,gx,gy,gz,mx,my,mz. Row numbers are
# indices into the data array below, and they are the Knight's (board id 66)
# ONLY - every other board id lays its IMU out somewhere else, so resolve those
# from BoardShim instead of reaching for these.
LOFF_P, LOFF_N = 9, 10    # lead-off status bitmasks, one bit per channel
ACCEL = [11, 12, 13]
GYRO = [14, 15, 16]
MAG = [17, 18, 19]
# IMU payloads are firmware float32 values passed through without SDK scaling;
# documented m/s^2, rad/s and uT units are not verified on this firmware.

# Electrode montage, board channel 1-8 in order. Occipital / parieto-occipital
# for SSVEP: the visual cortex sits at the back of the head, so this is where a
# flicker response actually shows up.
NAMES = ["Oz", "O1", "O2", "PO7", "PO8", "PO3", "PO4", "POz"]


def name_of(i):
    """Electrode label, or a plain index for boards with a different layout
    (the synthetic board has 16 channels, not 8)."""
    return NAMES[i] if i < len(NAMES) else f"ch{i + 1}"

# chon   - channel on
# rldadd - right leg drive add
#
# Gain MUST stay 12: brainflow's knight_imu.cpp hardcodes it into the
# ADC-count -> SDK-unit scale; electrode-level microvolts remain unverified.
GAIN = 12


def validate_eeg_channels(channels) -> list[int]:
    """Return a nonempty, unique Knight channel selection in board-row order."""
    message = "EEG channels must be a nonempty unique subset of integers 1-8."
    try:
        selected = list(channels)
    except TypeError:
        raise ValueError(message) from None
    if (not selected
            or any(isinstance(ch, bool) or not isinstance(ch, (int, np.integer))
                   or not 1 <= ch <= 8 for ch in selected)
            or len(set(selected)) != len(selected)):
        raise ValueError(message)
    return sorted(int(ch) for ch in selected)


def validate_knight_descriptor(board_id: int) -> dict:
    """Reject SDK layouts that do not match the pinned Knight IMU driver."""
    if board_id != 66:
        raise ValueError(f"Expected Knight IMU board 66, got {board_id}.")
    descriptor = BoardShim.get_board_descr(board_id)
    expected = {
        "num_rows": 22,
        "sampling_rate": 125,
        "package_num_channel": 0,
        "eeg_channels": list(range(1, 9)),
        "other_channels": [LOFF_P, LOFF_N, *ACCEL, *GYRO, *MAG],
        "timestamp_channel": 20,
        "marker_channel": 21,
    }
    for key, value in expected.items():
        if descriptor.get(key) != value:
            raise ValueError(
                f"Knight IMU descriptor mismatch for {key}: "
                f"expected {value!r}, got {descriptor.get(key)!r}."
            )
    return descriptor


def send_cmd(board, cmd):
    """Write while streaming; a successful write is not a firmware acknowledgement."""
    response = board.config_board(cmd)
    time.sleep(1.0)
    return f"write returned {response!r} (no firmware acknowledgement)"


def enable_eeg_channels(board, channels):
    """Apply the complete Knight channel/bias set after start_stream()."""
    channels = validate_eeg_channels(channels)
    for ch in range(1, 9):
        if ch not in channels:
            r1 = send_cmd(board, f"rldremove_{ch}")
            r2 = send_cmd(board, f"choff_{ch}")
            print(f"  channel {ch}: rldremove -> {r1} | choff -> {r2}")
    for ch in channels:
        r1 = send_cmd(board, f"chon_{ch}_{GAIN}")
        r2 = send_cmd(board, f"rldadd_{ch}")
        print(f"  channel {ch}: chon -> {r1} | rldadd -> {r2}")


def clean(win, rate, mains_hz=MAINS_HZ):
    """Strip mains hum and out-of-band junk. Copies input; never aliases it."""
    if mains_hz not in (50, 60):
        raise ValueError(f"mains_hz must be 50 or 60, got {mains_hz!r}")
    noise_type = NoiseTypes.FIFTY.value if mains_hz == 50 else NoiseTypes.SIXTY.value
    out = np.array(win, dtype=np.float64, order="C", copy=True)
    for ch in out:
        DataFilter.remove_environmental_noise(ch, rate, noise_type)
        DataFilter.perform_bandpass(ch, rate, 1.0, 40.0, 4,
                                    FilterTypes.BUTTERWORTH.value, 0.0)
    return out


def cca_reference(f, rate, n, harmonics=2):
    """Pure sin/cos pairs at f and its harmonics. The sin+cos pair lets CCA fit
    any phase, so we never need to know the brain's lag behind the stimulus."""
    t = np.arange(n) / rate
    rows = []
    for h in range(1, harmonics + 1):
        rows.append(np.sin(2 * np.pi * f * h * t))
        rows.append(np.cos(2 * np.pi * f * h * t))
    return np.vstack(rows)


def cca_score(X, Y):
    """Largest canonical correlation between EEG X (ch x n) and reference Y.

    Solved via QR + SVD rather than inverting covariances: same answer, but it
    does not blow up when two channels are nearly identical, which happens
    whenever electrodes share a common-mode signal.
    """
    X = X - X.mean(axis=1, keepdims=True)
    Y = Y - Y.mean(axis=1, keepdims=True)
    qx, _ = np.linalg.qr(X.T)
    qy, _ = np.linalg.qr(Y.T)
    sv = np.linalg.svd(qx.T @ qy, compute_uv=False)
    return float(np.clip(sv[0], 0.0, 1.0))


def cca_decode(win, rate, targets):
    """One score per candidate frequency, plus the argmax -- the actual decision."""
    n = win.shape[1]
    scores = [(f, cca_score(win, cca_reference(f, rate, n))) for f in targets]
    best = max(scores, key=lambda kv: kv[1])
    runner = sorted(s for _, s in scores)[-2] if len(scores) > 1 else 0.0
    margin = best[1] - runner
    return ("cca: " + "  ".join(f"{f:g}Hz {r:.2f}" for f, r in scores)
            + f"  -> {best[0]:g}Hz (margin {margin:+.2f})")


def ssvep_snr(win, rate, targets, labels):
    """SNR at each stimulus frequency, reported against `labels` -- which are
    the SELECTED channels, not all eight. Indexing NAMES directly here silently
    mislabels every channel after an excluded one.

    Power in the target bins versus the
    power in nearby bins. A flicker response is narrow, so comparing it to its
    own neighbourhood cancels out broadband noise and drift."""
    x = win - win.mean(axis=1, keepdims=True)
    x = x * np.hanning(x.shape[1])
    power = np.abs(np.fft.rfft(x, axis=1)) ** 2
    freqs = np.fft.rfftfreq(x.shape[1], 1.0 / rate)

    out = []
    for f in targets:
        parts = []
        for h, mult in (("f", 1), ("2f", 2)):
            fh = f * mult
            if fh >= rate / 2:
                continue
            sig_bins = np.abs(freqs - fh) <= 0.3
            # neighbourhood, skipping a guard band around the peak itself
            nb = (np.abs(freqs - fh) > 0.6) & (np.abs(freqs - fh) < 3.0)
            if not sig_bins.any() or not nb.any():
                continue
            sig = power[:, sig_bins].mean(axis=1)
            noise = power[:, nb].mean(axis=1)
            snr = 10 * np.log10(np.maximum(sig, 1e-12) / np.maximum(noise, 1e-12))
            best = int(np.argmax(snr))
            parts.append(f"{h} {snr[best]:5.1f}dB@{labels[best]}")
        out.append(f"{f:g}Hz[" + " ".join(parts) + "]")
    return " | ".join(out)


def describe_spectrum(win, rate):
    """Where does the energy live? Mains hum and electrode drift look identical
    in an rms number and completely different in a spectrum."""
    x = win - win.mean(axis=1, keepdims=True)
    x = x * np.hanning(x.shape[1])
    power = np.abs(np.fft.rfft(x, axis=1)) ** 2
    freqs = np.fft.rfftfreq(x.shape[1], 1.0 / rate)
    usable = freqs >= 0.5
    mains = (freqs >= 55) & (freqs < 65)
    alpha = (freqs >= 8) & (freqs < 12)
    drift = (freqs >= 0.5) & (freqs < 2)

    # Per channel, not averaged: a channel with bad contact should show a much
    # bigger mains share than its neighbours. That is the impedance test.
    tot = power[:, usable].sum(axis=1)
    tot[tot == 0] = 1e-12
    mains_ch = 100 * power[:, mains].sum(axis=1) / tot
    alpha_ch = 100 * power[:, alpha].sum(axis=1) / tot
    drift_ch = 100 * power[:, drift].sum(axis=1) / tot

    # Normalise each channel before averaging, otherwise one drifting
    # electrode at 200k rms decides the "dominant" frequency for all of them.
    norm = power[:, usable] / tot[:, None]
    dom = freqs[usable][np.argmax(norm.mean(axis=0))]
    return ("mains%: " + " ".join(f"{v:3.0f}" for v in mains_ch)
            + " | drift%: " + " ".join(f"{v:3.0f}" for v in drift_ch)
            + " | alpha%: " + " ".join(f"{v:3.0f}" for v in alpha_ch)
            + f" | dom {dom:5.1f}Hz")


def leadoff(value):
    """Lead-off byte -> per-channel flags. Bit set usually means 'electrode off'."""
    bits = int(value) & 0xFF
    return "".join("X" if bits >> i & 1 else "." for i in range(8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", action="store_true", help="use the headset instead of the fake board")
    ap.add_argument("--seconds", type=float, default=10)
    ap.add_argument("--quiet", action="store_true", help="silence brainflow's own logging")
    ap.add_argument("--settle", type=float, default=3.0, help="pause after connect before configuring")
    ap.add_argument("--fft", action="store_true", help="show where the signal energy actually sits")
    ap.add_argument("--filter", action="store_true", help="notch MAINS_HZ (50/60) + bandpass 1-40 Hz")
    ap.add_argument("--ssvep", default="",
                    help="stimulus frequencies to score, e.g. 12,15")
    ap.add_argument("--window", type=int, default=256,
                    help="rolling samples for the fft (256=2s, 512=4s sharper)")
    ap.add_argument("--channels",
                    help="only enable/show these, e.g. 2,3,6,7 (default: all EEG channels)")
    args = ap.parse_args()

    if args.quiet:
        BoardShim.disable_board_logger()

    # 1. Say which board and how to reach it.
    params = BrainFlowInputParams()
    if args.real:
        if not PORT:
            sys.exit("PORT_PATH is not set -- add it to .env (find it with: ls /dev/cu.*)")
        board_id = int(BoardIds.NEUROPAWN_KNIGHT_BOARD_IMU)
        params.serial_port = PORT
        print(f"port={PORT}")
    else:
        board_id = int(BoardIds.SYNTHETIC_BOARD)

    descriptor = (validate_knight_descriptor(board_id) if args.real
                  else BoardShim.get_board_descr(board_id))
    try:
        requested = ([int(c) for c in args.channels.split(",")] if args.channels is not None
                     else descriptor["eeg_channels"])
        eeg_rows = (validate_eeg_channels(requested) if args.real
                    else [r for r in descriptor["eeg_channels"] if r in requested])
    except ValueError as exc:
        ap.error(str(exc))
    rate = descriptor["sampling_rate"]
    print(f"board={descriptor['name']}  {rate} Hz  eeg rows={eeg_rows}")

    targets = [float(f) for f in args.ssvep.split(",")] if args.ssvep else []
    labels = [name_of(r - 1) for r in eeg_rows]
    print("            " + " ".join(f"{n:>8}" for n in labels))

    board = BoardShim(board_id, params)
    try:
        board.prepare_session()   # 2. open the serial connection
        if args.real:
            print(f"waiting {args.settle}s for boot chatter to finish...")
            time.sleep(args.settle)
        board.start_stream()      # host reader must run before channel commands
        if args.real:
            time.sleep(2.0)
            print("applying EEG channel and bias membership...")
            enable_eeg_channels(board, eeg_rows)
            board.get_board_data()  # discard configuration-period samples
        empty = 0
        roll = None               # rolling window, needed for any useful fft
        # the speller's own detector, so this line tests the thing that will type
        gestures = None
        if args.real:
            from ssvep_training.head import Gestures

            gestures = Gestures(rate)

        # 5. Drain the buffer twice a second and print the newest sample.
        end = time.time() + args.seconds
        while time.time() < end:
            time.sleep(0.5)
            data = board.get_board_data()        # shape: (rows, samples)
            if data.shape[1] == 0:
                empty += 1
                print("  ...no samples yet")
                if empty == 6:
                    print("  >> 3s with no data. The board is not framing packets.")
                    print("  >> Unplug/replug the USB-C to reset firmware state, then retry.")
                continue
            empty = 0
            newest = data[:, -1]                 # last column = most recent sample
            # Printing one sample of a 60 Hz-dominated signal every 0.5s just
            # samples random phase, which looks like noise however good the
            # contact is. RMS over the whole window is the honest summary.
            # 0.5s of data only resolves 2 Hz, too coarse to tell 60 Hz from
            # drift, and too short for the filters to settle. Keep ~2s rolling.
            win = data[eeg_rows, :]
            roll = win if roll is None else np.hstack([roll, win])[:, -args.window:]
            shown = clean(roll, rate) if args.filter else roll
            rms = np.sqrt(np.mean((shown - shown.mean(axis=1, keepdims=True)) ** 2, axis=1))
            eeg = " ".join(f"{v:8.0f}" for v in rms)
            tag = "filtered" if args.filter else "rms"
            line = f"[{data.shape[1]:3d}] {tag}: {eeg}"
            if args.fft and roll.shape[1] >= 128:
                line += "  " + describe_spectrum(shown, rate)
            if targets and roll.shape[1] >= 128:
                line += "  " + ssvep_snr(shown, rate, targets, labels)
                line += "  " + cca_decode(shown, rate, targets)
            if args.real:
                acc = " ".join(f"{newest[r]:6.2f}" for r in ACCEL)
                # Peak over the window, not the newest sample: a ~0.3s swipe
                # falls between twice-a-second snapshots. Median-subtracted
                # because a gyro axis sits at a bias that would dwarf it.
                gyr = " ".join(f"{np.abs(data[r, :] - np.median(data[r, :])).max():9.2f}" for r in GYRO)
                line += f" | loff P{leadoff(newest[LOFF_P])} N{leadoff(newest[LOFF_N])}"
                line += f" | accel: {acc}"
                line += f" | gyro pk: {gyr}"
                fired = gestures.feed(data[GYRO])
                if fired:                     # what the speller would have typed
                    line += f"   <<< GESTURE: {fired}"
            print(line)
    finally:
        # 6. Always release, or the serial port stays locked. stop_stream can
        # raise if we never got that far, and that would mask the real error.
        error = sys.exc_info()[1]
        try:
            board.stop_stream()
        except Exception:
            pass
        try:
            board.release_session()
        except Exception as cleanup_error:
            if error is None:
                raise
            print(f"Board release failed during cleanup: {cleanup_error}", file=sys.stderr)


if __name__ == "__main__":
    main()
