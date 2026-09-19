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

# BrainFlow 5.23 does not tag this board's IMU rows, so we take them from
# NeuroPawn's docs. Row numbers are indices into the data array below.
LOFF_P, LOFF_N = 9, 10    # lead-off status bitmasks, one bit per channel
ACCEL = [11, 12, 13]
GYRO = [14, 15, 16]

# Electrode montage, board channel 1-8 in order. Occipital / parieto-occipital
# for SSVEP: the visual cortex sits at the back of the head, so this is where a
# flicker response actually shows up.
NAMES = ["Oz", "O1", "O2", "PO7", "PO8", "PO3", "PO4", "POz"]

# chon   - channel on
# rldadd - right leg drive add
#
# Gain MUST stay 12: brainflow's knight_imu.cpp hardcodes 12 into its
# counts->microvolts scale factor, so any other gain gives wrong uV.
GAIN = 12


def send_cmd(board, cmd):
    """Send one config command and describe what came back.

    config_board writes the command first, THEN tries to read a reply. Once a
    channel is on, the Knight free-runs binary frames, so that "reply" is EEG
    packets and brainflow's .decode('utf-8') raises. The write already
    succeeded, so a decode error means sent-and-board-is-streaming, not failed.

    This firmware never acks. "quiet" means nothing was in the read buffer and
    "streaming" means we caught it mid-frame -- neither reports success.
    """
    try:
        r = board.config_board(cmd)
        return repr(r) if r else "quiet"
    except UnicodeDecodeError:
        return "sent (board streaming)"


def enable_eeg_channels(board, channels):
    """Knight EEG channels ship disabled - firmware needs these two commands each.

    Called BEFORE start_stream(), so the acquisition thread is not yet trying
    to parse 0xA0/0xC0 frames while these commands and the board's reply bytes
    are in flight. start_stream() then flushes the serial buffer for us.
    """
    for ch in channels:
        r1 = send_cmd(board, f"chon_{ch}_{GAIN}")   # channel on, at this gain
        time.sleep(0.3)
        r2 = send_cmd(board, f"rldadd_{ch}")        # tie into the bias/reference loop
        time.sleep(0.3)
        print(f"  channel {ch}: chon -> {r1} | rldadd -> {r2}")


def clean(win, rate):
    """Strip mains hum and out-of-band junk. Filters run in-place, per channel."""
    out = np.ascontiguousarray(win, dtype=np.float64)
    for ch in out:
        DataFilter.remove_environmental_noise(ch, rate, NoiseTypes.SIXTY.value)
        DataFilter.perform_bandpass(ch, rate, 1.0, 40.0, 4,
                                    FilterTypes.BUTTERWORTH.value, 0.0)
    return out


def ssvep_snr(win, rate, targets):
    """SNR at each stimulus frequency: power in the target bins versus the
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
            parts.append(f"{h} {snr[best]:5.1f}dB@{NAMES[best]}")
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

    # Per channel, not averaged: a channel with bad contact should show a much
    # bigger mains share than its neighbours. That is the impedance test.
    tot = power[:, usable].sum(axis=1)
    tot[tot == 0] = 1e-12
    mains_ch = 100 * power[:, mains].sum(axis=1) / tot
    alpha_ch = 100 * power[:, alpha].sum(axis=1) / tot

    # Normalise each channel before averaging, otherwise one drifting
    # electrode at 200k rms decides the "dominant" frequency for all of them.
    norm = power[:, usable] / tot[:, None]
    dom = freqs[usable][np.argmax(norm.mean(axis=0))]
    return ("mains%: " + " ".join(f"{v:3.0f}" for v in mains_ch)
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
    ap.add_argument("--filter", action="store_true", help="notch 60 Hz + bandpass 1-40 Hz")
    ap.add_argument("--ssvep", default="",
                    help="stimulus frequencies to score, e.g. 12,15")
    ap.add_argument("--window", type=int, default=256,
                    help="rolling samples for the fft (256=2s, 512=4s sharper)")
    ap.add_argument("--channels", default="",
                    help="only enable/show these, e.g. 2,3,6,7 (default: all 8)")
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

    eeg_rows = BoardShim.get_board_descr(board_id)["eeg_channels"]
    if args.channels:
        # Enabling only the good electrodes also takes the bad ones OUT of the
        # bias loop, which is the other thing worth testing.
        keep = {int(c) for c in args.channels.split(",")}
        eeg_rows = [r for r in eeg_rows if r in keep]
    rate = BoardShim.get_sampling_rate(board_id)
    print(f"board={BoardShim.get_board_descr(board_id)['name']}  {rate} Hz  eeg rows={eeg_rows}")

    targets = [float(f) for f in args.ssvep.split(",")] if args.ssvep else []
    labels = [NAMES[r - 1] if r - 1 < len(NAMES) else f"ch{r}" for r in eeg_rows]
    print("            " + " ".join(f"{n:>8}" for n in labels))

    board = BoardShim(board_id, params)
    board.prepare_session()       # 2. open the serial connection
    try:
        if args.real:             # 3. enable channels while NOT streaming
            # Opening the port resets the MCU, so it is still printing boot
            # chatter for a moment. The good run lost chon_1 to exactly this
            # and channel 1 read 0.0 for the whole session.
            print(f"waiting {args.settle}s for boot chatter to finish...")
            time.sleep(args.settle)
            print("enabling EEG channels...")
            enable_eeg_channels(board, eeg_rows)

        board.start_stream()      # 4. flushes serial, then samples flow into a ring buffer
        empty = 0
        roll = None               # rolling window, needed for any useful fft

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
                line += "  " + ssvep_snr(shown, rate, targets)
            if args.real:
                acc = " ".join(f"{newest[r]:6.2f}" for r in ACCEL)
                line += f" | loff P{leadoff(newest[LOFF_P])} N{leadoff(newest[LOFF_N])}"
                line += f" | accel: {acc}"
            print(line)
    finally:
        # 6. Always release, or the serial port stays locked. stop_stream can
        # raise if we never got that far, and that would mask the real error.
        try:
            board.stop_stream()
        except Exception:
            pass
        board.release_session()


if __name__ == "__main__":
    main()
