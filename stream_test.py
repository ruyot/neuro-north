"""Stream live data from the NeuroPawn Knight IMU board via BrainFlow.

Usage:
    python stream_test.py --port /dev/cu.usbserial-XXXX   # real board
    python stream_test.py --synthetic                     # no hardware needed

Knight IMU row layout (from BrainFlow's knight_imu.cpp):
    1-8   EEG (uV, gain 12)
    9-10  lead-off status P / N
    11-13 accel x/y/z (m/s^2)
    14-16 gyro x/y/z
    17-19 magnetometer x/y/z
"""
import argparse
import time

from brainflow.board_shim import BoardIds, BoardShim, BrainFlowInputParams

ACCEL_ROWS = [11, 12, 13]
GYRO_ROWS = [14, 15, 16]
MAG_ROWS = [17, 18, 19]


def enable_channels(board, channels, gain=12):
    """Knight channels are off by default and must be enabled after start_stream().

    The firmware drops commands sent too close together, so these delays follow
    NeuroPawn's documented startup sequence.
    """
    time.sleep(2)
    for ch in channels:
        time.sleep(1)
        board.config_board(f"chon_{ch}_{gain}")
        time.sleep(2)
        board.config_board(f"rldadd_{ch}")
        time.sleep(1)
        print(f"  enabled channel {ch}")


def fmt(values, width=7, prec=2):
    return " ".join(f"{v:{width}.{prec}f}" for v in values)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", help="serial port, e.g. /dev/cu.usbserial-XXXX")
    parser.add_argument("--synthetic", action="store_true", help="use BrainFlow's fake board")
    parser.add_argument("--seconds", type=float, default=10, help="how long to stream")
    parser.add_argument("--channels", type=int, nargs="+", default=list(range(1, 9)),
                        help="EEG channels to enable (1-8)")
    args = parser.parse_args()

    params = BrainFlowInputParams()
    if args.synthetic:
        board_id = BoardIds.SYNTHETIC_BOARD
    else:
        if not args.port:
            parser.error("--port is required unless --synthetic is set (find it with: ls /dev/cu.*)")
        board_id = BoardIds.NEUROPAWN_KNIGHT_BOARD_IMU
        params.serial_port = args.port

    BoardShim.disable_board_logger()
    descr = BoardShim.get_board_descr(board_id)
    eeg_channels = descr["eeg_channels"]
    print(f"Board: {descr['name']} | {descr['sampling_rate']} Hz")

    board = BoardShim(board_id, params)
    board.prepare_session()
    board.start_stream()
    try:
        if not args.synthetic:
            print(f"Enabling channels {args.channels} (~{4 * len(args.channels) + 2}s)...")
            enable_channels(board, args.channels)
            board.get_board_data()  # drop samples captured during setup

        end = time.time() + args.seconds
        while time.time() < end:
            time.sleep(0.5)
            data = board.get_board_data()  # rows = channels, cols = samples
            if data.shape[1] == 0:
                print("no samples yet...")
                continue
            latest = data[:, -1]
            line = f"[{data.shape[1]:3d}] EEG: {fmt(latest[eeg_channels], 9, 1)}"
            if not args.synthetic:
                line += (f" | acc: {fmt(latest[ACCEL_ROWS])}"
                         f" | gyro: {fmt(latest[GYRO_ROWS])}")
            print(line)
    finally:
        board.stop_stream()
        board.release_session()


if __name__ == "__main__":
    main()
