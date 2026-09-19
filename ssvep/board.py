"""
NeuroPawn Knight IMU board: connect, power up channels, read the ring buffer.

Every EEG channel is powered down when BrainFlow connects, so each one must be
switched on (`chon_{ch}_{gain}`) and added to the right-leg-drive noise
cancellation loop (`rldadd_{ch}`) after streaming starts. The firmware drops
commands sent too close together, so we use NeuroPawn's documented pauses:
~4 s per channel, ~34 s for all 8.

BrainFlow row layout for the Knight IMU (from BrainFlow's knight_imu.cpp):
    1-8   EEG (uV)
    9-10  lead-off status P / N
    11-13 accel x/y/z (m/s^2)
    14-16 gyro x/y/z
    17-19 magnetometer x/y/z
    20    timestamp
    21    marker
"""

from __future__ import annotations

import time

from brainflow.board_shim import BoardIds, BoardShim, BrainFlowInputParams

from . import config as cfg

ACCEL_ROWS = [11, 12, 13]
GYRO_ROWS = [14, 15, 16]
MAG_ROWS = [17, 18, 19]

SETUP_SECONDS_PER_CHANNEL = 4


def enable_channels(board_shim: BoardShim, channels, gain: int = cfg.CHANNEL_GAIN, log=print):
    """Power on each channel and add it to RLD. Call after start_stream()."""
    time.sleep(2)  # let the stream settle before sending config commands
    for ch in channels:
        time.sleep(1)
        board_shim.config_board(f"chon_{ch}_{gain}")
        time.sleep(2)
        board_shim.config_board(f"rldadd_{ch}")
        time.sleep(1)
        log(f"  enabled channel {ch}")


class KnightBoard:
    """Connect to, configure, and stream from a Knight IMU board (or BrainFlow's synthetic board)."""

    def __init__(self, serial_port: str | None, num_channels: int = cfg.NUM_CHANNELS,
                 gain: int = cfg.CHANNEL_GAIN, synthetic: bool = False):
        self.num_channels = num_channels
        self.gain = gain
        self.synthetic = synthetic

        params = BrainFlowInputParams()
        if synthetic:
            board_id = BoardIds.SYNTHETIC_BOARD
        else:
            board_id = cfg.BOARD_ID
            params.serial_port = serial_port

        BoardShim.disable_board_logger()
        self.board_shim = BoardShim(board_id, params)
        self.board_id = board_id
        self.eeg_channels = BoardShim.get_eeg_channels(board_id)[:num_channels]
        self.sr = BoardShim.get_sampling_rate(board_id)

    def start_stream(self, buffer_size: int = 450_000) -> None:
        """Open the session, start streaming, and power up every channel."""
        self.board_shim.prepare_session()
        self.board_shim.start_stream(buffer_size)
        if self.synthetic:
            print("Synthetic board streaming (no channel setup needed).")
            return
        print(f"Stream started - configuring channels (~{2 + SETUP_SECONDS_PER_CHANNEL * self.num_channels}s)...")
        enable_channels(self.board_shim, range(1, self.num_channels + 1), self.gain)
        print("Board ready.")

    def stop_stream(self) -> None:
        self.board_shim.stop_stream()
        self.board_shim.release_session()
        print("Stream stopped and session released.")

    def get_latest(self, num_samples: int):
        """Latest `num_samples` columns WITHOUT draining the ring buffer."""
        return self.board_shim.get_current_board_data(num_samples)
