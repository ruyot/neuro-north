"""
Open and close the Knight IMU board for the recording process

- prepare -> 3 s settle -> start_stream -> 2 s settle -> configure -> drain

"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field

from brainflow.board_shim import BoardIds, BoardShim, BrainFlowInputParams

from . import knight as stream  # stream.py's functions, via the one adapter file

SETTLE_SECONDS = 3.0        # stream.py's --settle default
BUFFER_SIZE = 450_000       # ring buffer samples (~1 h at 125 Hz)
DEFAULT_CHANNELS = range(1, 9)


@dataclass
class Board:
    """A streaming board plus what the rest of the code needs to read it."""
    shim: BoardShim
    board_id: int
    rate: int                                  # 125 Hz
    eeg_rows: list[int]                        # data rows holding EEG, in channel order
    names: list[str] = field(default_factory=list)
    marker_row: int = -1
    packet_row: int = -1
    timestamp_row: int = -1
    imu_rows: dict[str, list[int]] = field(default_factory=dict)

    def close(self) -> None:
        close_board(self)

    def __enter__(self) -> "Board":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def open_board(port: str | None = None, channels=DEFAULT_CHANNELS,
               settle: float = SETTLE_SECONDS, buffer_size: int = BUFFER_SIZE,
               quiet: bool = True) -> Board:
    eeg_rows = stream.validate_eeg_channels(channels)
    board_id = int(BoardIds.NEUROPAWN_KNIGHT_BOARD_IMU)
    descriptor = stream.validate_knight_descriptor(board_id)
    if quiet:
        BoardShim.disable_board_logger()

    port = port or stream.PORT
    if not port:
        raise RuntimeError("PORT_PATH is not set - add it to .env")
    params = BrainFlowInputParams()
    params.serial_port = port
    shim = BoardShim(board_id, params)
    board = Board(
        shim=shim, board_id=board_id, rate=descriptor["sampling_rate"],
        eeg_rows=eeg_rows, names=[stream.NAMES[row - 1] for row in eeg_rows],
        marker_row=descriptor["marker_channel"],
        packet_row=descriptor["package_num_channel"],
        timestamp_row=descriptor["timestamp_channel"],
        imu_rows={"accel": list(stream.ACCEL), "gyro": list(stream.GYRO),
                  "mag": list(stream.MAG)},
    )
    try:
        shim.prepare_session()
        print(f"port={port} - waiting {settle:g}s for boot chatter...")
        time.sleep(settle)
        shim.start_stream(buffer_size)
        time.sleep(2.0)
        stream.enable_eeg_channels(shim, eeg_rows)
        shim.get_board_data()  # discard configuration-period samples
    except BaseException:
        close_board(board)
        raise
    return board


def close_board(board: Board) -> None:
    try:
        board.shim.stop_stream()
    except Exception:
        pass
    try:
        board.shim.release_session()
    except Exception as exc:
        print(f"Board release failed during cleanup: {exc}", file=sys.stderr)
