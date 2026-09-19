"""
Open and close the Knight IMU board for the recording process

- prepare_session -> 3 s boot settle -> stream.enable_eeg_channels -> start_stream

"""

from __future__ import annotations

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

    def close(self) -> None:
        close_board(self)

    def __enter__(self) -> "Board":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def open_board(port: str | None = None, channels=DEFAULT_CHANNELS,
               settle: float = SETTLE_SECONDS, buffer_size: int = BUFFER_SIZE,
               quiet: bool = True, board_id: int | None = None) -> Board:
    """board_id: defaults to the Knight's IMU variant. Paradigms that never read
    the accelerometer can pass the plain Knight instead - asking for the IMU
    makes the firmware scan for it, and a headset whose IMU does not answer
    sits in that scan and never streams EEG."""
    if quiet:
        BoardShim.disable_board_logger()

    port = port or stream.PORT
    if not port:
        raise RuntimeError("PORT_PATH is not set - add it to .env")
    board_id = int(BoardIds.NEUROPAWN_KNIGHT_BOARD_IMU) if board_id is None else int(board_id)
    params = BrainFlowInputParams()
    params.serial_port = port

    wanted = set(channels)
    eeg_rows = [r for r in BoardShim.get_eeg_channels(board_id) if r in wanted]
    names = [stream.NAMES[r - 1] if r - 1 < len(stream.NAMES) else f"ch{r}" for r in eeg_rows]

    shim = BoardShim(board_id, params)
    shim.prepare_session()
    try:
        print(f"port={port} - waiting {settle:g}s for boot chatter, then enabling channels...")
        time.sleep(settle)
        stream.enable_eeg_channels(shim, eeg_rows)   # before streaming, stream.py's way
        shim.start_stream(buffer_size)               # flushes serial, samples start flowing
    except BaseException:
        shim.release_session()
        raise

    return Board(shim=shim, board_id=board_id, rate=BoardShim.get_sampling_rate(board_id),
                 eeg_rows=eeg_rows, names=names,
                 marker_row=BoardShim.get_marker_channel(board_id))


def close_board(board: Board) -> None:
    try:
        board.shim.stop_stream()
    except Exception:
        pass
    try:
        board.shim.release_session()
    except Exception:
        pass
