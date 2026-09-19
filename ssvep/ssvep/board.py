"""
Thin, well-documented wrapper around the NeuroPawn Knight board.

Why the board needs "setting up" before it streams anything useful
------------------------------------------------------------------
When BrainFlow first connects, every EEG channel on the Knight board is
*powered down*. If you start streaming immediately you just get flat lines.
Each channel has to be switched on and (optionally) tied into the board's
right-leg-drive (RLD / bias) circuit, which actively cancels common-mode
noise (mains hum, movement). That is what `configure_channels()` does with
two commands per channel:

    chon_{ch}_{gain}   -> turn channel `ch` ON with the given PGA gain
    rldadd_{ch}        -> add channel `ch` to the bias/RLD feedback loop

These commands are sent one-by-one with short sleeps, because the board's
firmware needs a moment to apply each register write. Configuring all 8
channels therefore takes ~20-30 s -- this is normal and only happens once
at start-up.
"""

from __future__ import annotations

import time

from brainflow.board_shim import BoardShim, BrainFlowInputParams

try:
    from . import config as cfg
except ImportError:  # run directly as a script: python ssvep/board.py
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from ssvep import config as cfg


class KnightBoard:
    """Connect to, configure, and stream from a NeuroPawn Knight board."""

    def __init__(self, serial_port: str, num_channels: int, gain: int = 12,
                 variant: str | None = None):
        self.num_channels = num_channels
        self.gain = gain
        self.variant = variant or cfg.BOARD_VARIANT

        # BrainFlow needs the serial port for real hardware; the synthetic
        # board is pure software and has none.
        params = BrainFlowInputParams()
        if self.variant != "synthetic":
            params.serial_port = serial_port

        self.board_shim = BoardShim(cfg.board_id(self.variant), params)
        self.board_id = self.board_shim.get_board_id()

        # Ask BrainFlow which rows of the data matrix hold EEG and the rate.
        # Keep only the channels actually wired up, so NUM_CHANNELS means
        # something instead of always handing back all 8 rows.
        self.eeg_channels = self.board_shim.get_exg_channels(self.board_id)[:num_channels]
        self.sr = self.board_shim.get_sampling_rate(self.board_id)

    # --------------------------------------------------------------------- #
    # Streaming lifecycle
    # --------------------------------------------------------------------- #
    def start_stream(self, buffer_size: int = 450_000) -> None:
        """Open the session, start streaming, and power up every channel."""
        self.board_shim.prepare_session()
        self.board_shim.start_stream(buffer_size)
        if self.sr != cfg.SAMPLING_RATE:
            print(f"WARNING: board reports {self.sr} Hz, config says "
                  f"{cfg.SAMPLING_RATE} Hz - check BOARD_VARIANT/firmware.")
        if self.variant == "synthetic":
            print("Synthetic board - skipping channel configuration.")
        else:
            print("Stream started - configuring channels (this takes a moment)...")
            time.sleep(2)  # let the stream settle before sending config commands
            self.configure_channels()
        print("Board ready.")

    def configure_channels(self) -> None:
        """Switch each channel on and add it to the bias/RLD loop."""
        for ch in range(1, self.num_channels + 1):
            time.sleep(0.5)
            on_cmd = f"chon_{ch}_{self.gain}"      # power channel on at `gain`
            self.board_shim.config_board(on_cmd)
            print(f"  sent {on_cmd}")

            time.sleep(1.0)
            rld_cmd = f"rldadd_{ch}"               # tie channel into bias drive
            self.board_shim.config_board(rld_cmd)
            print(f"  sent {rld_cmd}")
            time.sleep(0.5)

    def stop_stream(self) -> None:
        """Stop streaming and release the hardware session."""
        self.board_shim.stop_stream()
        self.board_shim.release_session()
        print("Stream stopped and session released.")

    # --------------------------------------------------------------------- #
    # Data access
    # --------------------------------------------------------------------- #
    def get_latest(self, num_samples: int):
        """
        Return the most recent `num_samples` columns WITHOUT draining the buffer.

        BrainFlow keeps incoming samples in a fixed-size ring buffer. There are
        two ways to read it:

          * get_board_data()          -> returns everything AND empties the
                                         buffer (destructive read).
          * get_current_board_data(n) -> returns a *copy* of the latest `n`
                                         samples and leaves the buffer intact.

        We always use the second form. Because it is non-destructive, the
        stimulus loop and the classifier can both "peek" at the freshest slice
        of EEG on every trial without ever losing samples or fighting over the
        buffer. The board keeps filling the ring in the background regardless.
        """
        return self.board_shim.get_current_board_data(num_samples)


if __name__ == "__main__":
    from brainflow.board_shim import BoardIds

    BoardShim.disable_board_logger()   # same as stream_test.py: no JSON spam

    assert cfg.board_id("imu") == BoardIds.NEUROPAWN_KNIGHT_BOARD_IMU.value == 66
    assert cfg.board_id("plain") == BoardIds.NEUROPAWN_KNIGHT_BOARD.value == 57
    assert cfg.board_id("synthetic") == BoardIds.SYNTHETIC_BOARD.value == -1
    try:
        cfg.board_id("nope")
    except ValueError as exc:
        print(f"bogus variant -> ValueError: {exc}")
    else:
        raise AssertionError("bogus variant should raise ValueError")

    imu = KnightBoard("/dev/null", 4, variant="imu")
    assert imu.eeg_channels == [1, 2, 3, 4], imu.eeg_channels
    assert imu.sr == 125, imu.sr

    plain = KnightBoard("/dev/null", 8, variant="plain")
    assert plain.sr == 125, plain.sr
    n_rows = BoardShim.get_board_descr(plain.board_id)["num_rows"]
    assert n_rows == 13, n_rows
    print(f"imu channels={imu.eeg_channels} sr={imu.sr}; plain rows={n_rows}")

    synth = KnightBoard("", 4, variant="synthetic")
    synth.start_stream()
    time.sleep(2)
    data = synth.get_latest(100)
    synth.stop_stream()
    assert data.size > 0, "synthetic board returned no data"
    print(f"synthetic sr={synth.sr} data={data.shape} channels={synth.eeg_channels}")

    # Drop the shims while the interpreter is still alive. BoardShim.__del__
    # calls into ctypes/numpy, which are already torn down at exit, so leaving
    # them to be collected at shutdown prints alarming tracebacks after a pass.
    del imu, plain, synth
    print("ALL CHECKS PASSED")
