"""Live scrolling graph of EEG + IMU from the NeuroPawn Knight IMU board.

Usage:
    python live_plot.py --port /dev/cu.usbserial-XXXX   # real board
    python live_plot.py --synthetic                     # no hardware needed

Press F to toggle filtering (1-40 Hz bandpass + 60 Hz notch) on the EEG traces.
"""
import argparse
import threading

import numpy as np
import pyqtgraph as pg
from brainflow.board_shim import BoardIds, BoardShim, BrainFlowInputParams
from brainflow.data_filter import DataFilter, DetrendOperations, FilterTypes
from pyqtgraph.Qt import QtCore, QtWidgets

from stream_test import ACCEL_ROWS, GYRO_ROWS, enable_channels

WINDOW_SECONDS = 5
REFRESH_MS = 50
COLORS = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#42d4f4", "#f032e6", "#bfef45"]
XYZ_COLORS = ["#e6194b", "#3cb44b", "#4363d8"]


class LivePlot:
    def __init__(self, board, board_id, has_imu):
        self.board = board
        self.sampling_rate = BoardShim.get_sampling_rate(board_id)
        self.eeg_channels = BoardShim.get_eeg_channels(board_id)[:8]
        self.num_points = WINDOW_SECONDS * self.sampling_rate
        self.has_imu = has_imu
        self.filtering = True
        self.pending_status = None  # set from the setup thread, applied on the GUI thread

        self.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        pg.setConfigOptions(antialias=True)
        self.win = pg.GraphicsLayoutWidget(title="Knight IMU - live")
        self.win.resize(1200, 900)

        self.status = self.win.addLabel("", row=0, col=0)
        self.set_status("Starting...")

        # One row per EEG channel, all sharing the time axis
        self.eeg_curves, self.eeg_plots = [], []
        for i, ch in enumerate(self.eeg_channels):
            p = self.win.addPlot(row=i + 1, col=0)
            p.setLabel("left", f"EEG {i + 1} (uV)")  # no units= so pyqtgraph doesn't rescale to kuV/nuV
            p.showGrid(x=True, y=True, alpha=0.2)
            p.setMouseEnabled(x=False, y=True)
            if i < len(self.eeg_channels) - 1:
                p.hideAxis("bottom")
            else:
                p.setLabel("bottom", "seconds")
            if self.eeg_plots:
                p.setXLink(self.eeg_plots[0])
            self.eeg_curves.append(p.plot(pen=pg.mkPen(COLORS[i % len(COLORS)], width=1)))
            self.eeg_plots.append(p)

        # IMU in a second column
        self.imu_curves = {}
        if has_imu:
            for r, (name, rows, units) in enumerate(
                [("Accel", ACCEL_ROWS, "m/s^2"), ("Gyro", GYRO_ROWS, "")]
            ):
                p = self.win.addPlot(row=1 + r * 4, col=1, rowspan=4, title=name)
                p.setLabel("left", name, units=units)
                p.showGrid(x=True, y=True, alpha=0.2)
                p.addLegend(offset=(5, 5))
                self.imu_curves[name] = [
                    (row, p.plot(pen=pg.mkPen(c, width=1.5), name=axis))
                    for row, c, axis in zip(rows, XYZ_COLORS, "xyz")
                ]
            self.win.ci.layout.setColumnStretchFactor(0, 3)
            self.win.ci.layout.setColumnStretchFactor(1, 2)

        self.win.keyPressEvent = self.on_key
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(REFRESH_MS)

    def set_status(self, text):
        self._status_text = text
        mode = "filtered 1-40 Hz + 60 Hz notch" if self.filtering else "raw"
        self.status.setText(f"{text}   |   EEG: {mode} (press F to toggle)")

    def on_key(self, event):
        if event.key() == QtCore.Qt.Key.Key_F:
            self.filtering = not self.filtering
            self.set_status(self._status_text)

    def filter_eeg(self, signal):
        DataFilter.detrend(signal, DetrendOperations.CONSTANT.value)
        if len(signal) >= self.sampling_rate:  # filters need ~1s of data to settle
            DataFilter.perform_bandpass(signal, self.sampling_rate, 1.0, 40.0, 4,
                                        FilterTypes.BUTTERWORTH_ZERO_PHASE.value, 0)
            DataFilter.perform_bandstop(signal, self.sampling_rate, 58.0, 62.0, 4,
                                        FilterTypes.BUTTERWORTH_ZERO_PHASE.value, 0)
        return signal

    def update(self):
        if self.pending_status is not None:
            self.set_status(self.pending_status)
            self.pending_status = None

        data = self.board.get_current_board_data(self.num_points)  # doesn't drain the buffer
        n = data.shape[1]
        if n == 0:
            return
        t = (np.arange(n) - n) / self.sampling_rate  # seconds, 0 = now

        for curve, ch in zip(self.eeg_curves, self.eeg_channels):
            signal = np.ascontiguousarray(data[ch], dtype=np.float64)
            if self.filtering:
                signal = self.filter_eeg(signal)
            curve.setData(t, signal)

        for pairs in self.imu_curves.values():
            for row, curve in pairs:
                curve.setData(t, data[row])

    def run(self):
        self.win.show()
        self.app.exec()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", help="serial port, e.g. /dev/cu.usbserial-XXXX")
    parser.add_argument("--synthetic", action="store_true", help="use BrainFlow's fake board")
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
    board = BoardShim(board_id, params)
    board.prepare_session()
    board.start_stream()

    plot = LivePlot(board, board_id, has_imu=not args.synthetic)

    if args.synthetic:
        plot.set_status("Synthetic board")
    else:
        # Enabling channels takes ~4s each; do it in the background so the IMU shows up right away.
        def post_status(text):
            plot.pending_status = text

        def setup():
            post_status(f"Enabling EEG channels {args.channels} (~{4 * len(args.channels) + 2}s)...")
            try:
                enable_channels(board, args.channels)
                post_status(f"Streaming - EEG channels {args.channels} on")
            except Exception as e:
                post_status(f"Channel setup failed: {e}")

        threading.Thread(target=setup, daemon=True).start()

    try:
        plot.run()
    finally:
        board.stop_stream()
        board.release_session()


if __name__ == "__main__":
    main()
