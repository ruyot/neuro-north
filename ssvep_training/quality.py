"""Transport checks for the Knight IMU's modulo-256 packet counter."""
import numpy as np


def packet_discontinuities(counter) -> int:
    values = np.asarray(counter)
    return int(np.count_nonzero(np.diff(values) % 256 != 1))


def trial_transport_ok(data, onset, rate, stop, history):
    start = onset - history
    end = onset + stop
    if start < 0 or end > data.shape[1]:
        return False
    return packet_discontinuities(data[0, start:end]) == 0
