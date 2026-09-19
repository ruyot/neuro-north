"""Head gestures from the headset's gyro. speller_ui's four edge panels were
always meant to be driven by head movement rather than by flicker."""

from __future__ import annotations

import numpy as np

from . import config

GESTURES = ("left", "right", "up", "down")   # same names as speller_ui.KEY_ACTIONS

# A GUESS, not a fact: nobody has read this board's gyro on a head yet. The
# Knight lies flat against the scalp, so z (index 2) should be the turn axis
# and y the nod; index 0 is roll, which is not a speller action. To correct it,
# run `python stream.py --real`, make one movement at a time and watch which
# `gyro pk` number jumps. Pairs read (negative, positive), so if a left turn
# types "right", swap that pair.
AXIS_GESTURES = {
    2: ("left", "right"),    # yaw, head turn
    1: ("up", "down"),       # pitch, nod
}

# edited by hand, so catch a typo on import, not in the child process
assert all(g in GESTURES for pair in AXIS_GESTURES.values() for g in pair)

# Firmware supplies float32 gyro readings without SDK scaling; the documented
# rad/s units have not been verified on this firmware. Thresholds stay relative
# to the sensor's resting jitter. This existing floor keeps a near-silent axis
# from collapsing the threshold or becoming a zero divisor; it is not calibrated.
NOISE_FLOOR = 2.0


def gyro_rows(board_id: int) -> list[int]:
    """Data rows holding gyro x,y,z. stream.py's constants are the Knight's rows
    only, so anything else has to come from BoardShim."""
    # late: brainflow is not installed everywhere this module is imported
    from brainflow.board_shim import BoardIds, BoardShim

    from . import knight as stream

    if board_id == int(BoardIds.NEUROPAWN_KNIGHT_BOARD_IMU):
        return list(stream.GYRO)   # BrainFlow 5.23 does not tag this board's IMU rows
    return BoardShim.get_gyro_channels(board_id)


class Gestures:
    """Rolling gyro in, at most one gesture name per head movement out."""

    def __init__(self, rate: int, threshold: float = config.GESTURE_THRESHOLD,
                 cooldown: float = config.GESTURE_COOLDOWN):
        self.threshold = threshold
        # Samples, not time.time(): the recorder hands us ring-buffer data, so
        # the wall clock runs ahead of it. Also what makes any rate work.
        self.hold = max(1, int(cooldown * rate))
        self.rest = np.zeros((3, 0))    # recent quiet samples, the reference's source
        self.baseline = None
        self.limit = None               # per-axis fire level, measured not configured
        self.since = self.hold          # armed from the very first sample

    def feed(self, gyro) -> str | None:
        n = gyro.shape[1]
        if not n:
            return None
        if self.baseline is None:
            # A gyro rests near zero but not at it. A first chunk landing
            # mid-swipe gets corrected by the rest buffer within a cooldown.
            self.baseline = gyro[:, 0].astype(float)

        dev = gyro - self.baseline[:, None]
        if self.limit is None:
            # No measured jitter yet means no scale to judge against, so
            # everything counts as rest until the first window fills.
            ratio = None
            quiet = np.ones(n, dtype=bool)
        else:
            # per axis: a noisier axis should be neither trigger-happy nor deaf
            ratio = np.abs(dev) / self.limit[:, None]
            quiet = (ratio < 1.0).all(axis=0)

        # Quiet samples only: the baseline would otherwise chase the swipe and
        # cancel it. A steady tilt or sensor bias is what it should absorb.
        if quiet.any():
            self.rest = np.hstack([self.rest, gyro[:, quiet]])[:, -self.hold:]
            self.baseline = self.rest.mean(axis=1)
            if self.rest.shape[1] >= self.hold:
                # MAD, not std: one stray sample moves it far less
                spread = np.abs(self.rest - self.baseline[:, None]).mean(axis=1)
                self.limit = np.maximum(spread * self.threshold, NOISE_FLOOR)

        # Without this, one swipe fires on nearly every sample it spans.
        self.since += n
        if ratio is None or self.since < self.hold:
            return None

        axes = list(AXIS_GESTURES)
        mapped = ratio[axes]
        hit = np.flatnonzero((mapped >= 1.0).any(axis=0))
        if not hit.size:
            return None
        col = hit[0]
        axis = axes[int(np.argmax(mapped[:, col]))]   # furthest past its own level wins
        self.since = 0
        low, high = AXIS_GESTURES[axis]
        return high if dev[axis, col] > 0 else low


def replay(path: str) -> None:
    """Run the detector over a saved session. Every recording already contains
    the gyro rows, so a calibration folder from the headset is real hardware
    evidence for the axis map and the threshold without a live board."""
    from .session import load_session

    data, meta = load_session(path)
    rate, rows = meta["rate"], gyro_rows(meta["board_id"])
    gyro = data[rows]
    det, fired = Gestures(rate), []
    for i in range(0, gyro.shape[1], 3):      # the chunk size the recorder feeds
        name = det.feed(gyro[:, i:i + 3])
        if name:
            fired.append((i / rate, name))

    peak = np.abs(gyro - np.median(gyro, axis=1, keepdims=True)).max(axis=1)
    print(f"{gyro.shape[1] / rate:.0f}s at {rate} Hz, gyro rows {rows}")
    print("axis  gestures       fires at   peak   ratio")
    for a in range(3):
        level = det.limit[a] if det.limit is not None else float("nan")
        names = "/".join(AXIS_GESTURES.get(a, ("unmapped",)))
        print(f"  {a}  {names:<13} {level:8.1f} {peak[a]:6.0f} {peak[a] / level:6.1f}x")
    print(f"{len(fired)} gestures" + (": " if fired else ""),
          ", ".join(f"{t:.0f}s {n}" for t, n in fired[:24]))


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:                     # a session folder: replay it
        replay(sys.argv[1])
        raise SystemExit

    def run(sig, rate, size=3):
        # the recorder drains every 20 ms, so feed() carries its own history
        g, fired = Gestures(rate), []
        for i in range(0, sig.shape[1], size):
            name = g.feed(sig[:, i:i + size])
            if name is not None:
                fired.append(name)
        return fired

    def noise(rate, seconds=10, seed=0):
        # uniform, not gaussian: a hard bound means these asserts cannot flake
        return np.random.default_rng(seed).uniform(-40, 40, (3, int(rate * seconds)))

    def swipe(rate, axis, amp=4000.0):
        sig = noise(rate, seconds=4, seed=1)
        n = int(0.32 * rate)        # a real head turn, and shorter than the cooldown
        # half a sine is one smooth movement; a second in, so jitter is measured
        sig[axis, rate:rate + n] += amp * np.sin(np.linspace(0, np.pi, n))
        return sig

    for rate in (125, 500):         # every window is derived from rate, none assume 125
        assert run(noise(rate), rate) == [], "a still head must fire nothing"
        fired = run(swipe(rate, 2), rate)
        assert fired == ["right"], f"one swipe is one gesture, got {fired} at {rate} Hz"
        assert run(swipe(rate, 2, -4000.0), rate) == ["left"], "the sign picks the direction"
        assert run(swipe(rate, 1), rate) == ["down"], "the pitch axis nods"
        assert run(swipe(rate, 1, -4000.0), rate) == ["up"], "the pitch axis nods"
        assert run(swipe(rate, 0), rate) == [], "the unmapped roll axis is not a gesture"

    tilted = noise(125) + np.array([[8000.0], [-3000.0], [2000.0]])
    assert run(tilted, 125) == [], "a steady offset is baseline, not a gesture"

    still = np.zeros((3, 125 * 10))
    still[1, ::50] = 1.0            # a silent sensor twitching one count now and then
    assert run(still, 125) == [], "single-bit jitter must not clear the noise floor"

    print("head.py self-check OK")
