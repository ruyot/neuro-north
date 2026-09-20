"""Head gestures from the headset's gyro. speller_ui's four edge panels were
always meant to be driven by head movement rather than by flicker."""

from __future__ import annotations

import numpy as np

from . import config

GESTURES = ("left", "right", "up", "down")   # same names as speller_ui.KEY_ACTIONS

# Fallback sensor-axis mapping. A fitted profile replaces these directions.
# Board artwork orientation does not establish the IMU chip's axis signs.
AXIS_GESTURES = {2: ("left", "right"), 1: ("up", "down")}

# BrainFlow passes firmware float32 IMU values through without unit conversion.
# These are not raw ADC counts. Keep the minimum in stream units; the guided
# setup measures actual movement magnitudes and direction vectors.
NOISE_FLOOR = config.GESTURE_MIN_SPEED


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
    """Detect sustained angular motion, with quiet rearming and sample timing."""

    def __init__(self, rate: int, threshold: float = config.GESTURE_THRESHOLD,
                 cooldown: float = config.GESTURE_COOLDOWN, profile=None):
        self.rate = rate
        self.threshold = threshold
        self.hold = max(1, round(cooldown * rate))
        self.warmup = max(2, round(.6 * rate))
        self.dwell = max(1, round(config.GESTURE_DWELL * rate))
        self.rearm = max(1, round(config.GESTURE_REARM * rate))
        self.rest = np.zeros((3, 0))
        self.baseline = None
        self.limit = None
        self.since = self.hold
        self.quiet_count = 0
        self.armed = True
        self.candidate = -1
        self.candidate_count = 0
        self.candidate_travel = 0.0
        self.active_direction = None
        self.travel = self.peak_travel = 0.0
        self.return_count = 0
        self.return_seen = False
        self.peak_ratio = 0.0
        directions = np.zeros((4, 3))
        for axis, pair in AXIS_GESTURES.items():
            directions[GESTURES.index(pair[0]), axis] = -1
            directions[GESTURES.index(pair[1]), axis] = 1
        self.directions = directions
        self.minimum = np.full(4, NOISE_FLOOR)
        self.profile = profile
        self.rest_reference = None
        if profile is not None:
            if profile.get("version") != 1 or profile.get("gestures") != list(GESTURES):
                raise ValueError("Unsupported IMU profile; rerun imu_setup")
            self.directions = np.asarray(profile['directions'], dtype=float)
            self.minimum = np.asarray(profile['minimum_speed'], dtype=float)
            if (self.directions.shape != (4, 3) or self.minimum.shape != (4,)
                    or not np.isfinite(self.directions).all() or not np.isfinite(self.minimum).all()
                    or (self.minimum <= 0).any()
                    or not np.allclose(np.linalg.norm(self.directions, axis=1), 1)):
                raise ValueError("Invalid IMU profile; rerun imu_setup")
            if 'rest_baseline' in profile and 'rest_noise' in profile:
                baseline = np.asarray(profile['rest_baseline'], dtype=float)
                noise = np.asarray(profile['rest_noise'], dtype=float)
                if (baseline.shape != (3,) or noise.shape != (3,)
                        or not np.isfinite(baseline).all() or not np.isfinite(noise).all()
                        or (noise < 0).any()):
                    raise ValueError('Invalid resting IMU measurements; rerun imu_setup')
                self.rest_reference = (baseline, noise)
        self.levels = None

    @classmethod
    def from_config(cls, rate):
        import json
        from pathlib import Path
        path = Path(config.GESTURE_PROFILE_PATH)
        profile = json.loads(path.read_text()) if path.exists() else None
        print(f"[head] profile: {path}" if profile else
              "[head] fallback axes; run python -m ssvep_training.imu_setup to learn directions")
        return cls(rate, profile=profile)

    def _update_rest(self, value):
        self.rest = np.column_stack((self.rest, value))[:, -self.warmup:]
        self.baseline = np.median(self.rest, axis=1)
        if self.rest.shape[1] >= self.warmup:
            if self.levels is None and self.rest_reference is not None:
                # Startup packets can contain almost constant stale motion.
                # Low variance does NOT make that a valid zero-rate baseline.
                # Use the explicitly measured resting reference, then let the
                # existing quiet-sample adaptation track fresh resting data.
                self.baseline, spread = (v.copy() for v in self.rest_reference)
                self.rest = np.zeros((3, 0))
            else:
                spread = 1.4826 * np.median(np.abs(self.rest - self.baseline[:, None]), axis=1)
            self.limit = np.maximum(self.threshold * spread, NOISE_FLOOR)
            projected = np.sqrt((self.directions ** 2) @ (spread ** 2))
            self.levels = np.maximum(self.minimum, self.threshold * projected)

    def feed(self, gyro) -> str | None:
        values = np.asarray(gyro, dtype=float)
        if values.ndim != 2 or values.shape[0] != 3 or not np.isfinite(values).all():
            raise ValueError("Gyro data must be finite with shape (3, samples)")
        last_hit = None
        for value in values.T:
            self.since += 1
            if self.levels is None:
                self._update_rest(value)
                continue
            dev = value - self.baseline
            self.peak_ratio = max(self.peak_ratio, float(np.max((self.directions @ dev) / self.levels)))
            quiet = (np.abs(dev) < self.limit * .5).all()
            # Baseline estimation needs genuinely quiet samples. Rearming only
            # needs motion to settle below gesture strength: using the much
            # tighter noise gate here can permanently reject normal head jitter.
            # Half the lowest activation level provides release hysteresis;
            # vector magnitude also rejects motion along an unmapped roll axis.
            settled = np.linalg.norm(dev) < .5 * np.min(self.levels)
            self.quiet_count = self.quiet_count + 1 if settled else 0
            if quiet:
                self._update_rest(value)
            if not self.armed:
                # Suppress the reverse rotation used to return to center.
                # Integrating in native units suffices: only a relative return
                # to the swipe's starting orientation is tested here.
                projection = float(self.active_direction @ dev)
                self.travel += projection / self.rate
                self.peak_travel = max(self.peak_travel, self.travel)
                # Gyro integration is not an absolute head-position estimate:
                # a partial return or bias error can leave a residual forever.
                # Also recognize a sustained reverse movement, then require the
                # SAME quiet/cooldown gate below. Holding the turned pose alone
                # never rearms, and the return itself cannot fire an action.
                reverse = (-projection > .5 * np.linalg.norm(self.limit)
                           and -projection >= .7 * np.linalg.norm(dev))
                self.return_count = self.return_count + 1 if reverse else 0
                # A completed return must survive corrective movement before
                # stillness. Otherwise an overshoot followed by a correction
                # can erase the integrated return and leave us locked forever.
                self.return_seen |= (self.return_count >= self.dwell
                                     or self.travel <= .35 * self.peak_travel)
                returned = self.return_seen
                if returned and self.since >= self.hold and self.quiet_count >= self.rearm:
                    self.armed = True
                else:
                    continue
            projections = self.directions @ dev
            ratios = projections / self.levels
            choice = int(np.argmax(ratios))
            # Require motion along the candidate direction, not primarily roll.
            aligned = projections[choice] >= .7 * np.linalg.norm(dev)
            if ratios[choice] < 1 or not aligned:
                self.candidate, self.candidate_count = -1, 0
                self.candidate_travel = 0.0
                continue
            if choice != self.candidate:
                self.candidate, self.candidate_count = choice, 0
                self.candidate_travel = 0.0
            self.candidate_count += 1
            self.candidate_travel += float(projections[choice]) / self.rate
            if self.candidate_count >= self.dwell:
                last_hit = GESTURES[choice]
                self.since, self.quiet_count = 0, 0
                self.armed = False
                self.active_direction = self.directions[choice]
                self.travel = self.peak_travel = self.candidate_travel
                self.return_count, self.return_seen = 0, False
                self.candidate, self.candidate_count = -1, 0
        return last_hit

    def diagnostics(self, reset_peak=False):
        """Expose the actual gate without changing detector state or thresholds."""
        returned = self.return_seen or self.travel <= .35 * self.peak_travel
        waiting = []
        if self.levels is None:
            waiting.append('warmup')
        elif not self.armed:
            if not returned:
                waiting.append('return')
            if self.quiet_count < self.rearm:
                waiting.append('stillness')
            if self.since < self.hold:
                waiting.append('cooldown')
        state = {'armed': self.armed, 'waiting': waiting,
                 'quiet_samples': self.quiet_count, 'quiet_required': self.rearm,
                 'return_seen': bool(self.return_seen), 'travel': self.travel,
                 'peak_travel': self.peak_travel, 'peak_ratio': self.peak_ratio,
                 'candidate_samples': self.candidate_count,
                 'baseline': self.baseline.tolist() if self.baseline is not None else None,
                 'quiet_limits': (self.limit * .5).tolist() if self.limit is not None else None,
                 'release_speed': float(.5 * np.min(self.levels)) if self.levels is not None else None,
                 'activation_levels': self.levels.tolist() if self.levels is not None else None}
        if reset_peak:
            self.peak_ratio = 0.0
        return state


def replay(path: str) -> None:
    """Run the detector over a saved session. Every recording already contains
    the gyro rows, so a calibration folder from the headset is real hardware
    evidence for the axis map and the threshold without a live board."""
    from .session import load_session

    data, meta = load_session(path)
    rate, rows = meta["rate"], gyro_rows(meta["board_id"])
    gyro = data[rows]
    det, fired = Gestures.from_config(rate), []
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
        print(f"  {a}  {names:<13} {level:8.4f} {peak[a]:6.4f} {peak[a] / level:6.1f}x")
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
    still[1, ::50] = .001            # a silent sensor twitching one count now and then
    assert run(still, 125) == [], "small isolated jitter must not trigger a gesture"

    print("head.py self-check OK")
