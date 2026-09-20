"""Display-independent frame schedule, shared by calibration and live UI."""
import math


def frames_per_cycle(frequency: float, refresh: float) -> int:
    period = round(refresh / frequency)
    if period < 2 or abs(refresh / period - frequency) > 0.15:
        raise ValueError(f"{frequency:g} Hz cannot use whole cycles at {refresh:.2f} Hz refresh; "
                         "set the display to 60 Hz (or 120 Hz) and restart")
    return period


def frame_is_on(frequency: float, frame: int, refresh: float) -> bool:
    period = frames_per_cycle(frequency, refresh)
    return frame % period < math.ceil(period / 2)
