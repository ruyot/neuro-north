from __future__ import annotations

import sys
from pathlib import Path

try:
    import stream
except ModuleNotFoundError:
    for parent in Path(__file__).resolve().parents:
        if (parent / "stream.py").exists():
            sys.path.insert(0, str(parent))
            break
    else:
        raise ModuleNotFoundError("needs stream.py") from None
    import stream

enable_eeg_channels = stream.enable_eeg_channels
clean = stream.clean
leadoff = stream.leadoff
NAMES = stream.NAMES
PORT = stream.PORT
LOFF_P = stream.LOFF_P
LOFF_N = stream.LOFF_N
ACCEL = stream.ACCEL
