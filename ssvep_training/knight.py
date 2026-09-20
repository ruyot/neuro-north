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
NAMES = stream.NAMES
PORT = stream.PORT
cca_score = stream.cca_score
cca_reference = stream.cca_reference
