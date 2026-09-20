"""Learn the four gesture directions for the headset's actual mounting.

python -m ssvep_training.imu_setup --channels 1,2,3,4
No flicker or typing. Hold each requested pose until told to return to center.
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
import numpy as np
from . import config
from .head import GESTURES, Gestures, gyro_rows


def fit_profile(rest, movements, rate):
    """Learn signed 3D directions and thresholds from two labeled turns each."""
    rest = np.asarray(rest, dtype=float)
    if rest.ndim != 2 or rest.shape[0] != 3 or not np.isfinite(rest).all():
        raise ValueError('Invalid stationary gyro data')
    baseline = np.median(rest, axis=1)
    noise = 1.4826 * np.median(np.abs(rest-baseline[:, None]), axis=1)
    directions, minimum, peaks = [], [], []
    for gesture in GESTURES:
        vectors, amplitudes = [], []
        for recording in movements[gesture]:
            dev = np.asarray(recording, dtype=float) - baseline[:, None]
            if dev.ndim != 2 or dev.shape[0] != 3 or not np.isfinite(dev).all():
                raise ValueError(f'{gesture}: invalid movement data')
            width = max(1, round(config.GESTURE_DWELL * rate))
            if dev.shape[1] < width:
                raise ValueError(f'{gesture}: not enough motion samples')
            smooth = np.array([np.convolve(row, np.ones(width)/width, mode='valid') for row in dev])
            strength = np.linalg.norm(smooth, axis=0)
            vector = smooth[:, int(strength.argmax())]
            amp = float(np.linalg.norm(vector))
            if amp <= max(1e-4, 6*np.linalg.norm(noise)):
                raise ValueError(f'{gesture}: movement too weak relative to rest; repeat a deliberate turn')
            vectors.append(vector/amp)
            amplitudes.append(amp)
        if len(vectors) < 2 or np.dot(vectors[0], vectors[1]) < .8:
            raise ValueError(f'{gesture}: repeats disagree; hold the pose until told to return')
        direction = np.mean(vectors, axis=0)
        direction /= np.linalg.norm(direction)
        directions.append(direction)
        projected_noise = float(np.sqrt(np.sum((direction*noise)**2)))
        minimum.append(max(1e-4, 6*projected_noise, .3*min(amplitudes)))
        peaks.append(amplitudes)
    directions = np.array(directions)
    if directions[0] @ directions[1] > -.65 or directions[2] @ directions[3] > -.65:
        raise ValueError('Opposite gestures do not rotate oppositely; repeat setup following the labels')
    if np.max(np.abs(directions[:2] @ directions[2:].T)) > .7:
        raise ValueError('Left/right and up/down motions are too similar; turn versus nod distinctly')
    return {'version': 1, 'gestures': list(GESTURES), 'directions': directions.tolist(),
            'minimum_speed': minimum, 'calibration_peaks': peaks,
            'rest_baseline': baseline.tolist(), 'rest_noise': noise.tolist(),
            'rate': rate, 'units': 'firmware stream units (no conversion assumed)',
            'created': time.strftime('%Y-%m-%d %H:%M:%S')}


def capture(board, seconds):
    """Discard old serial-buffer samples before each labeled segment."""
    board.shim.get_board_data()
    end = time.monotonic()+seconds
    chunks = []
    while time.monotonic() < end:
        time.sleep(.02)
        chunk = board.shim.get_board_data()
        if chunk.shape[1]:
            chunks.append(chunk)
    if not chunks:
        raise RuntimeError('No data from the IMU')
    data = np.hstack(chunks)
    if data.shape[1] < .7 * seconds * board.rate:
        raise RuntimeError('Too few samples for gesture setup')
    from .quality import packet_discontinuities
    if packet_discontinuities(data[0]):
        raise RuntimeError('Packet discontinuity during setup; retry the recording')
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--channels', default='1,2,3,4')
    parser.add_argument('--port')
    parser.add_argument('--check', action='store_true', help='only test the saved profile, without changing it')
    parser.add_argument('--seconds', type=float, default=20, help='length of the live direction check')
    args = parser.parse_args()
    channels = [int(c) for c in args.channels.split(',')]
    if not channels or not set(channels) <= set(range(1, 9)):
        parser.error('channels must be 1–8')
    from .board import open_board
    with open_board(port=args.port, channels=channels) as board:
        rows = gyro_rows(board.board_id)
        if not args.check:
            input('Wear the headset in its final position. Face center, stay still, then press ENTER. ')
            print('Measuring rest for 2 seconds. Keep still.', flush=True)
            rest_raw = capture(board, 2)
            raw = {'rest': rest_raw}
            movements = {name: [] for name in GESTURES}
            for name in GESTURES:
                for repetition in range(2):
                    input(f'Face CENTER. Prepare to turn {name.upper()} ({repetition+1}/2). Press ENTER while still. ')
                    for number in (3, 2, 1):
                        print(number, flush=True)
                        time.sleep(.4)
                    print(f'TURN {name.upper()} NOW, then HOLD. Do not return yet.', flush=True)
                    data = capture(board, 2)
                    raw[f'{name}_{repetition+1}'] = data
                    movements[name].append(data[rows])
                    print('Return gently to CENTER now.', flush=True)
            output = Path(config.GESTURE_PROFILE_PATH)
            output.parent.mkdir(parents=True, exist_ok=True)
            archive = output.parent / time.strftime('imu_setup_%Y%m%d_%H%M%S.npz')
            np.savez(archive, **raw)
            profile = fit_profile(rest_raw[rows], movements, board.rate)
            # Do not replace a usable existing profile if a new setup fails validation.
            temporary = output.with_suffix('.tmp.json')
            temporary.write_text(json.dumps(profile, indent=2))
            temporary.replace(output)
            print(f'Profile saved to {output}; labeled raw IMU data: {archive}')
        detector = Gestures.from_config(board.rate)
        input('Live check: face CENTER, press ENTER, keep still for 2 seconds. ')
        detector.feed(capture(board, 2)[rows])
        if detector.levels is not None:
            print('Thresholds (left/right/up/down):', np.round(detector.levels, 4).tolist())
        print(f'For {args.seconds:g}s, try each direction; return to center between gestures. No text is inserted.', flush=True)
        end = time.monotonic()+args.seconds
        while time.monotonic() < end:
            time.sleep(.02)
            data = board.shim.get_board_data()
            if data.shape[1]:
                name = detector.feed(data[rows])
                if name:
                    print(f'[head] {name}', flush=True)


if __name__ == '__main__':
    main()
