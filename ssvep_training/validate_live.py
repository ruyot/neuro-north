"""Fresh prospective A/B test; no old recordings or trained templates are used.

python -m ssvep_training.validate_live --channels 1,2,3,4 --blocks 10 --idle-trials 4
"""
from __future__ import annotations
import argparse
import json
import random
import time
from pathlib import Path
from . import config as cfg
from .cca_model import accept_prediction


def summarize(rows):
    from scipy.stats import binomtest
    targets = [r for r in rows if r['target'] is not None]
    valid = [r for r in targets if r['prediction'] >= 0 and not r['late_frames']]
    accepted = [r for r in valid if r['accepted']]
    idle = [r for r in rows if r['target'] is None]
    hits = sum(r['prediction'] == r['target'] for r in valid)
    result = {'attempted_target_trials': len(targets), 'valid_target_trials': len(valid),
              'correct_forced_choices': hits, 'accepted_target_trials': len(accepted),
              'correct_accepted': sum(r['prediction'] == r['target'] for r in accepted),
              'idle_trials': len(idle), 'idle_false_activations': sum(r['accepted'] for r in idle)}
    if valid:
        test = binomtest(hits, len(valid), .5, alternative='greater')
        ci = binomtest(hits, len(valid)).proportion_ci()
        result.update(forced_accuracy=hits/len(valid), accuracy_ci95=[ci.low, ci.high],
                      chance_p=test.pvalue)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--channels', required=True, help='physically connected channels, e.g. 1,2,3,4')
    parser.add_argument('--blocks', type=int, default=10, help='one A and one B trial per block')
    parser.add_argument('--idle-trials', type=int, default=4, help='center-fixation controls, interspersed')
    parser.add_argument('--seed', type=int, default=20260919)
    parser.add_argument('--port')
    parser.add_argument('--windowed', action='store_true')
    args = parser.parse_args()
    channels = [int(c) for c in args.channels.split(',')]
    if not channels or not set(channels) <= set(range(1, 9)) or len(set(channels)) != len(channels):
        parser.error('channels must be unique integers from 1 through 8')
    if args.blocks < 1 or args.idle_trials < 0 or cfg.N_TARGETS != 2:
        parser.error('requires two targets, positive blocks, and nonnegative idle trials')
    mapping = ' / '.join(f'{letter} = {frequency:g} Hz'
                         for letter, frequency in zip(cfg.TARGET_LETTERS, cfg.STIMULUS_FREQUENCIES))
    print(f'[test] {mapping}; {2 * args.blocks} target trials + {args.idle_trials} center trials')
    rng = random.Random(args.seed)
    schedule = []
    for _ in range(args.blocks):
        pair = [0, 1]
        rng.shuffle(pair)
        schedule.extend(pair)
    for _ in range(args.idle_trials):
        schedule.insert(rng.randrange(len(schedule)+1), None)
    path = Path(cfg.TRAINING_DATA_DIR) / time.strftime('validation_%Y%m%d_%H%M%S')
    path.mkdir(parents=True)
    from .recording import RecordingProcess
    recorder = RecordingProcess(port=args.port, channels=channels, session_dir=str(path), decoder='cca')
    recorder.start()
    win = None
    rows = []
    run = {'seed': args.seed, 'schedule': schedule, 'channels': channels,
           'frequencies': list(cfg.STIMULUS_FREQUENCIES), 'letters': list(cfg.TARGET_LETTERS),
           'threshold': cfg.CONFIDENCE_THRESHOLD, 'decoder': 'cca',
           'note': 'Prospective frozen-settings test. Idle means center fixation with both targets flashing.'}

    def save():
        payload = {**run, 'trials': rows, 'summary': summarize(rows)}
        temporary = path / 'validation.tmp.json'
        temporary.write_text(json.dumps(payload, indent=2, allow_nan=False))
        temporary.replace(path / 'validation.json')

    try:
        from psychopy import core, visual
        from .stimulus import build_window, build_stimuli, run_trial, wait_for_board, wait_for_key, show_message
        from .collect_training_data import WARNING
        win = build_window(fullscreen=not args.windowed)
        run['measured_refresh_hz'] = win.ssvep_refresh
        save()
        if not wait_for_key(win, WARNING) or not wait_for_board(win, recorder, 'Setting up Knight IMU...'):
            return
        squares, cues, labels = build_stimuli(win)
        center = visual.TextStim(win, text='+', pos=(0, 0), color='gray', height=.06)
        if not wait_for_key(win, f'Fresh A/B test: {mapping}\n\nLook at the red-cued square.\n'
                            'For CENTER trials, keep looking at the + between the squares.\n'
                            'Predictions are hidden until the test ends.\n\nSPACE starts; Escape stops.'):
            return
        for index, target in enumerate(schedule):
            label = 'CENTER (+)' if target is None else cfg.TARGET_LETTERS[target]
            show_message(win, f'{index+1}/{len(schedule)}: look at {label}')
            core.wait(cfg.CUE_DURATION)
            seen = recorder.prediction_count.value
            dropped = win.nDroppedFrames
            interval_start = len(win.frameIntervals)
            if not run_trial(win, squares, cues, -1 if target is None else target,
                             recorder, cfg.LIVE_MARKER, overlay=[*labels, center]):
                break
            late = win.nDroppedFrames - dropped
            intervals = list(win.frameIntervals[interval_start:])
            recorder.request_prediction()
            deadline = time.monotonic()+5
            while recorder.prediction_count.value == seen and time.monotonic() < deadline:
                if recorder.failed.value or not recorder.is_alive():
                    raise RuntimeError('Recorder failed; see terminal')
                core.wait(.01)
            responded = recorder.prediction_count.value != seen
            choice = recorder.last_prediction.value if responded else -1
            contrast = recorder.last_sigma.value if responded else float('-inf')
            accepted = not late and accept_prediction(choice, contrast, cfg.CONFIDENCE_THRESHOLD)
            rows.append({'trial': index+1, 'target': target, 'prediction': choice,
                         'decoy_contrast': contrast if abs(contrast) != float('inf') else None,
                         'accepted': bool(accepted), 'late_frames': late,
                         'frame_intervals_seconds': intervals})
            save()
            recorder.request_save()
            if not responded:
                raise RuntimeError('Prediction timed out; stopping to avoid stale responses')
        print(json.dumps(summarize(rows), indent=2))
        wait_for_key(win, 'Test finished. Results saved.\n\nPress SPACE to close.')
    finally:
        recorder.stop()
        recorder.join(timeout=15)
        if win is not None:
            win.close()
        save()
        print(f'Fresh test results: {path / "validation.json"}')


if __name__ == '__main__':
    main()
