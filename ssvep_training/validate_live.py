"""Prospective A/B and center test; optionally use a frozen TRCA calibration.

python -m ssvep_training.validate_live --channels 1,2,3,4 --blocks 10 --idle-trials 4
"""
from __future__ import annotations
import argparse
import json
import math
import random
import time
from pathlib import Path
from . import config as cfg
from .cca_model import accept_prediction


def accept_trca(choice, peak, minimum, late_frames=0):
    """Frozen peak-only gate, independent of CCA agreement/confidence."""
    return choice >= 0 and not late_frames and math.isfinite(peak) and peak >= minimum


def swap_schedule(blocks, rng):
    """Each block tests each side under each mapping, in randomized order."""
    result = []
    for _ in range(blocks):
        block = [(target, swapped) for swapped in (False, True) for target in (0, 1)]
        rng.shuffle(block)
        result.extend(block)
    return result


def position_choice(frequency_choice, swapped):
    return 1 - frequency_choice if swapped and frequency_choice in (0, 1) else frequency_choice


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
              'idle_trials': len(idle), 'idle_false_activations': sum(bool(r['accepted']) for r in idle)}
    if valid:
        test = binomtest(hits, len(valid), .5, alternative='greater')
        ci = binomtest(hits, len(valid)).proportion_ci()
        result.update(forced_accuracy=hits/len(valid), accuracy_ci95=[ci.low, ci.high],
                      chance_p=test.pvalue)
    if rows and all(r['accepted'] is None for r in rows):
        for key in ('accepted_target_trials', 'correct_accepted', 'idle_false_activations'):
            result.pop(key, None)
        result['rejection_gate_evaluated'] = False
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--channels', required=True, help='physically connected channels, e.g. 1,2,3,4')
    parser.add_argument('--blocks', type=int, default=10, help='one A and one B trial per block')
    parser.add_argument('--idle-trials', type=int, default=4, help='center-fixation controls, interspersed')
    parser.add_argument('--seed', type=int, default=20260919)
    parser.add_argument('--decoder', choices=['cca','trca'], default='cca')
    parser.add_argument('--session', help='frozen calibration required for TRCA')
    parser.add_argument('--trca-min-peak', type=float,
                        help='frozen minimum weighted TRCA correlation; omitted means score collection only')
    parser.add_argument('--port')
    parser.add_argument('--windowed', action='store_true')
    parser.add_argument('--swap-comparison', action='store_true',
                        help='CCA only: four trials/block, both positions under both frequency mappings')
    parser.add_argument("--stimulus", choices=["flicker", "motion"], default=cfg.STIMULUS_MODE)
    args = parser.parse_args()
    cfg.configure_stimulus(args.stimulus)
    if args.swap_comparison and (args.decoder != 'cca' or args.idle_trials != 0):
        parser.error('--swap-comparison requires --decoder cca --idle-trials 0')
    if args.trca_min_peak is not None and (args.decoder != 'trca' or not math.isfinite(args.trca_min_peak)):
        parser.error('--trca-min-peak requires TRCA and a finite threshold')
    channels = [int(c) for c in args.channels.split(',')]
    if not channels or not set(channels) <= set(range(1, 9)) or len(set(channels)) != len(channels):
        parser.error('channels must be unique integers from 1 through 8')
    if args.blocks < 1 or args.idle_trials < 0 or cfg.N_TARGETS != 2:
        parser.error('requires two targets, positive blocks, and nonnegative idle trials')
    if args.decoder == 'trca':
        if not args.session:
            parser.error('--decoder trca requires --session')
        from .trca_model import load_calibration
        try:
            calibration = load_calibration(args.session)
            if calibration.rows != channels:
                parser.error('TRCA channels must match the training session')
        except (ValueError, OSError) as exc:
            parser.error(str(exc))
        print(f'[test] Frozen TRCA calibration: {args.session}')
        print('[test] TRCA score collection only; no rejection gate.' if args.trca_min_peak is None else
              f'[test] Frozen gate: TRCA peak >= {args.trca_min_peak:g}; no margin requirement.')
    mapping = ' / '.join(f'{letter} = {frequency:g} Hz'
                         for letter, frequency in zip(cfg.TARGET_LETTERS, cfg.STIMULUS_FREQUENCIES))
    print(f'[test] {mapping}; {(4 if args.swap_comparison else 2) * args.blocks} target trials + {args.idle_trials} center trials')
    rng = random.Random(args.seed)
    schedule = []
    for _ in range(args.blocks):
        pair = [0, 1]
        rng.shuffle(pair)
        schedule.extend(pair)
    for _ in range(args.idle_trials):
        schedule.insert(rng.randrange(len(schedule)+1), None)
    swapped_schedule = [False] * len(schedule)
    if args.swap_comparison:
        paired = swap_schedule(args.blocks, rng)
        schedule = [target for target, _ in paired]
        swapped_schedule = [swapped for _, swapped in paired]
    path = Path(cfg.TRAINING_DATA_DIR) / time.strftime('validation_%Y%m%d_%H%M%S')
    path.mkdir(parents=True)
    from .recording import RecordingProcess
    recorder = RecordingProcess(port=args.port, channels=channels, session_dir=str(path),
                                decoder=args.decoder, predict_session=args.session if args.decoder == 'trca' else None)
    recorder.start()
    win = None
    rows = []
    run = {'seed': args.seed, 'schedule': schedule, 'channels': channels,
           'swap_comparison': args.swap_comparison, 'swapped_schedule': swapped_schedule,
           'frequencies': list(cfg.STIMULUS_FREQUENCIES), 'letters': list(cfg.TARGET_LETTERS),
           'threshold': cfg.CONFIDENCE_THRESHOLD if args.decoder == 'cca' else None, 'decoder': args.decoder,
           'calibration_session': args.session if args.decoder == 'trca' else None,
           'trca_score_kind': 'weighted template correlation' if args.decoder == 'trca' else None,
           'trca_min_peak': args.trca_min_peak, 'trca_min_margin': None,
           'note': 'Prospective frozen-settings test. Idle means center fixation with both targets flashing.'}

    def save():
        payload = {**run, 'trials': rows, 'summary': summarize(rows)}
        if args.swap_comparison:
            payload['by_mapping'] = {name: summarize([r for r in rows if r['swapped'] == swapped])
                                     for name, swapped in [('normal', False), ('swapped', True)]}
        temporary = path / 'validation.tmp.json'
        temporary.write_text(json.dumps(payload, indent=2, allow_nan=False))
        temporary.replace(path / 'validation.json')

    try:
        from psychopy import core, visual
        from .stimulus import build_window, build_stimuli, run_trial, wait_for_board, wait_for_key, show_message
        win = build_window(fullscreen=not args.windowed)
        run['measured_refresh_hz'] = win.ssvep_refresh
        save()
        if not wait_for_key(win, cfg.stimulus_warning()) or not wait_for_board(win, recorder, 'Setting up Knight IMU...'):
            return
        squares, cues, labels = build_stimuli(win)
        center = visual.TextStim(win, text='+', pos=(0, 0), color='gray', height=.06)
        if not wait_for_key(win, f'Fresh A/B test: {mapping}\n\nLook at the red-cued square.\n'
                            'For CENTER trials, keep looking at the + between the squares.\n'
                            'Predictions are hidden until the test ends.\n\nSPACE starts; Escape stops.'):
            return
        for index, target in enumerate(schedule):
            swapped = swapped_schedule[index]
            freqs = list(reversed(cfg.STIMULUS_FREQUENCIES)) if swapped else list(cfg.STIMULUS_FREQUENCIES)
            label = 'CENTER (+)' if target is None else cfg.TARGET_LETTERS[target]
            show_message(win, f'{index+1}/{len(schedule)}: look at {label}')
            core.wait(cfg.CUE_DURATION)
            seen = recorder.prediction_count.value
            dropped = win.nDroppedFrames
            interval_start = len(win.frameIntervals)
            if not run_trial(win, squares, cues, -1 if target is None else target,
                             recorder, cfg.LIVE_MARKER, overlay=[*labels, center], freqs=freqs):
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
            frequency_choice = choice
            choice = position_choice(choice, swapped)
            contrast = recorder.last_sigma.value if responded else float('-inf')
            accepted = not late and accept_prediction(choice, contrast, cfg.CONFIDENCE_THRESHOLD)
            if args.decoder == 'trca':
                accepted = (None if args.trca_min_peak is None else
                            accept_trca(choice, recorder.last_trca_peak.value, args.trca_min_peak, late))
            rows.append({'trial': index+1, 'target': target, 'prediction': choice,
                         'swapped': swapped, 'display_frequencies_left_right': freqs,
                         'frequency_prediction': frequency_choice,
                         'attended_frequency': freqs[target] if target is not None else None,
                         'decoy_contrast': contrast if abs(contrast) != float('inf') else None,
                         'accepted': accepted, 'late_frames': late,
                         'frame_intervals_seconds': intervals})
            if args.decoder == 'trca':
                rows[-1].update(trca_peak=recorder.last_trca_peak.value if choice >= 0 else None,
                                trca_margin=recorder.last_trca_margin.value if choice >= 0 else None)
            save()
            recorder.request_save()
            if not responded:
                raise RuntimeError('Prediction timed out; stopping to avoid stale responses')
        print(json.dumps(summarize(rows), indent=2))
        if args.swap_comparison:
            for name, swapped in [('normal: left 12 / right 15', False), ('swapped: left 15 / right 12', True)]:
                print(name, json.dumps(summarize([r for r in rows if r['swapped'] == swapped])))
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
