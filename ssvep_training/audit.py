"""Replay recordings without a headset/display: python -m ssvep_training.audit."""
from __future__ import annotations
import argparse
import glob
import json
from pathlib import Path
import numpy as np
from scipy.stats import binomtest
from . import config as cfg, cca_model
from .session import load_session, load_trials
from .quality import packet_discontinuities


def replay_validation(path, data, meta):
    """Join marker-99 epochs to completed trials using saved trial numbers.

    Use the recording's timing/frequencies, not the current experiment config.
    """
    from . import knight
    from .validate_live import summarize
    validation = json.loads((Path(path) / 'validation.json').read_text())
    settings = meta['processing']
    if tuple(settings['filter_band']) != (1.0, 40.0):
        raise ValueError('Replay clean() supports only the recorded 1–40 Hz filter')
    rate = meta['rate']
    history = round(settings['filter_history'] * rate)
    latency = round(settings['visual_latency'] * rate)
    length = round(settings['gaze_duration'] * rate)
    onsets = np.flatnonzero(data[meta['marker_row']] == cfg.LIVE_MARKER)
    frequencies = meta['frequencies']
    harmonics = max(1, min(sum(h*f < settings['filter_band'][1] for h in (1, 2, 3))
                           for f in frequencies))
    replay, mismatches = [], []
    for row in validation['trials']:
        index = row['trial'] - 1
        if not 0 <= index < len(onsets):
            raise ValueError(f"No onset marker for completed trial {row['trial']}")
        onset = int(onsets[index])
        end = onset + latency + length
        if onset < history or end > data.shape[1]:
            raise ValueError(f"Incomplete raw epoch for trial {row['trial']}")
        segment = data[meta['eeg_rows'], onset-history:end]
        epoch = knight.clean(segment, rate)[:, history+latency:].T
        score = cca_model.scores(epoch, rate, frequencies, harmonics, settings['cca_bands'])
        predicted = int(score.argmax())
        discontinuities = packet_discontinuities(data[0, onset-history:end])
        if not discontinuities and predicted != row['prediction']:
            mismatches.append(row['trial'])
        replay.append({**{k: row[k] for k in ('trial', 'target', 'prediction', 'accepted', 'late_frames')},
                       'replay_prediction': predicted, 'scores': score.tolist(),
                       'packet_discontinuities': discontinuities,
                       'channel_std_uv': epoch.std(axis=0).tolist()})
    per_target = []
    for target, frequency in enumerate(frequencies):
        rows = [r for r in replay if r['target'] == target and r['prediction'] >= 0
                and not r['late_frames'] and not r['packet_discontinuities']]
        per_target.append({'target': meta['letters'][target], 'frequency': frequency,
                           'valid': len(rows),
                           'correct': sum(r['replay_prediction'] == target for r in rows),
                           'median_scores': np.median([r['scores'] for r in rows], axis=0).tolist()
                           if rows else None})
    return {'summary': summarize(validation['trials']), 'per_target': per_target,
            'online_replay_mismatches': mismatches, 'trials': replay,
            'measured_refresh_hz': validation.get('measured_refresh_hz')}


def audit(path):
    data, meta = load_session(path)
    rate = meta['rate']
    timestamp = data[20]  # Knight IMU descriptor: host receive timestamps
    duration = float(timestamp[-1] - timestamp[0]) if len(timestamp) > 1 else 0
    result = {
        'session': Path(path).name, 'samples': data.shape[1], 'channels': meta['eeg_rows'],
        'nominal_rate': rate,
        'host_arrival_rate': (data.shape[1]-1)/duration if duration > 0 else None,
        'packet_discontinuities': packet_discontinuities(data[0]),
        'raw_std_uv': dict(zip(meta['names'], data[meta['eeg_rows']].std(axis=1).tolist())),
    }
    if (Path(path) / 'validation.json').exists():
        result['validation'] = replay_validation(path, data, meta)
        result['labeled_trials'] = sum(r['target'] is not None
                                      for r in result['validation']['trials'])
        return result
    trials = load_trials(path)
    n = len(trials.targets)
    result['labeled_trials'] = n
    if n:
        h = cca_model.harmonics_for(trials.freqs)
        for bands, name in [(1, 'plain_cca'), (cfg.CCA_BANDS, 'bank_cca')]:
            scores = np.array([cca_model.scores(trials.eeg[..., k], rate, trials.freqs, h, bands)
                               for k in range(n)])
            pred = scores.argmax(axis=1)
            hits = pred == trials.targets
            test = binomtest(int(hits.sum()), n, 1 / trials.n_targets, alternative='greater')
            ci = binomtest(int(hits.sum()), n).proportion_ci()
            confusion = np.zeros((trials.n_targets, trials.n_targets), dtype=int)
            np.add.at(confusion, (trials.targets, pred), 1)
            result[name] = {'correct': int(hits.sum()), 'total': n,
                            'accuracy': float(hits.mean()), 'confusion': confusion.tolist(),
                            'accuracy_ci95': [ci.low, ci.high], 'chance_p': test.pvalue}
            if name == 'bank_cca':
                contrast = np.array([cca_model.confidence(trials.eeg[..., k], rate, trials.freqs, h)
                                     for k in range(n)])
                accepted = contrast >= cfg.CONFIDENCE_THRESHOLD
                result[name]['heuristic_gate'] = {
                    'threshold': cfg.CONFIDENCE_THRESHOLD, 'accepted': int(accepted.sum()),
                    'correct': int(hits[accepted].sum()), 'rejected': int((~accepted).sum())}
        result['filtered_median_std_uv'] = dict(zip(trials.names,
            np.median(trials.eeg.std(axis=0), axis=1).tolist()))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session', help='single session; default all recordings')
    parser.add_argument('--output', help='write JSON report')
    args = parser.parse_args()
    paths = [args.session] if args.session else sorted(glob.glob(str(Path(cfg.TRAINING_DATA_DIR) / '*')))
    reports = [audit(p) for p in paths if (Path(p) / 'raw.npz').exists()]
    for r in reports:
        print(f"{r['session']}: {r['samples']} samples, packet discontinuities={r['packet_discontinuities']}")
        if 'validation' in r:
            v = r['validation']
            print('  Online summary:', json.dumps(v['summary']))
            for target in v['per_target']:
                print(f"  {target['target']} ({target['frequency']:g} Hz): "
                      f"{target['correct']}/{target['valid']} correct")
            print('  Online/raw replay mismatches:', v['online_replay_mismatches'])
        for name in ['plain_cca', 'bank_cca']:
            if name in r:
                s = r[name]
                print(f"  {name}: {s['correct']}/{s['total']} ({100*s['accuracy']:.1f}%) "
                      f"CI {100*s['accuracy_ci95'][0]:.0f}-{100*s['accuracy_ci95'][1]:.0f}% "
                      f"confusion={s['confusion']}")
        if not r['labeled_trials']:
            print('  Unlabeled live recording: accuracy cannot be measured.')
    if args.output:
        Path(args.output).write_text(json.dumps({'note': 'Exploratory replay using current processing; '
            'old sessions lack full timing/config metadata. CIs assume independent trials; '
            'reused/tuned recordings are not a prospective accuracy estimate. Host arrival '
            'rate is not a measurement of the ADC clock.', 'sessions': reports}, indent=2))


if __name__ == '__main__':
    main()
