"""Compare frozen CCA and trained TRCA on identical quality-checked trials.

python -m ssvep_training.evaluate_trca --session training_data/session_...
No live defaults are changed. Accuracy here is forced A/B choice, not idle safety.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from . import config as cfg
from . import cca_model
from .session import Trials, latest_session
from .trca_model import fit, load_calibration


def subset(trials, mask):
    return Trials(trials.eeg[..., mask], trials.targets[mask], trials.blocks[mask],
                  trials.rate, trials.names, trials.freqs, trials.letters, trials.rows)


def metrics(truth, predicted, count):
    matrix = np.zeros((count, count), dtype=int)
    for actual, choice in zip(truth, predicted):
        matrix[actual, choice] += 1
    return {'trials': len(truth), 'correct': int(np.sum(truth == predicted)),
            'accuracy': float(np.mean(truth == predicted)), 'confusion': matrix.tolist(),
            'per_target_accuracy': [float(row[i]/row.sum()) if row.sum() else None
                                    for i, row in enumerate(matrix)]}


def cca_predictions(trials):
    harmonics = cca_model.harmonics_for(trials.freqs)
    return np.array([int(cca_model.scores(trials.eeg[..., i], trials.rate,
                                        trials.freqs, harmonics).argmax())
                     for i in range(trials.eeg.shape[-1])])


def compare(trials):
    blocks = np.unique(trials.blocks)
    counts = np.bincount(trials.targets, minlength=trials.n_targets)
    if len(blocks) < 3 or min(counts) < 3:
        raise ValueError(f'Need at least 3 usable trials per target across 3 blocks; found {counts.tolist()}')
    cca = cca_predictions(trials)
    trca = np.full(len(trials.targets), -1, dtype=int)
    for block in blocks:
        test = trials.blocks == block
        model = fit(subset(trials, ~test))
        trca[test] = model.predict(trials.eeg[..., test])
    summary = {'leave_one_block_out': {
        'cca': metrics(trials.targets, cca, trials.n_targets),
        'trca': metrics(trials.targets, trca, trials.n_targets),
        'trca_only_correct': int(np.sum((trca == trials.targets) & (cca != trials.targets))),
        'cca_only_correct': int(np.sum((cca == trials.targets) & (trca != trials.targets)))}}
    # A later contiguous segment better reveals within-session drift. Settings
    # are identical to CV; do not select new hyperparameters using these results.
    cut = max(2, int(len(blocks)*2/3))
    test = np.isin(trials.blocks, blocks[cut:])
    train = subset(trials, ~test)
    if test.any() and min(np.bincount(train.targets, minlength=trials.n_targets)) >= 2:
        predictions = fit(train).predict(trials.eeg[..., test])
        summary['later_blocks_holdout'] = {
            'train_blocks': blocks[:cut].tolist(), 'test_blocks': blocks[cut:].tolist(),
            'cca': metrics(trials.targets[test], cca[test], trials.n_targets),
            'trca': metrics(trials.targets[test], predictions, trials.n_targets)}
    summary['trial_predictions'] = [dict(block=int(b), target=int(t), cca=int(c), trca=int(r))
                                    for b,t,c,r in zip(trials.blocks, trials.targets, cca, trca)]
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session', help='training session; default: newest session_*')
    parser.add_argument('--channels', help='explicit recorded-channel subset, e.g. 1,2; raw data stays unchanged')
    parser.add_argument('--test-session', help='optional separate labeled calibration session, never fitted')
    parser.add_argument("--stimulus", choices=["flicker", "motion"], default=cfg.STIMULUS_MODE)
    args = parser.parse_args()
    cfg.configure_stimulus(args.stimulus)
    path = args.session or latest_session()
    if not path:
        parser.error('No calibration found; run sh train_ssvep.sh first')
    try:
        channels = [int(c) for c in args.channels.split(',')] if args.channels else None
        trials = load_calibration(path, channels=channels)
        report = compare(trials)
        if args.test_session:
            if Path(path).resolve() == Path(args.test_session).resolve():
                raise ValueError('Training and test sessions must be different')
            test = load_calibration(args.test_session, channels=channels)
            if (test.rows != trials.rows or test.rate != trials.rate or test.freqs != trials.freqs
                    or not len(test.targets)):
                raise ValueError('Test session must contain usable data with matching channels/rate/frequencies')
            report['separate_session'] = {
                'session': args.test_session, 'rejected_trials': test.skipped,
                'cca': metrics(test.targets, cca_predictions(test), test.n_targets),
                'trca': metrics(test.targets, fit(trials).predict(test.eeg), test.n_targets)}
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    report.update(stimulus_method=cfg.stimulus_method(), session=str(path), channels=trials.rows, frequencies=trials.freqs,
                  usable_trials=len(trials.targets), rejected_trials=trials.skipped,
                  trca_filterbank=[cfg.FILTERBANK[0]], ensemble=cfg.USE_ENSEMBLE_TRCA,
                  note='Forced-choice accuracy. No TRCA idle rejection or autocomplete reliability has been validated.')
    print(f'Channels {trials.rows}; frequencies {trials.freqs}; usable {len(trials.targets)}, rejected {trials.skipped}')
    for name in ('leave_one_block_out', 'later_blocks_holdout', 'separate_session'):
        if name not in report:
            continue
        print(name.replace('_',' ') + ':')
        for method in ('cca','trca'):
            result = report[name][method]
            print(f"  {method.upper()}: {result['correct']}/{result['trials']} = {100*result['accuracy']:.1f}%")
            print(f"    confusion [true A/B rows, predicted A/B columns]: {result['confusion']}")
    suffix = '_ch' + '-'.join(map(str,trials.rows)) if args.channels else ''
    output = Path(cfg.RESULTS_DIR) / f'comparison_{Path(path).name}{suffix}.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False))
    print(f'Saved {output}\nNext: validate the frozen calibration on a separate recording; '
          'these results do not establish idle rejection.')


if __name__ == '__main__':
    main()
