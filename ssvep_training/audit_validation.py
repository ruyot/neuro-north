"""Replay a recorded validation against a separate frozen calibration.

No headset/display required. Diagnostic experiments do not change live settings.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from . import cca_model, config as cfg
from .quality import trial_transport_ok
from .session import analysis_slice, epoch_at, history_samples, load_session
from .trca_model import fit, load_calibration, scores as trca_scores


def confusion(targets, predictions):
    matrix = np.zeros((2, 2), dtype=int)
    for target, prediction in zip(targets, predictions):
        if target in (0, 1) and prediction in (0, 1):
            matrix[target, prediction] += 1
    return {'correct': int(matrix.trace()), 'total': int(matrix.sum()),
            'confusion': matrix.tolist()}


def audit(calibration, validation):
    data, meta = load_session(validation)
    log = json.loads((Path(validation) / 'validation.json').read_text())
    if log.get('swap_comparison'):
        raise ValueError('Mixed-layout diagnostic must be evaluated by mapping, not frozen-layout TRCA')
    methods = {'integer_frame_cycles_v1': 'flicker', 'motion_grating_reversal_v1': 'motion'}
    cfg.configure_stimulus(methods[meta['stimulus_method']])
    trials = load_calibration(calibration)
    _, train_meta = load_session(calibration)
    for key in ('rate', 'eeg_rows', 'frequencies', 'stimulus_method', 'flicker_duration', 'processing'):
        if meta[key] != train_meta[key]:
            raise ValueError(f'Calibration and validation differ in {key}')
    onsets = np.flatnonzero(data[meta['marker_row']] == cfg.LIVE_MARKER)
    rows = log['trials']
    if len(onsets) != len(rows) or [r['target'] for r in rows] != log['schedule'][:len(rows)]:
        raise ValueError('Marker count / trial log / cue schedule disagree; cannot safely align trials')
    model = fit(trials)
    harmonics = cca_model.harmonics_for(trials.freqs)
    details, epochs, targets = [], [], []
    for onset, row in zip(onsets, rows):
        epoch = epoch_at(data[trials.rows], int(onset), trials.rate)
        valid = (epoch is not None and np.isfinite(epoch).all()
                 and not np.any(epoch.std(axis=0) < 1e-9) and not row['late_frames']
                 and trial_transport_ok(data, int(onset), trials.rate,
                                        analysis_slice(trials.rate).stop, history_samples(trials.rate)))
        if not valid:
            details.append({'trial': row['trial'], 'valid': False})
            continue
        cs = cca_model.scores(epoch, trials.rate, trials.freqs, harmonics)
        ts = trca_scores(model, epoch)
        contrast = cca_model.confidence(epoch, trials.rate, trials.freqs, harmonics)
        epochs.append(epoch)
        targets.append(row['target'])
        details.append({'trial': row['trial'], 'valid': True, 'target': row['target'],
                        'cca': int(cs.argmax()), 'trca': int(ts.argmax()),
                        'cca_scores': cs.tolist(), 'trca_scores': ts.tolist(),
                        'live_cca_matches': (int(cs.argmax()) == row['prediction']
                            and row['decoy_contrast'] is not None
                            and bool(np.isclose(contrast, row['decoy_contrast'], atol=1e-8)))
                            if log['decoder'] == 'cca' else None})
    valid_rows = [r for r in details if r['valid']]
    report = {'calibration': str(calibration), 'validation': str(validation),
              'note': 'Forced choices, no rejection gate. Exploratory comparisons do not change live settings.',
              'marker_count': len(onsets), 'trial_count': len(rows),
              'excluded_quality_trials': len(rows) - len(valid_rows),
              'cca': confusion(targets, [r['cca'] for r in valid_rows]),
              'frozen_trca': confusion(targets, [r['trca'] for r in valid_rows]),
              'live_cca_mismatches': [r['trial'] for r in valid_rows if r['live_cca_matches'] is False],
              'trials': details}
    if not epochs:
        return report
    for name, values in [('calibration', np.moveaxis(trials.eeg, 2, 0)), ('validation', epochs)]:
        report[name + '_signal_summary'] = {
            'median_channel_std': np.median([e.std(axis=0) for e in values], axis=0).tolist(),
            'median_channel_correlation': np.median([np.corrcoef(e.T) for e in values], axis=0).tolist()}
    if cfg.STIMULUS_MODE == 'motion':
        # Explicitly exploratory: do not assume reversal-only references are
        # optimal, or promote a variant just because this test favors it.
        freqs = [f / 2 for f in trials.freqs]
        h = 4
        predict = lambda e: int(cca_model.scores(e, trials.rate, freqs, h).argmax())
        report['exploratory_full_cycle_cca'] = {
            'frequencies': freqs, 'harmonics': h,
            'calibration': confusion(trials.targets, [predict(e) for e in np.moveaxis(trials.eeg, 2, 0)]),
            'validation': confusion(targets, [predict(e) for e in epochs])}
    return report


def export_trca_evidence(report, destination, minimum=.1):
    """Export clearly labeled offline replay for the autocomplete adapter.

    Preserve the original CCA recording/log. Only quality-valid replayed trials
    supply evidence, and live RangeEvidence re-applies its selection policy.
    """
    from .validate_live import accept_trca, summarize
    source = Path(report['validation'])
    metadata = json.loads((source / 'session.json').read_text())
    original = json.loads((source / 'validation.json').read_text())
    rows = []
    for row in report['trials']:
        if not row['valid']:
            continue
        ordered = sorted(row['trca_scores'])
        peak, margin = ordered[-1], ordered[-1] - ordered[-2]
        rows.append({'trial': row['trial'], 'target': row['target'],
                     'prediction': row['trca'], 'trca_peak': peak, 'trca_margin': margin,
                     'accepted': accept_trca(row['trca'], peak, minimum), 'late_frames': 0})
    payload = {'decoder': 'trca', 'calibration_session': str(Path(report['calibration']).resolve()),
               'trca_min_peak': minimum, 'trials': rows, 'summary': summarize(rows),
               'provenance': {'kind': 'offline_replay', 'source_validation': str(source.resolve()),
                              'source_decoder': original['decoder']},
               'note': 'Offline frozen-TRCA replay, not a new prospective TRCA recording. '
                       'Autocomplete re-applies the requested live weak-score policy.'}
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / 'session.json').write_text(json.dumps(metadata, indent=2, allow_nan=False))
    (destination / 'validation.json').write_text(json.dumps(payload, indent=2, allow_nan=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--calibration', required=True)
    parser.add_argument('--validation', required=True)
    parser.add_argument('--export-evidence', help='directory for labeled offline TRCA language evidence')
    args = parser.parse_args()
    report = audit(args.calibration, args.validation)
    output = Path(cfg.RESULTS_DIR) / ('audit_' + Path(args.validation).name + '.json')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False))
    if args.export_evidence:
        export_trca_evidence(report, args.export_evidence)
        print(f'Offline TRCA language evidence: {args.export_evidence}')
    for name in ('cca', 'frozen_trca', 'live_cca_mismatches', 'exploratory_full_cycle_cca'):
        if name in report:
            print(name, report[name])
    print(f'Saved {output}')


if __name__ == '__main__':
    main()
