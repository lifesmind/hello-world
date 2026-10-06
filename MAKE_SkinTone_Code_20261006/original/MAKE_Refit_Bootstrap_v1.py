"""Patient-bootstrap PAD fitting and image-bootstrap DDI adaptation.

This fixed-fold sensitivity analysis refits 6 heads per bootstrap replicate.
It is not independent external replication and does not redo C selection.
Interrupted runs resume completed, verified replicate files.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

import make_reanalysis_common as c


STRATA = ['ALL_DDI', 'I-II', 'III-IV', 'V-VI']
METRICS = ['AUROC', 'sensitivity_FPR_le_0_50', 'achieved_FPR']
ARMS = ['PAD_baseline', 'target_weighted', 'target_weighted_minus_PAD_baseline']


def run(output_root):
    out = Path(output_root)
    data = c.load_inputs(out / 'inputs', out / 'plan/MAKE_Analysis_Plan_v1.json')
    c.init_output(out, data, [__file__])
    c.require(not (out / 'logs/refitting_statistics_audit.json').exists(), 'Refitting already completed; verify outputs instead')
    fit_audit = json.loads((out / 'logs/classifier_fit_audit.json').read_text())
    c.require(fit_audit['plan_sha256'] == data['plan_sha256'], 'Classifier plan changed')
    c.require(fit_audit['code_sha256'][Path(c.__file__).name] == c.file_sha(c.__file__), 'Fitting primitives changed')
    for name, expected in fit_audit['output_sha256'].items():
        c.require(c.file_sha(out / name) == expected, 'Fitted output changed: ' + name)
    started = c.utc_now()
    C = fit_audit['C_selected_from_PAD_only']
    pad, ddi = data['datasets']['PAD'], data['datasets']['DDI']
    n_pad, total = data['n_pad'], len(data['ids'])
    B = data['plan']['refitting_uncertainty']['replicates']
    code_sha = c.file_sha(__file__)
    with np.load(out / 'predictions/core_scores.npz', allow_pickle=False) as f:
        arms = f['DDI_arm_names'].tolist()
        original_scores = f['DDI_logits'][:, [arms.index('PAD_baseline'), arms.index('target_weighted')]].copy()
    originals = np.zeros((len(STRATA), len(ARMS), len(METRICS)))
    masks = [np.ones(len(ddi['y']), dtype=bool) if s == 'ALL_DDI' else ddi['tones'] == s for s in STRATA]
    for si, mask in enumerate(masks):
        for arm in (0, 1):
            pm = c.point_metrics(ddi['y'][mask], original_scores[mask, arm])
            originals[si, arm] = [pm[m] for m in METRICS]
        originals[si, 2] = originals[si, 1] - originals[si, 0]
    metrics = np.zeros((len(STRATA), len(ARMS), B, len(METRICS)))
    records, replicate_files = [], []
    with threadpool_limits(limits=data['plan']['classifier']['threads']):
        for b in range(B):
            path = out / f'models/refitting/replicate_{b:04d}.npz'
            if path.exists():
                with np.load(path, allow_pickle=False) as f:
                    c.require(str(f['plan_sha256']) == data['plan_sha256'] and str(f['code_sha256']) == code_sha, 'Resumed replicate provenance differs')
                    c.require(np.array_equal(f['DDI_image_ids'], ddi['ids']) and float(f['C']) == C, 'Resumed replicate input/order differs')
                    scores, evaluation_counts = f['DDI_logits'].copy(), f['DDI_bootstrap_counts'].copy()
                    registry = json.loads(str(f['model_registry_json']))
            else:
                pad_rng = np.random.default_rng(np.random.SeedSequence([20261004, 300, b]))
                ddi_rng = np.random.default_rng(np.random.SeedSequence([20261004, 301, b]))
                pad_counts = c.pad_cluster_weights(pad, 1, pad_rng)[0]
                evaluation_counts = c.stratified_bootstrap_weights(c.ddi_bootstrap_labels(ddi), 1, ddi_rng)[0]
                store = c.FitStore(data)
                multiplicity = np.zeros(total, dtype=np.uint16)
                multiplicity[:n_pad] = pad_counts
                baseline_id = f'refit_{b:04d}_PAD_baseline'
                mr = store.fit(baseline_id, multiplicity, C, purpose='bootstrap_PAD_only', bootstrap_replicate=b, fixed_C_from='PAD_final_CV')
                scores = np.full((data['n_ddi'], 2), np.nan)
                score_models = np.full(scores.shape, '', dtype='<U64')
                scores[:, 0] = store.predict(mr, n_pad + np.arange(data['n_ddi']))
                score_models[:, 0] = baseline_id
                for fold in range(5):
                    heldout = np.flatnonzero(ddi['fold'] == fold)
                    eligible = (ddi['fold'] != fold) & (ddi['tones'] == 'V-VI')
                    multiplicity = np.zeros(total, dtype=np.uint16)
                    multiplicity[:n_pad] = pad_counts
                    multiplicity[n_pad:] = evaluation_counts * eligible
                    model_id = f'refit_{b:04d}_target_weighted_fold_{fold}'
                    mr = store.fit(model_id, multiplicity, C, source_weighted=True, purpose='bootstrap_DDI_heldout_adaptation',
                                   heldout_fold=fold, bootstrap_replicate=b, fixed_C_from='PAD_final_CV')
                    c.require(not np.any(store.weights[mr][n_pad + heldout]), 'Bootstrap test fold enters fitting')
                    scores[heldout, 1] = store.predict(mr, n_pad + heldout)
                    score_models[heldout, 1] = model_id
                c.require(np.isfinite(scores).all(), 'Missing bootstrap predictions')
                registry = store.models
                c.save_npz(path, coefficients=np.stack(store.coefs), intercepts=np.array(store.intercepts), classes=np.array([0, 1]),
                           model_ids=np.array([r['model_id'] for r in store.models]), model_registry_json=np.array(json.dumps(registry, sort_keys=True)),
                           effective_sample_weights=np.stack(store.weights), training_multiplicities=np.stack(store.multiplicities),
                           analysis_image_ids=data['ids'], DDI_image_ids=ddi['ids'], DDI_logits=scores, DDI_model_ids=score_models,
                           DDI_bootstrap_counts=evaluation_counts, PAD_patient_image_multiplicities=pad_counts,
                           C=np.array(C), plan_sha256=np.array(data['plan_sha256']), code_sha256=np.array(code_sha),
                           feature_cache_id=np.array(data['plan']['feature_cache_id']))
            for si, mask in enumerate(masks):
                for arm in (0, 1):
                    pm = c.point_metrics(ddi['y'][mask], scores[mask, arm], evaluation_counts[mask])
                    metrics[si, arm, b] = [pm[m] for m in METRICS]
                metrics[si, 2, b] = metrics[si, 1, b] - metrics[si, 0, b]
            records.extend({**r, 'model_npz_path': str(path.relative_to(out))} for r in registry)
            replicate_files.append(path)
            if b == 0 or (b + 1) % 25 == 0:
                print(f'Refitting uncertainty: {b + 1}/{B} replicates saved', flush=True)
    rows = []
    for si, stratum in enumerate(STRATA):
        n, positive = int(masks[si].sum()), int(ddi['y'][masks[si]].sum())
        for ai, arm in enumerate(ARMS):
            for j, metric in enumerate(METRICS[:2]):
                lo, hi = c.percentile_ci(metrics[si, ai, :, j])
                rows.append({'stratum': stratum, 'arm_or_contrast': arm, 'metric': metric,
                             'n_evaluation_original': n, 'positive_original': positive, 'negative_original': n - positive,
                             'original_point_estimate': float(originals[si, ai, j]), 'CI_low': lo, 'CI_high': hi,
                             'bootstrap_mean': float(metrics[si, ai, :, j].mean()), 'bootstrap_SD': float(metrics[si, ai, :, j].std(ddof=1)),
                             'refitting_replicates': B, 'C_fixed': C,
                             'CI_type': 'paired percentile 95%; PAD patient cluster and DDI fold x tone x label image resampling; heads refitted',
                             'uncertainty_excluded': 'C selection; fold selection; encoder pretraining/fitting; unobserved DDI patient dependence',
                             'independent_external_replication': False})
    c.write_csv(out / 'results/Table_R5_refitting_robustness.csv', rows)
    c.write_csv(out / 'models/refitting_registry.csv', records)
    c.save_npz(out / 'uncertainty/refitting_metric_samples.npz', stratum_names=np.array(STRATA), arm_names=np.array(ARMS),
               metric_names=np.array(METRICS), original_point_metrics=originals, bootstrap_metrics=metrics,
               plan_sha256=np.array(data['plan_sha256']))
    c.write_json(out / 'logs/refitting_statistics_audit.json', {
        'started_utc': started, 'completed_utc': c.utc_now(), 'plan_sha256': data['plan_sha256'],
        'classifier_fit_audit_sha256': c.file_sha(out / 'logs/classifier_fit_audit.json'),
        'code_sha256': {Path(p).name: c.file_sha(p) for p in (__file__, c.__file__)}, 'runtime': c.environment(),
        'refitting_replicates': B, 'refitted_head_count': len(records), 'C_fixed_from_original_PAD_selection': C,
        'heldout_training_weight_checks': 'PASS', 'saved_coefficient_reconstruction_checks': 'PASS',
        'independent_external_replication': False, 'DDI_patient_independence_established': False,
        'historical_results_verified': False,
        'output_sha256': {str(p.relative_to(out)): c.file_sha(p) for p in replicate_files + [
            out / 'results/Table_R5_refitting_robustness.csv', out / 'models/refitting_registry.csv',
            out / 'uncertainty/refitting_metric_samples.npz']}})
    print('REFITTING STATISTICS: PASS', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True)
    args = p.parse_args()
    run(args.output)
