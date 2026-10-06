"""Paired conditional uncertainty for the locked MAKE reanalysis.

All effects are calculated before display rounding. Confidence intervals are
conditional on already fitted heads and the saved folds and lighter panels.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

import make_reanalysis_common as c


FAMILIES = ['PAD_baseline', 'target_unweighted', 'target_weighted', 'lighter_unweighted', 'lighter_weighted']
STRATA = ['ALL_DDI', 'I-II', 'III-IV', 'V-VI']
METRICS = ['AUROC', 'sensitivity_FPR_le_0_50', 'achieved_FPR']
CONTRASTS = [('target_weighted', 'PAD_baseline'), ('target_unweighted', 'PAD_baseline'),
             ('lighter_weighted', 'PAD_baseline'), ('lighter_unweighted', 'PAD_baseline'),
             ('target_weighted', 'lighter_weighted'), ('target_unweighted', 'lighter_unweighted'),
             ('target_weighted', 'target_unweighted')]


def run(output_root):
    out = Path(output_root)
    plan_path = out / 'plan/MAKE_Analysis_Plan_v1.json'
    data = c.load_inputs(out / 'inputs', plan_path)
    c.init_output(out, data, [__file__])
    c.require(not (out / 'logs/conditional_statistics_audit.json').exists(), 'Statistics already saved; verify outputs instead')
    fit_audit = json.loads((out / 'logs/classifier_fit_audit.json').read_text())
    c.require(fit_audit['plan_sha256'] == data['plan_sha256'], 'Classifier plan changed')
    for name, expected in fit_audit['output_sha256'].items():
        c.require(c.file_sha(out / name) == expected, 'Fitted output changed: ' + name)
    started = c.utc_now()
    with np.load(out / 'predictions/core_scores.npz', allow_pickle=False) as saved:
        ddi_scores, arm_names = saved['DDI_logits'].copy(), saved['DDI_arm_names'].tolist()
        pad_scores = saved['PAD_nested_logits'].copy()
    pad, ddi = data['datasets']['PAD'], data['datasets']['DDI']
    B = data['plan']['conditional_uncertainty']['replicates']
    rng = np.random.default_rng(np.random.SeedSequence([20261004, 400]))
    ddi_counts = c.stratified_bootstrap_weights(c.ddi_bootstrap_labels(ddi), B, rng)
    rng = np.random.default_rng(np.random.SeedSequence([20261004, 401]))
    pad_counts = c.pad_cluster_weights(pad, B, rng)
    family_points = np.zeros((len(STRATA), len(FAMILIES), len(METRICS)))
    family_samples = np.zeros((len(STRATA), len(FAMILIES), B, len(METRICS)))
    arm_rows, draw_rows, draw_summaries, transfer_rows = [], [], [], []
    all_draw_points = {}
    with threadpool_limits(limits=data['plan']['classifier']['threads']):
        pad_point = c.point_metrics(pad['y'], pad_scores)
        pad_boot = c.bootstrap_metrics(pad['y'], pad_scores, pad_counts)
        pad_row = {'population': 'PAD_nested_development', 'stratum': 'ALL_PAD', 'n': len(pad_scores),
                   'positive': int(pad['y'].sum()), 'negative': int((pad['y'] == 0).sum()),
                   'resampling_unit': 'source patient cluster', 'bootstrap_replicates': B,
                   'evaluation_type': 'nested patient-grouped development; PAD used during encoder pretraining'}
        for j, metric in enumerate(METRICS):
            lo, hi = c.percentile_ci(pad_boot[:, j])
            pad_row.update({metric: pad_point[metric], metric + '_CI_low': lo, metric + '_CI_high': hi})
        pad_row['retrospective_logit_threshold'] = pad_point['retrospective_logit_threshold']
        transfer_rows.append(pad_row)
        for si, stratum in enumerate(STRATA):
            mask = np.ones(data['n_ddi'], dtype=bool) if stratum == 'ALL_DDI' else ddi['tones'] == stratum
            y, w = ddi['y'][mask], ddi_counts[:, mask]
            n, positive = int(mask.sum()), int(y.sum())
            for fi, family in enumerate(FAMILIES):
                columns = [i for i, name in enumerate(arm_names) if name == family or name.startswith(family + '_draw_')]
                expected = 20 if family.startswith('lighter') else 1
                c.require(len(columns) == expected, 'Missing classifier arms')
                points, samples = [], []
                for col in columns:
                    point = c.point_metrics(y, ddi_scores[mask, col])
                    boot = c.bootstrap_metrics(y, ddi_scores[mask, col], w)
                    points.append([point[m] for m in METRICS])
                    samples.append(boot)
                    if expected == 20:
                        row = {'stratum': stratum, 'arm': family, 'draw_id': int(arm_names[col].split('_')[-1]),
                               'n_evaluation': n, 'positive': positive, 'negative': n - positive,
                               'bootstrap_replicates': B, 'uncertainty': 'conditional paired image bootstrap'}
                        for j, metric in enumerate(METRICS):
                            lo, hi = c.percentile_ci(boot[:, j])
                            row.update({metric: point[metric], metric + '_CI_low': lo, metric + '_CI_high': hi})
                        row['retrospective_logit_threshold'] = point['retrospective_logit_threshold']
                        draw_rows.append(row)
                points = np.array(points)
                means = points.mean(axis=0)
                mean_boot = np.mean(np.stack(samples), axis=0)
                family_points[si, fi] = means
                family_samples[si, fi] = mean_boot
                row = {'stratum': stratum, 'arm': family, 'n_evaluation': n, 'positive': positive, 'negative': n - positive,
                       'matched_draws': expected, 'point_estimator': 'mean of per-draw metrics' if expected == 20 else 'single saved score vector',
                       'bootstrap_replicates': B, 'uncertainty': 'conditional paired image bootstrap; fold x tone x label stratified'}
                for j, metric in enumerate(METRICS):
                    lo, hi = c.percentile_ci(mean_boot[:, j])
                    row.update({metric: float(means[j]), metric + '_CI_low': lo, metric + '_CI_high': hi})
                arm_rows.append(row)
                if family == 'PAD_baseline':
                    transfer_rows.append({'population': 'DDI_PAD_only_transfer', 'stratum': stratum, 'n': n,
                                          'positive': positive, 'negative': n - positive,
                                          'resampling_unit': 'image stratified by saved fold x tone x label',
                                          'evaluation_type': 'previously studied DDI cohort; no DDI row used for PAD head fitting',
                                          **{k: v for k, v in row.items() if k not in ('stratum', 'arm', 'n_evaluation', 'matched_draws', 'point_estimator')}})
                if expected == 20:
                    all_draw_points[(si, fi)] = points
                    summary = {'stratum': stratum, 'arm': family, 'matched_draws': expected,
                               'variability_type': 'between sampled lighter training panels; same evaluation images; not independent validation cohorts'}
                    for j, metric in enumerate(METRICS):
                        summary.update({metric + '_mean': float(means[j]), metric + '_SD': float(points[:, j].std(ddof=1)),
                                        metric + '_minimum': float(points[:, j].min()), metric + '_maximum': float(points[:, j].max()),
                                        metric + '_draw_percentile_2_5': float(np.quantile(points[:, j], 0.025)),
                                        metric + '_draw_percentile_97_5': float(np.quantile(points[:, j], 0.975))})
                    draw_summaries.append(summary)
            print(f'Conditional uncertainty: {stratum} completed ({B} shared bootstrap samples)', flush=True)

    contrasts = []
    for si, stratum in enumerate(STRATA):
        for left, right in CONTRASTS:
            li, ri = FAMILIES.index(left), FAMILIES.index(right)
            for j, metric in enumerate(METRICS[:2]):
                delta = float(family_points[si, li, j] - family_points[si, ri, j])
                lo, hi = c.percentile_ci(family_samples[si, li, :, j] - family_samples[si, ri, :, j])
                contrasts.append({'stratum': stratum, 'metric': metric, 'left_arm': left, 'right_arm': right,
                                  'left_value': float(family_points[si, li, j]), 'right_value': float(family_points[si, ri, j]),
                                  'difference_unrounded': delta, 'CI_low': lo, 'CI_high': hi,
                                  'bootstrap_replicates': B, 'CI_type': 'paired conditional percentile 95%; unadjusted',
                                  'difference_display_3dp': f'{delta:+.3f}'})
    interactions = []
    vi, bi, ti = STRATA.index('V-VI'), FAMILIES.index('PAD_baseline'), FAMILIES.index('target_weighted')
    for reference in ('I-II', 'III-IV'):
        si = STRATA.index(reference)
        for j, metric in enumerate(METRICS[:2]):
            delta_vvi = family_points[vi, ti, j] - family_points[vi, bi, j]
            delta_other = family_points[si, ti, j] - family_points[si, bi, j]
            samples = (family_samples[vi, ti, :, j] - family_samples[vi, bi, :, j]) - (family_samples[si, ti, :, j] - family_samples[si, bi, :, j])
            lo, hi = c.percentile_ci(samples)
            interactions.append({'metric': metric, 'reference_stratum': reference, 'V_VI_change': float(delta_vvi),
                                 'reference_change': float(delta_other), 'difference_in_changes': float(delta_vvi - delta_other),
                                 'CI_low': lo, 'CI_high': hi, 'bootstrap_replicates': B, 'CI_type': 'paired conditional percentile 95%; unadjusted'})
    for row in draw_rows:
        si = STRATA.index(row['stratum'])
        for j, metric in enumerate(METRICS[:2]):
            row[metric + '_change_from_PAD_baseline'] = row[metric] - float(family_points[si, 0, j])
    for row in draw_summaries:
        si = STRATA.index(row['stratum'])
        for j, metric in enumerate(METRICS[:2]):
            row[metric + '_mean_change_from_PAD_baseline'] = row[metric + '_mean'] - float(family_points[si, 0, j])
    results = out / 'results'
    c.write_csv(results / 'Table_R2_PAD_development_DDI_transfer.csv', transfer_rows)
    c.write_csv(results / 'Table_R3_DDI_arms.csv', arm_rows)
    c.write_csv(results / 'Table_R4_matched_controls.csv', contrasts)
    c.write_csv(results / 'Table_R6_repeated_draws.csv', draw_rows)
    c.write_csv(results / 'matched_draw_summary.csv', draw_summaries)
    c.write_csv(results / 'stratum_interactions.csv', interactions)
    groups, inverse = np.unique(pad['groups'], return_inverse=True)
    first_index = np.array([np.flatnonzero(inverse == g)[0] for g in range(len(groups))])
    c.save_npz(out / 'uncertainty/conditional_samples.npz', stratum_names=np.array(STRATA), family_names=np.array(FAMILIES),
               metric_names=np.array(METRICS), DDI_image_ids=ddi['ids'], DDI_bootstrap_counts=ddi_counts,
               PAD_patient_groups=groups, PAD_patient_bootstrap_counts=pad_counts[:, first_index],
               family_point_metrics=family_points, family_bootstrap_metrics=family_samples,
               PAD_point_metrics=np.array([pad_point[m] for m in METRICS]), PAD_bootstrap_metrics=pad_boot,
               plan_sha256=np.array(data['plan_sha256']))
    c.write_json(out / 'logs/conditional_statistics_audit.json', {
        'started_utc': started, 'completed_utc': c.utc_now(), 'plan_sha256': data['plan_sha256'],
        'classifier_fit_audit_sha256': c.file_sha(out / 'logs/classifier_fit_audit.json'),
        'code_sha256': {Path(p).name: c.file_sha(p) for p in (__file__, c.__file__)},
        'runtime': c.environment(), 'replicates': B, 'estimates_and_effects_computed_before_display_rounding': True,
        'intervals_are_conditional': True, 'multiple_comparison_adjustment': False,
        'PAD_patient_bootstrap_preserves_all_source_patient_images': True,
        'DDI_patient_independence_established': False, 'historical_results_verified': False,
        'output_sha256': {str(p.relative_to(out)): c.file_sha(p) for p in [
            results / 'Table_R2_PAD_development_DDI_transfer.csv', results / 'Table_R3_DDI_arms.csv',
            results / 'Table_R4_matched_controls.csv', results / 'Table_R6_repeated_draws.csv',
            results / 'matched_draw_summary.csv', results / 'stratum_interactions.csv',
            out / 'uncertainty/conditional_samples.npz']}})
    print('CONDITIONAL STATISTICS: PASS', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True)
    args = p.parse_args()
    run(args.output)
