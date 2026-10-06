"""Independently check saved weights, held-out scores and published numbers.

No models are refitted. NumPy and scikit-learn check saved numerical artifacts.
The first 16 conditional bootstrap samples are checked by literal resampling.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
from threadpoolctl import threadpool_limits

import make_reanalysis_common as c


def independent_metrics(y, scores, counts=None):
    if counts is not None:
        indexes = np.repeat(np.arange(len(y)), np.asarray(counts, dtype=int))
        y, scores = y[indexes], scores[indexes]
    fpr, tpr, _ = roc_curve(y, scores, drop_intermediate=False)
    return np.array([roc_auc_score(y, scores), max(tpr[fpr <= 0.50])])


def verify(output_root):
    out = Path(output_root)
    plan = json.loads((out / 'plan/MAKE_Analysis_Plan_v1.json').read_text())
    plan_sha = c.file_sha(out / 'plan/MAKE_Analysis_Plan_v1.json')
    for name, expected in plan['input_sha256'].items():
        c.require(c.file_sha(out / 'inputs' / name) == expected, 'Input checksum changed: ' + name)
    matrices, source_rows = [], []
    for ds in ('PAD', 'DDI'):
        matrices.append(np.load(out / f'inputs/features/{ds}_features_l2.npy', allow_pickle=False).astype(np.float64))
        source_rows.extend(c.read_csv(out / f'inputs/features/{ds}_feature_index.csv'))
    X = np.concatenate(matrices)
    ids = np.array([r['analysis_image_id'] for r in source_rows])
    lookup = {value: i for i, value in enumerate(ids)}
    y = np.array([int(r['target_label']) for r in source_rows])
    folds = np.array([int(r['cv_fold']) for r in source_rows])
    tones = np.array([r['skin_tone_group'] for r in source_rows])
    groups = np.array([r['split_unit_id'] for r in source_rows[:2268]])
    checked_heads = checked_predictions = 0
    audit_hashes = {}
    for name in ('classifier_fit_audit', 'conditional_statistics_audit', 'refitting_statistics_audit',
                 'DDI_similarity_screen_audit', 'proxy_group_sensitivity_audit'):
        path = out / f'logs/{name}.json'
        audit = json.loads(path.read_text())
        for relative, expected in audit['output_sha256'].items():
            c.require(c.file_sha(out / relative) == expected, 'Audited output checksum changed: ' + relative)
        audit_hashes[name] = c.file_sha(path)

    def check_models(f, registry):
        nonlocal checked_heads
        c.require(np.array_equal(f['analysis_image_ids'], ids) and np.array_equal(f['classes'], [0, 1]), 'Model feature/class order differs')
        for i, r in enumerate(registry):
            m = f['training_multiplicities'][i].astype(np.float64)
            w = f['effective_sample_weights'][i]
            coef, bias = f['coefficients'][i], f['intercepts'][i]
            c.require(str(f['model_ids'][i]) == r['model_id'] and int(r['storage_row_index']) == i, 'Model registry ordering differs')
            c.require(c.coefficient_sha(coef, [bias]) == r['coefficient_sha256'], 'Classifier coefficient checksum differs')
            c.require(c.sha_bytes(w.astype('<f8').tobytes()) == r['effective_sample_weight_sha256'], 'Fit-weight checksum differs')
            c.require(c.sha_bytes(m.astype('<f8').tobytes()) == r['training_multiplicity_sha256'], 'Training-row multiplicity checksum differs')
            counts = np.bincount(y, weights=m, minlength=2)
            class_factor = m.sum() / (2 * counts)
            source_factor = np.ones(len(ids))
            source_factor[2268:] = float(r['added_source_multiplier'])
            np.testing.assert_allclose(w, m * class_factor[y] * source_factor, rtol=1e-13, atol=1e-13)
            np.testing.assert_allclose(float(f['C'][i]) if f['C'].ndim else float(f['C']), float(r['C']), rtol=0, atol=0)
            c.require(int(m.sum()) == int(r['training_rows_with_multiplicity']), 'Fit row count differs')
            c.require(not np.any(w[m == 0]), 'Excluded fit rows have nonzero weights')
            checked_heads += 1

    with threadpool_limits(limits=2):
        core_registry = c.read_csv(out / 'models/core_models_registry.csv')
        with np.load(out / 'models/core_models.npz', allow_pickle=False) as f:
            check_models(f, core_registry)
            index = {r['model_id']: i for i, r in enumerate(core_registry)}
            for filename in ('per_image_predictions.csv', 'PAD_tuning_validation_predictions.csv'):
                by_model = {}
                for row in c.read_csv(out / 'predictions' / filename):
                    i = lookup[row['analysis_image_id']]
                    c.require(int(row['target_label']) == y[i], 'Prediction outcome label differs')
                    if 'image_sha256' in row:
                        c.require(row['image_sha256'] == source_rows[i]['image_sha256'], 'Prediction source image differs')
                    by_model.setdefault(row['model_id'], []).append((i, float(row['logit_score'])))
                for model_id, scores in by_model.items():
                    mr = index[model_id]
                    loc = np.array([s[0] for s in scores])
                    actual = np.array([s[1] for s in scores])
                    np.testing.assert_allclose(X[loc] @ f['coefficients'][mr] + f['intercepts'][mr], actual, rtol=1e-11, atol=1e-11)
                    c.require(not np.any(f['effective_sample_weights'][mr, loc]), 'Evaluation row also fitted')
                    if np.all(loc < 2268):
                        train = np.flatnonzero(f['training_multiplicities'][mr, :2268])
                        c.require(not set(groups[train]).intersection(groups[loc]), 'PAD patient spans training/evaluation')
                    elif np.all(loc >= 2268):
                        for fold in np.unique(folds[loc]):
                            test = 2268 + np.flatnonzero(folds[2268:] == fold)
                            c.require(not np.any(f['effective_sample_weights'][mr, test]), 'A DDI evaluation fold also fitted')
                    checked_predictions += len(loc)
        with np.load(out / 'predictions/core_scores.npz', allow_pickle=False) as f:
            ddi_scores, arms = f['DDI_logits'].copy(), f['DDI_arm_names'].tolist()
            pad_scores = f['PAD_nested_logits'].copy()
        families = ['PAD_baseline', 'target_unweighted', 'target_weighted', 'lighter_unweighted', 'lighter_weighted']
        strata = ['ALL_DDI', 'I-II', 'III-IV', 'V-VI']
        points = np.empty((4, 5, 2))
        masks = [np.ones(656, dtype=bool) if s == 'ALL_DDI' else tones[2268:] == s for s in strata]
        for si, mask in enumerate(masks):
            for fi, family in enumerate(families):
                cols = [i for i, a in enumerate(arms) if a == family or a.startswith(family + '_draw_')]
                points[si, fi] = np.mean([independent_metrics(y[2268:][mask], ddi_scores[mask, col]) for col in cols], axis=0)
        for row in c.read_csv(out / 'results/Table_R3_DDI_arms.csv'):
            si, fi = strata.index(row['stratum']), families.index(row['arm'])
            np.testing.assert_allclose([float(row['AUROC']), float(row['sensitivity_FPR_le_0_50'])], points[si, fi], rtol=0, atol=1e-12)
        with np.load(out / 'uncertainty/conditional_samples.npz', allow_pickle=False) as f:
            np.testing.assert_allclose(f['family_point_metrics'][:, :, :2], points, rtol=0, atol=1e-12)
            np.testing.assert_allclose(f['PAD_point_metrics'][:2], independent_metrics(y[:2268], pad_scores), rtol=0, atol=1e-12)
            # Literal duplication and sklearn's independent ROC implementation.
            si, mask = 3, masks[3]
            for b in range(16):
                counts = f['DDI_bootstrap_counts'][b, mask]
                for fi, family in enumerate(families):
                    cols = [i for i, a in enumerate(arms) if a == family or a.startswith(family + '_draw_')]
                    oracle = np.mean([independent_metrics(y[2268:][mask], ddi_scores[mask, col], counts) for col in cols], axis=0)
                    np.testing.assert_allclose(f['family_bootstrap_metrics'][si, fi, b, :2], oracle, rtol=0, atol=1e-12)
            for row in c.read_csv(out / 'results/Table_R4_matched_controls.csv'):
                si = strata.index(row['stratum'])
                li, ri = families.index(row['left_arm']), families.index(row['right_arm'])
                j = 0 if row['metric'] == 'AUROC' else 1
                expected = points[si, li, j] - points[si, ri, j]
                np.testing.assert_allclose(float(row['difference_unrounded']), expected, rtol=0, atol=1e-12)
                ci = np.percentile(f['family_bootstrap_metrics'][si, li, :, j] - f['family_bootstrap_metrics'][si, ri, :, j], [2.5, 97.5])
                np.testing.assert_allclose([float(row['CI_low']), float(row['CI_high'])], ci, rtol=0, atol=1e-12)
        panels = c.read_csv(out / 'splits/DDI_adaptation_samples.csv')
        by_panel = {}
        for row in panels:
            by_panel.setdefault(row['sampling_panel_id'], []).append(row)
        for fold in range(5):
            target = [r for r in panels if r['family'] == 'target' and int(r['heldout_fold']) == fold]
            quotas = [sum(int(r['target_label']) == label for r in target) for label in (0, 1)]
            for draw in range(20):
                rows = by_panel[f'lighter_draw_{draw:02d}_fold_{fold}']
                c.require(len({r['analysis_image_id'] for r in rows}) == len(rows), 'Lighter sampling uses replacement')
                c.require([sum(int(r['target_label']) == label for r in rows) for label in (0, 1)] == quotas, 'Lighter control class counts differ')
                c.require(all(folds[lookup[r['analysis_image_id']]] != fold for r in rows), 'Lighter panel contains test fold')
        with np.load(out / 'uncertainty/refitting_metric_samples.npz', allow_pickle=False) as summary:
            for b, path in enumerate(sorted((out / 'models/refitting').glob('replicate_*.npz'))):
                with np.load(path, allow_pickle=False) as f:
                    registry = json.loads(str(f['model_registry_json']))
                    check_models(f, registry)
                    index = {str(mid): i for i, mid in enumerate(f['model_ids'])}
                    for col in range(2):
                        for mid in np.unique(f['DDI_model_ids'][:, col]):
                            mr = index[str(mid)]
                            loc = np.flatnonzero(f['DDI_model_ids'][:, col] == mid)
                            np.testing.assert_allclose(X[2268 + loc] @ f['coefficients'][mr] + f['intercepts'][mr], f['DDI_logits'][loc, col], rtol=1e-11, atol=1e-11)
                            c.require(not np.any(f['effective_sample_weights'][mr, 2268 + loc]), 'Refit prediction image also fitted')
                            checked_predictions += len(loc)
                    _, inverse = np.unique(groups, return_inverse=True)
                    for g in range(inverse.max() + 1):
                        c.require(len(np.unique(f['PAD_patient_image_multiplicities'][inverse == g])) == 1, 'Refit PAD patient images split in bootstrap')
                    if b < 16:
                        for si, mask in enumerate(masks):
                            for ai in range(2):
                                oracle = independent_metrics(y[2268:][mask], f['DDI_logits'][mask, ai], f['DDI_bootstrap_counts'][mask])
                                np.testing.assert_allclose(summary['bootstrap_metrics'][si, ai, b, :2], oracle, rtol=0, atol=1e-12)
            for row in c.read_csv(out / 'results/Table_R5_refitting_robustness.csv'):
                si = strata.index(row['stratum'])
                ai = ['PAD_baseline', 'target_weighted', 'target_weighted_minus_PAD_baseline'].index(row['arm_or_contrast'])
                j = 0 if row['metric'] == 'AUROC' else 1
                ci = np.percentile(summary['bootstrap_metrics'][si, ai, :, j], [2.5, 97.5])
                np.testing.assert_allclose([float(row['CI_low']), float(row['CI_high'])], ci, rtol=0, atol=1e-12)
        proxy_members = c.read_csv(out / 'splits/DDI_proxy_image_groups.csv')
        proxy_lookup = {r['analysis_image_id']: r for r in proxy_members}
        with np.load(out / 'models/proxy_group_models.npz', allow_pickle=False) as f:
            registry = c.read_csv(out / 'models/proxy_group_models_registry.csv')
            check_models(f, registry)
            index = {r['model_id']: i for i, r in enumerate(registry)}
            by_model = {}
            for row in c.read_csv(out / 'predictions/proxy_group_per_image_predictions.csv'):
                if row['model_id'] == 'PAD_baseline':
                    continue
                by_model.setdefault(row['model_id'], []).append(row)
            for mid, rows in by_model.items():
                mr = index[mid]
                loc = np.array([lookup[r['analysis_image_id']] for r in rows])
                np.testing.assert_allclose(X[loc] @ f['coefficients'][mr] + f['intercepts'][mr], [float(r['logit_score']) for r in rows], rtol=1e-11, atol=1e-11)
                train = np.flatnonzero(f['training_multiplicities'][mr, 2268:]) + 2268
                train_groups = {proxy_lookup[ids[i]]['proxy_group_id'] for i in train}
                test_groups = {proxy_lookup[ids[i]]['proxy_group_id'] for i in loc}
                c.require(not train_groups.intersection(test_groups), 'Similarity component also fitted')
                checked_predictions += len(loc)
    result = {'completed_utc': c.utc_now(), 'status': 'PASS', 'primary_plan_sha256': plan_sha,
              'code_sha256': c.file_sha(__file__), 'audits_verified_sha256': audit_hashes,
              'classifier_heads_with_coefficients_and_fit_weights_verified': checked_heads,
              'saved_heldout_and_tuning_logits_reconstructed': checked_predictions,
              'independent_AUROC_and_step_ROC_checks': 'sklearn roc_auc_score/roc_curve; literal resampling for first 16 bootstrap samples',
              'matched_lighter_panel_count_and_outcome_checks': '100 panels; 20 draws x 5 heldout folds',
              'published_unrounded_differences_and_percentile_limits_checked': True,
              'patient_independence_for_DDI_established': False, 'historical_results_verified': False}
    # Verification is read-only with respect to analytical outputs; this log can be refreshed.
    c.write_file(out / 'logs/independent_verification.json', c.json_bytes(result), immutable=False)
    print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True)
    args = p.parse_args()
    verify(args.output)
