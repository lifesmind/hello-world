"""Post hoc image-similarity grouping sensitivity; never patient identification.

Keep the original analysis unchanged. Move each connected similarity component
to its minimum original fold, keeping all singleton assignments unchanged.
The secondary plan is recorded before any secondary classifier is fitted.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

import make_reanalysis_common as c


def run(output_root):
    out = Path(output_root)
    data = c.load_inputs(out / 'inputs', out / 'plan/MAKE_Analysis_Plan_v1.json')
    c.init_output(out, data, [__file__])
    c.require(not (out / 'logs/proxy_group_sensitivity_audit.json').exists(), 'Secondary grouping sensitivity already completed')
    ddi = data['datasets']['DDI']
    pairs = c.read_csv(out / 'results/DDI_similarity_candidate_pairs.csv')
    ids = ddi['ids'].tolist()
    lookup = {image: i for i, image in enumerate(ids)}
    parent = list(range(len(ids)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for pair in pairs:
        a, b = find(lookup[pair['left_image_id']]), find(lookup[pair['right_image_id']])
        parent[max(a, b)] = min(a, b)
    components = {}
    for i in range(len(ids)):
        components.setdefault(find(i), []).append(i)
    groups = np.full(len(ids), '', dtype='<U96')
    folds = np.full(len(ids), -1, dtype=np.int8)
    memberships = []
    for members in components.values():
        group = 'image_similarity:' + min(ids[i] for i in members)
        fold = int(ddi['fold'][members].min())
        for i in members:
            groups[i], folds[i] = group, fold
            memberships.append({'analysis_image_id': ids[i], 'proxy_group_id': group, 'component_size': len(members),
                                'original_cv_fold': int(ddi['fold'][i]), 'proxy_cv_fold': fold,
                                'skin_tone_group': ddi['tones'][i], 'target_label': int(ddi['y'][i]), 'patient_id_inferred': False})
    membership_path = out / 'splits/DDI_proxy_image_groups.csv'
    c.write_csv(membership_path, sorted(memberships, key=lambda r: r['analysis_image_id']))
    plan = {'stage': 'post_hoc_image_similarity_grouping_sensitivity', 'locked_utc': c.utc_now(),
            'primary_metrics_already_inspected': True, 'secondary_scores_not_yet_computed': True,
            'registered_preregistration': False, 'primary_plan_sha256': data['plan_sha256'],
            'similarity_pair_csv_sha256': c.file_sha(out / 'results/DDI_similarity_candidate_pairs.csv'),
            'group_and_fold_csv_sha256': c.file_sha(membership_path),
            'code_sha256': {Path(p).name: c.file_sha(p) for p in (__file__, c.__file__)},
            'grouping': 'connected components of the locked DCT-hash distance <=4 diagnostic pairs; singleton images retained',
            'fold_rule': 'component assigned to minimum of its original saved folds; all singleton assignments preserved',
            'images': len(ids), 'image_similarity_components': len(components), 'non_singleton_components': sum(len(m) > 1 for m in components.values()),
            'images_moved_to_another_fold': int((folds != ddi['fold']).sum()),
            'C': 'reuse final PAD-only selected C; no tuning on DDI', 'arms': ['PAD_baseline', 'target_unweighted', 'target_weighted'],
            'baseline': 'reuse original PAD-only scores',
            'resampling': '10000 uniform component bootstrap draws shared across arms; include all images with component multiplicity',
            'bootstrap_seed': [20261004, 702], 'CI': 'conditional unadjusted percentile 95%',
            'patient_independence_established': False, 'historical_Table_S13_reproduced': False,
            'historical_subset_or_folds_reconstructed': False, 'primary_folds_or_primary_results_modified': False}
    plan_path = out / 'plan/MAKE_Proxy_Group_Plan_v1.json'
    c.write_json(plan_path, plan)
    secondary_sha = c.file_sha(plan_path)
    fit_audit = json.loads((out / 'logs/classifier_fit_audit.json').read_text())
    C = fit_audit['C_selected_from_PAD_only']
    with np.load(out / 'predictions/core_scores.npz', allow_pickle=False) as f:
        cols = f['DDI_arm_names'].tolist()
        baseline = f['DDI_logits'][:, cols.index('PAD_baseline')].copy()
    names = ['PAD_baseline', 'target_unweighted', 'target_weighted']
    scores = np.full((len(ids), 3), np.nan)
    model_ids = np.full(scores.shape, '', dtype='<U64')
    scores[:, 0], model_ids[:, 0] = baseline, 'PAD_baseline'
    store = c.FitStore(data)
    arm_rows, contrast_rows = [], []
    strata = ['ALL_DDI', 'I-II', 'III-IV', 'V-VI']
    points = np.zeros((4, 3, 3))
    samples = np.zeros((4, 3, 10000, 3))
    metric_names = ['AUROC', 'sensitivity_FPR_le_0_50', 'achieved_FPR']
    with threadpool_limits(limits=data['plan']['classifier']['threads']):
        for fold in range(5):
            heldout = np.flatnonzero(folds == fold)
            added = np.flatnonzero((folds != fold) & (ddi['tones'] == 'V-VI'))
            c.require(not set(groups[added]).intersection(groups[heldout]), 'Similarity component leaks across training and testing')
            for col, weighted in ((1, False), (2, True)):
                m = np.zeros(len(data['ids']), dtype=np.uint16)
                m[:data['n_pad']] = 1
                m[data['n_pad'] + added] = 1
                model_id = f'proxy_{names[col]}_fold_{fold}'
                mr = store.fit(model_id, m, C, source_weighted=weighted, purpose='post_hoc_proxy_group_sensitivity',
                               heldout_fold=fold, secondary_plan_sha256=secondary_sha)
                c.require(not np.any(store.weights[mr][data['n_pad'] + heldout]), 'Secondary heldout image fitted')
                scores[heldout, col] = store.predict(mr, data['n_pad'] + heldout)
                model_ids[heldout, col] = model_id
        c.require(np.isfinite(scores).all(), 'Missing secondary heldout score')
        unique, inverse = np.unique(groups, return_inverse=True)
        rng = np.random.default_rng(np.random.SeedSequence([20261004, 702]))
        component_counts = rng.multinomial(len(unique), np.full(len(unique), 1 / len(unique)), size=10000).astype(np.uint16)
        weights = component_counts[:, inverse]
        for si, stratum in enumerate(strata):
            mask = np.ones(len(ids), dtype=bool) if stratum == 'ALL_DDI' else ddi['tones'] == stratum
            for col, arm in enumerate(names):
                p = c.point_metrics(ddi['y'][mask], scores[mask, col])
                boot = c.bootstrap_metrics(ddi['y'][mask], scores[mask, col], weights[:, mask])
                points[si, col] = [p[m] for m in metric_names]
                samples[si, col] = boot
                row = {'stratum': stratum, 'arm': arm, 'n_images': int(mask.sum()), 'n_proxy_components': len(np.unique(groups[mask])),
                       'positive': int(ddi['y'][mask].sum()), 'secondary_plan_sha256': secondary_sha}
                for j, metric in enumerate(metric_names):
                    lo, hi = c.percentile_ci(boot[:, j])
                    row.update({metric: p[metric], metric + '_CI_low': lo, metric + '_CI_high': hi})
                arm_rows.append(row)
            for col in (1, 2):
                for j, metric in enumerate(metric_names[:2]):
                    lo, hi = c.percentile_ci(samples[si, col, :, j] - samples[si, 0, :, j])
                    contrast_rows.append({'stratum': stratum, 'arm': names[col], 'reference_arm': 'PAD_baseline', 'metric': metric,
                                          'baseline': float(points[si, 0, j]), 'adapted': float(points[si, col, j]),
                                          'difference_unrounded': float(points[si, col, j] - points[si, 0, j]), 'CI_low': lo, 'CI_high': hi,
                                          'CI_type': 'conditional component bootstrap percentile 95%; post hoc',
                                          'historical_Table_S13_reproduced': False})
    store.save(out / 'models/proxy_group_models')
    c.save_npz(out / 'predictions/proxy_group_scores.npz', DDI_image_ids=ddi['ids'], DDI_arm_names=np.array(names), DDI_logits=scores,
               DDI_model_ids=model_ids, proxy_group_ids=groups, proxy_fold=folds, secondary_plan_sha256=np.array(secondary_sha))
    c.write_csv(out / 'predictions/proxy_group_per_image_predictions.csv', [
        {'analysis_image_id': image, 'target_label': int(ddi['y'][i]), 'skin_tone_group': ddi['tones'][i],
         'proxy_group_id': groups[i], 'heldout_proxy_fold': int(folds[i]), 'arm': arm, 'logit_score': float(scores[i, col]),
         'model_id': model_ids[i, col], 'feature_cache_id': data['plan']['feature_cache_id'], 'secondary_plan_sha256': secondary_sha}
        for i, image in enumerate(ids) for col, arm in enumerate(names)])
    c.write_csv(out / 'results/Table_RS13_proxy_group_arms.csv', arm_rows)
    c.write_csv(out / 'results/Table_RS13_proxy_group_changes.csv', contrast_rows)
    c.save_npz(out / 'uncertainty/proxy_group_samples.npz', stratum_names=np.array(strata), arm_names=np.array(names),
               metric_names=np.array(metric_names), component_ids=unique, component_bootstrap_counts=component_counts,
               point_metrics=points, bootstrap_metrics=samples, secondary_plan_sha256=np.array(secondary_sha))
    c.write_json(out / 'logs/proxy_group_sensitivity_audit.json', {'completed_utc': c.utc_now(), 'secondary_plan': plan,
                  'secondary_plan_sha256': secondary_sha, 'classifier_fit_audit_sha256': c.file_sha(out / 'logs/classifier_fit_audit.json'),
                  'fitted_heads': len(store.models), 'component_separation_checks': 'PASS', 'saved_coefficient_reconstruction_checks': 'PASS',
                  'primary_results_modified': False, 'historical_Table_S13_reproduced': False,
                  'output_sha256': {str(p.relative_to(out)): c.file_sha(p) for p in [
                      plan_path, membership_path, out / 'models/proxy_group_models.npz', out / 'models/proxy_group_models_registry.csv',
                      out / 'predictions/proxy_group_scores.npz', out / 'predictions/proxy_group_per_image_predictions.csv',
                      out / 'results/Table_RS13_proxy_group_arms.csv', out / 'results/Table_RS13_proxy_group_changes.csv',
                      out / 'uncertainty/proxy_group_samples.npz']}})
    print('POST HOC PROXY GROUP SENSITIVITY: PASS;', len(components), 'components;', int((folds != ddi['fold']).sum()), 'fold moves')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True)
    args = p.parse_args()
    run(args.output)
