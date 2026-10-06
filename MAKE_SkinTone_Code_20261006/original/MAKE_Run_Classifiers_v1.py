"""Fit PAD development models and held-out DDI controlled comparisons.

Usage: python MAKE_Run_Classifiers_v1.py --inputs INPUT_DIR --plan PLAN.json --output OUT_DIR
The companion make_reanalysis_common.py must be beside this file.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from threadpoolctl import threadpool_limits

import make_reanalysis_common as common


def run(input_root, plan_path, output_root):
    data = common.load_inputs(input_root, plan_path)
    out = common.init_output(output_root, data, [__file__, common.__file__])
    common.require(not (out / 'logs/classifier_fit_audit.json').exists(), 'Core classifiers already fitted; verify saved outputs instead of refitting')
    started = common.utc_now()
    pad, ddi = data['datasets']['PAD'], data['datasets']['DDI']
    store = common.FitStore(data)
    total, n_pad = len(data['ids']), data['n_pad']
    tuning_predictions, tuning_metrics, inner_membership, samples = [], [], [], []

    def tune(scope, pool, local_folds):
        means = []
        for C in data['plan']['classifier']['C_grid']:
            fold_aucs = []
            for k in range(5):
                train, test = pool[local_folds != k], pool[local_folds == k]
                common.require(not set(pad['groups'][train]).intersection(pad['groups'][test]), 'Patient leakage inside PAD tuning')
                m = np.zeros(total, dtype=np.uint16)
                m[train] = 1
                model_id = f'{scope}_C_{C:g}_fold_{k}'
                mr = store.fit(model_id, m, C, purpose='PAD_C_tuning', scope=scope, validation_fold=k)
                scores = store.predict(mr, test)
                auc = float(roc_auc_score(pad['y'][test], scores))
                fold_aucs.append(auc)
                tuning_metrics.append({'scope': scope, 'C': float(C), 'validation_fold': k, 'AUROC': auc,
                                       'validation_n': len(test), 'model_id': model_id})
                for idx, score in zip(test, scores, strict=True):
                    tuning_predictions.append({'analysis_image_id': pad['ids'][idx], 'target_label': int(pad['y'][idx]),
                                               'scope': scope, 'C': float(C), 'validation_fold': k,
                                               'logit_score': float(score), 'model_id': model_id})
            means.append(float(np.mean(fold_aucs)))
        best = max(means)
        selected = next(C for C, value in zip(data['plan']['classifier']['C_grid'], means, strict=True) if best - value <= 1e-12)
        print(f'{scope}: selected C={selected:g}', flush=True)
        return selected, means

    with threadpool_limits(limits=data['plan']['classifier']['threads']):
        all_pad = np.arange(n_pad)
        C, final_means = tune('PAD_final_CV', all_pad, pad['fold'])
        nested_scores = np.full(n_pad, np.nan)
        nested_model_ids = np.full(n_pad, '', dtype='<U64')
        nested_selected = []
        for outer in range(5):
            train = np.flatnonzero(pad['fold'] != outer)
            train = train[np.argsort(pad['ids'][train], kind='stable')]
            test = np.flatnonzero(pad['fold'] == outer)
            common.require(not set(pad['groups'][train]).intersection(pad['groups'][test]), 'Patient leakage in outer PAD fold')
            cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=20261004 + 100 + outer)
            inner = np.full(len(train), -1, dtype=np.int8)
            for k, (_, valid) in enumerate(cv.split(pad['X'][train], pad['y'][train], groups=pad['groups'][train])):
                inner[valid] = k
            common.require((inner >= 0).all(), 'Inner split omitted PAD rows')
            inner_membership.extend({'outer_fold': outer, 'analysis_image_id': pad['ids'][idx],
                                     'split_unit_id': pad['groups'][idx], 'inner_fold': int(inner[i])}
                                    for i, idx in enumerate(train))
            inner_C, _ = tune(f'PAD_outer_{outer}_inner_CV', train, inner)
            nested_selected.append(float(inner_C))
            m = np.zeros(total, dtype=np.uint16)
            m[train] = 1
            model_id = f'PAD_nested_outer_{outer}'
            mr = store.fit(model_id, m, inner_C, purpose='PAD_nested_outer_heldout', outer_fold=outer)
            nested_scores[test] = store.predict(mr, test)
            nested_model_ids[test] = model_id
        common.require(np.isfinite(nested_scores).all() and (nested_model_ids != '').all(), 'Incomplete PAD nested predictions')
        m = np.zeros(total, dtype=np.uint16)
        m[:n_pad] = 1
        baseline_row = store.fit('PAD_baseline', m, C, purpose='PAD_only_transfer', scope='all_PAD')
        arm_names = ['PAD_baseline', 'target_unweighted', 'target_weighted']
        arm_names += [f'lighter_unweighted_draw_{i:02d}' for i in range(20)]
        arm_names += [f'lighter_weighted_draw_{i:02d}' for i in range(20)]
        ddi_scores = np.full((data['n_ddi'], len(arm_names)), np.nan)
        ddi_model_ids = np.full(ddi_scores.shape, '', dtype='<U64')
        ddi_scores[:, 0] = store.predict(baseline_row, n_pad + np.arange(data['n_ddi']))
        ddi_model_ids[:, 0] = 'PAD_baseline'
        for fold in range(5):
            heldout = np.flatnonzero(ddi['fold'] == fold)
            target = np.flatnonzero((ddi['fold'] != fold) & (ddi['tones'] == 'V-VI'))
            quotas = np.bincount(ddi['y'][target], minlength=2)
            panels = [('target', -1, target)]
            for draw in range(20):
                rng = np.random.default_rng(np.random.SeedSequence([20261004, 200, draw, fold]))
                selected = []
                for label in (0, 1):
                    pool = np.flatnonzero((ddi['fold'] != fold) & (ddi['tones'] != 'V-VI') & (ddi['y'] == label))
                    pool = pool[np.argsort(ddi['ids'][pool], kind='stable')]
                    common.require(len(pool) >= quotas[label], 'Insufficient lighter matches')
                    selected.extend(rng.choice(pool, size=int(quotas[label]), replace=False).tolist())
                selected = np.array(selected, dtype=int)
                common.require(np.array_equal(np.bincount(ddi['y'][selected], minlength=2), quotas), 'Outcome matching failed')
                panels.append(('lighter', draw, selected))
            for family, draw, selected in panels:
                common.require(not set(heldout).intersection(selected), 'DDI test image enters classifier training')
                panel_id = f'{family}_draw_{draw:02d}_fold_{fold}'
                samples.extend({'sampling_panel_id': panel_id, 'family': family, 'draw_id': draw, 'heldout_fold': fold,
                                'analysis_image_id': ddi['ids'][idx], 'target_label': int(ddi['y'][idx]),
                                'skin_tone_group': ddi['tones'][idx], 'DDI_feature_row_index': int(idx)} for idx in selected)
                for weighted in (False, True):
                    kind = 'weighted' if weighted else 'unweighted'
                    arm = f'{family}_{kind}' if family == 'target' else f'{family}_{kind}_draw_{draw:02d}'
                    col = arm_names.index(arm)
                    m = np.zeros(total, dtype=np.uint16)
                    m[:n_pad] = 1
                    m[n_pad + selected] = 1
                    model_id = f'{arm}_fold_{fold}'
                    mr = store.fit(model_id, m, C, source_weighted=weighted, purpose='DDI_heldout_adaptation',
                                   heldout_fold=fold, arm=arm, sampling_panel_id=panel_id, fixed_C_from='PAD_final_CV')
                    common.require(not np.any(store.weights[mr][n_pad + heldout]), 'Heldout row has nonzero fit weight')
                    ddi_scores[heldout, col] = store.predict(mr, n_pad + heldout)
                    ddi_model_ids[heldout, col] = model_id
            print(f'DDI fold {fold + 1}/5: target + 20 matched lighter panels fitted', flush=True)
        common.require(np.isfinite(ddi_scores).all() and (ddi_model_ids != '').all(), 'Incomplete heldout DDI scores')

    store.save(out / 'models/core_models')
    common.save_npz(out / 'predictions/core_scores.npz', PAD_nested_logits=nested_scores, PAD_nested_model_ids=nested_model_ids,
                    PAD_image_ids=pad['ids'], DDI_logits=ddi_scores, DDI_model_ids=ddi_model_ids,
                    DDI_image_ids=ddi['ids'], DDI_arm_names=np.array(arm_names), plan_sha256=np.array(data['plan_sha256']))
    rows = []
    for ds, scores, models, arms in [('PAD', nested_scores[:, None], nested_model_ids[:, None], ['PAD_nested_development']),
                                    ('DDI', ddi_scores, ddi_model_ids, arm_names)]:
        d = data['datasets'][ds]
        for i, image in enumerate(d['rows']):
            for col, arm in enumerate(arms):
                rows.append({'dataset': ds, 'analysis_image_id': image['analysis_image_id'], 'image_sha256': image['image_sha256'],
                             'feature_row_index': i, 'feature_cache_id': data['plan']['feature_cache_id'],
                             'target_label': int(d['y'][i]), 'skin_tone_group': d['tones'][i], 'heldout_fold': int(d['fold'][i]),
                             'arm': arm, 'logit_score': float(scores[i, col]), 'model_id': models[i, col],
                             'evaluation_type': 'nested_patient_grouped_development' if ds == 'PAD' else
                                                ('PAD_only_transfer' if arm == 'PAD_baseline' else 'image_fold_within_DDI_adaptation')})
    common.write_csv(out / 'predictions/per_image_predictions.csv', rows)
    common.write_csv(out / 'predictions/PAD_tuning_validation_predictions.csv', tuning_predictions)
    common.write_csv(out / 'results/PAD_tuning_fold_AUROC.csv', tuning_metrics)
    common.write_csv(out / 'splits/PAD_nested_inner_folds.csv', inner_membership)
    common.write_csv(out / 'splits/DDI_adaptation_samples.csv', samples)
    audit = {'started_utc': started, 'completed_utc': common.utc_now(), 'stage': 'classifier_fitting',
             'plan_sha256': data['plan_sha256'], 'code_sha256': {Path(p).name: common.file_sha(p) for p in (__file__, common.__file__)},
             'runtime': common.environment(), 'feature_cache_id': data['plan']['feature_cache_id'],
             'C_selected_from_PAD_only': float(C), 'PAD_final_CV_mean_AUROC_by_C': dict(zip(map(str, data['plan']['classifier']['C_grid']), final_means)),
             'PAD_nested_outer_selected_C': nested_selected, 'fitted_head_count': len(store.models),
             'PAD_nested_predictions': len(nested_scores), 'DDI_predictions_per_arm': data['n_ddi'], 'DDI_arms': arm_names,
             'heldout_row_training_weight_checks': 'PASS', 'saved_coefficient_reconstruction_checks': 'PASS',
             'historical_results_verified': False,
             'output_sha256': {str(p.relative_to(out)): common.file_sha(p) for folder in ('models', 'predictions', 'results', 'splits') for p in sorted((out / folder).glob('*')) if p.is_file()}}
    common.write_json(out / 'logs/classifier_fit_audit.json', audit)
    print('CLASSIFIER FITTING: PASS', flush=True)
    print('Heads:', len(store.models), '| final C:', C, '| all per-image heldout predictions saved', flush=True)
    return audit


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs', required=True)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    run(args.inputs, args.plan, args.output)
