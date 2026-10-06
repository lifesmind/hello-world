"""Shared, auditable primitives for the MAKE exploratory reanalysis.

All paths are arguments. No credentials, network calls or image downloads.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import platform
import warnings
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def json_bytes(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode()


def write_file(path, data, immutable=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and immutable:
        require(path.read_bytes() == data, 'Existing file changed: ' + str(path))
        return
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_bytes(data)
    temporary.replace(path)


def write_json(path, value):
    write_file(path, json_bytes(value))


def write_csv(path, rows, fields=None):
    rows = list(rows)
    if fields is None:
        fields = list(dict.fromkeys(k for row in rows for k in row))
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    write_file(path, stream.getvalue().encode('utf-8-sig'))


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def save_npz(path, **arrays):
    stream = io.BytesIO()
    np.savez_compressed(stream, **arrays)
    write_file(path, stream.getvalue())


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def environment():
    return {'python': platform.python_version(), 'packages': {p: version(p) for p in
            ('numpy', 'scipy', 'scikit-learn', 'pandas', 'threadpoolctl', 'matplotlib')}}


def load_inputs(input_root, plan_path):
    root, plan_path = Path(input_root), Path(plan_path)
    plan = json.loads(plan_path.read_text())
    require(plan['locked_before_new_classifier_fit_or_score_inspection'], 'Plan not locked')
    for name, expected in plan['input_sha256'].items():
        require(file_sha(root / name) == expected, 'Input hash mismatch: ' + name)
    require(environment() == plan['runtime'], 'Classifier runtime differs from locked plan')
    audit = json.loads((root / 'logs/feature_extraction_audit_20261004T171454019752Z.json').read_text())
    spec = json.loads((root / 'features/feature_spec.json').read_text())
    split = json.loads((root / 'splits/split_spec_v1.json').read_text())
    require(file_sha(root / 'features/feature_spec.json') == audit['feature_spec_sha256'], 'Feature-spec audit mismatch')
    require(audit['feature_cache_id'] == file_sha(root / 'features/feature_spec.json') == plan['feature_cache_id'], 'Wrong feature cache')
    require(file_sha(root / 'splits/split_spec_v1.json') == spec['split_spec_sha256'], 'Split-spec mismatch')
    datasets = {}
    for name, count, positive in (('PAD', 2268, 1070), ('DDI', 656, 171)):
        index = read_csv(root / f'features/{name}_feature_index.csv')
        manifest = read_csv(root / f'manifests/{name}_labeled_manifest.csv')
        folds = read_csv(root / f'splits/{name}_cv_folds.csv')
        matrix = np.load(root / f'features/{name}_features_l2.npy', allow_pickle=False)
        require(matrix.shape == (count, 1152) and matrix.dtype == np.float32, 'Invalid feature shape/dtype')
        require(np.isfinite(matrix).all() and np.allclose(np.linalg.norm(matrix, axis=1), 1, atol=2e-5), 'Invalid L2 features')
        require(len(index) == len(manifest) == len(folds) == count, 'Row-count mismatch')
        require(len({r['analysis_image_id'] for r in index}) == count, 'Duplicate image ID')
        by_id = {r['analysis_image_id']: r for r in folds}
        for i, (r, m) in enumerate(zip(index, manifest, strict=True)):
            require(all(r[k] == v for k, v in m.items()), 'Index/manifest row mismatch')
            require(all(r[k] == v for k, v in by_id[r['analysis_image_id']].items()), 'Saved fold mismatch')
            require(int(r['feature_row_index']) == i and r['feature_cache_id'] == plan['feature_cache_id'], 'Feature-row cache mismatch')
            require(sha_bytes(matrix[i].tobytes()) == r['feature_row_sha256'], 'Feature-row checksum mismatch')
        for relative, path in ((f'data/Analysis_Inputs_v1/{name}_labeled_manifest.csv', root / f'manifests/{name}_labeled_manifest.csv'),
                               (f'splits/Analysis_Inputs_v1/{name}_cv_folds.csv', root / f'splits/{name}_cv_folds.csv')):
            require(file_sha(path) == split['output_sha256'][relative], 'Preparation-output hash mismatch')
        target = np.array([int(r['target_label']) for r in index], dtype=np.int8)
        require(target.sum() == positive and set(target) == {0, 1}, 'Unexpected labels')
        fold = np.array([int(r['cv_fold']) for r in index], dtype=np.int8)
        require(set(fold) == set(range(5)), 'Unexpected fold numbering')
        groups = np.array([r['split_unit_id'] for r in index])
        if name == 'PAD':
            require(len(np.unique(groups)) == 1370, 'Unexpected PAD patient count')
            for g in np.unique(groups):
                require(len(np.unique(fold[groups == g])) == 1, 'PAD patient spans saved folds')
        datasets[name] = {'rows': index, 'X': matrix.astype(np.float64), 'y': target, 'fold': fold,
                          'ids': np.array([r['analysis_image_id'] for r in index]), 'groups': groups,
                          'tones': np.array([r['skin_tone_group'] for r in index])}
    require(not set(r['image_sha256'] for r in datasets['PAD']['rows']).intersection(
            r['image_sha256'] for r in datasets['DDI']['rows']), 'Exact cross-source image duplicate')
    X = np.ascontiguousarray(np.concatenate([datasets['PAD']['X'], datasets['DDI']['X']]))
    y = np.concatenate([datasets['PAD']['y'], datasets['DDI']['y']])
    ids = np.concatenate([datasets['PAD']['ids'], datasets['DDI']['ids']])
    return {'root': root, 'plan': plan, 'plan_path': plan_path, 'plan_sha256': file_sha(plan_path),
            'datasets': datasets, 'X': X, 'y': y, 'ids': ids, 'n_pad': 2268, 'n_ddi': 656}


def init_output(output_root, data, code_paths):
    out = Path(output_root)
    out.mkdir(parents=True, exist_ok=True)
    write_file(out / 'plan/MAKE_Analysis_Plan_v1.json', data['plan_path'].read_bytes())
    for name in data['plan']['input_sha256']:
        write_file(out / 'inputs' / name, (data['root'] / name).read_bytes())
    for path in code_paths:
        path = Path(path)
        write_file(out / 'code' / path.name, path.read_bytes())
    return out


def coefficient_sha(coef, intercept):
    return sha_bytes(np.concatenate([np.asarray(coef, dtype='<f8').ravel(), np.asarray(intercept, dtype='<f8').ravel()]).tobytes())


class FitStore:
    def __init__(self, data):
        self.data = data
        self.models, self.coefs, self.intercepts, self.weights, self.multiplicities = [], [], [], [], []

    def fit(self, model_id, multiplicity, C, source_weighted=False, **tags):
        d = self.data
        m = np.asarray(multiplicity, dtype=np.float64)
        require(m.shape == (len(d['ids']),) and np.isfinite(m).all() and (m >= 0).all(), 'Invalid training multiplicities')
        require(np.equal(m, np.floor(m)).all(), 'Multiplicities must be integer')
        counts = np.bincount(d['y'], weights=m, minlength=2)
        require((counts > 0).all(), 'Training bootstrap lacks an outcome class')
        n_pad = float(m[:d['n_pad']].sum())
        n_added = float(m[d['n_pad']:].sum())
        require(n_pad > 0, 'No PAD training rows')
        multiplier = float(np.sqrt(n_pad / n_added)) if source_weighted else 1.0
        require(not source_weighted or n_added > 0, 'Cannot weight an empty target source')
        factors = np.ones(len(m), dtype=np.float64)
        factors[d['n_pad']:] = multiplier
        class_factors = float(counts.sum()) / (2.0 * counts)
        weights = m * factors * class_factors[d['y']]
        train = np.flatnonzero(m)
        train = train[np.argsort(d['ids'][train], kind='stable')]
        model = None
        attempts = []
        for limit in d['plan']['classifier']['max_iter_attempts']:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always', ConvergenceWarning)
                model = LogisticRegression(C=float(C), l1_ratio=0.0, solver='lbfgs', class_weight=None,
                                           fit_intercept=True, tol=d['plan']['classifier']['tol'], max_iter=limit)
                model.fit(d['X'][train], d['y'][train], sample_weight=weights[train])
            did_not_converge = any(issubclass(w.category, ConvergenceWarning) for w in caught)
            other_warnings = [str(w.message) for w in caught if not issubclass(w.category, ConvergenceWarning)]
            require(not other_warnings, 'Unexpected fitting warning: ' + repr(other_warnings))
            attempts.append({'max_iter': limit, 'n_iter': int(model.n_iter_[0]), 'convergence_warning': did_not_converge})
            if not did_not_converge:
                break
        require(not did_not_converge, 'Classifier failed to converge: ' + model_id)
        require(np.array_equal(model.classes_, [0, 1]) and np.isfinite(model.coef_).all(), 'Invalid classifier')
        require(model_id not in {r['model_id'] for r in self.models}, 'Repeated model ID')
        row = len(self.models)
        record = {'model_id': model_id, 'storage_row_index': row, 'C': float(C), 'classes': '0,1',
                  'training_unique_rows': len(train), 'training_rows_with_multiplicity': int(m.sum()),
                  'PAD_rows_with_multiplicity': int(n_pad), 'added_rows_with_multiplicity': int(n_added),
                  'positive_rows_with_multiplicity': int(counts[1]), 'negative_rows_with_multiplicity': int(counts[0]),
                  'class_factor_0': float(class_factors[0]), 'class_factor_1': float(class_factors[1]),
                  'added_source_multiplier': multiplier, 'effective_weight_sum': float(weights.sum()),
                  'coefficient_sha256': coefficient_sha(model.coef_, model.intercept_),
                  'effective_sample_weight_sha256': sha_bytes(weights.astype('<f8').tobytes()),
                  'training_multiplicity_sha256': sha_bytes(m.astype('<f8').tobytes()),
                  'fit_attempts_json': json.dumps(attempts, separators=(',', ':')),
                  'plan_sha256': d['plan_sha256'], 'feature_cache_id': d['plan']['feature_cache_id'], **tags}
        self.models.append(record)
        self.coefs.append(model.coef_[0].copy())
        self.intercepts.append(float(model.intercept_[0]))
        self.weights.append(weights)
        self.multiplicities.append(m.astype(np.uint16))
        # Mathematical reconstruction is validated for every fitted head.
        reconstructed = d['X'][train] @ self.coefs[-1] + self.intercepts[-1]
        require(np.allclose(reconstructed, model.decision_function(d['X'][train]), rtol=1e-11, atol=1e-11), 'Saved coefficient reconstruction failed')
        return row

    def predict(self, model_row, global_rows):
        return self.data['X'][global_rows] @ self.coefs[model_row] + self.intercepts[model_row]

    def save(self, prefix):
        prefix = Path(prefix)
        save_npz(prefix.with_suffix('.npz'), coefficients=np.stack(self.coefs), intercepts=np.array(self.intercepts),
                 classes=np.array([0, 1]), C=np.array([r['C'] for r in self.models]),
                 model_ids=np.array([r['model_id'] for r in self.models]),
                 analysis_image_ids=self.data['ids'], effective_sample_weights=np.stack(self.weights),
                 training_multiplicities=np.stack(self.multiplicities),
                 plan_sha256=np.array(self.data['plan_sha256']), feature_cache_id=np.array(self.data['plan']['feature_cache_id']))
        write_csv(prefix.with_name(prefix.name + '_registry.csv'), self.models)


def point_metrics(y, scores, weights=None, fpr_limit=0.50):
    y, scores = np.asarray(y), np.asarray(scores)
    weights = np.ones(len(y), dtype=np.float64) if weights is None else np.asarray(weights, dtype=np.float64)
    require(y.shape == scores.shape == weights.shape and np.isfinite(scores).all(), 'Invalid metric inputs')
    pos, neg = y == 1, y == 0
    require(weights[pos].sum() > 0 and weights[neg].sum() > 0, 'Metric lacks an outcome class')
    order = np.argsort(-scores[neg], kind='stable')
    cumulative = np.cumsum(weights[neg][order])
    boundary_index = np.flatnonzero(cumulative > fpr_limit * weights[neg].sum())[0]
    boundary = scores[neg][order][boundary_index]
    # Strict > excludes an entire tie block that would exceed the FPR budget.
    sensitivity = float(weights[pos][scores[pos] > boundary].sum() / weights[pos].sum())
    achieved_fpr = float(weights[neg][scores[neg] > boundary].sum() / weights[neg].sum())
    return {'AUROC': float(roc_auc_score(y, scores, sample_weight=weights)),
            'sensitivity_FPR_le_0_50': sensitivity, 'achieved_FPR': achieved_fpr,
            'retrospective_logit_threshold': float(np.nextafter(boundary, np.inf))}


def stratified_bootstrap_weights(labels, repeats, rng):
    labels = np.asarray(labels)
    weights = np.zeros((repeats, len(labels)), dtype=np.uint16)
    for label in sorted(np.unique(labels)):
        loc = np.flatnonzero(labels == label)
        weights[:, loc] = rng.multinomial(len(loc), np.full(len(loc), 1.0 / len(loc)), size=repeats)
    return weights


def ddi_bootstrap_labels(ddi):
    return np.array([f'{fold}:{tone}:{label}' for fold, tone, label in
                     zip(ddi['fold'], ddi['tones'], ddi['y'], strict=True)])


def pad_cluster_weights(pad, repeats, rng):
    groups, inverse = np.unique(pad['groups'], return_inverse=True)
    counts = rng.multinomial(len(groups), np.full(len(groups), 1.0 / len(groups)), size=repeats).astype(np.uint16)
    return counts[:, inverse]


def bootstrap_metrics(y, scores, weights, fpr_limit=0.50, batch_size=250):
    """Exact empirical weighted AUC, with 0.5 credit for tied scores.

    Multinomial row counts are equivalent to explicitly duplicating bootstrap
    observations. Batches bound memory; paired contrasts reuse the same counts.
    """
    y, scores = np.asarray(y), np.asarray(scores)
    weights = np.asarray(weights)
    pos, neg = y == 1, y == 0
    sp, sn = scores[pos], scores[neg]
    pair = (sp[:, None] > sn[None, :]).astype(np.float64) + 0.5 * (sp[:, None] == sn[None, :])
    order = np.argsort(-sn, kind='stable')
    auc, sensitivity, fpr = [], [], []
    for start in range(0, len(weights), batch_size):
        batch = weights[start:start + batch_size].astype(np.float64)
        wp, wn = batch[:, pos], batch[:, neg]
        np_, nn_ = wp.sum(axis=1), wn.sum(axis=1)
        require((np_ > 0).all() and (nn_ > 0).all(), 'Bootstrap replicate lacks a class')
        auc.append(((wp @ pair) * wn).sum(axis=1) / (np_ * nn_))
        cumulative = np.cumsum(wn[:, order], axis=1)
        boundary_indices = np.argmax(cumulative > fpr_limit * nn_[:, None], axis=1)
        boundaries = sn[order][boundary_indices]
        sensitivity.append((wp * (sp[None, :] > boundaries[:, None])).sum(axis=1) / np_)
        fpr.append((wn * (sn[None, :] > boundaries[:, None])).sum(axis=1) / nn_)
    return np.column_stack([np.concatenate(auc), np.concatenate(sensitivity), np.concatenate(fpr)])


def percentile_ci(values):
    return [float(x) for x in np.quantile(values, [0.025, 0.975], method='linear')]
