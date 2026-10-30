"""Final publication E4 evaluator for frozen heads on MRA-MIDAS features.

This script reproduces the analysis reported in the manuscript:
- image-level AUROC;
- seven prespecified frozen linear heads;
- no MRA-MIDAS refitting, recalibration or model selection;
- 10,000 percentile bootstrap resamples clustered by midas_record_id.

The released midas_record_id is treated only as a resampling cluster, not as a
verified patient identifier.
"""
from __future__ import annotations
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def load_manifest(path: str, n_rows: int) -> pd.DataFrame:
    m = pd.read_csv(path)
    require({'midas_record_id', 'label'} <= set(m.columns),
            'Manifest must contain midas_record_id and label')
    if 'feature_row_index' in m.columns:
        m = m.sort_values('feature_row_index').reset_index(drop=True)
        require(np.array_equal(m.feature_row_index.to_numpy(dtype=int), np.arange(len(m))),
                'feature_row_index must be zero-based and contiguous')
    require(len(m) == n_rows, 'Manifest row count must match feature rows')
    y = m.label.to_numpy(dtype=int)
    require(set(np.unique(y)) == {0, 1}, 'Manifest must contain both binary classes')
    require(m.midas_record_id.notna().all(), 'Missing midas_record_id')
    return m


def run(args) -> None:
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    X = np.load(args.features, allow_pickle=False)
    require(X.dtype == np.float32, 'Features must be float32')
    require(X.ndim == 2 and X.shape[1] == 1152, 'Expected 1,152-dimensional features')
    require(np.isfinite(X).all(), 'Features contain non-finite values')
    require(np.max(np.abs(np.linalg.norm(X, axis=1) - 1.0)) < 2e-6,
            'Features are not L2-normalised')

    m = load_manifest(args.manifest, len(X))
    y = m.label.to_numpy(dtype=int)
    records = m.midas_record_id.astype(str).to_numpy()

    models = np.load(args.models, allow_pickle=False)
    required = {'coefficients', 'intercepts', 'model_ids'}
    require(required <= set(models.files), 'Deployment NPZ missing required arrays')
    coef = np.asarray(models['coefficients'], dtype=np.float64)
    intercepts = np.asarray(models['intercepts'], dtype=np.float64)
    model_ids = [str(x) for x in models['model_ids']]
    require(coef.shape == (7, 1152), 'Expected seven 1,152-dimensional heads')
    require(intercepts.shape == (7,), 'Expected seven intercepts')
    require(len(model_ids) == 7, 'Expected seven model IDs')

    registry = pd.read_csv(args.registry)
    if 'model_id' in registry.columns:
        require(model_ids == registry.model_id.astype(str).tolist(),
                'Deployment registry order does not match model_ids')

    logits = X.astype(np.float64) @ coef.T + intercepts[None, :]
    point_auc = np.array([roc_auc_score(y, logits[:, j]) for j in range(7)])

    unique_records, record_index = np.unique(records, return_inverse=True)
    n_records = len(unique_records)
    rng = np.random.default_rng(args.seed)
    boot_auc = np.full((7, args.bootstrap_rounds), np.nan, dtype=np.float64)

    for b in range(args.bootstrap_rounds):
        sampled = rng.integers(0, n_records, size=n_records)
        record_counts = np.bincount(sampled, minlength=n_records)
        w = record_counts[record_index].astype(np.float64)
        if w[y == 1].sum() == 0 or w[y == 0].sum() == 0:
            continue
        for j in range(7):
            boot_auc[j, b] = roc_auc_score(y, logits[:, j], sample_weight=w)

    rows = []
    for j, model_id in enumerate(model_ids):
        valid = boot_auc[j, np.isfinite(boot_auc[j])]
        require(len(valid) > 0, 'No valid bootstrap samples')
        lo, hi = np.percentile(valid, [2.5, 97.5])
        rows.append({
            'model_id': model_id,
            'AUROC': float(point_auc[j]),
            'CI95_low': float(lo),
            'CI95_high': float(hi),
            'bootstrap_valid': int(len(valid)),
            'bootstrap_rounds': int(args.bootstrap_rounds),
            'bootstrap_seed': int(args.seed),
            'n_images': int(len(y)),
            'n_negative_images': int((y == 0).sum()),
            'n_positive_images': int((y == 1).sum()),
            'n_midas_record_id': int(n_records),
            'resampling_unit': 'midas_record_id',
        })

    results = pd.DataFrame(rows)
    results.to_csv(out / 'E4_MRA_MIDAS_7heads_AUROC.csv', index=False)

    pred = m.copy()
    probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -50, 50)))
    for j, model_id in enumerate(model_ids):
        safe = ''.join(c if c.isalnum() or c in '_-' else '_' for c in model_id)
        pred[f'logit__{safe}'] = logits[:, j]
        pred[f'probability__{safe}'] = probs[:, j]
    pred.to_csv(out / 'E4_MRA_MIDAS_per_image_predictions.csv', index=False)

    np.savez_compressed(
        out / 'E4_MRA_MIDAS_record_cluster_bootstrap.npz',
        model_ids=np.asarray(model_ids),
        auroc_samples=boot_auc,
        seed=np.asarray(args.seed),
        rounds=np.asarray(args.bootstrap_rounds),
        resampling_unit=np.asarray('midas_record_id'),
    )

    print('E4 EXTERNAL EVALUATION COMPLETE')
    print(results[['model_id', 'AUROC', 'CI95_low', 'CI95_high']].to_string(index=False))
    print('Images:', len(y), 'record groups:', n_records)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--models', required=True)
    p.add_argument('--registry', required=True)
    p.add_argument('--features', required=True)
    p.add_argument('--manifest', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--bootstrap-rounds', type=int, default=10000)
    p.add_argument('--seed', type=int, default=0)
    run(p.parse_args())
