"""Descriptive post hoc within-fold ranking audit; no fitting or inference."""
from pathlib import Path
import hashlib, json, argparse
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--scores',type=Path,default=Path('analysis/make_submission_20261006/evidence/MAKE_Extension_20261006/predictions/extension_scores.npz'))
parser.add_argument('--output',type=Path,default=Path(__file__).resolve().parents[1])
args=parser.parse_args()
root=args.output.resolve();source=args.scores.resolve()
plan=json.loads((root/'audit/within_fold_diagnostic_plan.json').read_text())
assert hashlib.sha256(source.read_bytes()).hexdigest()==plan['input_score_sha256']
rows=[]
with np.load(source,allow_pickle=False) as z:
    labels=z['labels']; folds=z['folds']; tones=z['tones']
    for arm in z['arms']:
        scores=z['score__'+str(arm)]
        for cohort in ['ALL_DDI','I-II','III-IV','V-VI']:
            cohort_mask=np.ones(len(labels),dtype=bool) if cohort=='ALL_DDI' else tones==cohort
            for fold in range(5):
                mask=cohort_mask&(folds==fold)
                for draw in range(scores.shape[1]):
                    rows.append({'arm':str(arm),'cohort':cohort,'heldout_fold':fold,'draw_id':draw,
                        'n':int(mask.sum()),'positive':int(labels[mask].sum()),
                        'AUROC':roc_auc_score(labels[mask],scores[mask,draw])})
df=pd.DataFrame(rows);df.to_csv(root/'tables/within_fold_AUROC_draws.csv',index=False)
summary=df.groupby(['arm','cohort'],sort=False).agg(mean_within_fold_AUROC=('AUROC','mean'),
    minimum_fold_panel_AUROC=('AUROC','min'),maximum_fold_panel_AUROC=('AUROC','max'),
    fold_panel_estimates=('AUROC','size')).reset_index()
summary.to_csv(root/'tables/within_fold_AUROC_summary.csv',index=False)
print(summary[summary.cohort=='V-VI'].to_string(index=False))
