"""Input integrity, explicit weighted fitting and artifact linkage."""
from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import importlib.metadata as metadata
import json
import warnings

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def array_sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def runtime():
    return {k: metadata.version(k) for k in ['numpy', 'pandas', 'scipy', 'scikit-learn', 'threadpoolctl']}


def load(base, output):
    base, output = Path(base), Path(output)
    for folder in ['plan','models','predictions','splits','results','uncertainty','logs','figures','documents']:
        (output/folder).mkdir(parents=True,exist_ok=True)
    pp = output/'plan/MAKE_Exploratory_Extension_20261006_v1.json'
    p = json.loads(pp.read_text())
    ph = sha(pp)
    require(ph == pp.with_suffix('.sha256').read_text().split()[0], 'Protocol bytes changed')
    for name, expected in p['base_asset_sha256'].items():
        require(sha(base/name) == expected, 'Base asset changed: '+name)
    frames, arrays = {}, {}
    for ds in ['PAD', 'DDI']:
        frames[ds] = pd.read_csv(base/f'inputs/features/{ds}_feature_index.csv', keep_default_na=False)
        xf = np.load(base/f'inputs/features/{ds}_features_l2.npy', allow_pickle=False)
        require(xf.dtype == np.float32 and xf.shape == (len(frames[ds]), 1152), 'Feature shape/dtype mismatch')
        require(np.isfinite(xf).all() and np.max(abs(np.linalg.norm(xf, axis=1)-1)) < 2e-6, 'Invalid L2 features')
        require(np.array_equal(frames[ds].feature_row_index.values, np.arange(len(xf))), 'Feature row order changed')
        require(all(array_sha(xf[j]) == frames[ds].iloc[j].feature_row_sha256 for j in range(len(xf))), 'Feature row SHA mismatch')
        arrays[ds] = xf.astype(np.float64)
    pad, ddi = frames['PAD'], frames['DDI']
    require(len(pad)==2268 and len(ddi)==656, 'Unexpected cohort counts')
    require(pad.analysis_image_id.is_unique and ddi.analysis_image_id.is_unique, 'Duplicate analysis IDs')
    require(set(ddi.skin_tone_group) == {'I-II','III-IV','V-VI'}, 'Unknown DDI tone')
    require(set(ddi.target_label.astype(int)) == {0,1} and set(ddi.cv_fold.astype(int)) == set(range(5)), 'DDI labels/folds invalid')
    core = np.load(base/'predictions/core_scores.npz', allow_pickle=False)
    require(np.array_equal(core['DDI_image_ids'], ddi.analysis_image_id.values), 'Saved core score ordering mismatch')
    cm = np.load(base/'models/core_models.npz', allow_pickle=False)
    return {'base':base, 'out':output, 'plan':p, 'plan_sha':ph, 'PAD':pad, 'DDI':ddi,
            'X':np.vstack([arrays['PAD'], arrays['DDI']]), 'XP':arrays['PAD'], 'XD':arrays['DDI'],
            'y':np.r_[pad.target_label.astype(int), ddi.target_label.astype(int)],
            'yp':pad.target_label.to_numpy(dtype=int), 'yd':ddi.target_label.to_numpy(dtype=int),
            'tones':ddi.skin_tone_group.to_numpy(dtype=str), 'fold':ddi.cv_fold.to_numpy(dtype=int),
            'ids':np.r_[pad.analysis_image_id.to_numpy(dtype=str), ddi.analysis_image_id.to_numpy(dtype=str)],
            'n_pad':len(pad), 'core':core, 'core_models':cm}


class ModelStore:
    def __init__(self, data):
        self.d=data
        self.coef=[]; self.intercept=[]; self.weights=[]; self.registry=[]

    def add(self, model_id, coef, intercept, weights, **tags):
        require(model_id not in [m['model_id'] for m in self.registry], 'Duplicate model ID')
        coef=np.asarray(coef,dtype=np.float64).ravel(); weights=np.asarray(weights,dtype=np.float64)
        require(coef.shape==(1152,) and weights.shape==(len(self.d['y']),), 'Invalid saved model shape')
        require(np.isfinite(coef).all() and np.isfinite(intercept) and np.isfinite(weights).all(), 'Nonfinite model')
        row=len(self.coef); self.coef.append(coef); self.intercept.append(float(intercept));self.weights.append(weights)
        self.registry.append({'model_id':model_id,'storage_row_index':row,'coefficient_sha256':array_sha(coef),
                              'effective_sample_weight_sha256':array_sha(weights),'training_unique_rows':int((weights>0).sum()),
                              'PAD_fit_rows':int((weights[:self.d['n_pad']]>0).sum()),
                              'DDI_fit_rows':int((weights[self.d['n_pad']:]>0).sum()),
                              'effective_weight_sum':float(weights.sum()),'protocol_sha256':self.d['plan_sha'],**tags})
        return row

    def predict(self,row,X):
        return np.asarray(X) @ self.coef[row] + self.intercept[row]

    def fit(self, model_id, weights, *, C=10.0, kind='L2', heldout_fold=-1, purpose='DDI_OOF', **tags):
        weights=np.asarray(weights,dtype=np.float64)
        if heldout_fold>=0:
            require(not np.any(weights[self.d['n_pad']:][self.d['fold']==heldout_fold]), 'Outer-test row in fitting')
        selected=np.flatnonzero(weights>0)
        require(set(self.d['y'][selected])=={0,1}, 'Training panel lacks a class')
        X=self.d['X'][selected]
        if kind=='DFR_L1':
            X=(X-self.d['scaler_mean'])/self.d['scaler_scale']
        attempts=[]
        for max_iter in self.d['plan']['L2_fit']['max_iter_attempts']:
            kwargs={'C':C,'tol':1e-7,'max_iter':max_iter,'class_weight':None,'random_state':20261006}
            if kind=='DFR_L1':
                kwargs.update(penalty='l1',solver='liblinear',intercept_scaling=1.)
            else:
                kwargs.update(l1_ratio=0.,solver='lbfgs')
            model=LogisticRegression(**kwargs)
            with warnings.catch_warnings(record=True) as seen:
                warnings.simplefilter('always')
                model.fit(X,self.d['y'][selected],sample_weight=weights[selected])
            warned=any(issubclass(w.category,ConvergenceWarning) for w in seen)
            attempts.append({'max_iter':max_iter,'n_iter':int(model.n_iter_.max()),'convergence_warning':warned})
            if not warned: break
        require(not warned, 'Failed to converge: '+model_id)
        b=model.coef_[0].copy(); a=float(model.intercept_[0])
        if kind=='DFR_L1':
            b=b/self.d['scaler_scale']; a-=float(self.d['scaler_mean']@b)
        raw=self.d['X'][selected]@b+a
        require(np.max(abs(raw-model.decision_function(X)))<1e-9, 'Coefficient transform reconstruction failed')
        return self.add(model_id,b,a,weights,C=float(C),kind=kind,heldout_fold=int(heldout_fold),purpose=purpose,
                        fit_attempts_json=json.dumps(attempts,separators=(',',':')),**tags)

    def ensemble(self, model_id, rows, *, C, heldout_fold, purpose, **tags):
        b=np.mean([self.coef[j] for j in rows],axis=0);a=float(np.mean([self.intercept[j] for j in rows]))
        # Mean fit weights record union of component support, not a single ensemble objective.
        w=np.mean([self.weights[j] for j in rows],axis=0)
        return self.add(model_id,b,a,w,C=float(C),kind='DFR_ENSEMBLE',heldout_fold=int(heldout_fold),purpose=purpose,
                        components_json=json.dumps([self.registry[j]['model_id'] for j in rows]),
                        effective_weights_note='Mean component support; ensemble is coefficient average, not a separately fitted objective',**tags)

    def save(self, prefix):
        prefix=Path(prefix)
        np.savez_compressed(prefix.with_suffix('.npz'),coefficients=np.asarray(self.coef),intercepts=np.asarray(self.intercept),
                            effective_sample_weights=np.asarray(self.weights),model_ids=np.array([r['model_id'] for r in self.registry]),
                            analysis_image_ids=self.d['ids'],protocol_sha256=np.array(self.d['plan_sha']))
        pd.DataFrame(self.registry).to_csv(prefix.with_name(prefix.name+'_registry.csv'),index=False,float_format='%.17g')


def pooled_weights(d, selected, source_weighted=True, group=False):
    n_pad=d['n_pad'];w=np.zeros(len(d['y']));inc=np.r_[np.arange(n_pad),n_pad+np.asarray(selected)]
    counts=np.bincount(d['y'][inc],minlength=2)
    require(np.all(counts>0), 'Pooled fitting lacks class')
    w[inc]=len(inc)/(2*counts[d['y'][inc]])
    multiplier=float(np.sqrt(n_pad/len(selected))) if source_weighted else 1.
    w[n_pad+selected]*=multiplier
    if group:
        for label in [0,1]:
            ly=np.asarray(selected)[d['yd'][selected]==label]
            for tone in ['I-II','III-IV','V-VI']:
                ids=ly[d['tones'][ly]==tone]
                require(len(ids)>0,'Group-weighting cell empty')
                w[n_pad+ids]*=len(ly)/(3*len(ids))
    return w


def ddi_only_weights(d,selected,group=False):
    selected=np.asarray(selected);w=np.zeros(len(d['y']))
    if not group:
        counts=np.bincount(d['yd'][selected],minlength=2)
        w[d['n_pad']+selected]=len(selected)/(2*counts[d['yd'][selected]])
    else:
        for tone in ['I-II','III-IV','V-VI']:
            for label in [0,1]:
                ix=selected[(d['tones'][selected]==tone)&(d['yd'][selected]==label)]
                require(len(ix)>0,'DDI six-group cell empty')
                w[d['n_pad']+ix]=len(selected)/(6*len(ix))
    return w


def balanced_panel(d,pool,rng):
    groups=[np.asarray(pool)[(d['tones'][pool]==t)&(d['yd'][pool]==y)]
            for t in ['I-II','III-IV','V-VI'] for y in [0,1]]
    count=min(map(len,groups));require(count>0,'Balanced sampling group empty')
    return np.sort(np.concatenate([rng.choice(g,size=count,replace=False) for g in groups]))


def rows_for_panel(d,panel_id,purpose,fold,draw,selected):
    return [{'panel_id':panel_id,'purpose':purpose,'heldout_fold':int(fold),'draw_id':int(draw),
             'DDI_feature_row_index':int(i),'analysis_image_id':d['DDI'].iloc[i].analysis_image_id,
             'target_label':int(d['yd'][i]),'skin_tone_group':str(d['tones'][i])} for i in selected]


def raw_probabilities(scores):
    return expit(scores)
