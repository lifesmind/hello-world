"""Independent model/heldout/CI reconstruction and E4 validation smoke checks."""
from pathlib import Path
import argparse
import json
import hashlib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score,roc_curve
from threadpoolctl import threadpool_limits
import make_extension_common as c
from MAKE_Compute_Extension_Statistics import samples,point
from MAKE_Evaluate_MRA_External import validate_features_and_manifest


def run(base,output):
    d=c.load(base,output);out=d['out'];z=np.load(out/'predictions/extension_scores.npz',allow_pickle=False)
    models=np.load(out/'models/extension_models.npz',allow_pickle=False)
    reg=pd.read_csv(out/'models/extension_models_registry.csv',keep_default_na=False)
    names=models['model_ids'].tolist();lookup={v:i for i,v in enumerate(names)}
    old=d['core_models'];old_lookup={v:i for i,v in enumerate(old['model_ids'].tolist())}
    c.require(names==reg.model_id.tolist(),'Registry row order differs')
    code_reconstruction=0;ensemble_checks=0;inner_checks=0
    inner=pd.read_csv(out/'splits/DFR_inner_folds.csv')
    with threadpool_limits(limits=1):
        for j,row in reg.iterrows():
            b=models['coefficients'][j];w=models['effective_sample_weights'][j]
            c.require(c.array_sha(b)==row.coefficient_sha256,'Coefficient SHA mismatch')
            c.require(c.array_sha(w)==row.effective_sample_weight_sha256,'Weight SHA mismatch')
            fold=int(row.heldout_fold)
            if fold>=0:c.require(not w[d['n_pad']:][d['fold']==fold].any(),'Nonzero outer-test fit weight')
            if row.purpose=='DFR_inner_tuning':
                val=inner[(inner.outer_fold==fold)&(inner.inner_fold==int(row.inner_fold))].DDI_feature_row_index.to_numpy()
                c.require(not w[d['n_pad']+val].any(),'DFR inner validation in fitting');inner_checks+=1
            if row.kind=='DFR_L1':
                c.require(not w[:d['n_pad']].any(),'DFR refit includes PAD rows')
                selected=np.flatnonzero(w[d['n_pad']:]>0)
                counts=[int(((d['tones'][selected]==t)&(d['yd'][selected]==y)).sum()) for t in ['I-II','III-IV','V-VI'] for y in [0,1]]
                c.require(len(set(counts))==1 and counts[0]>0,'DFR sampling not exactly six-group balanced')
            if row.kind=='DFR_ENSEMBLE':
                parts=json.loads(row.components_json);ii=[lookup[n] for n in parts]
                c.require(len(ii)==20,'DFR ensemble component count')
                c.require(np.max(abs(b-models['coefficients'][ii].mean(axis=0)))<1e-12,'DFR coefficient average mismatch')
                c.require(abs(models['intercepts'][j]-models['intercepts'][ii].mean())<1e-12,'DFR intercept average mismatch')
                ensemble_checks+=1
        for arm in z['arms']:
            a=z['score__'+arm];mid=z['model__'+arm]
            for draw in range(a.shape[1]):
                for name in np.unique(mid[:,draw]):
                    selected=np.flatnonzero(mid[:,draw]==name)
                    m=models if name in lookup else old;ix=lookup[name] if name in lookup else old_lookup[name]
                    reconstructed=d['XD'][selected]@m['coefficients'][ix]+m['intercepts'][ix]
                    c.require(np.max(abs(reconstructed-a[selected,draw]))<1e-10,'Heldout score reconstruction mismatch')
                    c.require(not m['effective_sample_weights'][ix,d['n_pad']+selected].any(),'Heldout image has fitting weight')
                    code_reconstruction+=len(selected)
    panels=pd.read_csv(out/'splits/adaptation_panels.csv.gz');nested_checks=0;matching_checks=0
    disease=d['DDI'].disease.astype(str).to_numpy()
    for fold in range(5):
        for draw in range(20):
            def selected(name):
                return panels[(panels.panel_id==f'{name}_f{fold}_d{draw:02d}')].DDI_feature_row_index.to_numpy(dtype=int)
            previous=set()
            for size in [25,50,100]:
                ix=selected('target_budget_'+str(size));pool=np.flatnonzero((d['fold']!=fold)&(d['tones']=='V-VI'))
                quota=max(1,min(size-1,int(np.floor(size*d['yd'][pool].mean()+.5))))
                c.require(len(ix)==size and len(set(ix))==size and previous<=set(ix),'Learning subsets not nested/unique')
                c.require((d['fold'][ix]!=fold).all() and (d['tones'][ix]=='V-VI').all() and d['yd'][ix].sum()==quota,'Learning quota/fold invalid')
                previous=set(ix);nested_checks+=1
            tx=selected('diagnosis_matched_target');lx=selected('diagnosis_matched_lighter')
            tc=pd.DataFrame({'disease':disease[tx],'y':d['yd'][tx]}).value_counts().sort_index()
            lc=pd.DataFrame({'disease':disease[lx],'y':d['yd'][lx]}).value_counts().sort_index()
            c.require(tc.equals(lc),'Diagnosis matching invalid');c.require(len(tx)==len(lx),'Budget mismatch')
            c.require((d['tones'][tx]=='V-VI').all() and (d['tones'][lx]!='V-VI').all(),'Matched source group invalid')
            c.require((d['fold'][tx]!=fold).all() and (d['fold'][lx]!=fold).all(),'Matched test leakage');matching_checks+=1
    bs=np.load(out/'uncertainty/paired_bootstrap_samples.npz',allow_pickle=False)
    index=json.loads((out/'uncertainty/sample_index.json').read_text())
    metric=pd.read_csv(out/'results/all_arm_metrics.csv');contrasts=pd.read_csv(out/'results/all_paired_contrasts.csv')
    group=json.loads((out/'predictions/arm_experiment_groups.json').read_text());lookup_bs={}
    point_checks=0;literal_boot_checks=0;ci_checks=0
    with threadpool_limits(limits=1):
        for key,desc in index.items():
            cohort,arm=desc['cohort'],desc['arm'];mask=np.ones(len(d['yd']),bool) if cohort=='ALL_DDI' else d['tones']==cohort
            y=d['yd'][mask];a=z['score__'+arm][mask];w=bs['image_counts'][:,mask]
            pts=[];first=[]
            for draw in range(a.shape[1]):
                score=a[:,draw];expected=roc_auc_score(y,score);computed=point(y,score)
                c.require(abs(expected-computed[0])<1e-14,'Point AUC mismatch')
                fpr,tpr,threshold=roc_curve(y,score,drop_intermediate=False)
                c.require(abs(tpr[fpr<=.5].max()-computed[1])<1e-14,'Point ROC sensitivity mismatch')
                pts.append(computed);boot=samples(y,score,w[:4]);first.append(boot)
                for b in range(4):
                    yy=np.repeat(y,w[b]);ss=np.repeat(score,w[b]);ff,tt,_=roc_curve(yy,ss,drop_intermediate=False)
                    c.require(abs(roc_auc_score(yy,ss)-boot[b,0])<1e-12,'Literal bootstrap AUC mismatch')
                    c.require(abs(tt[ff<=.5].max()-boot[b,1])<1e-12,'Literal bootstrap sensitivity mismatch');literal_boot_checks+=1
            pts=np.array(pts);v=bs['sample_metrics'][int(key)];lookup_bs[cohort,arm]=v
            rr=metric[(metric.cohort==cohort)&(metric.arm==arm)].iloc[0]
            c.require(abs(rr.AUROC-pts[:,0].mean())<1e-14,'Reported mean AUC mismatch')
            lo,hi=np.quantile(v[:,0],[.025,.975]);c.require(max(abs(lo-rr.AUROC_CI_low),abs(hi-rr.AUROC_CI_high))<1e-14,'Reported AUC CI mismatch')
            first=np.stack(first,axis=1)
            expected=first[:,0,:] if a.shape[1]==1 else np.einsum('bdm,bd->bm',first,bs['panel_counts_'+group[arm]][:4])/20
            c.require(np.max(abs(expected-v[:4]))<1e-12,'Saved paired image/panel resampling mismatch');point_checks+=1
        for _,r in contrasts.iterrows():
            k=0 if r.metric=='AUROC' else 1
            l=metric[(metric.cohort==r.cohort)&(metric.arm==r.left_arm)].iloc[0]
            q=metric[(metric.cohort==r.cohort)&(metric.arm==r.right_arm)].iloc[0]
            expected=float(l[r.metric]-q[r.metric]);diff=lookup_bs[r.cohort,r.left_arm][:,k]-lookup_bs[r.cohort,r.right_arm][:,k]
            lo,hi=np.quantile(diff,[.025,.975])
            c.require(abs(expected-r.difference_unrounded)<1e-14,'Unrounded contrast arithmetic mismatch')
            c.require(max(abs(lo-r.CI_low),abs(hi-r.CI_high))<1e-14,'Paired contrast CI mismatch');ci_checks+=1
    # These are synthetic INPUT-VALIDATION fixtures only. No synthetic study result/table/figure is produced.
    rng=np.random.default_rng(73);xx=rng.normal(size=(6,1152)).astype(np.float32);xx/=np.linalg.norm(xx,axis=1,keepdims=True)
    manifest=pd.DataFrame({'analysis_image_id':[f'TEST{i}' for i in range(6)],'feature_row_index':range(6),
                            'feature_row_sha256':[c.array_sha(v) for v in xx],
                            'image_sha256':[hashlib.sha256(f'TEST_ONLY{i}'.encode()).hexdigest() for i in range(6)],
                            'patient_key':['p0','p0','p1','p2','p2','p3'],'lesion_key':['l0','l0','l1','l2','l2','l3'],
                            'diagnosis_code':['NEG','NEG','POS','NEG','NEG','POS'],'target_label':[0,0,1,0,0,1],
                            'skin_tone_group':['I-II']*6,'modality':['clinical']*6,'biopsy_confirmed':[1]*6,'eligible':[1]*6})
    label={'dataset_doi':'10.71718/15nz-jv40','reviewed_by':'SYNTHETIC_TEST_ONLY','reviewed_utc':'TEST_ONLY',
           'histopathology_binary_endpoint_reviewed':True,'intermediate_and_uncertain_excluded':True,'clinical_only_reviewed':True,
           'diagnosis_to_binary':{'NEG':0,'POS':1,'INTERMEDIATE':'EXCLUDE'}}
    spec=d['plan']['feature_spec'];passed=validate_features_and_manifest(xx,manifest,spec,d['plan'],label,[])
    c.require(passed.all(),'Valid validation fixture rejected');failure_cases=0
    cases=[]
    def changed(column,row,value):
        mm=manifest.copy();mm.loc[row,column]=value;return mm
    cases += [(changed('modality',0,'dermoscopy'),spec,[]),(changed('patient_key',0,''),spec,[]),
              (changed('target_label',2,0),spec,[]),(changed('biopsy_confirmed',0,0),spec,[]),
              (changed('diagnosis_code',2,'INTERMEDIATE'),spec,[]),(changed('feature_row_index',0,5),spec,[]),
              (changed('feature_row_sha256',0,'WRONG'),spec,[]),(changed('analysis_image_id',1,'TEST0'),spec,[]),
              (manifest,{**spec,'revision':'WRONG'},[]),(manifest,spec,[manifest.image_sha256.iloc[0]])]
    mm=manifest.copy();mm.loc[1,['diagnosis_code','target_label']]=['POS',1];cases.append((mm,spec,[]))
    for mm,ss,overlap in cases:
        try:validate_features_and_manifest(xx,mm,ss,d['plan'],label,overlap)
        except ValueError:failure_cases+=1
        else:raise AssertionError('Invalid external fixture accepted')
    c.write_json(out/'logs/independent_extension_verification.json',{
        'completed_utc':c.now(),'status':'PASS','protocol_sha256':d['plan_sha'],'code_sha256':c.sha(__file__),
        'saved_heads_verified':len(reg),'heldout_and_transfer_predictions_reconstructed':code_reconstruction,
        'DFR_ensemble_reconstruction_checks':ensemble_checks,'DFR_inner_validation_exclusion_checks':inner_checks,
        'nested_learning_panel_checks':nested_checks,'exact_diagnosis_matching_panel_pair_checks':matching_checks,
        'metric_rows_verified':point_checks,'literal_sklearn_bootstrap_checks':literal_boot_checks,'unrounded_paired_contrast_CI_checks':ci_checks,
        'E4_valid_synthetic_input_fixture_passes':1,'E4_invalid_synthetic_input_fixtures_rejected':failure_cases,
        'synthetic_fixtures_exported_as_scientific_results':False,'real_MRA_MIDAS_validation_performed':False,
        'DDI_patient_independence_established':False,
    })
    print('INDEPENDENT VERIFICATION PASS:',len(reg),'heads,',code_reconstruction,'predictions,',ci_checks,'contrasts; E4 only validation fixtures')


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--base',required=True);ap.add_argument('--output',required=True)
    a=ap.parse_args();run(a.base,a.output)
