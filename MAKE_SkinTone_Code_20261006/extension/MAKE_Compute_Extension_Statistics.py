"""Paired image-and-panel conditional uncertainty; all contrasts unrounded."""
from __future__ import annotations
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits
import make_extension_common as c


def resample_counts(strata,B,rng):
    w=np.zeros((B,len(strata)),dtype=np.uint16)
    for label in sorted(set(strata)):
        ix=np.flatnonzero(strata==label)
        w[:,ix]=rng.multinomial(len(ix),np.full(len(ix),1/len(ix)),size=B)
    return w


def point(y,s):
    sn=np.sort(s[y==0])[::-1];threshold=sn[int(np.floor(.5*len(sn)))]
    return np.array([roc_auc_score(y,s),np.mean(s[y==1]>threshold),np.mean(sn>threshold)],dtype=float)


def samples(y,s,w,batch_size=250):
    """Weighted empirical AUROC with exact tie credit, plus descriptive ROC sensitivity."""
    order=np.argsort(s,kind='stable');ordered=s[order];ys=y[order]
    starts=np.r_[0,np.flatnonzero(np.diff(ordered)!=0)+1]
    pos=np.flatnonzero(y==1);neg=np.flatnonzero(y==0);negative_order=neg[np.argsort(-s[neg],kind='stable')]
    result=np.empty((len(w),3))
    for a in range(0,len(w),batch_size):
        b=w[a:a+batch_size].astype(np.float64);np_=b[:,pos].sum(axis=1);nn_=b[:,neg].sum(axis=1)
        c.require((np_>0).all() and (nn_>0).all(),'Bootstrap cohort lacks a class')
        bs=b[:,order];wp=np.add.reduceat(bs*ys[None,:],starts,axis=1)
        wn=np.add.reduceat(bs*(1-ys)[None,:],starts,axis=1)
        au=(wp*(np.cumsum(wn,axis=1)-.5*wn)).sum(axis=1)/(np_*nn_)
        cum=np.cumsum(b[:,negative_order],axis=1);boundary=np.argmax(cum>.5*nn_[:,None],axis=1)
        th=s[negative_order][boundary]
        sens=(b[:,pos]*(s[pos][None,:]>th[:,None])).sum(axis=1)/np_
        fpr=(b[:,neg]*(s[neg][None,:]>th[:,None])).sum(axis=1)/nn_
        result[a:a+len(b)]=np.column_stack([au,sens,fpr])
    return result


def ci(x):
    return np.quantile(x,[.025,.975],axis=0,method='linear')


def run(base,output):
    d=c.load(base,output);out=d['out'];started=c.now();p=d['plan'];B=p['uncertainty']['replicates']
    c.require(not (out/'logs/extension_statistics_audit.json').exists(),'Statistics exist; verify instead')
    fit=json.loads((out/'logs/extension_fit_audit.json').read_text());c.require(fit['protocol_sha256']==d['plan_sha'],'Fitted protocol changed')
    for n,h in fit['output_sha256'].items():c.require(c.sha(out/n)==h,'Fit artifact changed: '+n)
    saved=np.load(out/'predictions/extension_scores.npz',allow_pickle=False);arms=saved['arms'].tolist()
    groups=json.loads((out/'predictions/arm_experiment_groups.json').read_text())
    strata=np.array([f'{f}:{t}:{y}' for f,t,y in zip(d['fold'],d['tones'],d['yd'])])
    rng=np.random.default_rng(np.random.SeedSequence([20261006,800]))
    weights=resample_counts(strata,B,rng)
    panel_weights={g:np.random.default_rng(np.random.SeedSequence([20261006,810,j])).multinomial(20,np.full(20,.05),size=B)
                   for j,g in enumerate(['BASE','E1','E2','E3'])}
    cohorts=['ALL_DDI','I-II','III-IV','V-VI'];metrics=['AUROC','sensitivity_FPR_le_0_50','achieved_FPR']
    values={};bs_values={};rows=[];draw_rows=[]
    with threadpool_limits(limits=1):
        for cohort in cohorts:
            mask=np.ones(len(d['yd']),dtype=bool) if cohort=='ALL_DDI' else d['tones']==cohort
            y=d['yd'][mask];ww=weights[:,mask]
            for arm in arms:
                scores=saved['score__'+arm][mask];K=scores.shape[1]
                pts=np.array([point(y,scores[:,k]) for k in range(K)])
                boots=np.stack([samples(y,scores[:,k],ww) for k in range(K)],axis=1)
                mean=pts.mean(axis=0)
                if K==1:mean_bs=boots[:,0,:]
                else:mean_bs=np.einsum('bdm,bd->bm',boots,panel_weights[groups[arm]])/K
                limits=ci(mean_bs);values[cohort,arm]=mean;bs_values[cohort,arm]=mean_bs
                row={'cohort':cohort,'arm':arm,'n':len(y),'positive':int(y.sum()),'negative':int(len(y)-y.sum()),
                     'training_panels':K,'point_estimator':'mean of per-panel metrics' if K>1 else 'single prespecified score vector',
                     'bootstrap_replicates':B,'uncertainty':'paired fold x tone x outcome image bootstrap'+(' + training-panel bootstrap' if K>1 else ''),
                     'inference_scope':'exploratory; conditional on fitted heads and saved folds; no DDI patient-independence claim'}
                for j,m in enumerate(metrics):
                    row[m]=float(mean[j]);row[m+'_CI_low']=float(limits[0,j]);row[m+'_CI_high']=float(limits[1,j])
                    row[m+'_panel_SD']=float(pts[:,j].std(ddof=1)) if K>1 else 0.
                rows.append(row)
                for k in range(K):
                    draw_rows.append({'cohort':cohort,'arm':arm,'draw_id':k,**{m:float(pts[k,j]) for j,m in enumerate(metrics)}})
            print('Paired uncertainty complete:',cohort,B,'samples',flush=True)
    pairs=[('target_weighted','PAD_baseline'),('target_unweighted','PAD_baseline'),('target_weighted','lighter_weighted')]
    pairs += [(a,'PAD_baseline') for a in arms if a not in ['PAD_baseline','target_weighted','target_unweighted','lighter_weighted']]
    pairs += [('target_weighted',a) for a in ['all_DDI_unweighted','all_DDI_source_weighted','all_DDI_group_reweighted',
                                           'all_DDI_budget_matched','DDI_only_class_balanced','DDI_only_group_balanced','DFR_style']]
    pairs += [('target_budget_50','target_budget_25'),('target_budget_100','target_budget_50'),('target_weighted','target_budget_100'),
              ('diagnosis_matched_target','diagnosis_matched_lighter'),('diagnosis_matched_target','target_weighted')]
    contrasts=[]
    for cohort in cohorts:
        for left,right in pairs:
            difference=values[cohort,left]-values[cohort,right];limits=ci(bs_values[cohort,left]-bs_values[cohort,right])
            for j,m in enumerate(metrics[:2]):
                contrasts.append({'cohort':cohort,'metric':m,'left_arm':left,'right_arm':right,
                                  'left_value':float(values[cohort,left][j]),'right_value':float(values[cohort,right][j]),
                                  'difference_unrounded':float(difference[j]),'CI_low':float(limits[0,j]),'CI_high':float(limits[1,j]),
                                  'bootstrap_replicates':B,'CI_type':'paired exploratory95% percentile; unadjusted',
                                  'difference_display_3dp':f'{difference[j]:+.3f}'})
    metrics_df=pd.DataFrame(rows);contrast_df=pd.DataFrame(contrasts)
    metrics_df.to_csv(out/'results/all_arm_metrics.csv',index=False,float_format='%.17g')
    contrast_df.to_csv(out/'results/all_paired_contrasts.csv',index=False,float_format='%.17g')
    pd.DataFrame(draw_rows).to_csv(out/'results/panel_level_metrics.csv',index=False,float_format='%.17g')
    e1=['PAD_baseline','target_weighted','target_unweighted','lighter_weighted','all_DDI_unweighted','all_DDI_source_weighted',
        'all_DDI_group_reweighted','all_DDI_budget_matched','DDI_only_class_balanced','DDI_only_group_balanced','DFR_style']
    metrics_df[metrics_df.arm.isin(e1)].to_csv(out/'results/Table_E1_baselines.csv',index=False,float_format='%.17g')
    metrics_df[metrics_df.arm.isin(['PAD_baseline','target_budget_25','target_budget_50','target_budget_100','target_weighted'])].to_csv(out/'results/Table_E2_learning_curve.csv',index=False,float_format='%.17g')
    metrics_df[metrics_df.arm.isin(['diagnosis_matched_target','diagnosis_matched_lighter'])].to_csv(out/'results/Table_E3_diagnosis_matched.csv',index=False,float_format='%.17g')
    cc=pd.read_csv(out/'results/E3_diagnosis_common_support.csv')
    coverage=cc.groupby('heldout_fold').agg(target_available=('target_available','sum'),matched_per_source=('matched_per_source','sum'),
                                         target_rows_not_retained=('target_rows_not_retained','sum')).reset_index()
    coverage['target_fraction_retained']=coverage.matched_per_source/coverage.target_available
    coverage.to_csv(out/'results/Table_E3_matching_coverage.csv',index=False,float_format='%.17g')
    sample_keys={};sample_array=[]
    for cohort in cohorts:
        for arm in arms:
            sample_keys[len(sample_array)]={'cohort':cohort,'arm':arm};sample_array.append(bs_values[cohort,arm])
    np.savez_compressed(out/'uncertainty/paired_bootstrap_samples.npz',sample_metrics=np.array(sample_array),
                        image_counts=weights,**{'panel_counts_'+k:v for k,v in panel_weights.items()},
                        metrics=np.array(metrics),protocol_sha256=np.array(d['plan_sha']))
    c.write_json(out/'uncertainty/sample_index.json',sample_keys)
    audit={'started_utc':started,'completed_utc':c.now(),'protocol_sha256':d['plan_sha'],'bootstrap_replicates':B,
           'code_sha256':c.sha(__file__),'metric_rows':len(rows),'contrast_rows':len(contrasts),
           'effects_calculated_before_rounding':True,'MRA_MIDAS_evaluated':False,
           'output_sha256':{str(f.relative_to(out)):c.sha(f) for folder in ['results','uncertainty'] for f in sorted((out/folder).glob('*')) if f.is_file()}}
    c.write_json(out/'logs/extension_statistics_audit.json',audit)
    print('STATISTICS PASS; E1/E2/E3 saved; no MRA result fabricated',flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--base',required=True);ap.add_argument('--output',required=True)
    a=ap.parse_args();run(a.base,a.output)
