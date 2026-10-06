"""Run the explicitly exploratory E1-E3 extension and freeze E4 heads."""
from __future__ import annotations
import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

import make_extension_common as c


def run(base, output):
    d=c.load(base,output);out=d['out'];started=c.now()
    c.require(not (out/'logs/extension_fit_audit.json').exists(),'Fits already saved; verify instead of replacing')
    c.require(c.runtime()['scikit-learn']==d['plan']['runtime']['scikit-learn'],'Classifier package version differs from recorded protocol')
    scaler=StandardScaler().fit(d['XP']);d['scaler_mean']=scaler.mean_;d['scaler_scale']=scaler.scale_
    store=c.ModelStore(d);D=d['plan']['draws'];n=len(d['DDI'])
    scores={};score_models={};group={};panels=[];tuning=[];inner_rows=[];coverage=[];budgets=[]
    new_names=['all_DDI_unweighted','all_DDI_source_weighted','all_DDI_group_reweighted',
               'DDI_only_class_balanced','DDI_only_group_balanced','DFR_style']
    draw_names=['all_DDI_budget_matched','target_budget_25','target_budget_50','target_budget_100',
                'diagnosis_matched_target','diagnosis_matched_lighter']
    for name in new_names+draw_names:
        width=D if name in draw_names else 1
        scores[name]=np.full((n,width),np.nan);score_models[name]=np.full((n,width),'',dtype='<U120')
        group[name]='E3' if name.startswith('diagnosis') else ('E2' if name.startswith('target_budget') else 'E1')
    old=d['core'];names=old['DDI_arm_names'].tolist()
    for name in ['PAD_baseline','target_weighted','target_unweighted','lighter_weighted']:
        cols=[j for j,a in enumerate(names) if a==name or a.startswith(name+'_draw_')]
        scores[name]=old['DDI_logits'][:,cols].copy();score_models[name]=old['DDI_model_ids'][:,cols].copy()
        group[name]='BASE'
    deployment_rows=[]

    def record_prediction(name,fold,row,draw=0):
        ix=np.flatnonzero(d['fold']==fold)
        scores[name][ix,draw]=store.predict(row,d['XD'][ix])
        score_models[name][ix,draw]=store.registry[row]['model_id']

    def add_budget(name,fold,draw,ix):
        budgets.append({'arm':name,'heldout_fold':fold,'draw_id':draw,'DDI_fit_rows':len(ix),
                        'DDI_positive':int(d['yd'][ix].sum()),'DDI_negative':int(len(ix)-d['yd'][ix].sum())})

    def dfr(pool,fold):
        # Three-fold nested tuning; no DDI outer-test information enters scaling or selection.
        local_y=np.array([f'{t}:{y}' for t,y in zip(d['tones'][pool],d['yd'][pool])])
        seed=20261006+500+(fold if fold>=0 else 99)
        skf=StratifiedKFold(n_splits=3,shuffle=True,random_state=seed)
        splits=list(skf.split(np.zeros(len(pool)),local_y));grid=d['plan']['E1']['DFR_style']['C_grid'];means=[]
        for inner,(_,valid) in enumerate(splits):
            inner_rows.extend({'outer_fold':fold,'analysis_image_id':d['DDI'].iloc[i].analysis_image_id,
                               'DDI_feature_row_index':int(i),'inner_fold':inner} for i in pool[valid])
        for C in grid:
            values=[]
            for inner,(train_local,valid_local) in enumerate(splits):
                train,valid=pool[train_local],pool[valid_local]
                c.require(not set(train)&set(valid),'DFR inner leakage')
                for draw in range(4):
                    rng=np.random.default_rng(np.random.SeedSequence([20261006,600,fold+1,inner,draw]))
                    ix=c.balanced_panel(d,train,rng)
                    w=np.zeros(len(d['y']));w[d['n_pad']+ix]=1.
                    mid=f'DFR_tune_f{fold}_i{inner}_d{draw}_C{C:g}'
                    row=store.fit(mid,w,C=C,kind='DFR_L1',heldout_fold=fold,purpose='DFR_inner_tuning',
                                  inner_fold=inner,draw_id=draw,arm='DFR_style')
                    pred=store.predict(row,d['XD'][valid])>0
                    group_acc=[float(np.mean(pred[(d['tones'][valid]==t)&(d['yd'][valid]==y)]==y))
                               for t in ['I-II','III-IV','V-VI'] for y in [0,1]]
                    worst=min(group_acc);values.append(worst)
                    tuning.append({'outer_fold':fold,'inner_fold':inner,'draw_id':draw,'C':C,
                                   'worst_six_group_accuracy':worst,'model_id':mid,'validation_n':len(valid)})
            means.append(float(np.mean(values)))
        best=max(means);C=next(v for v,m in zip(grid,means) if best-m<=1e-12)
        members=[]
        for draw in range(D):
            rng=np.random.default_rng(np.random.SeedSequence([20261006,610,fold+1,draw]))
            ix=c.balanced_panel(d,pool,rng);w=np.zeros(len(d['y']));w[d['n_pad']+ix]=1.
            pid=f'DFR_refit_f{fold}_d{draw:02d}';panels.extend(c.rows_for_panel(d,pid,'DFR_refit',fold,draw,ix))
            row=store.fit(pid,w,C=C,kind='DFR_L1',heldout_fold=fold,purpose='DFR_component',
                          draw_id=draw,arm='DFR_style',sampling_panel_id=pid)
            members.append(row)
        ensemble=store.ensemble(f'DFR_ensemble_f{fold}',members,C=C,heldout_fold=fold,
                                purpose='DDI_OOF' if fold>=0 else 'E4_DEPLOYMENT',arm='DFR_style')
        print(f'DFR fold {fold}: internal selected C={C:g}, twenty balanced components',flush=True)
        return ensemble

    with threadpool_limits(limits=1):
        for fold in range(5):
            pool=np.flatnonzero(d['fold']!=fold)
            target=pool[d['tones'][pool]=='V-VI'];light=pool[d['tones'][pool]!='V-VI']
            quota=np.bincount(d['yd'][target],minlength=2)
            for name in new_names:
                if name=='DFR_style':row=dfr(pool,fold)
                else:
                    if name.startswith('DDI_only'):w=c.ddi_only_weights(d,pool,group=name.endswith('group_balanced'))
                    else:w=c.pooled_weights(d,pool,source_weighted=name!='all_DDI_unweighted',group=name=='all_DDI_group_reweighted')
                    row=store.fit(f'{name}_f{fold}',w,heldout_fold=fold,arm=name)
                record_prediction(name,fold,row);add_budget(name,fold,-1,pool)
            # Same-budget all-tone panels, balanced only to the target outcomes.
            for draw in range(D):
                rng=np.random.default_rng(np.random.SeedSequence([20261006,200,fold,draw]))
                ix=np.sort(np.concatenate([rng.choice(pool[d['yd'][pool]==y],size=int(quota[y]),replace=False) for y in [0,1]]))
                pid=f'all_DDI_budget_matched_f{fold}_d{draw:02d}'
                panels.extend(c.rows_for_panel(d,pid,'E1_budget_matched',fold,draw,ix))
                row=store.fit(pid,c.pooled_weights(d,ix),heldout_fold=fold,arm='all_DDI_budget_matched',draw_id=draw,sampling_panel_id=pid)
                record_prediction('all_DDI_budget_matched',fold,row,draw);add_budget('all_DDI_budget_matched',fold,draw,ix)
            # Nested prefixes in the two outcome-specific permutations.
            for draw in range(D):
                rng=np.random.default_rng(np.random.SeedSequence([20261006,300,fold,draw]))
                permutations=[rng.permutation(target[d['yd'][target]==y]) for y in [0,1]]
                for budget in [25,50,100]:
                    npos=min(budget-1,max(1,int(np.floor(budget*quota[1]/len(target)+.5))))
                    ix=np.sort(np.r_[permutations[0][:budget-npos],permutations[1][:npos]])
                    c.require(len(ix)==budget,'Learning-curve quota failed')
                    name=f'target_budget_{budget}';pid=f'{name}_f{fold}_d{draw:02d}'
                    panels.extend(c.rows_for_panel(d,pid,'E2_learning_curve',fold,draw,ix))
                    row=store.fit(pid,c.pooled_weights(d,ix),heldout_fold=fold,arm=name,draw_id=draw,sampling_panel_id=pid)
                    record_prediction(name,fold,row,draw);add_budget(name,fold,draw,ix)
            # Maximum exact diagnosis common support; never silently match malignancy alone.
            cells=[]
            diseases=d['DDI'].disease.astype(str).to_numpy()
            for disease in sorted(set(diseases[target])):
                for label in [0,1]:
                    tt=target[(diseases[target]==disease)&(d['yd'][target]==label)]
                    ll=light[(diseases[light]==disease)&(d['yd'][light]==label)]
                    if not len(tt):continue
                    q=min(len(tt),len(ll))
                    coverage.append({'heldout_fold':fold,'disease':disease,'target_label':label,'target_available':len(tt),
                                     'lighter_available':len(ll),'matched_per_source':q,'target_rows_not_retained':len(tt)-q})
                    if q:cells.append((tt,ll,q))
            for draw in range(D):
                rng=np.random.default_rng(np.random.SeedSequence([20261006,400,fold,draw]))
                tix=[];lix=[]
                for tt,ll,q in cells:
                    tix.extend(rng.choice(tt,size=q,replace=False));lix.extend(rng.choice(ll,size=q,replace=False))
                tix=np.sort(tix);lix=np.sort(lix)
                count_t=pd.DataFrame({'disease':diseases[tix],'y':d['yd'][tix]}).value_counts().sort_index()
                count_l=pd.DataFrame({'disease':diseases[lix],'y':d['yd'][lix]}).value_counts().sort_index()
                c.require(count_t.equals(count_l),'Exact diagnosis/outcome matching failed')
                for name,ix in [('diagnosis_matched_target',tix),('diagnosis_matched_lighter',lix)]:
                    pid=f'{name}_f{fold}_d{draw:02d}';panels.extend(c.rows_for_panel(d,pid,'E3_diagnosis_match',fold,draw,ix))
                    row=store.fit(pid,c.pooled_weights(d,ix),heldout_fold=fold,arm=name,draw_id=draw,sampling_panel_id=pid)
                    record_prediction(name,fold,row,draw);add_budget(name,fold,draw,ix)
            print(f'Outer fold {fold+1}/5 complete: all controls, learning sizes and {len(tix)} exact diagnosis matches/source',flush=True)
        # Final heads are frozen for future independent external validation.
        cm=d['core_models'];bi=cm['model_ids'].tolist().index('PAD_baseline')
        baseline=store.add('E4_PAD_baseline',cm['coefficients'][bi],cm['intercepts'][bi],cm['effective_sample_weights'][bi],
                           C=10.,kind='L2',heldout_fold=-1,purpose='E4_DEPLOYMENT',arm='PAD_baseline',inherited_from='base core_models.npz')
        deployment_rows.append(baseline)
        full=np.arange(n);tv=full[d['tones']=='V-VI']
        for name,ix in [('target_weighted_full',tv),('all_DDI_source_weighted_full',full),('all_DDI_group_reweighted_full',full),
                        ('DDI_only_class_balanced_full',full),('DDI_only_group_balanced_full',full)]:
            if name.startswith('DDI_only'):w=c.ddi_only_weights(d,ix,group='group_balanced' in name)
            else:w=c.pooled_weights(d,ix,group='group_reweighted' in name)
            row=store.fit('E4_'+name,w,heldout_fold=-1,purpose='E4_DEPLOYMENT',arm=name)
            deployment_rows.append(row)
        dr=dfr(full,-1);deployment_rows.append(dr)
    c.require(all(np.isfinite(a).all() for a in scores.values()),'Missing heldout prediction')
    store.save(out/'models/extension_models')
    # Preserve only prespecified final deployable heads in a compact independent file.
    np.savez_compressed(out/'models/E4_deployment_models.npz',coefficients=np.array([store.coef[j] for j in deployment_rows]),
                        intercepts=np.array([store.intercept[j] for j in deployment_rows]),
                        model_ids=np.array([store.registry[j]['model_id'] for j in deployment_rows]),
                        protocol_sha256=np.array(d['plan_sha']),feature_spec_sha256=np.array(c.sha(d['base']/'inputs/features/feature_spec.json')))
    pd.DataFrame([store.registry[j] for j in deployment_rows]).to_csv(out/'models/E4_deployment_registry.csv',index=False,float_format='%.17g')
    np.savez_compressed(out/'models/DFR_source_scaler.npz',mean=d['scaler_mean'],scale=d['scaler_scale'])
    saved={f'score__{k}':v for k,v in scores.items()};saved.update({f'model__{k}':v for k,v in score_models.items()})
    saved.update(image_ids=d['DDI'].analysis_image_id.to_numpy(dtype=str),labels=d['yd'],tones=d['tones'],folds=d['fold'],
                 arms=np.array(list(scores)),protocol_sha256=np.array(d['plan_sha']))
    np.savez_compressed(out/'predictions/extension_scores.npz',**saved)
    prediction_rows=[]
    for name,a in scores.items():
        for draw in range(a.shape[1]):
            for i in range(n):
                prediction_rows.append({'analysis_image_id':d['DDI'].iloc[i].analysis_image_id,
                                        'feature_row_index':i,'image_sha256':d['DDI'].iloc[i].image_sha256,
                                        'target_label':int(d['yd'][i]),'skin_tone_group':d['tones'][i],
                                        'heldout_fold':int(d['fold'][i]),'arm':name,'draw_id':draw,
                                        'logit_score':float(a[i,draw]),'model_id':score_models[name][i,draw],
                                        'model_file':'base:models/core_models.npz' if group[name]=='BASE' else 'models/extension_models.npz',
                                        'protocol_sha256':d['plan_sha']})
    pd.DataFrame(prediction_rows).to_csv(out/'predictions/DDI_per_image_predictions.csv.gz',index=False,float_format='%.17g',compression={'method':'gzip','mtime':0})
    pd.DataFrame(panels).to_csv(out/'splits/adaptation_panels.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    pd.DataFrame(inner_rows).to_csv(out/'splits/DFR_inner_folds.csv',index=False)
    pd.DataFrame(tuning).to_csv(out/'results/DFR_inner_tuning.csv',index=False,float_format='%.17g')
    pd.DataFrame(coverage).to_csv(out/'results/E3_diagnosis_common_support.csv',index=False)
    pd.DataFrame(budgets).to_csv(out/'results/annotation_budgets.csv',index=False)
    c.write_json(out/'predictions/arm_experiment_groups.json',group)
    audit={'started_utc':started,'completed_utc':c.now(),'protocol_sha256':d['plan_sha'],'runtime':c.runtime(),
           'fitted_and_inherited_heads':len(store.registry),'head_counts_by_purpose':pd.Series([r['purpose'] for r in store.registry]).value_counts().to_dict(),
           'new_results_are_exploratory':True,'named_old_extension_plan_recovered':False,'MRA_MIDAS_evaluated':False,
           'code_sha256':{p.name:c.sha(p) for p in [Path(__file__),Path(c.__file__)]},
           'output_sha256':{str(p.relative_to(out)):c.sha(p) for folder in ['models','predictions','splits','results'] for p in sorted((out/folder).glob('*')) if p.is_file()}}
    c.write_json(out/'logs/extension_fit_audit.json',audit)
    print('EXTENSION FITTING PASS:',len(store.registry),'saved heads,',len(prediction_rows),'image predictions',flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--base',required=True);ap.add_argument('--output',required=True)
    args=ap.parse_args();run(args.base,args.output)
