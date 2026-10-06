"""Reconstruct every DFR inner validation score and verify selected C."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import make_extension_common as c


def run(base,output):
    d=c.load(base,output);out=d['out'];m=np.load(out/'models/extension_models.npz',allow_pickle=False)
    registry=pd.read_csv(out/'models/extension_models_registry.csv',keep_default_na=False).set_index('model_id')
    split=pd.read_csv(out/'splits/DFR_inner_folds.csv');tuning=pd.read_csv(out/'results/DFR_inner_tuning.csv')
    predictions=[];checks=0
    with threadpool_limits(limits=1):
        for _,r in tuning.iterrows():
            model=registry.loc[r.model_id];row=int(model.storage_row_index)
            ix=split[(split.outer_fold==r.outer_fold)&(split.inner_fold==r.inner_fold)].DDI_feature_row_index.to_numpy(dtype=int)
            s=d['XD'][ix]@m['coefficients'][row]+m['intercepts'][row];pred=s>0
            acc=[float(np.mean(pred[(d['tones'][ix]==t)&(d['yd'][ix]==y)]==y)) for t in ['I-II','III-IV','V-VI'] for y in [0,1]]
            c.require(abs(min(acc)-r.worst_six_group_accuracy)<1e-14,'DFR internal tuning accuracy changed')
            for i,v in zip(ix,s):
                predictions.append({'analysis_image_id':d['DDI'].iloc[i].analysis_image_id,'feature_row_index':int(i),
                                    'outer_fold':int(r.outer_fold),'inner_fold':int(r.inner_fold),'draw_id':int(r.draw_id),
                                    'C':float(r.C),'target_label':int(d['yd'][i]),'skin_tone_group':d['tones'][i],
                                    'logit_score':float(v),'model_id':r.model_id,'provenance':'reconstructed from saved coefficients; no refitting'})
            checks+=1
        for f in sorted(tuning.outer_fold.unique()):
            mean=tuning[tuning.outer_fold==f].groupby('C').worst_six_group_accuracy.mean().sort_index()
            best=mean.max();chosen=next(float(C) for C,value in mean.items() if best-value<=1e-12)
            ensemble=registry.loc[f'DFR_ensemble_f{f}'];c.require(abs(float(ensemble.C)-chosen)<1e-14,'DFR C selection does not match internal rule')
    dest=out/'predictions/DFR_inner_validation_predictions.csv.gz'
    pd.DataFrame(predictions).to_csv(dest,index=False,float_format='%.17g',compression={'method':'gzip','mtime':0})
    c.write_json(out/'logs/DFR_tuning_reconstruction.json',{'completed_utc':c.now(),'status':'PASS','protocol_sha256':d['plan_sha'],
                 'internal_tuning_metrics_verified':checks,'internal_validation_predictions_reconstructed':len(predictions),
                 'outer_and_final_deployment_C_selections_verified':6,'model_refitting_performed':False,'code_sha256':c.sha(__file__),
                 'source_sha256':{n:c.sha(out/n) for n in ['models/extension_models.npz','splits/DFR_inner_folds.csv','results/DFR_inner_tuning.csv']},
                 'output_sha256':c.sha(dest)})
    print('DFR INTERNAL TUNING PASS:',checks,'tuning rows,',len(predictions),'validation predictions,6 C selections')


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--base',required=True);ap.add_argument('--output',required=True)
    a=ap.parse_args();run(a.base,a.output)
