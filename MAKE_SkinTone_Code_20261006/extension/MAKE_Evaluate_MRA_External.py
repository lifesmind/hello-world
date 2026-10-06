"""Evaluate frozen heads on a REVIEWED MRA-MIDAS feature/label manifest.

No network access, raw-data upload, model selection, refitting or automatic
diagnosis mapping. Requires actual external inputs; never supplies results from
synthetic data. The validation functions are separately smoke-tested.
"""
from __future__ import annotations
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

import make_extension_common as c
from MAKE_Compute_Extension_Statistics import samples


REQUIRED=['analysis_image_id','feature_row_index','feature_row_sha256','image_sha256',
          'patient_key','lesion_key','diagnosis_code','target_label','skin_tone_group',
          'modality','biopsy_confirmed','eligible']


def validate_features_and_manifest(X,m,feature_spec,protocol,label_audit,source_image_hashes):
    c.require(set(REQUIRED).issubset(m.columns),'Missing manifest columns: '+str(sorted(set(REQUIRED)-set(m.columns))))
    c.require(X.dtype==np.float32 and X.shape==(len(m),1152),'MRA feature shape/dtype invalid')
    c.require(np.isfinite(X).all() and np.max(abs(np.linalg.norm(X,axis=1)-1))<2e-6,'MRA features not finite/L2 normalized')
    c.require(np.array_equal(m.feature_row_index.to_numpy(dtype=int),np.arange(len(X))),'MRA feature index mismatch')
    c.require(m.analysis_image_id.is_unique and m.image_sha256.is_unique,'Duplicate MRA image identifiers or byte hashes')
    c.require(all(c.array_sha(X[i])==str(m.iloc[i].feature_row_sha256) for i in range(len(X))),'MRA feature-row hash mismatch')
    reference=protocol['feature_spec']
    keys=['repo_id','revision','dimension','dtype','feature_method','image_conversion','image_processor_class',
          'image_processor_backend','image_processor_config']
    for k in keys:c.require(feature_spec.get(k)==reference.get(k),'Encoder/processor mismatch: '+k)
    for fn in ['model.safetensors','config.json','preprocessor_config.json']:
        c.require(feature_spec.get('model_source_files',{}).get(fn,{}).get('sha256')==reference['model_source_files'][fn]['sha256'],
                  'Pinned encoder file hash mismatch: '+fn)
    c.require(label_audit.get('dataset_doi')=='10.71718/15nz-jv40','Dataset label-audit DOI mismatch')
    c.require(bool(label_audit.get('reviewed_by')) and bool(label_audit.get('reviewed_utc')),'Reviewed diagnosis mapping required')
    for field in ['histopathology_binary_endpoint_reviewed','intermediate_and_uncertain_excluded','clinical_only_reviewed']:
        c.require(label_audit.get(field) is True,'Unreviewed endpoint: '+field)
    mapping=label_audit.get('diagnosis_to_binary',{})
    c.require(len(mapping)>0 and all(v in [0,1,'EXCLUDE'] for v in mapping.values()),'Invalid diagnosis-to-binary mapping')
    c.require(set(m.eligible.astype(str)) <= {'0','1'},'Eligibility must be reviewed binary0/1')
    selected=m.eligible.astype(int).to_numpy()==1
    c.require(selected.any(),'No eligible external clinical biopsy images')
    s=m.loc[selected].copy()
    c.require((s.patient_key.astype(str).str.len()>0).all() and (s.lesion_key.astype(str).str.len()>0).all(),'Missing patient/lesion key')
    c.require((s.modality=='clinical').all() and (s.biopsy_confirmed.astype(str)=='1').all(),'Only biopsied clinical views eligible')
    c.require(set(s.skin_tone_group) <= {'I-II','III-IV','V-VI','UNKNOWN'},'Invalid reviewed tone groups')
    c.require(set(s.target_label.astype(str)) <= {'0','1'},'Invalid included binary labels')
    for _,row in s.iterrows():
        code=str(row.diagnosis_code)
        c.require(code in mapping and mapping[code] != 'EXCLUDE','Excluded or unmapped diagnosis included: '+code)
        c.require(int(row.target_label)==mapping[code],'Diagnosis/outcome mapping mismatch')
    for _,rows in s.groupby(['patient_key','lesion_key'],dropna=False):
        c.require(rows.target_label.nunique()==1 and rows.skin_tone_group.nunique()==1,'Conflicting labels/tone in a lesion')
    overlap=set(s.image_sha256)&set(source_image_hashes)
    c.require(not overlap,'External images overlap source head-fitting cohorts by byte hash')
    c.require(set(s.target_label.astype(int))=={0,1},'External cohort lacks an endpoint class')
    return selected


def run(args):
    out=Path(args.output);c.require(not out.exists() or not list(out.iterdir()),'External output directory must be empty')
    protocol=json.loads(Path(args.protocol).read_text());ph=c.sha(args.protocol)
    models=np.load(args.models,allow_pickle=False)
    c.require(str(models['protocol_sha256'])==ph,'Deployment models/protocol mismatch')
    registry=pd.read_csv(args.registry);c.require(models['model_ids'].tolist()==registry.model_id.tolist(),'Deployment registry mismatch')
    for j,row in registry.iterrows():c.require(c.array_sha(models['coefficients'][j])==row.coefficient_sha256,'Deployment coefficient changed')
    review=json.loads(Path(args.rights_review).read_text())
    for key in ['noncommercial_research_use_reviewed','processing_environment_reviewed']:
        c.require(review.get(key) is True,'MRA data-use review missing: '+key)
    c.require(bool(review.get('agreement_version_or_date')) and bool(review.get('reviewed_by')),'Identify accepted dataset agreement and reviewer')
    X=np.load(args.features,allow_pickle=False);m=pd.read_csv(args.manifest,keep_default_na=False)
    spec=json.loads(Path(args.feature_spec).read_text());label_audit=json.loads(Path(args.label_audit).read_text())
    source_hashes=[]
    for ds in ['PAD','DDI']:
        source_hashes.extend(pd.read_csv(Path(args.base)/f'inputs/features/{ds}_feature_index.csv').image_sha256.tolist())
    eligible=validate_features_and_manifest(X,m,spec,protocol,label_audit,source_hashes)
    included=m.loc[eligible].copy();scores=X[eligible].astype(np.float64)@models['coefficients'].T+models['intercepts']
    for j,model_id in enumerate(models['model_ids']):
        included['logit__'+model_id]=scores[:,j]
    lesion_rows=[]
    for (patient,lesion),views in included.groupby(['patient_key','lesion_key'],sort=True):
        lesion_rows.append({'patient_key':patient,'lesion_key':lesion,'target_label':int(views.target_label.iloc[0]),
                            'skin_tone_group':views.skin_tone_group.iloc[0],'n_clinical_views':len(views),
                            **{'logit__'+str(model_id):float(views['logit__'+str(model_id)].mean()) for model_id in models['model_ids']}})
    lesions=pd.DataFrame(lesion_rows);patients,inverse=np.unique(lesions.patient_key.to_numpy(dtype=str),return_inverse=True)
    B=protocol['uncertainty']['replicates'];rng=np.random.default_rng(np.random.SeedSequence([20261006,900]))
    patient_counts=rng.multinomial(len(patients),np.full(len(patients),1/len(patients)),size=B)
    all_weights=patient_counts[:,inverse];metrics=[];contrasts=[];boot={}
    cohorts=['ALL_MRA','I-II','III-IV','V-VI']
    with threadpool_limits(limits=1):
        for cohort in cohorts:
            mask=np.ones(len(lesions),dtype=bool) if cohort=='ALL_MRA' else lesions.skin_tone_group.to_numpy()==cohort
            y=lesions.target_label.to_numpy(dtype=int)[mask];ww=all_weights[:,mask]
            for model_id in models['model_ids']:
                row={'cohort':cohort,'model_id':str(model_id),'lesions':len(y),'positive':int(y.sum()),'negative':int(len(y)-y.sum()),
                     'patients':int(lesions.loc[mask,'patient_key'].nunique()),'bootstrap_requested':B,
                     'unit':'lesion score averaged over clinical views; shared patient-cluster bootstrap'}
                if set(y)!={0,1}:
                    row.update(status='NOT_ESTIMABLE_NO_TWO_CLASSES',bootstrap_valid=0);metrics.append(row);continue
                valid=(ww[:,y==1].sum(axis=1)>0)&(ww[:,y==0].sum(axis=1)>0)
                s=lesions['logit__'+str(model_id)].to_numpy()[mask]
                vv=np.full((B,3),np.nan);vv[valid]=samples(y,s,ww[valid])
                row.update(status='ESTIMATED' if valid.sum()>=250 else 'INSUFFICIENT_VALID_BOOTSTRAPS',AUROC=float(roc_auc_score(y,s)),
                           bootstrap_valid=int(valid.sum()))
                if valid.sum()>=250:
                    limits=np.quantile(vv[valid,0],[.025,.975]);row.update(AUROC_CI_low=float(limits[0]),AUROC_CI_high=float(limits[1]))
                boot[cohort,str(model_id)]=vv;metrics.append(row)
            ids=models['model_ids'].tolist();base_id=ids[0];target_id=ids[1]
            pairs=[(i,base_id) for i in ids[1:]]+[(target_id,i) for i in ids[2:]]
            for left,right in pairs:
                if (cohort,left) not in boot:continue
                delta=boot[cohort,left][:,0]-boot[cohort,right][:,0];valid=np.isfinite(delta)
                lm=next(r for r in metrics if r['cohort']==cohort and r['model_id']==left)
                rm=next(r for r in metrics if r['cohort']==cohort and r['model_id']==right)
                rr={'cohort':cohort,'left_model':left,'right_model':right,'difference_unrounded':lm['AUROC']-rm['AUROC'],
                    'bootstrap_valid':int(valid.sum()),'CI_type':'paired patient-cluster percentile95%; unadjusted exploratory'}
                if valid.sum()>=250:
                    limits=np.quantile(delta[valid],[.025,.975]);rr.update(CI_low=float(limits[0]),CI_high=float(limits[1]))
                contrasts.append(rr)
    for folder in ['predictions','results','logs','uncertainty']:(out/folder).mkdir(parents=True,exist_ok=True)
    included.to_csv(out/'predictions/MRA_eligible_image_predictions.csv.gz',index=False,float_format='%.17g')
    lesions.to_csv(out/'predictions/MRA_lesion_predictions.csv',index=False,float_format='%.17g')
    pd.DataFrame(metrics).to_csv(out/'results/Table_E4_external_validation.csv',index=False,float_format='%.17g')
    pd.DataFrame(contrasts).to_csv(out/'results/Table_E4_paired_contrasts.csv',index=False,float_format='%.17g')
    np.savez_compressed(out/'uncertainty/patient_cluster_counts.npz',patient_counts=patient_counts,patients=patients,
                        lesion_patient_index=inverse,protocol_sha256=np.array(ph))
    c.write_json(out/'logs/external_validation_audit.json',{'completed_utc':c.now(),'protocol_sha256':ph,
                 'input_sha256':{k:c.sha(getattr(args,k)) for k in ['features','manifest','feature_spec','label_audit','rights_review','models','registry']},
                 'patients':len(patients),'lesions':len(lesions),'included_images':int(eligible.sum()),'excluded_images':int((~eligible).sum()),
                 'training_image_byte_hash_overlap':0,'cross_dataset_patient_disjointness':'not established from released DDI patient identifiers',
                 'encoder_used_PAD_pretraining':True,'model_refitting_or_selection_on_external_data':False,
                 'public_upload_performed':False,'software':c.runtime()})
    print('EXTERNAL VALIDATION COMPLETE:',len(patients),'patients,',len(lesions),'lesions; actual MRA inputs only')


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    for name in ['base','protocol','models','registry','features','manifest','feature-spec','label-audit','rights-review','output']:
        ap.add_argument('--'+name,required=True)
    run(ap.parse_args())
