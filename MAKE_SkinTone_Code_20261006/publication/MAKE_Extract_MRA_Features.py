"""Extract real MRA clinical-image features with the original pinned pipeline.

Local files only. No automatic download, diagnosis mapping, fitting or upload.
Requires the existing float32 CUDA encoder and a reviewed image manifest.
This adapter has not been executed on MRA images in this publication revision.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.metadata as metadata
import json
import platform
import numpy as np
import pandas as pd

REQUIRED = ['analysis_image_id','image_relative_path','patient_key','lesion_key',
            'diagnosis_code','target_label','skin_tone_group','modality',
            'biopsy_confirmed','eligible']

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def require(ok,message):
    if not ok:raise ValueError(message)

def run(args):
    for name in ['manifest','protocol','rights_review']:
        require(Path(getattr(args,name)).is_file(),'Required real input missing: '+name)
    require(Path(args.image_dir).is_dir(),'Actual MRA image directory missing')
    require(Path(args.model_dir).is_dir(),'Existing pinned encoder directory missing')
    out=Path(args.output)
    require(not out.exists() or not any(out.iterdir()),'Output directory must be empty')
    protocol=json.loads(Path(args.protocol).read_text())
    reference=protocol['feature_spec']
    rights=json.loads(Path(args.rights_review).read_text())
    for field in ['noncommercial_research_use_reviewed','processing_environment_reviewed']:
        require(rights.get(field) is True,'Data-use review required: '+field)
    require(rights.get('reviewed_by') and rights.get('agreement_version_or_date'),'Identify source agreement and reviewer')
    manifest=pd.read_csv(args.manifest,keep_default_na=False)
    require(set(REQUIRED)<=set(manifest),'Missing manifest columns: '+str(sorted(set(REQUIRED)-set(manifest))))
    require(len(manifest)>0 and manifest.analysis_image_id.is_unique,'Empty or duplicate image identifiers')
    images_root=Path(args.image_dir).resolve()
    image_paths=[]
    for name in manifest.image_relative_path:
        relative=Path(str(name))
        require(not relative.is_absolute() and '..' not in relative.parts,'Unsafe image path')
        image=(images_root/relative).resolve()
        require(image.is_relative_to(images_root) and image.is_file(),'Actual image missing or outside image root: '+str(name))
        image_paths.append(image)
    model_dir=Path(args.model_dir)
    for filename in ['model.safetensors','config.json','preprocessor_config.json']:
        require((model_dir/filename).is_file(),'Pinned model file missing: '+filename)
        require(sha(model_dir/filename)==reference['model_source_files'][filename]['sha256'],
                'Pinned model file hash differs: '+filename)
    # Imports occur after input gates; checking missing inputs needs no GPU packages.
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor,AutoModel
    require(torch.cuda.is_available(),'CUDA GPU required by the original extraction pipeline')
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    processor=AutoImageProcessor.from_pretrained(str(model_dir),local_files_only=True,backend='pil')
    processor_config=json.loads(processor.to_json_string())
    require(type(processor).__name__==reference['image_processor_class'],'Image processor class differs')
    require(getattr(processor,'backend',None)==reference['image_processor_backend'],'Image processor backend differs')
    require(processor_config==reference['image_processor_config'],'Image processor configuration differs')
    model=AutoModel.from_pretrained(str(model_dir),local_files_only=True,use_safetensors=True,
                                   dtype=torch.float32,attn_implementation='sdpa').to('cuda').eval()
    model.requires_grad_(False)
    require(model.config.vision_config.hidden_size==1152 and model.config.vision_config.image_size==448,
            'Encoder dimensions differ')
    require(all(p.dtype==torch.float32 and p.device.type=='cuda' for p in model.parameters()),'Encoder must use float32 CUDA')
    require(reference['feature_method']=='get_image_features(return_dict=True).pooler_output then float32 L2 normalization',
            'Feature method differs from original extractor')
    features=[]
    for start in range(0,len(image_paths),4):
        batch=[]
        for path in image_paths[start:start+4]:
            with Image.open(path) as image:
                image.load()
                batch.append(image.convert('RGB'))
        inputs=processor(images=batch,return_tensors='pt').to('cuda')
        require(tuple(inputs['pixel_values'].shape)==(len(batch),3,448,448),'Processed image shape differs')
        with torch.inference_mode():
            pooled=model.get_image_features(**inputs,return_dict=True).pooler_output.float()
            require(tuple(pooled.shape)==(len(batch),1152) and torch.isfinite(pooled).all().item(), 'Invalid encoder output')
            require((pooled.norm(dim=1)>0).all().item(),'Zero feature vector')
            normalized=torch.nn.functional.normalize(pooled,p=2,dim=1).cpu().numpy()
        features.append(np.ascontiguousarray(normalized,dtype=np.float32))
    X=np.ascontiguousarray(np.concatenate(features),dtype=np.float32)
    require(np.max(abs(np.linalg.norm(X,axis=1)-1))<2e-6,'L2 normalization check failed')
    hashes=[sha(p) for p in image_paths]
    require(len(set(hashes))==len(hashes),'Duplicate image bytes in MRA input')
    if 'image_sha256' in manifest:
        require(manifest.image_sha256.tolist()==hashes,'Source image bytes differ from supplied hashes')
    manifest['image_sha256']=hashes
    manifest['feature_row_index']=np.arange(len(X))
    manifest['feature_row_sha256']=[hashlib.sha256(row.tobytes()).hexdigest() for row in X]
    spec={key:reference[key] for key in ['repo_id','revision','dimension','dtype','feature_method',
                                        'image_conversion','image_processor_class','image_processor_backend',
                                        'image_processor_config','model_source_files']}
    spec.update(batch_size=4,attention_implementation='sdpa',cuda_matmul_allow_tf32=False,cudnn_allow_tf32=False,
                extraction_code_sha256=sha(__file__),input_manifest_sha256=sha(args.manifest),
                protocol_sha256=sha(args.protocol),python=platform.python_version(),
                packages={p:metadata.version(p) for p in ['torch','transformers','Pillow','numpy']},
                cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0),
                classifier_fitted=False,predictions_produced=False,public_upload_performed=False)
    out.mkdir(parents=True,exist_ok=True)
    np.save(out/'MRA_features_l2.npy',X,allow_pickle=False)
    manifest.to_csv(out/'MRA_feature_manifest.csv',index=False)
    (out/'MRA_feature_spec.json').write_text(json.dumps(spec,indent=2)+'\n')
    (out/'SHA256SUMS').write_text(''.join(sha(out/p)+'  '+p+'\n' for p in
                                    ['MRA_features_l2.npy','MRA_feature_manifest.csv','MRA_feature_spec.json']))
    print('REAL MRA FEATURES COMPLETE:',len(X),'images; no clinical results computed')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['image-dir','manifest','protocol','rights-review','model-dir','output']:
        p.add_argument('--'+name,required=True)
    run(p.parse_args())
