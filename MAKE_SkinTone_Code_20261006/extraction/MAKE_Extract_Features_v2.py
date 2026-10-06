"""Traceable, resumable frozen MedSigLIP features for the NEW MAKE reanalysis.

No classifier fitting, predictions, AUROC, CI, or historical-result verification.
Run in Colab after MAKE_Load_MedSigLIP_v1.py, passing its three exported objects.
Source images stay in the user's Drive/runtime; this script does not publish them.
"""
import csv
import hashlib
import io
import json
import os
import platform
import shutil
import stat
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

import numpy as np
from PIL import Image

REVISION = '9cea28a1a1195f665105faa6e8544c112fd960a4'
PREP_CODE_SHA = '568b77717e9052b75bf096d30a2fa7395c0e6f70727e7f027c83a4c5d66b53c7'
ARCHIVES = {
    'imgs_part_1.zip': (1245184680, '0ab44f60938bf57445e12f518a8878954cc734e6b0aec6d01194e2d26b4b2dca'),
    'imgs_part_2.zip': (1126646990, 'e2d9a3cbd58e823f5ae33163c48643e7d1b54ae3f9e145f01f8e9f16a363a60b'),
    'imgs_part_3.zip': (1220565093, 'ecc4ef10143a43e1d01cb736773148607a78b530417eb76f2c38ad24bf5d0d2c'),
    'DDI - Diverse Dermatology Images_files-2.zip': (238670195, '40bd0bbd6cb72fd5fa85dfa382720c6ac834366c0758763fab5871fc591f9522'),
}
BATCH_SIZE = 4
CHUNK_SIZE = 128
DIMENSION = 1152


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + '\n').encode('utf-8')


def atomic_write(path, data, immutable=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if immutable and path.exists():
        require(path.read_bytes() == data, 'Existing result differs; not overwritten: ' + str(path))
        return
    temp = path.with_name(path.name + '.tmp')
    try:
        with temp.open('wb') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def read_csv(path):
    data = Path(path).read_bytes()
    reader = csv.DictReader(io.StringIO(data.decode('utf-8-sig'), newline=''))
    fields, rows = reader.fieldnames, list(reader)
    require(fields and len(fields) == len(set(fields)), 'Invalid CSV header')
    require(all(None not in r and all(v is not None for v in r.values()) for r in rows), 'Malformed CSV row')
    return rows, fields


def read_inputs(root):
    spec_path = root / 'splits/Analysis_Inputs_v1/split_spec_v1.json'
    spec = json.loads(spec_path.read_text(encoding='utf-8'))
    require(spec.get('analysis_type') == 'new_reanalysis' and spec.get('n_folds') == 5, 'Unexpected split specification')
    require(file_sha(root / 'code/MAKE_Prepare_Data_v1.py') == PREP_CODE_SHA, 'Preparation code changed')
    output = {}
    for name, dataset, count, positives in [('PAD', 'PAD-UFES-20', 2268, 1070), ('DDI', 'DDI', 656, 171)]:
        rel = f'data/Analysis_Inputs_v1/{name}_labeled_manifest.csv'
        split_rel = f'splits/Analysis_Inputs_v1/{name}_cv_folds.csv'
        for p in [rel, split_rel]:
            require(file_sha(root / p) == spec['output_sha256'][p], 'Input differs from the prepared split: ' + p)
        rows, fields = read_csv(root / rel)
        split_rows, split_fields = read_csv(root / split_rel)
        require(len(rows) == count and sum(int(r['target_label']) for r in rows) == positives, 'Unexpected image/label counts')
        require(len({r['analysis_image_id'] for r in rows}) == count, 'Duplicate analysis image ID')
        require(len({r['image_sha256'] for r in rows}) == count, 'Exact duplicate in the analysis set')
        require(len(split_rows) == count, 'Split row count differs')
        split_by_id = {r['analysis_image_id']: r for r in split_rows}
        require(len(split_by_id) == count, 'Duplicate split identifier')
        for row in rows:
            require(row['analysis_dataset_id'] == dataset, 'Dataset identifier changed')
            require(row['target_label'] in ('0', '1') and row['cv_fold'] in ('0', '1', '2', '3', '4'), 'Invalid label or fold')
            require(all(row[c] == split_by_id[row['analysis_image_id']][c] for c in split_fields), 'Manifest and split disagree')
            safe_relative(row['image_relative_path'])
            image_id = row['img_id'] if name == 'PAD' else row['DDI_file']
            require(row['analysis_image_id'] == dataset + ':' + image_id, 'Image identifier changed')
            if name == 'PAD':
                require(row['primary_eligibility'] == 'include', 'Excluded PAD image entered the analysis')
                require(row['split_unit_id'] == 'PAD-UFES-20:' + row['patient_id'], 'PAD patient key changed')
                require(int(row['target_label']) == {'BCC': 1, 'SCC': 1, 'MEL': 1, 'ACK': 0, 'SEK': 0, 'NEV': 0}[row['diagnostic']], 'PAD label changed')
            else:
                require(row['malignant'] in ('True', 'False') and int(row['target_label']) == int(row['malignant'] == 'True'), 'Released DDI label changed')
        if name == 'PAD':
            patient_folds = {}
            for row in rows:
                key = row['split_unit_id']
                require(key not in patient_folds or patient_folds[key] == row['cv_fold'], 'PAD patient crosses folds')
                patient_folds[key] = row['cv_fold']
        output[name] = (rows, fields)
    require(not {r['image_sha256'] for r in output['PAD'][0]}.intersection(r['image_sha256'] for r in output['DDI'][0]), 'Exact cross-dataset overlap')
    return output, spec_path


def safe_relative(name):
    p = PurePosixPath(name)
    require(not p.is_absolute() and '..' not in p.parts and '\\' not in name and ':' not in name, 'Unsafe archive/image path')
    return p


def stage_archive(source, stage_dir, expected_size, expected_sha):
    require(source.is_file() and source.stat().st_size == expected_size, 'Source ZIP missing or size changed: ' + source.name)
    staged = stage_dir / source.name
    if staged.is_file() and staged.stat().st_size == expected_size and file_sha(staged) == expected_sha:
        return staged
    require(shutil.disk_usage(stage_dir).free > expected_size + 256 * 1024**2, 'Colab disk has insufficient space for the source ZIP')
    temp = staged.with_name(staged.name + '.tmp')
    digest = hashlib.sha256()
    copied = 0
    with source.open('rb') as src, temp.open('wb') as dst:
        for block in iter(lambda: src.read(8 * 1024 * 1024), b''):
            dst.write(block)
            digest.update(block)
            copied += len(block)
    require(copied == expected_size and digest.hexdigest() == expected_sha, 'Source ZIP hash changed: ' + source.name)
    os.replace(temp, staged)
    return staged


def restore_images(root, runtime, inputs):
    stage_dir = runtime / 'feature_stage_archives_v1'
    stage_dir.mkdir(parents=True, exist_ok=True)
    locations = {'PAD': runtime / 'PAD_UFES_20', 'DDI': runtime / 'DDI'}
    records = []
    for archive_name, (size, digest) in ARCHIVES.items():
        name = 'PAD' if archive_name.startswith('imgs_part_') else 'DDI'
        selected = inputs[name][0]
        if name == 'PAD':
            folder = archive_name[:-4]
            selected = [r for r in selected if PurePosixPath(r['image_relative_path']).parts[0] == folder]
            source = root / 'data/zr7vgbcyr2-1/images' / archive_name
        else:
            source = root / 'data' / archive_name
        require(selected, 'No images assigned to an expected source archive')
        print('ตรวจ ZIP และเตรียมภาพ:', archive_name, flush=True)
        staged = stage_archive(source, stage_dir, size, digest)
        with ZipFile(staged) as archive:
            infos = archive.infolist()
            require(len({i.filename for i in infos}) == len(infos), 'Duplicate ZIP entry')
            for info in infos:
                safe_relative(info.filename)
                require(not stat.S_ISLNK(info.external_attr >> 16), 'ZIP symlink rejected')
            info_by_name = {i.filename: i for i in infos}
            by_basename = {}
            for info in infos:
                if not info.is_dir():
                    by_basename.setdefault(PurePosixPath(info.filename).name, []).append(info.filename)
            members = {}
            for row in selected:
                relative = row['image_relative_path']
                if relative in info_by_name:
                    member = relative
                else:
                    candidates = by_basename.get(PurePosixPath(relative).name, [])
                    require(len(candidates) == 1, 'Missing or ambiguous ZIP image: ' + relative)
                    member = candidates[0]
                members[relative] = member
            needed = sum(info_by_name[members[r['image_relative_path']]].file_size for r in selected)
            require(shutil.disk_usage(runtime).free > needed + 256 * 1024**2, 'Colab disk has insufficient space for images')
            for n, row in enumerate(selected, 1):
                relative = row['image_relative_path']
                destination = locations[name] / relative
                if not destination.is_file() or file_sha(destination) != row['image_sha256']:
                    data = archive.read(members[relative])
                    require(sha(data) == row['image_sha256'], 'Source image hash differs: ' + row['analysis_image_id'])
                    atomic_write(destination, data)
                if n % 250 == 0 or n == len(selected):
                    print(f'  {n}/{len(selected)} ภาพ', flush=True)
        records.append({'archive': archive_name, 'bytes': size, 'sha256': digest,
                        'selected_images': len(selected),
                        'member_mapping': [{'analysis_image_id': r['analysis_image_id'],
                                            'image_relative_path': r['image_relative_path'],
                                            'archive_member': members[r['image_relative_path']]}
                                           for r in selected]})
        # Only remove this script's verified runtime copy. The Drive original is retained.
        staged.unlink()
    return locations, records


def open_verified_image(row, image_root):
    data = (image_root / row['image_relative_path']).read_bytes()
    require(sha(data) == row['image_sha256'], 'Image hash changed before encoding: ' + row['analysis_image_id'])
    with Image.open(io.BytesIO(data)) as image:
        image.load()
        require(image.size == (int(row['image_width']), int(row['image_height'])) and image.mode == row['image_mode'], 'Image dimensions/mode changed')
        return image.convert('RGB')


def validate_features(features, count):
    require(features.dtype == np.float32 and features.shape == (count, DIMENSION), 'Invalid feature dtype or shape')
    require(np.isfinite(features).all(), 'Non-finite feature value')
    require(np.allclose(np.linalg.norm(features, axis=1), 1.0, rtol=0, atol=1e-4), 'Feature is not L2 normalized')


def save_chunk(path, rows, features, cache_id):
    validate_features(features, len(rows))
    stream = io.BytesIO()
    np.savez(stream, features=features,
             image_ids=np.asarray([r['analysis_image_id'] for r in rows]),
             image_sha256=np.asarray([r['image_sha256'] for r in rows]),
             cache_id=np.asarray(cache_id), feature_sha256=np.asarray(sha(features.tobytes(order='C'))))
    atomic_write(path, stream.getvalue(), immutable=True)


def load_chunk(path, rows, cache_id):
    with np.load(path, allow_pickle=False) as data:
        require(str(data['cache_id'].item()) == cache_id, 'Feature cache specification changed')
        require(data['image_ids'].tolist() == [r['analysis_image_id'] for r in rows], 'Feature cache row order changed')
        require(data['image_sha256'].tolist() == [r['image_sha256'] for r in rows], 'Feature cache image hashes changed')
        features = data['features'].copy()
        require(str(data['feature_sha256'].item()) == sha(features.tobytes(order='C')), 'Feature cache contents changed')
    validate_features(features, len(rows))
    return features


def process_dataset(name, rows, fields, image_root, output_dir, cache_id, encode):
    chunks = output_dir / 'chunks'
    chunks.mkdir(parents=True, exist_ok=True)
    matrices, chunk_hashes = [], {}
    for start in range(0, len(rows), CHUNK_SIZE):
        selected = rows[start:start + CHUNK_SIZE]
        path = chunks / f'{name}_{start:05d}_{start + len(selected):05d}.npz'
        if path.exists():
            features = load_chunk(path, selected, cache_id)
            status = 'ใช้ช่วงที่บันทึกไว้'
        else:
            batches = []
            for offset in range(0, len(selected), BATCH_SIZE):
                batch = selected[offset:offset + BATCH_SIZE]
                images = [open_verified_image(r, image_root) for r in batch]
                try:
                    encoded = encode(images)
                    validate_features(encoded, len(batch))
                    batches.append(encoded)
                finally:
                    for image in images:
                        image.close()
            features = np.concatenate(batches, axis=0)
            save_chunk(path, selected, features, cache_id)
            # Verify persisted bytes and their image-to-row binding before continuing.
            features = load_chunk(path, selected, cache_id)
            status = 'ประมวลผลและบันทึกแล้ว'
        matrices.append(features)
        chunk_hashes[str(path.relative_to(output_dir))] = file_sha(path)
        print(f'{name}: {start + len(selected)}/{len(rows)} — {status}', flush=True)
    matrix = np.concatenate(matrices, axis=0)
    validate_features(matrix, len(rows))
    matrix_stream = io.BytesIO()
    np.save(matrix_stream, matrix, allow_pickle=False)
    matrix_path = output_dir / f'{name}_features_l2.npy'
    atomic_write(matrix_path, matrix_stream.getvalue(), immutable=True)
    added = ['feature_row_index', 'feature_row_sha256', 'feature_cache_id']
    require(not set(added).intersection(fields), 'Feature columns already present')
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=fields + added)
    writer.writeheader()
    for i, row in enumerate(rows):
        writer.writerow({**row, 'feature_row_index': i, 'feature_row_sha256': sha(matrix[i].tobytes()), 'feature_cache_id': cache_id})
    index_path = output_dir / f'{name}_feature_index.csv'
    atomic_write(index_path, stream.getvalue().encode('utf-8-sig'), immutable=True)
    return {'image_count': len(rows), 'feature_shape': list(matrix.shape),
            'matrix_path': str(matrix_path), 'matrix_sha256': file_sha(matrix_path),
            'index_path': str(index_path), 'index_sha256': file_sha(index_path),
            'chunk_sha256': chunk_hashes}


def extract(model, processor, model_info, project_root='/content/drive/MyDrive/MAKE_Reanalysis', runtime_root='/content/MAKE_Reanalysis'):
    import torch

    root, runtime = Path(project_root), Path(runtime_root)
    runtime.mkdir(parents=True, exist_ok=True)
    require(torch.cuda.is_available(), 'GPU disconnected: reconnect and run the model-loading cell first')
    require(model_info['repo_id'] == 'google/medsiglip-448' and model_info['revision'] == REVISION, 'Wrong model revision')
    require(not model.training and all(not p.requires_grad for p in model.parameters()), 'Encoder must be frozen and in evaluation mode')
    require(all(p.dtype == torch.float32 and p.device.type == 'cuda' for p in model.parameters()), 'Encoder must use float32 on CUDA')
    processor_config = json.loads(processor.to_json_string())
    require(processor_config == model_info['image_processor_config'] and getattr(processor, 'backend', None) == 'pil', 'Processor changed since loading')
    require(model.config.vision_config.hidden_size == DIMENSION and model.config.vision_config.image_size == 448, 'Unexpected vision configuration')
    require(not torch.backends.cuda.matmul.allow_tf32 and not torch.backends.cudnn.allow_tf32, 'TF32 settings changed')
    print('1/3 ตรวจข้อมูลที่แบ่งชุดไว้และการบันทึกลง Drive', flush=True)
    inputs, split_spec_path = read_inputs(root)
    source_code = Path(__file__).resolve()
    code_sha = file_sha(source_code)
    atomic_write(root / 'code' / source_code.name, source_code.read_bytes(), immutable=True)
    # A small real write checks Drive availability; mounted Drive disk-usage estimates may be misleading.
    probe = root / 'features' / 'MedSigLIP_v1' / '.write_probe.tmp'
    atomic_write(probe, b'0' * (1024 * 1024))
    probe.unlink()
    packages = {p: version(p) for p in ('torch', 'transformers', 'huggingface_hub', 'Pillow', 'safetensors', 'numpy')}
    for p, value in model_info['environment']['packages'].items():
        require(packages[p] == value, 'Environment changed since loading: ' + p)
    feature_spec = {
        'stage': 'frozen_image_feature_extraction', 'analysis_type': 'new_reanalysis',
        'historical_results_verified': False, 'classifier_fitted': False, 'predictions_produced': False,
        'repo_id': model_info['repo_id'], 'revision': REVISION,
        'model_source_files': model_info['source_files'], 'model_load_code_sha256': model_info['code_sha256'],
        'image_processor_class': type(processor).__name__, 'image_processor_backend': 'pil',
        'image_processor_config': processor_config,
        'image_conversion': 'PIL load then convert RGB; alpha discarded; no EXIF transpose or augmentation',
        'feature_method': 'get_image_features(return_dict=True).pooler_output then float32 L2 normalization',
        'dimension': DIMENSION, 'dtype': 'float32', 'batch_size': BATCH_SIZE, 'chunk_size': CHUNK_SIZE,
        'row_order': 'unchanged labeled-manifest order; zero-based feature_row_index',
        'split_spec_sha256': file_sha(split_spec_path),
        'manifest_sha256': {name: file_sha(root / f'data/Analysis_Inputs_v1/{name}_labeled_manifest.csv') for name in inputs},
        'source_archives': {name: {'bytes': v[0], 'sha256': v[1]} for name, v in ARCHIVES.items()},
        'code_sha256': code_sha, 'python': platform.python_version(), 'packages': packages,
        'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0), 'attention_implementation': 'sdpa',
        'cuda_matmul_allow_tf32': False, 'cudnn_allow_tf32': False,
        'deterministic_algorithms_enabled': torch.are_deterministic_algorithms_enabled(),
        'known_limitations': ['PAD used during encoder pretraining according to its model card',
                              'DDI folds are image-level; released metadata do not identify patients',
                              'Near-duplicate checks have not been performed',
                              'Full curated PAD pool does not reconstruct the historical subset']
    }
    cache_id = sha(json_bytes(feature_spec))
    output_dir = root / 'features/MedSigLIP_v1' / cache_id
    atomic_write(output_dir / 'feature_spec.json', json_bytes(feature_spec), immutable=True)
    print('2/3 เตรียมภาพจาก ZIP เดิม ไม่ต้องดาวน์โหลดข้อมูลใหม่', flush=True)
    locations, archive_audit = restore_images(root, runtime, inputs)

    def encode(images):
        batch = processor(images=images, return_tensors='pt').to('cuda')
        require(tuple(batch['pixel_values'].shape) == (len(images), 3, 448, 448), 'Unexpected processed image shape')
        with torch.inference_mode():
            pooled = model.get_image_features(**batch, return_dict=True).pooler_output.float()
            require(tuple(pooled.shape) == (len(images), DIMENSION), 'Unexpected pooled feature shape')
            require(torch.isfinite(pooled).all().item() and (pooled.norm(dim=1) > 0).all().item(), 'Invalid pooled features')
            features = torch.nn.functional.normalize(pooled, p=2, dim=1).cpu().numpy()
        return np.ascontiguousarray(features, dtype=np.float32)

    print('3/3 ประมวลผลภาพและบันทึกเป็นช่วง ๆ', flush=True)
    results = {name: process_dataset(name, rows, fields, locations[name], output_dir, cache_id, encode)
               for name, (rows, fields) in inputs.items()}
    audit = {'completed_utc': datetime.now(timezone.utc).isoformat(), 'feature_cache_id': cache_id,
             'feature_spec_path': str(output_dir / 'feature_spec.json'),
             'feature_spec_sha256': file_sha(output_dir / 'feature_spec.json'),
             'model_load_evidence': model_info, 'archives_verified_this_run': archive_audit,
             'results': results, 'historical_results_verified': False,
             'classifier_fitted': False, 'predictions_produced': False}
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    audit_path = root / 'logs' / ('feature_extraction_audit_' + stamp + '.json')
    atomic_write(audit_path, json_bytes(audit), immutable=True)
    print('FEATURE EXTRACTION: PASS')
    print('PAD: 2268 x 1152')
    print('DDI: 656 x 1152')
    print('Saved features and per-image index:', output_dir)
    print('Audit:', audit_path)
    print('NEXT: classifier experiments and numerical uncertainty estimates')
    return audit
