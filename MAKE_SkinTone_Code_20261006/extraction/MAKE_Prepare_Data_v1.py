#!/usr/bin/env python3
"""Prepare traceable labels and CV assignments for the NEW MAKE reanalysis.

Run locally: python MAKE_Prepare_Data_v1.py --project-root /path/to/MAKE_Reanalysis
Run in Colab: runpy.run_path(path_to_this_file)['prepare'](project_root)
This preparation stage does not fit a model, score images, or verify old metrics.
"""
import argparse
import csv
import hashlib
import io
import json
import platform
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

import numpy as np
import sklearn
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

VERSION = 'Analysis_Inputs_v1'
SEED = 20261004
N_FOLDS = 5
PAD_PRIMARY_SHA = '86c1444b7b0367b478629fffc5ce6c97371526f40faa6f7150aa9086b6a537cb'
PAD_ALL_SHA = '0ad273d8a5b12b50a7988430ba6c440a10b5f33cbdb0a0412faf4c53a4bf04bc'
DDI_ZIP_SHA = '40bd0bbd6cb72fd5fa85dfa382720c6ac834366c0758763fab5871fc591f9522'
DDI_METADATA_SHA = '534bdb3d95943eefb9bb5691d5fc00e5c0f132326bea24b28f750e9f6087a838'
PAD_LABELS = {'BCC': 1, 'SCC': 1, 'MEL': 1, 'ACK': 0, 'SEK': 0, 'NEV': 0}
PAD_TONES = {'1.0': 'I-II', '2.0': 'I-II', '3.0': 'III-IV',
             '4.0': 'III-IV', '5.0': 'V-VI', '6.0': 'V-VI', '': 'UNKNOWN'}
DDI_TONES = {'12': 'I-II', '34': 'III-IV', '56': 'V-VI'}
DDI_LABELS = {'True': 1, 'False': 0}
ADDED_COLUMNS = ['analysis_dataset_id', 'analysis_image_id', 'target_label',
                 'target_definition_id', 'skin_tone_group', 'split_unit_id',
                 'split_unit_type', 'cv_fold']


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def file_sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_csv(data):
    reader = csv.DictReader(io.StringIO(data.decode('utf-8-sig'), newline=''))
    rows = list(reader)
    fields = reader.fieldnames or []
    require(len(fields) == len(set(fields)), 'Duplicate CSV columns')
    require(all(None not in r and all(v is not None for v in r.values()) for r in rows),
            'CSV row lengths do not match the header')
    return rows, fields


def csv_bytes(rows, fields):
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode('utf-8-sig')


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + '\n').encode('utf-8')


def check_images(rows, id_column):
    require(len({r[id_column] for r in rows}) == len(rows), 'Repeated image identifier')
    require(all(re.fullmatch(r'[0-9a-f]{64}', r['image_sha256']) for r in rows),
            'Invalid image SHA-256')
    for row in rows:
        path = PurePosixPath(row['image_relative_path'])
        require(not path.is_absolute() and '..' not in path.parts and path.name == row[id_column],
                'Invalid image path or image identifier')


def enrich(rows, fields, dataset):
    require(not set(fields).intersection(ADDED_COLUMNS), 'Analysis columns already exist')
    enriched = []
    for source in rows:
        row = dict(source)
        if dataset == 'PAD-UFES-20':
            require(source['diagnostic'] in PAD_LABELS, 'Unknown PAD diagnosis')
            require(source['fitspatrick'] in PAD_TONES, 'Unexpected PAD skin-tone value')
            require(source['patient_id'] and source['split_patient_key'] == 'PAD-UFES-20:' + source['patient_id'],
                    'Invalid PAD patient grouping key')
            require(source['primary_eligibility'] == 'include', 'Quarantined PAD row entered the primary set')
            label = PAD_LABELS[source['diagnostic']]
            tone = PAD_TONES[source['fitspatrick']]
            definition = 'PAD_BCC_SCC_MEL_positive_v1'
            image_id = source['img_id']
            unit_id = source['split_patient_key']
            unit_type = 'source_patient'
        else:
            require(source['malignant'] in DDI_LABELS, 'Unexpected released DDI malignant value')
            require(source['skin_tone'] in DDI_TONES, 'Unexpected released DDI skin-tone value')
            label = DDI_LABELS[source['malignant']]
            tone = DDI_TONES[source['skin_tone']]
            definition = 'DDI_released_malignant_positive_v1'
            image_id = source['DDI_file']
            unit_id = 'DDI:image:' + image_id
            unit_type = 'source_image'
        row.update(analysis_dataset_id=dataset, analysis_image_id=dataset + ':' + image_id,
                   target_label=label, target_definition_id=definition, skin_tone_group=tone,
                   split_unit_id=unit_id, split_unit_type=unit_type, cv_fold=-1)
        enriched.append(row)
    return enriched


def assign_folds(rows, patient_grouped):
    # Sort only the splitter input. The output manifest retains the source row order.
    ordered = sorted(rows, key=lambda r: r['analysis_image_id'])
    y = np.asarray([r['target_label'] for r in ordered], dtype=int)
    x = np.zeros((len(ordered), 1), dtype=np.uint8)
    if patient_grouped:
        groups = np.asarray([r['split_unit_id'] for r in ordered])
        splitter = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        iterator = splitter.split(x, y, groups)
    else:
        strata = np.asarray([r['skin_tone_group'] + '|' + str(r['target_label']) for r in ordered])
        require(min(Counter(strata).values()) >= N_FOLDS, 'Too few DDI images for the fixed stratification')
        splitter = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        iterator = splitter.split(x, strata)
    for fold, (train_indices, test_indices) in enumerate(iterator):
        train_units = {ordered[int(i)]['split_unit_id'] for i in train_indices}
        test_units = {ordered[int(i)]['split_unit_id'] for i in test_indices}
        require(not train_units.intersection(test_units), 'A split unit crosses train and validation')
        require(set(y[train_indices]) == {0, 1} and set(y[test_indices]) == {0, 1},
                'A fold does not contain both outcome classes')
        for index in test_indices:
            row = ordered[int(index)]
            require(row['cv_fold'] == -1, 'A row was assigned twice')
            row['cv_fold'] = fold
    require(all(0 <= r['cv_fold'] < N_FOLDS for r in rows), 'A row has no fold')
    unit_folds = defaultdict(set)
    for row in rows:
        unit_folds[row['split_unit_id']].add(row['cv_fold'])
    require(all(len(folds) == 1 for folds in unit_folds.values()), 'A split unit appears in multiple folds')
    return splitter.__class__.__name__


def count_rows(rows, by_fold=False):
    output = []
    folds = range(N_FOLDS) if by_fold else [None]
    for fold in folds:
        selected = rows if fold is None else [r for r in rows if r['cv_fold'] == fold]
        for tone in ['ALL', 'I-II', 'III-IV', 'V-VI', 'UNKNOWN']:
            subset = selected if tone == 'ALL' else [r for r in selected if r['skin_tone_group'] == tone]
            if not subset:
                continue
            output.append({'dataset': rows[0]['analysis_dataset_id'], 'cv_fold': '' if fold is None else fold,
                           'skin_tone_group': tone, 'image_count': len(subset),
                           'positive_count': sum(r['target_label'] for r in subset),
                           'negative_count': sum(1 - r['target_label'] for r in subset),
                           'split_unit_count': len({r['split_unit_id'] for r in subset}),
                           'split_unit_type': subset[0]['split_unit_type']})
    return output


def prepare(project_root):
    root = Path(project_root).expanduser().resolve()
    pad_dir = root / 'data' / 'PAD_Curation_v1'
    pad_path = pad_dir / 'primary_manifest.csv'
    pad_all_path = pad_dir / 'all_images_with_decisions.csv'
    ddi_path = root / 'data' / 'ddi_image_manifest_v1.csv'
    ddi_zip_path = root / 'data' / 'DDI - Diverse Dermatology Images_files-2.zip'
    for path in [pad_path, pad_all_path, ddi_path, ddi_zip_path]:
        require(path.is_file(), 'Required input is missing: ' + str(path))
    pad_data, pad_all_data, ddi_data = pad_path.read_bytes(), pad_all_path.read_bytes(), ddi_path.read_bytes()
    require(sha(pad_data) == PAD_PRIMARY_SHA, 'PAD primary manifest is not the curated snapshot')
    require(sha(pad_all_data) == PAD_ALL_SHA, 'PAD all-image manifest is not the reviewed snapshot')
    require(ddi_zip_path.stat().st_size == 238670195 and file_sha(ddi_zip_path) == DDI_ZIP_SHA,
            'DDI source ZIP is not the audited snapshot')
    pad_source, pad_fields = read_csv(pad_data)
    pad_all, _ = read_csv(pad_all_data)
    ddi_source, ddi_fields = read_csv(ddi_data)
    require(len(pad_source) == 2268 and len(pad_all) == 2298 and len(ddi_source) == 656, 'Unexpected image counts')
    require(len({r['patient_id'] for r in pad_source}) == 1370, 'Unexpected PAD patient count')
    check_images(pad_source, 'img_id')
    check_images(ddi_source, 'DDI_file')
    require(len({r['image_sha256'] for r in pad_source}) == 2268, 'Exact duplicate in PAD primary set')
    require(len({r['image_sha256'] for r in ddi_source}) == 656, 'Exact duplicate in DDI')
    require(not {r['image_sha256'] for r in pad_all}.intersection(r['image_sha256'] for r in ddi_source),
            'Exact cross-source duplicate detected')
    with ZipFile(ddi_zip_path) as archive:
        metadata_bytes = archive.read('ddi_metadata.csv')
        require(sha(metadata_bytes) == DDI_METADATA_SHA, 'DDI metadata snapshot changed')
        metadata, metadata_fields = read_csv(metadata_bytes)
        require(set(metadata_fields).issubset(ddi_fields), 'DDI manifest lost a source metadata column')
        source_by_file = {r['DDI_file']: r for r in metadata}
        require(len(source_by_file) == 656 and set(source_by_file) == {r['DDI_file'] for r in ddi_source},
                'DDI manifest and source metadata identifiers differ')
        for row in ddi_source:
            require(all(row[c] == source_by_file[row['DDI_file']][c] for c in metadata_fields),
                    'A DDI source metadata value changed')
            require(sha(archive.read(row['DDI_file'])) == row['image_sha256'],
                    'DDI manifest image hash differs from the source ZIP: ' + row['DDI_file'])
    pad = enrich(pad_source, pad_fields, 'PAD-UFES-20')
    ddi = enrich(ddi_source, ddi_fields, 'DDI')
    require(Counter(r['target_label'] for r in pad) == {1: 1070, 0: 1198}, 'Unexpected PAD outcome counts')
    require(Counter(r['target_label'] for r in ddi) == {1: 171, 0: 485}, 'Unexpected DDI outcome counts')
    algorithms = {'PAD-UFES-20': assign_folds(pad, True), 'DDI': assign_folds(ddi, False)}
    # Check preservation of every input cell after deriving labels and fold columns.
    for before, after, fields in [(pad_source, pad, pad_fields), (ddi_source, ddi, ddi_fields)]:
        require(all(all(a[c] == b[c] for c in fields) for a, b in zip(before, after)), 'A source value was modified')
    for tone in DDI_TONES.values():
        for fold in range(N_FOLDS):
            require({r['target_label'] for r in ddi if r['skin_tone_group'] == tone and r['cv_fold'] == fold} == {0, 1},
                    'A DDI skin-tone/fold cell has only one outcome class')
    counts = count_rows(pad) + count_rows(ddi)
    fold_counts = count_rows(pad, True) + count_rows(ddi, True)
    split_fields = ADDED_COLUMNS + ['image_sha256']
    payloads = {
        f'data/{VERSION}/PAD_labeled_manifest.csv': csv_bytes(pad, pad_fields + ADDED_COLUMNS),
        f'data/{VERSION}/DDI_labeled_manifest.csv': csv_bytes(ddi, ddi_fields + ADDED_COLUMNS),
        f'splits/{VERSION}/PAD_cv_folds.csv': csv_bytes([{c: r[c] for c in split_fields} for r in pad], split_fields),
        f'splits/{VERSION}/DDI_cv_folds.csv': csv_bytes([{c: r[c] for c in split_fields} for r in ddi], split_fields),
        f'results/data_preparation_v1/dataset_counts.csv': csv_bytes(counts, list(counts[0])),
        f'results/data_preparation_v1/fold_counts.csv': csv_bytes(fold_counts, list(fold_counts[0])),
    }
    script_bytes = Path(__file__).read_bytes()
    payloads['code/MAKE_Prepare_Data_v1.py'] = script_bytes
    input_hashes = {str(p.relative_to(root)): h for p, h in [
        (pad_path, sha(pad_data)), (pad_all_path, sha(pad_all_data)),
        (ddi_path, sha(ddi_data)), (ddi_zip_path, DDI_ZIP_SHA)]}
    spec = {
        'analysis_type': 'new_reanalysis', 'stage': 'label_and_split_preparation', 'version': VERSION,
        'seed': SEED, 'n_folds': N_FOLDS, 'fold_numbering': '0,1,2,3,4',
        'splitter_input_order': 'ascending analysis_image_id; output manifests retain source row order',
        'split_algorithms': algorithms, 'PAD_grouping': 'unchanged source patient_id with dataset prefix',
        'PAD_label_mapping': PAD_LABELS, 'DDI_label_mapping': DDI_LABELS,
        'DDI_grouping': 'source image identifier; patient-level separation is NOT established',
        'DDI_stratification': 'released skin_tone group and released malignant label jointly',
        'missing_PAD_skin_tone': 'retain for classifier fitting; UNKNOWN for skin-tone summaries; do not impute',
        'DDI_status': 'previously studied cohort; within-DDI adaptation remains exploratory',
        'use_of_fold_files': 'PAD folds for development/C selection; DDI folds only for within-DDI analyses. A PAD-only model may score all DDI images without using any DDI row for fitting.',
        'scope': 'dataset-defined binary labels; not evaluated clinical referral urgency',
        'label_preservation': 'retain every source value, including DDI disease and malignant; do not relabel diagnoses',
        'near_duplicate_checks': 'not performed at this stage; image folds do not prove different patients',
        'historical_subset_status': 'full curated PAD pool differs from the historical 1164-image subset; these files do not reconstruct historical manifests',
        'model_fitting_or_prediction_performed': False, 'historical_metrics_verified': False,
        'is_registered_preregistration': False,
        'input_sha256': input_hashes, 'DDI_metadata_sha256': DDI_METADATA_SHA,
        'runtime': {'python': platform.python_version(), 'numpy': np.__version__, 'scikit_learn': sklearn.__version__},
        'output_sha256': {name: sha(data) for name, data in payloads.items()},
    }
    payloads[f'splits/{VERSION}/split_spec_v1.json'] = json_bytes(spec)
    # Preflight every fixed path before writing any output. Reruns reuse identical bytes.
    for name, data in payloads.items():
        path = root / name
        require(not path.exists() or path.read_bytes() == data,
                'Existing v1 output differs; stopped without overwriting it: ' + str(path))
    for name, data in payloads.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            with path.open('xb') as stream:
                stream.write(data)
        require(file_sha(path) == sha(data), 'Saved output hash mismatch: ' + str(path))
    now = datetime.now(timezone.utc)
    audit = {**spec, 'completed_at_utc': now.isoformat(),
             'checks_passed': ['source snapshots', 'source cell preservation', 'outcome counts',
                               'exact image hash overlap', 'PAD patient separation',
                               'complete fold assignment', 'DDI tone/label fold coverage', 'saved output hashes'],
             'output_sha256': {name: sha(data) for name, data in payloads.items()},
             'counts': counts, 'fold_counts': fold_counts}
    audit_path = root / 'logs' / ('label_split_audit_' + now.strftime('%Y%m%dT%H%M%S%fZ') + '.json')
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    with audit_path.open('xb') as stream:
        stream.write(json_bytes(audit))
    print('PASS: labels / source hashes / source values / patient groups / CV folds / saved files')
    for dataset in ['PAD-UFES-20', 'DDI']:
        total = next(r for r in counts if r['dataset'] == dataset and r['skin_tone_group'] == 'ALL')
        print(f"{dataset}: {total['image_count']} images; positive={total['positive_count']}; negative={total['negative_count']}; {total['split_unit_count']} {total['split_unit_type']} units")
    print('PAD: every source patient stays in one CV fold.')
    print('DDI: image-level folds; released metadata provide no patient identifiers.')
    print('Saved manifests:', root / 'data' / VERSION)
    print('Saved splits:', root / 'splits' / VERSION)
    print('Audit:', audit_path)
    print('NEXT: feature extraction. Model fitting and AUROC/CI calculation have not started.')
    return audit


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, default=Path('/content/drive/MyDrive/MAKE_Reanalysis'))
    prepare(parser.parse_args().project_root)
