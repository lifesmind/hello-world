"""Diagnostic image-similarity screen, never used to change scored folds."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import zipfile
from pathlib import Path, PurePosixPath

import numpy as np
from PIL import Image
from scipy.fft import dctn

import make_reanalysis_common as c


def run(output_root, source_zip):
    out = Path(output_root)
    data = c.load_inputs(out / 'inputs', out / 'plan/MAKE_Analysis_Plan_v1.json')
    c.init_output(out, data, [__file__])
    archive = Path(source_zip)
    expected = data['plan']['input_sha256']['features/feature_spec.json']
    spec = json.loads((out / 'inputs/features/feature_spec.json').read_text())
    evidence = spec['source_archives']['DDI - Diverse Dermatology Images_files-2.zip']
    c.require(archive.stat().st_size == evidence['bytes'] and c.file_sha(archive) == evidence['sha256'], 'DDI source ZIP differs from encoder source')
    rows, fingerprints = [], []
    ddi = data['datasets']['DDI']
    with zipfile.ZipFile(archive) as z:
        members = {}
        for entry in z.infolist():
            if entry.is_dir():
                continue
            pure = PurePosixPath(entry.filename)
            c.require(not pure.is_absolute() and '..' not in pure.parts, 'Unsafe source ZIP member')
            members.setdefault(pure.name, []).append(entry)
        for i, row in enumerate(ddi['rows']):
            options = members.get(PurePosixPath(row['image_relative_path']).name, [])
            c.require(len(options) == 1, 'Ambiguous DDI image member')
            raw = z.read(options[0])
            c.require(hashlib.sha256(raw).hexdigest() == row['image_sha256'], 'DDI image bytes changed')
            with Image.open(io.BytesIO(raw)) as im:
                im.load()
                rgb = im.convert('RGB')
            c.require(rgb.size == (int(row['image_width']), int(row['image_height'])), 'DDI dimensions changed')
            small = rgb.convert('L').resize((32, 32), Image.Resampling.LANCZOS)
            block = dctn(np.asarray(small, dtype=np.float64), type=2, norm='ortho')[:8, :8].ravel()
            bits = block > np.median(block[1:])
            fingerprint = sum(int(value) << k for k, value in enumerate(bits))
            pixel_sha = hashlib.sha256(c.json_bytes({'mode': 'RGB', 'width': rgb.width, 'height': rgb.height}) + rgb.tobytes()).hexdigest()
            fingerprints.append(fingerprint)
            rows.append({'analysis_image_id': ddi['ids'][i], 'image_sha256': row['image_sha256'], 'cv_fold': int(ddi['fold'][i]),
                         'decoded_RGB_sha256_with_dimensions': pixel_sha, 'perceptual_DCT_hash_64bit_hex': f'{fingerprint:016x}'})
    pairs = []
    for i, left in enumerate(fingerprints):
        for j in range(i + 1, len(fingerprints)):
            distance = (left ^ fingerprints[j]).bit_count()
            identical_pixels = rows[i]['decoded_RGB_sha256_with_dimensions'] == rows[j]['decoded_RGB_sha256_with_dimensions']
            if distance <= 4 or identical_pixels:
                pairs.append({'left_image_id': ddi['ids'][i], 'right_image_id': ddi['ids'][j], 'perceptual_hash_distance': distance,
                              'decoded_RGB_identical': identical_pixels, 'left_fold': int(ddi['fold'][i]), 'right_fold': int(ddi['fold'][j]),
                              'crosses_saved_folds': bool(ddi['fold'][i] != ddi['fold'][j]), 'patient_identity_inferred': False})
    c.write_csv(out / 'results/DDI_similarity_fingerprints.csv', rows)
    c.write_csv(out / 'results/DDI_similarity_candidate_pairs.csv', pairs,
                fields=['left_image_id', 'right_image_id', 'perceptual_hash_distance', 'decoded_RGB_identical', 'left_fold', 'right_fold', 'crosses_saved_folds', 'patient_identity_inferred'])
    c.write_json(out / 'logs/DDI_similarity_screen_audit.json', {'completed_utc': c.utc_now(), 'plan_sha256': data['plan_sha256'],
                  'source_zip_sha256': evidence['sha256'], 'source_zip_bytes': evidence['bytes'],
                  'code_sha256': {Path(p).name: c.file_sha(p) for p in (__file__, c.__file__)},
                  'method': data['plan']['duplicate_screen'], 'images_screened': len(rows), 'candidate_pairs': len(pairs),
                  'candidate_pairs_crossing_folds': sum(r['crosses_saved_folds'] for r in pairs),
                  'decoded_RGB_identical_pairs': sum(r['decoded_RGB_identical'] for r in pairs),
                  'folds_or_training_samples_changed': False, 'DDI_patient_independence_established': False,
                  'output_sha256': {str(p.relative_to(out)): c.file_sha(p) for p in [out / 'results/DDI_similarity_fingerprints.csv', out / 'results/DDI_similarity_candidate_pairs.csv']}})
    print('DDI similarity screen:', len(rows), 'images;', len(pairs), 'candidate pairs; no fold changes')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--source-zip', required=True)
    args = parser.parse_args()
    run(args.output, args.source_zip)
