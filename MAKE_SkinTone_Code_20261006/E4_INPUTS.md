# Real E4 input requirements

Status: not executed. These are field definitions, not completed author attestations.

## Reviewed image CSV

`analysis_image_id,image_relative_path,patient_key,lesion_key,diagnosis_code,target_label,skin_tone_group,modality,biopsy_confirmed,eligible`

- One row per actual clinical image. Preserve nonempty patient and lesion identifiers from the release.
- `image_relative_path` is relative to the actual local image directory.
- `target_label`: 0 or 1 for included images, following a reviewed diagnosis mapping.
- `skin_tone_group`: I-II, III-IV, V-VI or UNKNOWN; do not invent missing Fitzpatrick labels.
- `modality`: clinical for all eligible rows. Dermoscopic views are excluded.
- `biopsy_confirmed` and `eligible`: reviewed 0/1 fields. All included rows require a biopsy-confirmed endpoint.
- Intermediate and uncertain diagnoses must be excluded. The extractor does not decide eligibility or alter labels.

The extractor adds actual image SHA256, sequential zero-based feature rows and the SHA256 of each contiguous float32 feature row. It emits `MRA_features_l2.npy`, `MRA_feature_manifest.csv` and `MRA_feature_spec.json` from real inputs only.

## Label audit JSON

The evaluator requires `dataset_doi` = `10.71718/15nz-jv40`, nonempty `reviewed_by` and `reviewed_utc`, and true reviewed fields `histopathology_binary_endpoint_reviewed`, `intermediate_and_uncertain_excluded` and `clinical_only_reviewed`. `diagnosis_to_binary` maps the actual diagnosis codes to 0, 1 or `EXCLUDE`. Review against source documentation and the intended clinical endpoint before setting these flags.

## Data-use review JSON

Both extractor and evaluator require actual author review: `noncommercial_research_use_reviewed`, `processing_environment_reviewed`, `agreement_version_or_date`, `reviewed_by`. Do not treat blank templates, automatically filled flags or the public dataset webpage as evidence of acceptance of an agreement.

The release intentionally does not provide affirmative placeholder audit JSON files. Without real images/features and reviewed source metadata there are no E4 scientific results to add to the paper.
