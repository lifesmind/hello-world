# E4 MRA-MIDAS publication inputs

Status: completed for the manuscript E4 analysis.

This file documents the inputs consumed by the final public evaluator. It does not redistribute MRA-MIDAS photographs, the 1,152-dimensional feature matrix, row-level clinical metadata or per-image predictions.

## Final feature matrix

File used in the publication analysis:

- `MRA_MIDAS_E4_features_l2.npy`
- dtype: float32
- shape: `(1579, 1152)`
- every row finite and L2-normalised
- encoder: `google/medsiglip-448`
- pinned revision: `9cea28a1a1195f665105faa6e8544c112fd960a4`
- image size: 448 × 448
- pooled image feature: `get_image_features(..., return_dict=True).pooler_output`
- final operation: float32 L2 normalisation

The public local-image adapter is `publication/MAKE_Extract_MRA_Features.py`.

## Final manifest

The publication evaluator requires one row per feature row. Minimum required columns are:

`midas_record_id,label`

Optional but recommended traceability columns are:

`feature_row_index,midas_file_name,local_file_name,endpoint_group`

If `feature_row_index` is present, it must be a zero-based sequence matching the feature matrix.

The binary label is 1 for the positive endpoint and 0 for the negative endpoint.

## Cohort definition

MRA-MIDAS v1.0 clinical-image rows were filtered as follows:

- released control flag must be `no`;
- released distance must be 6-inch or 1-foot;
- malignant basal cell carcinoma, squamous cell carcinoma and melanoma are positive;
- diagnoses beginning with `benign-` are negative;
- intermediate or otherwise non-mapped diagnoses are excluded.

Audit counts for the publication E4 cohort:

- clinical-image rows considered: 1,935
- endpoint-review exclusions: 349
- endpoint-defined rows before image-file matching: 1,586
- endpoint-defined rows without a matching image file: 7
- final images: 1,579
- positive images: 692
- negative images: 887
- unique `midas_record_id` groups: 567
- mixed-label record groups: 67

The released `midas_record_id` is not asserted to be a patient identifier. Mixed-label groups are therefore allowed and are resampled as intact statistical clusters.

## Frozen deployment heads

Required files:

- `models/E4_deployment_models.npz`
- `models/E4_deployment_registry.csv`

The NPZ contains seven frozen coefficient vectors of length 1,152 and seven intercepts. All seven heads were fitted and saved before MRA-MIDAS scoring. No E4 head is refitted, recalibrated or selected on MRA-MIDAS.

## Final estimand and uncertainty

- point estimate: image-level AUROC;
- bootstrap unit: `midas_record_id`;
- bootstrap replicates: 10,000;
- random seed: 0;
- interval: 2.5th and 97.5th percentiles of valid record-cluster bootstrap AUROCs.

These intervals are record-clustered and must not be described as patient-level confidence intervals.

## Final aggregate results

The seven-head aggregate output is public in:

`publication/E4_MRA_MIDAS_7heads_AUROC.csv`

The DFR ensemble had the highest observed AUROC, 0.7379 (95% record-clustered CI 0.7016–0.7719). This is an observed external ranking, not a basis for retrospective model selection.

## Source-use restrictions

Dataset access and redistribution remain governed by the MRA-MIDAS custodian. This repository intentionally excludes raw clinical photographs, restricted metadata, feature arrays and individual predictions.
