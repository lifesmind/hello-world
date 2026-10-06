# MAKE skin-tone annotation study — source code

**Targeted Skin-Tone Annotation for Skin Lesion Classification: Annotation Efficiency, Stronger Baselines and Diagnosis Matching with Frozen MedSigLIP Features**

Sakchai Tangprasert and Suwit Chanthasen. Source release: 6 October 2026; publication-code update: 7 October 2026.

This directory contains the original PAD/DDI reanalysis, the separately recorded exploratory E1–E3 extension, the prepared E4 evaluator, and publication plotting/verification recipes. **E4 has not been executed on real MRA-MIDAS data.** An input-validation test is not an external performance result.

## Scope

Only Python source, instructions and dependency records are included. No clinical photographs, pretrained checkpoints, feature vectors, row-level clinical metadata, fitted parameters, individual predictions, numerical archive or manuscript files are included. Dataset access and redistribution conditions remain with their custodians. Public code alone does not supply the private numerical inputs needed to reproduce the reported numbers.

## Layout

| Directory | Purpose |
| --- | --- |
| `extraction/` | Original data preparation, pinned MedSigLIP loading and PAD/DDI extraction scripts |
| `original/` | Classifiers, fixed-head intervals, refitting bootstrap, similarity screen and independent verification |
| `extension/` | E1 baselines, E2 nested annotation budgets, E3 exact diagnosis matching, DFR tuning and saved-head E4 evaluation |
| `publication/` | 600-dpi scientific plotting, within-fold diagnostics, document verification and the prepared MRA extraction adapter |

The archived scripts are retained as executed. Their historical filenames and embedded report text describe those original execution stages; the current paper title is above. `publication/plot_extension_publication.py` produces Figures 5 and 7 without the in-image E1/E3 prefixes. `publication/integrate_manuscript.py` retains the exact Figure 1 and Figure 6 recipes, including the corrected full-label x coordinate of 165.6.

## Numerical reproduction with authorized inputs

Obtain the immutable numerical archives from the corresponding author subject to source-data conditions. Extract the original archive as `BASE` (`MAKE_Reanalysis_v1`) and the extension archive as `EXT` (`MAKE_Extension_20261006`). Do not use test images to choose penalties or update a classifier. These commands verify and plot retained results:

```bash
python -m pip install -r original/requirements.txt
python original/MAKE_Verify_Results_v1.py --output BASE
python extension/MAKE_Verify_Extensions.py --base BASE --output EXT
python extension/MAKE_Reconstruct_DFR_Tuning.py --base BASE --output EXT
python publication/plot_original_results_600dpi.py --output BASE
python publication/check_within_fold.py --scores EXT/predictions/extension_scores.npz --output PUBLICATION
python publication/plot_extension_publication.py --root PUBLICATION
python publication/verify_publication.py --root PUBLICATION
```

Replace `BASE`, `EXT`, and `PUBLICATION` with the extracted roots. `PUBLICATION` is the private review package's `publication` directory containing the final Word/PDF files and full-precision result tables. Scripts record full model and sampling provenance. Review each script's command-line help before starting a new fit; refitting can be substantially more expensive than verification. The original feature scripts are designed for the original Colab layout and read the user's existing authentication through Colab Secrets, without logging values.

## Independent MRA-MIDAS evaluation (not yet executed)

Use the dataset custodian's official access path: https://aimi.stanford.edu/datasets/mra-midas-Multimodal-Image-Dataset-for-AI-based-Skin-Cancer (DOI 10.71718/15nz-jv40). Retain actual clinical photographs locally, patient/lesion identifiers, histopathology labels, modality and biopsy status. The binary endpoint must exclude intermediate and uncertain diagnoses as specified in the archived protocol. Do not infer labels or patient identifiers from filenames.

The original encoder is `google/medsiglip-448`, revision `9cea28a1a1195f665105faa6e8544c112fd960a4`: float32 CUDA, PIL backend, RGB conversion without EXIF transposition or augmentation, SDPA, TF32 disabled, `get_image_features(return_dict=True).pooler_output` and float32 L2 normalization, 1,152 dimensions. The new adapter verifies the existing encoder file hashes and processor configuration against that same archived protocol and performs this exact feature operation. The adapter is prepared source, not evidence that real MRA extraction succeeded.

```bash
python publication/MAKE_Extract_MRA_Features.py \
  --image-dir MRA_IMAGES --manifest MRA_reviewed_image_manifest.csv \
  --protocol EXT/plan/MAKE_Exploratory_Extension_20261006_v1.json \
  --rights-review MRA_rights_review.json --model-dir PINNED_MODEL_DIR \
  --output MRA_FEATURES

python extension/MAKE_Evaluate_MRA_External.py \
  --base BASE --protocol EXT/plan/MAKE_Exploratory_Extension_20261006_v1.json \
  --models EXT/models/E4_deployment_models.npz \
  --registry EXT/models/E4_deployment_registry.csv \
  --features MRA_FEATURES/MRA_features_l2.npy \
  --manifest MRA_FEATURES/MRA_feature_manifest.csv \
  --feature-spec MRA_FEATURES/MRA_feature_spec.json \
  --label-audit MRA_label_audit.json --rights-review MRA_rights_review.json \
  --output E4_REAL_RESULTS
```

The evaluator uses the seven previously saved heads without MRA model selection or fitting, averages clinical-view logits by lesion, and computes paired patient-cluster intervals. It verifies image/feature hashes, endpoint mapping and byte-level non-overlap with the development data. DDI cross-dataset patient independence cannot be certified from its released identifiers. See `E4_INPUTS.md` for the exact input schema. No example feature vectors or invented review attestations are supplied.

## Integrity

`SHA256SUMS` covers every other file in this source release. Historical archive hashes (not downloads) are:

- Original numerical archive: `bb22bcafdef2e13a85131d503b962dc8089ba96107ae0fcb96f4364fd31bd3a7`
- Exploratory extension archive: `b52595468f8f6386efea3ff89fd5dea755c32c271eddb4ff055ed4442ea7e7d2`
- Extension protocol: `58a6164a2b2fb03941887679e6cf21ebb1271d5b5522ea1c4ac04e05456f86b9`

Questions about authorized numerical inputs or source-code reuse: sakchai.t@sci.kmutnb.ac.th.

## Publication update: 7 October 2026

The author-supplied MIT LICENSE and CITATION.cff accompany this update. Figures 5 and 7 have no in-image experiment prefixes, and Figure 5 correctly labels per-arm intervals. The supplied figure exports record Matplotlib 3.11.2; use publication/requirements-figures-20261007.txt for this renderer. Older original result plots retain their recorded 3.10.8 environment. Font and backend changes can alter file bytes without changing estimates. Annotation budgets count training labels under known released outcomes/diagnoses, and do not establish prospective screening or annotation-cost savings.
