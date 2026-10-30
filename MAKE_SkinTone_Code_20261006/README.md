# MAKE skin-tone annotation study — source code

**Targeted Skin-Tone Annotation for Skin Lesion Classification: Annotation Efficiency, Stronger Baselines and Diagnosis Matching with Frozen MedSigLIP Features**

Sakchai Tangprasert and Suwit Chanthasen. Source release: 6 October 2026; publication-code update: 30 October 2026.

This directory contains the original PAD/DDI reanalysis, the separately recorded exploratory E1–E3 extension, the completed external E4 MRA-MIDAS evaluation code, and publication plotting/verification recipes. E4 was run with seven prespecified frozen heads and no MRA-MIDAS refitting or retrospective model selection.

## Scope

Only Python source, instructions and dependency records are included. No clinical photographs, pretrained checkpoints, feature vectors, row-level clinical metadata, fitted parameters, individual predictions, numerical archive or manuscript files are included. Dataset access and redistribution conditions remain with their custodians. Public code alone does not supply the restricted numerical inputs needed to reproduce all reported numbers.

## Layout

| Directory | Purpose |
| --- | --- |
| `extraction/` | Original data preparation, pinned MedSigLIP loading and PAD/DDI extraction scripts |
| `original/` | Classifiers, fixed-head intervals, refitting bootstrap, similarity screen and independent verification |
| `extension/` | E1 baselines, E2 nested annotation budgets, E3 exact diagnosis matching, DFR tuning and final saved-head E4 evaluation |
| `publication/` | 600-dpi scientific plotting, within-fold diagnostics, document verification and the MRA feature-extraction adapter |

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

Replace `BASE`, `EXT`, and `PUBLICATION` with the extracted roots. Scripts record full model and sampling provenance. Review each script's command-line help before starting a new fit; refitting can be substantially more expensive than verification.

## Independent MRA-MIDAS evaluation (E4; completed)

MRA-MIDAS v1.0 was accessed through the Stanford AIMI data-access channel (DOI 10.71718/15nz-jv40). The publication E4 cohort used released non-control clinical images at 6-inch or 1-foot distance. Malignant basal cell carcinoma, squamous cell carcinoma and melanoma were mapped to the positive endpoint; diagnoses beginning with `benign-` were mapped to the negative endpoint; intermediate or otherwise non-mapped diagnoses were excluded.

The final E4 cohort contains 1,579 images (692 positive, 887 negative) from 567 `midas_record_id` groups. Of 1,935 clinical-image rows, 349 were endpoint-review exclusions; seven additional endpoint-defined rows lacked a matching image file in the accessed file index. Sixty-seven record groups contain both endpoint labels, so `midas_record_id` is treated only as a resampling cluster and is not interpreted as a patient identifier.

Images were encoded with `google/medsiglip-448`, revision `9cea28a1a1195f665105faa6e8544c112fd960a4`, using the same 448 × 448 float32 frozen-encoder pipeline as the PAD/DDI analyses. The saved feature matrix has shape 1,579 × 1,152 and each row is L2-normalised.

Seven deployment heads saved before E4 scoring were applied without refitting. Point AUROC is computed at image level. Uncertainty uses 10,000 percentile bootstrap resamples clustered by `midas_record_id` with NumPy seed 0. These are record-clustered, not patient-level, confidence intervals.

| Frozen head | AUROC | 95% record-clustered CI |
| --- | ---: | ---: |
| E4_PAD_baseline | 0.7081 | 0.6721–0.7425 |
| E4_target_weighted_full | 0.6556 | 0.6171–0.6934 |
| E4_all_DDI_source_weighted_full | 0.7139 | 0.6778–0.7487 |
| E4_all_DDI_group_reweighted_full | 0.7102 | 0.6737–0.7453 |
| E4_DDI_only_class_balanced_full | 0.7141 | 0.6775–0.7485 |
| E4_DDI_only_group_balanced_full | 0.7089 | 0.6723–0.7436 |
| DFR_ensemble_f-1 | 0.7379 | 0.7016–0.7719 |

The aggregate table above is also stored in `publication/E4_MRA_MIDAS_7heads_AUROC.csv`. Raw MRA-MIDAS photographs, feature arrays, row-level metadata and per-image predictions are not redistributed.

With authorized local inputs, the final scoring step is reproduced by:

```bash
python extension/MAKE_Evaluate_MRA_External.py \
  --models EXT/models/E4_deployment_models.npz \
  --registry EXT/models/E4_deployment_registry.csv \
  --features MRA_MIDAS_E4_features_l2.npy \
  --manifest E4_primary_manifest_FINAL.csv \
  --output E4_FINAL_RESULTS \
  --bootstrap-rounds 10000 --seed 0
```

The final evaluator writes the seven-head AUROC table, per-image logits/probabilities and the record-cluster bootstrap samples. See `E4_INPUTS.md` for the exact publication-E4 input schema and endpoint definition. The local feature-extraction adapter in `publication/MAKE_Extract_MRA_Features.py` implements the same pinned MedSigLIP feature operation for authorized image files.

## Integrity

`SHA256SUMS` in the repository should be regenerated after this E4 publication-code update so that it covers the synchronized files. Historical archive hashes (not downloads) remain:

- Original numerical archive: `bb22bcafdef2e13a85131d503b962dc8089ba96107ae0fcb96f4364fd31bd3a7`
- Exploratory extension archive: `b52595468f8f6386efea3ff89fd5dea755c32c271eddb4ff055ed4442ea7e7d2`
- Extension protocol: `58a6164a2b2fb03941887679e6cf21ebb1271d5b5522ea1c4ac04e05456f86b9`

Questions about authorized numerical inputs or source-code reuse: sakchai.t@sci.kmutnb.ac.th.

## Publication update: 30 October 2026

The author-supplied MIT LICENSE and CITATION.cff accompany this release. The 30 October update synchronizes the public E4 documentation and evaluator with the completed MRA-MIDAS analysis reported in the manuscript. Figures 5 and 7 have no in-image experiment prefixes, and Figure 5 correctly labels per-arm intervals. Annotation budgets count training labels under known released outcomes/diagnoses and do not establish prospective screening or annotation-cost savings.
