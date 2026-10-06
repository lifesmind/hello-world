"""Record a distinct exploratory extension protocol before any NEW fitting."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import importlib.metadata as metadata
import json
import sys


def main(base, output):
    assets = ['inputs/features/PAD_features_l2.npy', 'inputs/features/DDI_features_l2.npy',
              'inputs/features/PAD_feature_index.csv', 'inputs/features/DDI_feature_index.csv',
              'inputs/features/feature_spec.json', 'models/core_models.npz',
              'models/core_models_registry.csv', 'predictions/core_scores.npz',
              'plan/MAKE_Analysis_Plan_v1.json']
    p = {
        'protocol_id': 'MAKE_Exploratory_Extension_20261006_v1',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'classification': 'Exploratory extension recorded before NEW fitting; original DDI results already known; NOT a preregistration',
        'previously_named_plan_unavailable': 'a2e188c2…ca4ac4',
        'not_a_reconstruction_or_replacement_of_previously_named_plan': True,
        'manuscript_policy': 'Retain the designated 20261006_MDPI manuscripts and 32-reference bibliography; create separate insertion text and evidence',
        'source_archive': {'name': 'MAKE_Reanalysis_20261005.zip', 'sha256': 'bb22bcafdef2e13a85131d503b962dc8089ba96107ae0fcb96f4364fd31bd3a7'},
        'base_asset_sha256': {n: hashlib.sha256((base/n).read_bytes()).hexdigest() for n in assets},
        'feature_spec': json.loads((base/'inputs/features/feature_spec.json').read_text()),
        'feature_type': '1152 float32 L2-normalized frozen MedSigLIP features; float64 head fitting',
        'fixed_C_L2': 10.0, 'seed': 20261006, 'draws': 20,
        'outer_test_splits': 'Reuse saved five DDI image folds stratified by tone x outcome; every DDI adaptation/tuning row outside test fold; DDI patient independence NOT established',
        'L2_fit': {'solver': 'lbfgs', 'tol': 1e-7, 'max_iter_attempts': [4000, 8000, 16000],
                   'class_weights': 'N/(2*N_class), unweighted included PAD+DDI counts',
                   'source_multiplier': 'sqrt(N_PAD/N_DDI_added), applied after class balancing in source-weighted arms; 1 otherwise'},
    }
    p['E1'] = {
        'arms': ['all_DDI_unweighted', 'all_DDI_source_weighted', 'all_DDI_group_reweighted',
                 'all_DDI_budget_matched', 'DDI_only_class_balanced', 'DDI_only_group_balanced', 'DFR_style'],
        'all_DDI': 'All non-test DDI rows; larger annotation budget than target-only addition, explicitly reported',
        'group_reweighting': 'Keep pooled outcome weights and DDI source multiplier; multiply DDI by within-outcome inverse-tone factor N_DDI_y/(3*N_DDI_tone_y); PAD factor unchanged',
        'all_DDI_budget_matched': '20 panels from all non-test tones, exact target-VVI positive/negative counts and total budget; without replacement; source-weighted',
        'DDI_only': 'Frozen encoder, no PAD head-fitting rows; L2 with class balance or equal weight in six tone x outcome groups',
        'DFR_style': {
            'scope': 'Published DFR last-layer principles adapted to frozen MedSigLIP; not Waterbirds/CelebA replication or encoder retraining',
            'primary_code_url': 'https://github.com/PolinaKirichenko/deep_feature_reweighting/blob/main/dfr_evaluate_spurious.py',
            'paper_url': 'https://arxiv.org/abs/2204.02937',
            'preprocessing': 'StandardScaler fit only on PAD; fixed source transform for all DFR tuning/final heads',
            'head': 'L1 LogisticRegression liblinear tol1e-7 intercept_scaling1; no class weights on exactly balanced panels',
            'groups': 'Six DDI tone x binary-outcome groups; equal n=min(group count), without replacement',
            'C_grid': [0.01, 0.03, 0.07, 0.1, 0.3, 0.7, 1.0],
            'tuning': 'Three inner folds stratified by six groups within outer training pool; four balanced panels per inner fold; same panels for all C; maximize mean worst-six-group accuracy at logit0; ties smallest C; outer test never used',
            'final': '20 balanced last-layer fits; average coefficients/intercepts in original feature coordinates; no PAD head-fitting rows',
        },
    }
    p['E2'] = {
        'budgets': [25, 50, 100, 'ALL_NONTEST_VVI'], 'arms': ['target_source_weighted'],
        'sampling': '20 nested per-outcome permutations per outer fold; same prefixes for25/50/100; positives=round-half-up(budget*non-test-VVI prevalence), min1/max budget-1; ALL uses saved full-target-weighted results; one draw combines its five-fold panels',
    }
    p['E3'] = {
        'arms': ['diagnosis_matched_target', 'diagnosis_matched_lighter'],
        'matching': 'Exact released disease string x binary outcome; non-test DDI only; quota=min(VVI count,I-II/III-IV count) in each cell; 20 paired panels without replacement; report all unsupported and discarded rows',
        'fit': 'PAD+matched panel source-weighted L2 C10; identical diagnosis/outcome counts, total budget and weights in paired heads',
        'estimand': 'Diagnosis-overlap adaptation support evaluated on unchanged full held-out images; NOT full-pool exact matching or causal skin-tone effect',
    }
    p['uncertainty'] = {
        'replicates': 5000, 'confidence': 0.95,
        'image_resampling': 'Shared multinomial counts within saved fold x tone x outcome; no DDI patient-cluster CI claim',
        'panel_resampling': '20-panel arms use multinomial counts over20 draw IDs; counts shared within experiment, E3 pairs and E2 nested sizes; conditional on fixed fitted heads/splits, includes Monte Carlo panel variability, no model refitting',
        'primary': 'VVI AUROC; secondary other tones/overall; mean per-draw AUROC, NOT AUROC of averaged random-panel scores',
        'secondary': 'Retrospective sensitivity at empirical FPR<=0.50; evaluation-derived ROC threshold, NOT clinical operating threshold',
        'multiplicity': 'Unadjusted exploratory95% percentile CIs; no confirmatory significance or best-arm selection claim',
    }
    p['deployment_for_E4'] = {
        'heads': ['PAD_baseline', 'target_weighted_full', 'all_DDI_source_weighted_full', 'all_DDI_group_reweighted_full',
                  'DDI_only_class_balanced_full', 'DDI_only_group_balanced_full', 'DFR_style_full'],
        'policy': 'Fit before external outcomes available; same formulas; DFR C via internal3-fold DDI tuning plus20 refits; prespecified comparisons, no MRA-based selection',
    }
    p['E4_pending'] = {
        'dataset': 'MRA-MIDAS DOI10.71718/15nz-jv40', 'status': 'INPUTS_ABSENT; no external result reported',
        'endpoint': 'Biopsy-confirmed malignant vs benign lesions; intermediate/uncertain diagnoses excluded by reviewed manifest; never relabeled benign automatically',
        'modality': 'Clinical photos only, dermoscopy excluded; mean clinical-view logit per lesion',
        'unit': 'Lesion AUROC, paired patient-cluster bootstrap; missing patient keys, conflicting lesion labels/tone, duplicate IDs or missing classes fail validation',
        'probability_policy': 'Uncalibrated sigmoid for reconstruction only; no clinical decision claim',
        'rights': 'Check dataset-specific agreement before importing/uploading MRA; no public raw-data/feature-cache upload',
    }
    p['rights_policy'] = 'Private author-review analysis outputs; source raw images/features/metadata excluded from new package; no Zenodo/public dataset/model deposit performed'
    p['runtime'] = {'python': sys.version.split()[0], **{k: metadata.version(k) for k in
                    ['numpy', 'pandas', 'scipy', 'scikit-learn', 'threadpoolctl', 'matplotlib', 'python-docx', 'Pillow']}}
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError('Protocol exists; do not overwrite recorded protocol')
    output.write_text(json.dumps(p, indent=2, ensure_ascii=False)+'\n')
    h = hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_suffix('.sha256').write_text(h+'  '+output.name+'\n')
    print('New exploratory protocol recorded:', h)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    main(args.base, args.output)
