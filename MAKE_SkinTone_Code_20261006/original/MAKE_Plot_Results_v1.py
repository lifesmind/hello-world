"""Publication graphics from saved unrounded result tables; no image generation."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import make_reanalysis_common as c


def run(output_root):
    out = Path(output_root)
    c.write_file(out / 'code' / Path(__file__).name, Path(__file__).read_bytes())
    figures = out / 'figures'
    figures.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'axes.titlesize': 12, 'savefig.dpi': 300, 'svg.fonttype': 'none'})
    provenance = {'generated_utc': c.utc_now(), 'code_sha256': c.file_sha(__file__),
                  'primary_plan_sha256': c.file_sha(out / 'plan/MAKE_Analysis_Plan_v1.json'),
                  'graphics_method': 'matplotlib from saved, unrounded CSV values; PNG at 300 dpi and vector SVG', 'figures': []}

    def save(fig, stem, rows, source_paths, caption, selector):
        data_path = figures / (stem + '_data.csv')
        c.write_csv(data_path, rows)
        paths = []
        for extension in ('png', 'svg'):
            path = figures / (stem + '.' + extension)
            fig.savefig(path, bbox_inches='tight', facecolor='white')
            paths.append({'path': str(path.relative_to(out)), 'sha256': c.file_sha(path)})
        plt.close(fig)
        provenance['figures'].append({'figure_id': stem, 'caption': caption, 'selection': selector,
            'plotted_data': {'path': str(data_path.relative_to(out)), 'sha256': c.file_sha(data_path)},
            'source_tables': [{'path': str(p.relative_to(out)), 'sha256': c.file_sha(p)} for p in source_paths], 'outputs': paths})

    transfer_path = out / 'results/Table_R2_PAD_development_DDI_transfer.csv'
    source = c.read_csv(transfer_path)
    rows = [r for r in source if r['stratum'] in ('ALL_PAD', 'I-II', 'III-IV', 'V-VI')]
    labels = ['PAD nested CV\n(pretraining overlap)', 'DDI I-II', 'DDI III-IV', 'DDI V-VI']
    values = np.array([float(r['AUROC']) for r in rows])
    lower = np.array([float(r['AUROC_CI_low']) for r in rows])
    upper = np.array([float(r['AUROC_CI_high']) for r in rows])
    fig, (ax, value_ax) = plt.subplots(1, 2, figsize=(7.8, 3.7), layout='constrained',
                                     sharey=True, gridspec_kw={'width_ratios': [3.1, 1.15], 'wspace': 0.02})
    y = np.arange(len(rows))[::-1]
    ax.errorbar(values, y, xerr=[values-lower, upper-values], fmt='o', color='#185a7d', capsize=4, lw=1.5)
    for i, r in enumerate(rows):
        value_ax.text(0.03, y[i], f'{values[i]:.3f}  (n={r["n"]})', ha='left', va='center', fontsize=9)
    value_ax.set_xlim(0, 1)
    value_ax.axis('off')
    ax.axvline(0.5, color='#888888', ls='--', lw=1)
    ax.set(yticks=y, yticklabels=labels, xlim=(0.32, 1.0), ylim=(-0.6, 3.6), xlabel='AUROC with conditional 95% confidence interval')
    fig.suptitle('PAD development and DDI transfer', fontsize=12)
    save(fig, 'Figure_R1_transfer', rows, [transfer_path],
         'PAD patient-grouped nested development and PAD-only DDI transfer. Intervals use 10000 conditional bootstrap samples. PAD appeared in encoder pretraining, so its development AUROC does not establish independence from encoder training.',
         'ALL_PAD plus the three DDI skin-tone strata, PAD-only head')

    controls_path = out / 'results/Table_R4_matched_controls.csv'
    all_controls = c.read_csv(controls_path)
    rows = [r for r in all_controls if r['stratum']=='V-VI' and r['metric']=='AUROC']
    labels = ['V-VI added, weighted', 'V-VI added, unweighted', 'Lighter added, weighted mean', 'Lighter added, unweighted mean',
              'V-VI minus lighter, weighted', 'V-VI minus lighter, unweighted', 'Weighted minus unweighted V-VI']
    delta = np.array([float(r['difference_unrounded']) for r in rows])
    lower = np.array([float(r['CI_low']) for r in rows])
    upper = np.array([float(r['CI_high']) for r in rows])
    fig, ax = plt.subplots(figsize=(8.0, 4.6), layout='constrained')
    y = np.arange(len(rows))[::-1]
    ax.errorbar(delta, y, xerr=[delta-lower, upper-delta], fmt='o', color='#185a7d', capsize=4, lw=1.5)
    ax.axvline(0, color='#888888', ls='--', lw=1)
    ax.set(yticks=y, yticklabels=labels, xlim=(-0.018, 0.405), ylim=(-0.6, 6.6), xlabel='AUROC difference with paired conditional 95% confidence interval', title='Controlled comparisons in DDI V-VI')
    save(fig, 'Figure_R2_controls', rows, [controls_path],
         'All arms are evaluated on the same 207 DDI V-VI images. Lighter controls average 20 samples matching the positive and negative training counts of V-VI addition. Intervals use 10000 paired, fold-by-tone-by-label stratified image bootstrap samples, conditional on fitted heads and panels; comparisons are unadjusted.',
         'V-VI AUROC contrasts')

    refit_path = out / 'results/Table_R5_refitting_robustness.csv'
    proxy_path = out / 'results/Table_RS13_proxy_group_changes.csv'
    conditional = next(r for r in rows if r['left_arm']=='target_weighted' and r['right_arm']=='PAD_baseline')
    refit = next(r for r in c.read_csv(refit_path) if r['stratum']=='V-VI' and r['arm_or_contrast']=='target_weighted_minus_PAD_baseline' and r['metric']=='AUROC')
    proxy = next(r for r in c.read_csv(proxy_path) if r['stratum']=='V-VI' and r['arm']=='target_weighted' and r['metric']=='AUROC')
    rows = [{'analysis':'Fixed fitted heads', 'point':conditional['difference_unrounded'], 'CI_low':conditional['CI_low'], 'CI_high':conditional['CI_high'], 'resamples':10000},
            {'analysis':'Heads refitted with fixed C and folds', 'point':refit['original_point_estimate'], 'CI_low':refit['CI_low'], 'CI_high':refit['CI_high'], 'resamples':500},
            {'analysis':'Image similarity groups post hoc', 'point':proxy['difference_unrounded'], 'CI_low':proxy['CI_low'], 'CI_high':proxy['CI_high'], 'resamples':10000}]
    delta = np.array([float(r['point']) for r in rows]);lower=np.array([float(r['CI_low']) for r in rows]);upper=np.array([float(r['CI_high']) for r in rows])
    fig, ax = plt.subplots(figsize=(7.6, 3.2), layout='constrained')
    y = np.array([2,1,0])
    ax.errorbar(delta,y,xerr=[delta-lower,upper-delta],fmt='o',color='#185a7d',capsize=4,lw=1.5)
    ax.axvline(0,color='#888888',ls='--',lw=1)
    ax.set(yticks=y,yticklabels=[r['analysis'] for r in rows],xlim=(-0.02,0.43),ylim=(-0.6,2.6),
           xlabel='V-VI AUROC difference with 95% confidence interval',title='Uncertainty in weighted V-VI data addition')
    save(fig,'Figure_R3_uncertainty',rows,[controls_path,refit_path,proxy_path],
         'The point estimate is calculated on original observations. The 500-replicate sensitivity refits the PAD head and five DDI adaptation heads per replicate, with PAD patient-cluster and DDI stratified image resampling. C, encoder and folds stay fixed. Similarity grouping is a separately declared post hoc analysis, not patient identification. None is independent external replication.',
         'V-VI target_weighted minus PAD_baseline, three uncertainty analyses')

    draws_path=out/'results/Table_R6_repeated_draws.csv'
    rows=[r for r in c.read_csv(draws_path) if r['stratum']=='V-VI']
    fig,ax=plt.subplots(figsize=(6.8,3.8),layout='constrained')
    for i,arm in enumerate(['lighter_unweighted','lighter_weighted']):
        selected=[r for r in rows if r['arm']==arm]
        values=np.array([float(r['AUROC_change_from_PAD_baseline']) for r in selected])
        offsets=np.linspace(-0.12,0.12,len(selected))
        ax.scatter(np.full(len(selected),i)+offsets,values,s=25,color=['#997144','#185a7d'][i],alpha=0.8)
        ax.plot([i-0.2,i+0.2],[values.mean(),values.mean()],color='black',lw=2)
    ax.axhline(0,color='#888888',ls='--',lw=1)
    ax.set(xticks=[0,1],xticklabels=['Lighter addition without source weight','Lighter addition with source weight'],
           xlim=(-0.5,1.5),ylabel='V-VI AUROC change from PAD-only head',title='Variability across 20 matched lighter samples')
    ax.tick_params(axis='x',labelsize=9)
    save(fig,'Figure_R4_repeated_draws',rows,[draws_path],
         'Each dot is a per-draw out-of-fold AUROC change, on the same V-VI evaluation images. Horizontal marks are draw means. The dots describe training-panel sampling variability and do not represent independent patient cohorts.',
         'V-VI per-draw lighter control AUROC changes; deterministic visual offsets, no analytic jitter')
    c.write_json(figures/'figure_provenance.json',provenance)
    print('FIGURES: PASS; 4 PNG/SVG figures with exact source CSV and SHA256 provenance')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True)
    args=p.parse_args();run(args.output)
