"""Render Figures 5 and 7 from unchanged, full-precision result tables."""
from pathlib import Path
import argparse
import hashlib
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

LABELS = {
    'PAD_baseline': 'PAD-only transfer',
    'target_weighted': 'V–VI addition, weighted',
    'target_unweighted': 'V–VI addition, unweighted',
    'lighter_weighted': 'Outcome-matched lighter, weighted',
    'all_DDI_unweighted': 'All-DDI addition, unweighted',
    'all_DDI_source_weighted': 'All-DDI addition, weighted',
    'all_DDI_group_reweighted': 'All-DDI addition, tone reweighted',
    'all_DDI_budget_matched': 'All-DDI, matched annotation budget',
    'DDI_only_class_balanced': 'DDI-only, class balanced',
    'DDI_only_group_balanced': 'DDI-only, six groups balanced',
    'DFR_style': 'DFR-style last-layer refit',
}

def main(root):
    root = Path(root)
    data = pd.read_csv(root/'tables/Table_E1_baselines.csv')
    v = data[data.cohort == 'V-VI'].set_index('arm')
    co = pd.read_csv(root/'tables/all_paired_contrasts.csv')
    cv = co[(co.cohort == 'V-VI') & (co.metric == 'AUROC')]
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':9,
                         'axes.spines.top':False, 'axes.spines.right':False,
                         'savefig.facecolor':'white', 'axes.labelcolor':'#172334',
                         'text.color':'#172334'})
    output = root/'figures'
    output.mkdir(exist_ok=True)
    records = []
    def save(fig, name, source, filters, title):
        fig.tight_layout()
        for suffix in ['png','pdf','svg']:
            fig.savefig(output/(name+'.'+suffix), dpi=600, bbox_inches='tight')
        plt.close(fig)
        records.append({'figure':name, 'source':source,
                        'source_sha256':hashlib.sha256((root/source).read_bytes()).hexdigest(),
                        'filters':filters, 'title':title,
                        'change':'Removed experiment-prefix text; corrected Figure 5 axis label (per-arm, not paired, intervals); en dashes in tone ranges. Estimates, intervals, axes limits and colors unchanged.'})
    arms = list(LABELS)
    fig, ax = plt.subplots(figsize=(7.5,4.8))
    for i, arm in enumerate(arms):
        r = v.loc[arm]
        color = '#1F5A99' if arm.startswith('target') else ('#208171' if arm.startswith('DDI_only') else '#75859A')
        ax.errorbar(r.AUROC, i, xerr=[[r.AUROC-r.AUROC_CI_low],[r.AUROC_CI_high-r.AUROC]],
                    fmt='o', capsize=3, color=color, ms=5)
    ax.set_yticks(range(len(arms)), [LABELS[a] for a in arms])
    ax.invert_yaxis()
    ax.axvline(.5, ls='--', lw=.8, color='#BBBBBB')
    ax.set_xlim(.36,.94)
    ax.set_xlabel('V–VI AUROC with exploratory 95% CI')
    ax.grid(axis='x',alpha=.17)
    title = 'Domain adaptation and bias baselines'
    ax.set_title(title,loc='left',fontweight='bold',pad=12)
    save(fig,'Figure5_baselines','tables/Table_E1_baselines.csv',{'cohort':'V-VI','arms':arms},title)
    fig, ax = plt.subplots(figsize=(7.2,2.8))
    pairs = [('target_weighted','lighter_weighted'),('diagnosis_matched_target','diagnosis_matched_lighter')]
    labels = ['Outcome-matched controls\n165–166 images/source/fold',
              'Exact diagnosis-overlap controls\n114–125 images/source/fold']
    for i,(left,right) in enumerate(pairs):
        r = cv[(cv.left_arm==left)&(cv.right_arm==right)].iloc[0]
        ax.errorbar(r.difference_unrounded,i,
                    xerr=[[r.difference_unrounded-r.CI_low],[r.CI_high-r.difference_unrounded]],
                    fmt='o',capsize=4,color='#1F5A99',ms=6)
    ax.axvline(0,color='#606B78',lw=.9,ls='--')
    ax.set_yticks([0,1],labels)
    ax.invert_yaxis()
    ax.set_xlim(-.04,.34)
    ax.set_xlabel('V–VI AUROC difference: target minus lighter (95% CI)')
    ax.grid(axis='x',alpha=.15)
    title = 'Disease-mix sensitivity'
    ax.set_title(title,loc='left',fontweight='bold')
    save(fig,'Figure7_diagnosis','tables/all_paired_contrasts.csv',{'cohort':'V-VI','metric':'AUROC','pairs':pairs},title)
    (root/'audit/figure_5_7_revision.json').write_text(json.dumps({'figures':records},indent=2)+'\n')

if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True)
    main(parser.parse_args().root)
