"""Promote completed five-fold controls to the current report evidence.

Run only after python -m src.dl.report_controls has completed both tasks:
    python -m src.dl.finalize_results
"""
from __future__ import annotations

import json
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, roc_curve

from config import FIG_DIR, ROOT, TABLE_DIR
from src.dl.train_eegnet import (plot_roc_pr_curves,
                                plot_saliency_analysis, plot_saliency_topomaps)
from src.dl.report_controls import INCLUDED
from src.features.extract_features import load_features
from src.utils.io import epoch_path
from src.verify_report import verify_dl_controls


def plot_comparison(summary, prevalence):
    """Use separate chance references for balanced accuracy/AUC and AP."""
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(summary))
    for offset, key, label, color in [
            (-1, 'balanced_accuracy', 'Balanced accuracy', '#4c72b0'),
            (0, 'roc_auc', 'ROC-AUC', '#55a868'),
            (1, 'pr_auc', 'AP', '#c44e52')]:
        ax.bar(x + offset * 0.26, summary[key + '_mean'], 0.26,
               yerr=summary[key + '_std'], label=label, color=color, capsize=3)
    ax.axhline(0.5, ls='--', color='0.5', lw=1, label='BA / ROC-AUC reference: 0.50')
    ax.axhline(prevalence, ls=':', color='0.4', lw=1,
               label=f'AP prevalence reference: {prevalence:.3f}')
    ax.set_xticks(x, summary['model'], fontsize=9)
    ax.set(ylabel='Score', ylim=(0, 1.13),
           title='Current participant-wise results (mean ± sample SD over five folds)')
    ax.legend(loc='upper center', ncol=3, fontsize=8, frameon=False)
    ax.grid(axis='y', ls=':', alpha=0.4)
    fig.tight_layout()
    fig.savefig(FIG_DIR / 'ml_vs_dl_comparison.png', dpi=200, bbox_inches='tight')
    plt.close(fig)


def plot_controls(directory):
    """Report layout with legible axes at A4 text width."""
    fig = plt.figure(figsize=(8, 7.4))
    grid = fig.add_gridspec(2, 2, height_ratios=[1, 0.85])
    axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1]),
            fig.add_subplot(grid[1, :])]
    for task, color in [('target', '#d55e00'), ('distractor', '#0072b2')]:
        data = np.load(directory / f'{task}_oof.npz')
        summary = json.loads((directory / f'{task}_summary.json').read_text())
        fpr, tpr, _ = roc_curve(data['y'], data['oof_probs'])
        axes[0].plot(fpr, tpr, color=color,
                     label=f"{task.capitalize()}: AUC {summary['oof_roc_auc']:.3f}")
        axes[1].plot(data['times'] * 1000, data['saliency'].sum(axis=0), color=color,
                     label=f"{task.capitalize()}: {summary['saliency']['peak_ms']:.0f} ms")
        axes[2].scatter(data['effect'], data['saliency'].mean(axis=1), s=15,
                        alpha=0.65, color=color,
                        label=f"{task.capitalize()}: r={summary['saliency']['spatial_r']:.3f}")
    axes[0].plot([0, 1], [0, 1], 'k--', lw=1)
    axes[0].set(xlabel='False positive rate', ylabel='True positive rate',
                title='Pooled OOF discrimination')
    axes[1].axvspan(300, 600, color='0.9', zorder=0)
    axes[1].set(xlabel='Time (ms)', ylabel='Normalised saliency mass',
                title='Outer-test temporal saliency')
    axes[2].set(xlabel='Pooled OOF ERP difference, 300–600 ms (µV)',
                ylabel='Mean normalised saliency', title='Channel-wise correspondence')
    for ax in axes:
        ax.legend(fontsize=8)
        ax.tick_params(labelsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / 'eegnet_controls_oof.png', dpi=200, bbox_inches='tight')
    plt.close(fig)


def main():
    import mne

    directory = TABLE_DIR / 'eegnet_controls'
    _, y, groups, _ = load_features()
    verification = verify_dl_controls(y, groups)
    if not verification or set(verification) != {'target', 'distractor'}:
        raise RuntimeError('Both five-fold tasks must pass independent verification')
    target = np.load(directory / 'target_oof.npz')
    summary = json.loads((directory / 'target_summary.json').read_text())
    row = {k: v for k, v in summary.items() if k not in {
        'positive', 'n_trials', 'n_positive', 'n_subjects', 'saliency'}}
    row['model'] = 'EEGNet'
    pd.DataFrame([row]).to_csv(TABLE_DIR / 'eegnet_metrics_real.csv', index=False)
    fold_columns = ['fold', 'n_test', 'epochs_run', 'balanced_accuracy', 'roc_auc',
                    'pr_auc', 'f1_macro', 'f1_target', 'sensitivity', 'specificity']
    folds = pd.read_csv(directory / 'target_folds.csv')
    folds[fold_columns].to_csv(TABLE_DIR / 'eegnet_folds_real.csv', index=False)
    np.savez_compressed(TABLE_DIR / 'eegnet_oof_real.npz', y=y, groups=groups,
                        oof_probs=target['oof_probs'], fold_id=target['fold_id'])
    np.savez_compressed(TABLE_DIR / 'eegnet_saliency_real.npz',
                        saliency=target['saliency'], times=target['times'],
                        p300_effect=target['effect'], ch_names=target['ch_names'],
                        fold_saliency=target['fold_saliency'],
                        interpretation=np.array('outer-test positives; fold L1 normalisation; positive-count pooling'))

    # Use the final baseline run's saved OOF scores for paired comparisons.
    lda = np.load(TABLE_DIR / 'lda_channel_controls_oof.npz')
    np.testing.assert_array_equal(lda['y'], y)
    np.testing.assert_array_equal(lda['groups'], groups)
    baseline = np.load(TABLE_DIR / 'baseline_oof.npz')
    np.testing.assert_array_equal(baseline['y'], y)
    np.testing.assert_array_equal(baseline['groups'], groups)
    lr_probs = baseline['logistic_regression']
    predictions = {'EEGNet': target['oof_probs'], 'LDA': baseline['lda'],
                   'Logistic Regression': lr_probs}
    participants = []
    for group in np.unique(groups):
        mask = groups == group
        participant = {'subject': INCLUDED[int(group)]}
        for name, probs in predictions.items():
            participant['auc_' + name] = roc_auc_score(y[mask], probs[mask])
            participant['bacc_' + name] = balanced_accuracy_score(y[mask], probs[mask] >= 0.5)
        participants.append(participant)
    participants = pd.DataFrame(participants)
    participants.to_csv(TABLE_DIR / 'ml_vs_dl_per_subject.csv', index=False)
    paired = []
    for model in ['LDA', 'Logistic Regression']:
        for metric in ['auc', 'bacc']:
            delta = participants[metric + '_EEGNet'] - participants[metric + '_' + model]
            paired.append(dict(baseline=model, metric=metric, median_gain=float(delta.median()),
                               n_higher=int((delta > 0).sum()), p=float(wilcoxon(delta).pvalue)))
    (TABLE_DIR / 'ml_vs_dl_paired.json').write_text(json.dumps(paired, indent=2) + '\n')
    comparison = pd.concat([pd.read_csv(TABLE_DIR / 'baseline_metrics.csv'),
                            pd.DataFrame([row])], ignore_index=True)
    comparison.to_csv(TABLE_DIR / 'ml_vs_dl_comparison.csv', index=False)
    plot_comparison(comparison, y.mean())
    plot_controls(directory)
    plot_roc_pr_curves(y, target['oof_probs'], FIG_DIR / 'eegnet_roc_pr_real.png')
    plot_saliency_analysis(target['saliency'], target['times'], target['effect'],
                           FIG_DIR / 'eegnet_saliency_real.png')
    info = mne.read_epochs(epoch_path('sub-01'), preload=False, verbose='error').info
    plot_saliency_topomaps(target['saliency'], np.round(target['times'], 6), target['effect'], info,
                            FIG_DIR / 'eegnet_saliency_topomap_real.png')
    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    for name in ['LDA-12_bins', 'LDA-128_bins', 'LDA-402_features']:
        probs = baseline['lda'] if name == 'LDA-402_features' else lda[name]
        fpr, tpr, _ = roc_curve(y, probs)
        ax.plot(fpr, tpr, label=f'{name.replace("_", " ")}: pooled AUC {roc_auc_score(y, probs):.3f}')
    probs = target['oof_probs']
    fpr, tpr, _ = roc_curve(y, probs)
    ax.plot(fpr, tpr, label=f'EEGNet: pooled AUC {roc_auc_score(y, probs):.3f}')
    ax.plot([0, 1], [0, 1], 'k--', lw=1)
    ax.set_xlabel('False positive rate', fontsize=13)
    ax.set_ylabel('True positive rate', fontsize=13)
    ax.set_title('Participant-disjoint OOF channel controls', fontsize=14)
    ax.tick_params(labelsize=12)
    ax.legend(loc='lower right', fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / 'lda_channel_controls_roc.png', dpi=200)
    plt.close(fig)
    for name in ['eegnet_roc_pr_real.png', 'eegnet_saliency_real.png',
                 'eegnet_saliency_topomap_real.png', 'ml_vs_dl_comparison.png',
                 'lda_channel_controls_roc.png', 'eegnet_controls_oof.png',
                 'eegnet_controls_topomaps.png', 'baseline_roc_curves.png',
                 'baseline_pr_curves.png']:
        for destination in [ROOT / 'report/figures', ROOT / 'docs/report_figures']:
            shutil.copyfile(FIG_DIR / name, destination / name)
    print('Current EEGNet evidence and figures replaced with the completed five-fold run.')
    print(json.dumps(row, indent=2))


if __name__ == '__main__':
    main()
