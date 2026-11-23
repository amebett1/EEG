"""Check final-report evidence: python -m src.verify_report (no PyTorch needed)."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, roc_auc_score)
from sklearn.model_selection import GroupKFold

from config import FEATURE_DIR, TABLE_DIR
from src.erp.plot_erp import collect_group_erps
from src.features.extract_features import load_features
from src.features.rank_features import feature_auc


def metrics(y, p):
    pred = p >= 0.5
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return dict(balanced_accuracy=balanced_accuracy_score(y, pred),
                roc_auc=roc_auc_score(y, p), pr_auc=average_precision_score(y, p),
                f1_macro=f1_score(y, pred, average='macro'),
                sensitivity=tp / (tp + fn), specificity=tn / (tn + fp))


def verify_dl_controls(reference_y, reference_groups):
    directory = TABLE_DIR / 'eegnet_controls'
    if not directory.exists():
        return None
    results = {}
    reference_splits = list(GroupKFold(5).split(reference_y, reference_y, reference_groups))
    for positive in ['target', 'distractor']:
        data = np.load(directory / f'{positive}_oof.npz')
        folds = pd.read_csv(directory / f'{positive}_folds.csv')
        summary = json.loads((directory / f'{positive}_summary.json').read_text())
        y, groups, p = data['y'], data['groups'], data['oof_probs']
        assert len(folds) == 5 and set(np.unique(data['fold_id'])) == set(range(1, 6))
        assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
        if positive == 'target':
            np.testing.assert_array_equal(y, reference_y)
            np.testing.assert_array_equal(groups, reference_groups)
        for k, (_, original_test) in enumerate(reference_splits):
            mask = data['fold_id'] == k + 1
            np.testing.assert_array_equal(np.unique(groups[mask]),
                                          np.unique(reference_groups[original_test]))
            fold = np.load(directory / f'{positive}_fold_{k + 1}.npz')
            np.testing.assert_array_equal(fold['test_indices'], np.flatnonzero(mask))
            np.testing.assert_array_equal(fold['test_labels'], y[mask])
            np.testing.assert_allclose(fold['probs'], p[mask], atol=1e-12)
            np.testing.assert_allclose(fold['saliency'] / fold['saliency'].sum(),
                                       data['fold_saliency'][k], atol=1e-12)
            for key, value in metrics(y[mask], p[mask]).items():
                np.testing.assert_allclose(value, folds.iloc[k][key], atol=1e-12)
            np.testing.assert_allclose(folds.iloc[k]['n_positive'], y[mask].sum())
        for key, value in metrics(y, p).items():
            np.testing.assert_allclose(value, summary['oof_' + key], atol=1e-12)
            np.testing.assert_allclose(folds[key].mean(), summary[key + '_mean'], atol=1e-12)
            np.testing.assert_allclose(folds[key].std(), summary[key + '_std'], atol=1e-12)
        np.testing.assert_allclose(data['saliency'],
                                   np.average(data['fold_saliency'], axis=0,
                                              weights=folds['n_positive']), atol=1e-12)
        win = (np.round(data['times'] * 1000) >= 300) & (np.round(data['times'] * 1000) <= 600)
        mass = data['saliency'][:, win].sum() / data['saliency'].sum()
        spatial_r = np.corrcoef(data['saliency'].mean(axis=1), data['effect'])[0, 1]
        np.testing.assert_allclose(mass, summary['saliency']['mass_300_600'], atol=1e-12)
        np.testing.assert_allclose(spatial_r, summary['saliency']['spatial_r'], atol=1e-12)
        results[positive] = dict(passed=True, folds_verified=5, n_trials=len(y),
                                 n_positive=int(y.sum()), pooled=metrics(y, p),
                                 saliency=summary['saliency'])
    return results


def main():
    _, y, groups, _ = load_features()
    saved = np.load(TABLE_DIR / 'eegnet_oof_real.npz')
    np.testing.assert_array_equal(saved['y'], y)
    np.testing.assert_array_equal(saved['groups'], groups)
    folds = pd.DataFrame([metrics(y[te], saved['oof_probs'][te])
                          for _, te in GroupKFold(5).split(y, y, groups)])
    stored = pd.read_csv(TABLE_DIR / 'eegnet_folds_real.csv')
    summary = pd.read_csv(TABLE_DIR / 'eegnet_metrics_real.csv').iloc[0]
    pooled = metrics(y, saved['oof_probs'])
    for key in folds:
        np.testing.assert_allclose(folds[key], stored[key], atol=1e-12)
        np.testing.assert_allclose(folds[key].mean(), summary[key + '_mean'], atol=1e-12)
        np.testing.assert_allclose(folds[key].std(), summary[key + '_std'], atol=1e-12)
        np.testing.assert_allclose(pooled[key], summary['oof_' + key], atol=1e-12)
    participant_table = pd.read_csv(TABLE_DIR / 'ml_vs_dl_per_subject.csv')
    baseline = np.load(TABLE_DIR / 'baseline_oof.npz')
    np.testing.assert_array_equal(baseline['y'], y)
    np.testing.assert_array_equal(baseline['groups'], groups)
    baseline_table = pd.read_csv(TABLE_DIR / 'baseline_metrics.csv')
    for _, row in baseline_table.iterrows():
        name = row['model'].lower().replace(' ', '_')
        scores = baseline[name]
        actual = pd.DataFrame([metrics(y[te], scores[te])
                               for _, te in GroupKFold(5).split(y, y, groups)])
        for key in actual:
            np.testing.assert_allclose(actual[key].mean(), row[key + '_mean'], atol=1e-12)
            np.testing.assert_allclose(actual[key].std(), row[key + '_std'], atol=1e-12)
    for group, (_, row) in zip(np.unique(groups), participant_table.iterrows()):
        mask = groups == group
        actual = metrics(y[mask], saved['oof_probs'][mask])
        np.testing.assert_allclose(actual['roc_auc'], row['auc_EEGNet'], atol=1e-12)
        np.testing.assert_allclose(actual['balanced_accuracy'], row['bacc_EEGNet'], atol=1e-12)
        for model in ['LDA', 'Logistic Regression']:
            scores = baseline[model.lower().replace(' ', '_')]
            actual = metrics(y[mask], scores[mask])
            np.testing.assert_allclose(actual['roc_auc'], row['auc_' + model], atol=1e-12)
            np.testing.assert_allclose(actual['balanced_accuracy'], row['bacc_' + model], atol=1e-12)
    saved_paired = json.loads((TABLE_DIR / 'ml_vs_dl_paired.json').read_text())
    assert len(saved_paired) == 4
    for row in saved_paired:
        delta = (participant_table[row['metric'] + '_EEGNet']
                 - participant_table[row['metric'] + '_' + row['baseline']])
        np.testing.assert_allclose(delta.median(), row['median_gain'], atol=1e-12)
        np.testing.assert_allclose(stats.wilcoxon(delta).pvalue, row['p'], atol=1e-12)
        assert int((delta > 0).sum()) == row['n_higher']
    paired = {}
    for model in ['LDA', 'Logistic Regression']:
        d = participant_table['auc_EEGNet'] - participant_table['auc_' + model]
        paired[model] = dict(auc_median_gain=float(d.median()),
                             n_higher=int((d > 0).sum()),
                             wilcoxon_p=float(stats.wilcoxon(d).pvalue))
    feature_table = pd.read_csv(FEATURE_DIR / 'all_subjects_features.csv')
    ranking = feature_auc(feature_table).set_index('feature').sort_index()
    old_ranking = pd.read_csv(TABLE_DIR / 'v4_feature_auc.csv').set_index('feature').sort_index()
    np.testing.assert_allclose(ranking[['auc', 'strength']], old_ranking[['auc', 'strength']], atol=1e-12)
    erp_table, evokeds = collect_group_erps()
    old_table = pd.read_csv(TABLE_DIR / 'v3_p300_group_results.csv')
    pd.testing.assert_frame_equal(erp_table, old_table, check_exact=False, atol=1e-10, rtol=1e-10)
    included = erp_table[~erp_table.task_misunderstood]
    delta = included.oddball_diff_uv.to_numpy()
    t_result = stats.ttest_1samp(delta, 0)
    old_erp = json.loads((TABLE_DIR / 'v3_p300_summary.json').read_text())
    for key, value in dict(difference_mean_uv=delta.mean(), difference_sd_uv=delta.std(ddof=1),
                           paired_t=t_result.statistic, paired_t_p=t_result.pvalue,
                           cohens_dz=delta.mean() / delta.std(ddof=1)).items():
        np.testing.assert_allclose(value, old_erp[key], atol=1e-12)
    # Exploratory distractor contrasts; retain participants as statistical units.
    channels = ['Fz', 'Cz', 'Pz']
    means = {}
    for condition, values in evokeds.items():
        means[condition] = np.array([ev.copy().pick(channels).crop(0.3, 0.5).data.mean(axis=1) * 1e6
                                    for ev in values])
    p3a = {}
    for j, ch in enumerate(channels):
        d = means['distractor'][:, j] - means['standard'][:, j]
        test = stats.ttest_1samp(d, 0)
        p3a[ch] = dict(difference_mean_uv=float(d.mean()), difference_sd_uv=float(d.std(ddof=1)),
                       paired_t=float(test.statistic), p=float(test.pvalue))
    relative = ((means['distractor'][:, 0] - means['distractor'][:, 2])
                - (means['target'][:, 0] - means['target'][:, 2]))
    p3a['relative_frontal'] = dict(mean_uv=float(relative.mean()), sd_uv=float(relative.std(ddof=1)),
                                  positive_count=int((relative > 0).sum()))
    sal = np.load(TABLE_DIR / 'eegnet_saliency_real.npz')
    a, times = sal['saliency'], sal['times']
    win = (np.round(times * 1000) >= 300) & (np.round(times * 1000) <= 600)
    reference = dict(peak_ms=float(times[np.argmax(a.mean(axis=0))] * 1000),
                     mass_300_600=float(a[:, win].sum() / a.sum()),
                     spatial_r=float(np.corrcoef(a.mean(axis=1), sal['p300_effect'])[0, 1]))
    result = dict(passed=True, eegnet_pooled=pooled, eegnet_folds_verified=5,
                  baseline_models_verified=len(baseline_table),
                  paired_participant_comparisons=paired,
                  eegnet_folds_at_50_epochs=int((stored.epochs_run == 50).sum()),
                  ranked_features_verified=len(ranking), erp_participants_verified=len(erp_table),
                  p3a_300_500=p3a, outer_test_saliency=reference)
    result['new_eegnet_controls'] = verify_dl_controls(y, groups)
    assert result['new_eegnet_controls'] is not None
    np.testing.assert_allclose(reference['spatial_r'],
                               result['new_eegnet_controls']['target']['saliency']['spatial_r'], atol=1e-12)
    (TABLE_DIR / 'report_verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
