"""Check final-report evidence: python -m src.verify_report (no PyTorch needed)."""
from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, roc_auc_score)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

from config import FEATURE_DIR, FIG_DIR, ROOT, TABLE_DIR
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
    for group, (_, row) in zip(np.unique(groups), participant_table.iterrows()):
        mask = groups == group
        actual = metrics(y[mask], saved['oof_probs'][mask])
        np.testing.assert_allclose(actual['roc_auc'], row['auc_EEGNet'], atol=1e-12)
        np.testing.assert_allclose(actual['balanced_accuracy'], row['bacc_EEGNet'], atol=1e-12)
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
    win = (times >= 0.3) & (times <= 0.6)
    reference = dict(peak_ms=float(times[np.argmax(a.mean(axis=0))] * 1000),
                     mass_300_600=float(a[:, win].sum() / a.sum()),
                     spatial_r=float(np.corrcoef(a.mean(axis=1), sal['p300_effect'])[0, 1]))
    # Notebook output is historical evidence, not a newly rerun saliency experiment.
    nb_path = ROOT / 'notebooks/04_deep_learning.ipynb'
    nb = json.loads(nb_path.read_text())
    cell = next(c for c in nb['cells'] if 'show_saliency(sal,' in ''.join(c['source']))
    texts = [''.join(o['text']) for o in cell['outputs'] if 'text' in o]
    if not texts or not all(v in texts[0] for v in ['508 ms', '40.5%', 'r = 0.15']):
        raise ValueError('Held-out notebook saliency output changed; review report values')
    pictures = [o['data']['image/png'] for o in cell['outputs'] if 'image/png' in o.get('data', {})]
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    (FIG_DIR / 'eegnet_saliency_holdout_notebook.png').write_bytes(base64.b64decode(pictures[0]))
    _, te = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42).split(y, y, groups))
    result = dict(passed=True, eegnet_pooled=pooled, eegnet_folds_verified=5,
                  paired_participant_comparisons=paired,
                  eegnet_folds_at_50_epochs=int((stored.epochs_run == 50).sum()),
                  ranked_features_verified=len(ranking), erp_participants_verified=len(erp_table),
                  p3a_300_500=p3a, reference_saliency=reference,
                  holdout_saliency=dict(source=str(nb_path.relative_to(ROOT)), rerun=False,
                                       n_test=len(te), n_subjects=len(np.unique(groups[te])),
                                       n_target=int(y[te].sum()), peak_ms=508,
                                       mass_300_600=0.405, spatial_r=0.15,
                                       correlation_reference='whole-cohort trial-pooled ERP effect'))
    (TABLE_DIR / 'report_verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
