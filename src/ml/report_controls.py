"""Reproduce final-report controls without PyTorch or raw-data cleaning.

Run: python -m src.ml.report_controls
The fixed 100 ms bins cover the full epoch. No test-driven tuning is performed.
"""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from config import FIG_DIR, SUBJECTS, TABLE_DIR, TASK_MISUNDERSTOOD
from src.features.extract_features import KEY_CHANNELS, load_features
from src.ml.train_baseline import evaluate_model, get_models
from src.utils.io import load_epochs


def main():
    subjects = [s for s in SUBJECTS if s not in TASK_MISUNDERSTOOD]
    matrices, labels, groups = [], [], []
    channel_names = None
    for i, sub in enumerate(subjects):
        ep = load_epochs(sub)
        if channel_names is None:
            channel_names = ep.ch_names
        if ep.ch_names != channel_names:
            raise ValueError(f"Channel order mismatch: {sub}")
        data = ep.get_data(copy=False) * 1e6
        # Sample indices avoid floating-point ambiguity at bin boundaries.
        # Bins are [-200,-100), ..., [900,1000] ms (last includes endpoint).
        means = np.stack([data[:, :, j:(j + 25 if j < 275 else 301)].mean(axis=2)
                          for j in range(0, 300, 25)], axis=2)
        matrices.append(means)
        labels.append((ep.events[:, 2] == ep.event_id['target']).astype(int))
        groups.append(np.full(len(ep), i))
        print(f"{sub}: {len(ep)} trials", flush=True)
    binned, y, g = np.concatenate(matrices), np.concatenate(labels), np.concatenate(groups)
    features, fy, fg, _ = load_features()
    saved = np.load(TABLE_DIR / 'eegnet_oof_real.npz')
    for actual, expected in [(y, fy), (g, fg), (y, saved['y']), (g, saved['groups'])]:
        np.testing.assert_array_equal(actual, expected)
    picks = [channel_names.index(ch) for ch in KEY_CHANNELS]
    inputs = {'LDA-12 bins': binned[:, picks].reshape(len(y), -1),
              'LDA-128 bins': binned.reshape(len(y), -1),
              'LDA-402 features': features}
    rows, predictions = [], {}
    for name, x in inputs.items():
        print(f"Evaluating {name}: {x.shape}", flush=True)
        row, _, probs = evaluate_model(name, get_models()['LDA'], x, y, g)
        row.update(n_channels=128 if name == 'LDA-128 bins' else 12, n_features=x.shape[1])
        rows.append(row)
        predictions[name.replace(' ', '_')] = probs
        print(row, flush=True)
    # Fail if the historical LDA result cannot be reproduced to report precision.
    old = pd.read_csv(TABLE_DIR / 'baseline_metrics.csv').set_index('model').loc['LDA']
    for key in old.index:
        np.testing.assert_allclose(rows[-1][key], old[key], atol=5e-4, rtol=0)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(TABLE_DIR / 'lda_channel_controls.csv', index=False)
    np.savez_compressed(TABLE_DIR / 'lda_channel_controls_oof.npz', y=y, groups=g, **predictions)
    splits = list(GroupKFold(5).split(np.zeros(len(y)), y, g))
    from sklearn.metrics import roc_curve, roc_auc_score
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, probs in [(k.replace('_', ' '), v) for k, v in predictions.items()] + [('EEGNet', saved['oof_probs'])]:
        fpr, tpr, _ = roc_curve(y, probs)
        ax.plot(fpr, tpr, label=f'{name}: pooled AUC {roc_auc_score(y, probs):.3f}')
    ax.plot([0, 1], [0, 1], 'k--', lw=1)
    ax.set(xlabel='False positive rate', ylabel='True positive rate', title='Participant-disjoint OOF channel controls')
    ax.legend(loc='lower right', fontsize=9)
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / 'lda_channel_controls_roc.png', dpi=200)
    plt.close(fig)
    details = {'n_trials': len(y), 'n_target': int(y.sum()), 'n_participants': len(subjects),
               'bin_edges_ms': list(range(-200, 1001, 100)), 'last_bin_includes_1000ms': True,
               'splitter': 'GroupKFold(n_splits=5)', 'standardisation': 'outer train only',
               'classifier': "LDA(solver='lsqr', shrinkage='auto'), empirical priors",
               'alignment': 'epoch/feature/EEGNet labels and groups identical row by row',
               'folds': [{'fold': k + 1, 'test_subjects': [subjects[i] for i in np.unique(g[te])],
                          'n_test': len(te)} for k, (_, te) in enumerate(splits)]}
    (TABLE_DIR / 'lda_channel_controls.json').write_text(json.dumps(details, indent=2) + '\n')
    print('Saved channel controls; historical LDA reproduced.', flush=True)


if __name__ == '__main__':
    main()
