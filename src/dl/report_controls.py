"""Train no-button EEGNet control and explain every outer test fold.

Run: python -u -m src.dl.report_controls
Uses the original target-task outer participant folds for both tasks. Completed
folds can be resumed; historical benchmark outputs are never overwritten.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import platform
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from config import (EVENT_ID, FIG_DIR, MODEL_DIR, RANDOM_SEED, SUBJECTS,
                    TABLE_DIR, TASK_MISUNDERSTOOD)
from src.dl.train_eegnet import (compute_metrics, get_splits, predict_proba,
                                saliency_map, summarize, train_eegnet)
from src.utils.io import epoch_path, load_epochs

CONTROL_DIR = TABLE_DIR / "eegnet_controls"
INCLUDED = [s for s in SUBJECTS if s not in TASK_MISUNDERSTOOD]


def task_labels(events, positive):
    if positive not in ("target", "distractor"):
        raise ValueError("Positive condition must be target or distractor")
    if not np.isin(events[:, 2], [EVENT_ID["standard"], EVENT_ID[positive]]).all():
        raise ValueError("Unexpected condition in binary task")
    return (events[:, 2] == EVENT_ID[positive]).astype(np.int64)


def load_task(positive):
    arrays, labels, groups = [], [], []
    info, times = None, None
    for i, sub in enumerate(INCLUDED):
        ep = load_epochs(sub, conditions=("standard", positive))
        if info is None:
            info, times = ep.info.copy(), ep.times.copy()
        if ep.ch_names != info.ch_names or not np.array_equal(times, ep.times):
            raise ValueError(f"Channel/time mismatch: {sub}")
        if not ep.metadata["correct"].all():
            raise ValueError(f"Incorrect trials in {sub}")
        if positive == "distractor" and ep.metadata["responded"].any():
            raise ValueError(f"Button response in no-button control: {sub}")
        arrays.append((ep.get_data(copy=False) * 1e6).astype(np.float32))
        labels.append(task_labels(ep.events, positive))
        groups.append(np.full(len(ep), i, dtype=np.int64))
        print(f"Loaded {positive}: {sub}, {len(ep)} trials", flush=True)
    return np.concatenate(arrays), np.concatenate(labels), np.concatenate(groups), times, info


def matched_splits(groups, reference_y, reference_groups):
    splits, _ = get_splits(reference_y, reference_groups)
    if set(np.unique(groups)) != set(np.unique(reference_groups)):
        raise ValueError("Tasks must contain the same participants")
    return [(np.flatnonzero(~np.isin(groups, np.unique(reference_groups[te]))),
             np.flatnonzero(np.isin(groups, np.unique(reference_groups[te]))))
            for _, te in splits]


def describe_saliency(sal, times, effect, ch_names):
    win = (np.round(times * 1000) >= 300) & (np.round(times * 1000) <= 600)
    spatial = sal.mean(axis=1)
    order = np.argsort(spatial)[::-1]
    return dict(peak_ms=float(times[np.argmax(sal.mean(axis=0))] * 1000),
                mass_300_600=float(sal[:, win].sum() / sal.sum()),
                window_samples=int(win.sum()),
                spatial_r=float(np.corrcoef(spatial, effect)[0, 1]),
                pz_rank=int(np.flatnonzero(order == ch_names.index("Pz"))[0] + 1),
                top_channels=[ch_names[i] for i in order[:10]])


def run_task(positive, args, reference_y, reference_groups):
    X, y, groups, times, info = load_task(positive)
    if positive == "target":
        np.testing.assert_array_equal(y, reference_y)
        np.testing.assert_array_equal(groups, reference_groups)
    splits = matched_splits(groups, reference_y, reference_groups)
    oof = np.full(len(y), np.nan)
    fold_id = np.full(len(y), -1)
    fold_rows, fold_sals, fold_effects, n_positive = [], [], [], []
    win = (np.round(times * 1000) >= 300) & (np.round(times * 1000) <= 600)
    for k, (tr, te) in enumerate(splits):
        checkpoint = CONTROL_DIR / f"{positive}_fold_{k + 1}.npz"
        record = CONTROL_DIR / f"{positive}_fold_{k + 1}.json"
        test_subjects = [INCLUDED[i] for i in np.unique(groups[te])]
        if np.intersect1d(groups[tr], groups[te]).size:
            raise ValueError("Participant leakage")
        if checkpoint.exists() and record.exists():
            saved = np.load(checkpoint)
            np.testing.assert_array_equal(saved["test_indices"], te)
            np.testing.assert_array_equal(saved["test_labels"], y[te])
            probs, sal, effect = saved["probs"], saved["saliency"], saved["effect"]
            row = json.loads(record.read_text())
            print(f"Resumed {positive} fold {k + 1}", flush=True)
        else:
            t0 = time.monotonic()
            tr_idx, va_idx = train_test_split(tr, test_size=0.15, stratify=y[tr],
                                             random_state=RANDOM_SEED + k)
            print(f"START {positive} fold {k + 1}: train={len(tr_idx)}, val={len(va_idx)}, "
                  f"test={len(te)}, subjects={test_subjects}", flush=True)
            model, history, scale = train_eegnet(
                X[tr_idx], y[tr_idx], X[va_idx], y[va_idx],
                epochs=args.epochs, batch_size=args.batch_size, eval_batch_size=args.batch_size,
                patience=15, lr=1e-3, seed=RANDOM_SEED + k,
                device=torch.device("cuda"), verbose=True)
            probs = predict_proba(model, X[te], scale, batch_size=args.batch_size)
            positive_indices = te[y[te] == 1]
            sal = saliency_map(model, X[positive_indices], scale,
                               batch_size=min(32, args.batch_size))
            effect = (X[positive_indices][:, :, win].mean(axis=(0, 2))
                      - X[te[y[te] == 0]][:, :, win].mean(axis=(0, 2)))
            row = dict(fold=k + 1, n_test=len(te), n_positive=len(positive_indices),
                       epochs_run=len(history["train_loss"]), scale=scale,
                       seconds=time.monotonic() - t0, test_subjects=test_subjects,
                       **compute_metrics(y[te], probs),
                       **describe_saliency(sal, times, effect, info.ch_names))
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            torch.save(dict(state_dict=model.state_dict(), scale=scale, positive=positive,
                            test_subjects=test_subjects),
                       MODEL_DIR / f"eegnet_control_{positive}_fold_{k + 1}.pt")
            np.savez_compressed(checkpoint, test_indices=te, test_labels=y[te], probs=probs,
                                saliency=sal, effect=effect)
            record.write_text(json.dumps(row, indent=2) + "\n")
            (CONTROL_DIR / f"{positive}_history_{k + 1}.json").write_text(json.dumps(history) + "\n")
            del model
            gc.collect()
            torch.cuda.empty_cache()
            print(f"DONE {positive} fold {k + 1}: AUC={row['roc_auc']:.4f}, "
                  f"BAcc={row['balanced_accuracy']:.4f}, saliency r={row['spatial_r']:.3f}, "
                  f"time={row['seconds']:.0f}s", flush=True)
        oof[te], fold_id[te] = probs, k + 1
        fold_rows.append(row)
        fold_sals.append(sal / sal.sum())
        fold_effects.append(effect)
        n_positive.append(int(y[te].sum()))
    if not np.isfinite(oof).all() or (fold_id < 1).any():
        raise ValueError("Missing OOF predictions")
    # Unit-mass normalisation prevents logit/gradient scale differences between
    # independently trained models from dominating the pooled explanation.
    pooled_sal = np.average(fold_sals, axis=0, weights=n_positive)
    pooled_effect = X[y == 1][:, :, win].mean(axis=(0, 2)) - X[y == 0][:, :, win].mean(axis=(0, 2))
    folds = pd.DataFrame(fold_rows)
    summary = summarize(f"EEGNet-{positive}-control", folds, y, oof)
    summary.update(positive=positive, n_trials=len(y), n_positive=int(y.sum()),
                   n_subjects=len(INCLUDED),
                   saliency=describe_saliency(pooled_sal, times, pooled_effect, info.ch_names))
    folds.to_csv(CONTROL_DIR / f"{positive}_folds.csv", index=False)
    np.savez_compressed(CONTROL_DIR / f"{positive}_oof.npz", y=y, groups=groups, fold_id=fold_id,
                        oof_probs=oof, saliency=pooled_sal, times=times,
                        effect=pooled_effect, fold_saliency=np.array(fold_sals),
                        fold_effect=np.array(fold_effects), ch_names=np.array(info.ch_names))
    (CONTROL_DIR / f"{positive}_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    del X
    gc.collect()
    return summary


def plot_results():
    from sklearn.metrics import roc_curve
    import mne
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for positive, color in [("target", "#d55e00"), ("distractor", "#0072b2")]:
        data = np.load(CONTROL_DIR / f"{positive}_oof.npz")
        summary = json.loads((CONTROL_DIR / f"{positive}_summary.json").read_text())
        fpr, tpr, _ = roc_curve(data["y"], data["oof_probs"])
        axes[0].plot(fpr, tpr, color=color,
                     label=f"{positive}: pooled AUC {summary['oof_roc_auc']:.3f}")
        axes[1].plot(data["times"] * 1000, data["saliency"].sum(axis=0), color=color,
                     label=f"{positive}: peak {summary['saliency']['peak_ms']:.0f} ms")
        axes[2].scatter(data["effect"], data["saliency"].mean(axis=1), color=color, s=12,
                        alpha=0.65, label=f"{positive}: r={summary['saliency']['spatial_r']:.2f}")
    axes[0].plot([0, 1], [0, 1], "k--", lw=1)
    axes[0].set(xlabel="False positive rate", ylabel="True positive rate", title="OOF discrimination")
    axes[1].axvspan(300, 600, color="0.9", zorder=0)
    axes[1].set(xlabel="Time (ms)", ylabel="Normalised saliency mass", title="Outer-test saliency")
    axes[2].set(xlabel="Pooled held-out ERP difference (µV)", ylabel="Normalised mean saliency",
                title="Channel-wise correspondence")
    for ax in axes:
        ax.legend(fontsize=8)
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "eegnet_controls_oof.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    # Separate topographies for each task, using the same montage as the epochs.
    info = mne.read_epochs(epoch_path(INCLUDED[0]), preload=False, verbose="error").info
    fig, axes = plt.subplots(2, 2, figsize=(9, 8))
    for row, positive in enumerate(["target", "distractor"]):
        data = np.load(CONTROL_DIR / f"{positive}_oof.npz")
        win = (np.round(data["times"] * 1000) >= 300) & (np.round(data["times"] * 1000) <= 600)
        for col, (values, cmap, label) in enumerate([
                (data["saliency"][:, win].mean(axis=1), "magma", "normalised saliency"),
                (data["effect"], "RdBu_r", "ERP difference (µV)")]):
            im, _ = mne.viz.plot_topomap(values, info, axes=axes[row, col], show=False, cmap=cmap)
            fig.colorbar(im, ax=axes[row, col], fraction=0.04)
            axes[row, col].set_title(f"{positive} vs standard: {label}", fontsize=10)
    fig.suptitle("All five outer test folds, 300–600 ms")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "eegnet_controls_topomaps.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("GPU unavailable: run in the GPU-enabled execution environment")
    torch.set_num_threads(4)
    CONTROL_DIR.mkdir(parents=True, exist_ok=True)
    reference = np.load(TABLE_DIR / "eegnet_oof_real.npz")
    manifest = dict(epochs=args.epochs, batch_size=args.batch_size, eval_batch_size=args.batch_size,
                    lr=1e-3, patience=15, val_size=0.15, seed=RANDOM_SEED,
                    inner_split="stratified trials, matching original benchmark",
                    outer_split="original target GroupKFold participants, reused for both tasks",
                    subjects=INCLUDED, torch=torch.__version__, cuda=torch.version.cuda,
                    gpu=torch.cuda.get_device_name(0), python=platform.python_version(),
                    saliency="outer-test positives only; fold L1 normalisation, positive-count weighting",
                    erp_reference="test-only within each fold; pooled test trials for aggregate",
                    code_sha256={name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
                                 for name in [__file__, str(Path(__file__).with_name("train_eegnet.py"))]},
                    epochs_inventory={s: dict(size=epoch_path(s).stat().st_size,
                                              mtime_ns=epoch_path(s).stat().st_mtime_ns) for s in INCLUDED})
    path = CONTROL_DIR / "manifest.json"
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError("Existing control configuration differs; preserve outputs and use a separate run")
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)
    for positive in ["target", "distractor"]:
        run_task(positive, args, reference["y"], reference["groups"])
    plot_results()
    print("Both tasks complete; all explanations use unseen outer-test participants.", flush=True)


if __name__ == "__main__":
    main()
