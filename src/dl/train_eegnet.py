"""V6: EEGNet cho phân loại P300 đơn trial (standard vs target).

EEGNet-8,2 theo Lawhern et al. (2018), viết bằng PyTorch thuần (không cần braindecode).
Đầu vào: epoch thô (n_trials, 128 kênh, 301 mẫu @ 250 Hz), đơn vị µV.

Đánh giá giống V5 (src/ml/train_baseline.py) để so sánh trực tiếp:
GroupKFold theo người khi đủ người, StratifiedKFold khi chỉ có 1 người.

Chạy thử:
    python src/dl/train_eegnet.py --synthetic            # dữ liệu mô phỏng
    python src/dl/train_eegnet.py --subjects sub-01      # 1 người thật (cần epoch + mne)
"""
from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    GroupKFold,
    StratifiedKFold,
    train_test_split,
)

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import (
    FIG_DIR,
    MODEL_DIR,
    N_CHANNELS,
    RANDOM_SEED,
    SFREQ,
    TABLE_DIR,
    TMAX,
    TMIN,
)

N_TIMES = int(round((TMAX - TMIN) * SFREQ)) + 1  # 301


# ---------------------------------------------------------------------------
# Mô hình
# ---------------------------------------------------------------------------
class EEGNet(nn.Module):
    """EEGNet (Lawhern et al., 2018).

    Block 1: Conv thời gian (F1 bộ lọc, dài kern_length) -> BN
             -> Depthwise conv không gian (D bộ lọc/kênh thời gian, max-norm 1) -> BN -> ELU
             -> AvgPool(1, 4) -> Dropout
    Block 2: Separable conv (depthwise 1x16 + pointwise F2) -> BN -> ELU -> AvgPool(1, 8) -> Dropout
    Phân loại: Flatten -> Linear (max-norm 0.25)
    """

    def __init__(self, n_channels: int = N_CHANNELS, n_times: int = N_TIMES, n_classes: int = 2,
                 F1: int = 8, D: int = 2, F2: Optional[int] = None, kern_length: int = SFREQ // 2,
                 dropout: float = 0.5):
        super().__init__()
        F2 = F2 or F1 * D
        self.n_channels, self.n_times = n_channels, n_times

        self.temporal = nn.Sequential(
            nn.Conv2d(1, F1, (1, kern_length), padding=(0, kern_length // 2), bias=False),
            nn.BatchNorm2d(F1),
        )
        self.spatial = nn.Conv2d(F1, F1 * D, (n_channels, 1), groups=F1, bias=False)
        self.block1_rest = nn.Sequential(
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8), groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, 1, bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout),
        )
        with torch.no_grad():
            n_flat = self._features(torch.zeros(1, 1, n_channels, n_times)).shape[1]
        self.classifier = nn.Linear(n_flat, n_classes)

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.temporal(x)
        x = self.spatial(x)
        x = self.block1_rest(x)
        x = self.block2(x)
        return x.flatten(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:  # (B, C, T) -> (B, 1, C, T)
            x = x.unsqueeze(1)
        return self.classifier(self._features(x))

    @torch.no_grad()
    def apply_max_norm(self) -> None:
        """Ràng buộc max-norm như bản gốc: bộ lọc không gian <= 1, lớp dense <= 0.25."""
        for layer, max_val, dim in ((self.spatial, 1.0, (1, 2, 3)), (self.classifier, 0.25, 1)):
            w = layer.weight
            norm = w.norm(2, dim=dim, keepdim=True).clamp(min=max_val)
            w.mul_(max_val / norm)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Dữ liệu
# ---------------------------------------------------------------------------
def load_real_dataset(subjects: Optional[Sequence[str]] = None
                      ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Đọc epoch thật qua src.utils.io (cần mne và derivatives/v1/epochs). Trả về X theo µV."""
    from src.utils.io import load_dataset  # import muộn: mne chỉ cần khi có dữ liệu thật
    X, y, groups = load_dataset(subjects)
    return (X * 1e6).astype(np.float32), y.astype(np.int64), groups


def real_data_available(subjects: Optional[Sequence[str]] = None) -> bool:
    try:
        import mne  # noqa: F401
        from src.utils.io import epoch_path
    except ImportError:
        return False
    from config import SUBJECTS
    return all(epoch_path(s).exists() for s in (subjects or SUBJECTS))


def make_synthetic_dataset(n_subjects: int = 5, n_trials: int = 400, target_ratio: float = 0.12,
                           n_channels: int = N_CHANNELS, n_times: int = N_TIMES, sfreq: int = SFREQ,
                           p300_amp: float = 3.0, noise_amp: float = 10.0,
                           seed: int = RANDOM_SEED) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Dữ liệu thử cùng định dạng epoch thật: (trial, 128 kênh, 301 mẫu), µV.

    - Mọi trial: N1/P2 thị giác (~100/200 ms) ở vùng chẩm + nhiễu 1/f + alpha ~10 Hz.
    - Trial target: thêm sóng P300 (~350-450 ms, trễ dao động từng trial) phân bố vùng đỉnh.
    - Mỗi người có biên độ, độ trễ và phân bố không gian hơi khác nhau -> có ý nghĩa khi chia theo người.
    SNR thấp có chủ ý (P300 ~3 µV so với nhiễu ~10 µV) để bài toán không quá dễ.
    """
    rng = np.random.default_rng(seed)
    t = TMIN + np.arange(n_times) / sfreq
    ch = np.arange(n_channels)

    X = np.empty((n_subjects * n_trials, n_channels, n_times), dtype=np.float32)
    y = np.zeros(n_subjects * n_trials, dtype=np.int64)
    groups = np.repeat(np.arange(n_subjects), n_trials)

    for s in range(n_subjects):
        parietal_center = 80 + rng.normal(0, 4)
        p_topo = np.exp(-0.5 * ((ch - parietal_center) / 18) ** 2)       # "vùng đỉnh"
        o_topo = np.exp(-0.5 * ((ch - 105) / 15) ** 2)                    # "vùng chẩm"
        amp_s = p300_amp * rng.uniform(0.7, 1.3)
        lat_s = rng.uniform(0.35, 0.45)
        mixing = rng.normal(0, 1, (n_channels, 16)) / 4                   # nhiễu có tương quan giữa các kênh

        sl = slice(s * n_trials, (s + 1) * n_trials)
        y_s = (rng.random(n_trials) < target_ratio).astype(np.int64)
        y[sl] = y_s

        for i in range(n_trials):
            # Nhiễu 1/f: tích lũy nhiễu trắng rồi bỏ xu hướng, trộn qua các nguồn chung
            src = np.cumsum(rng.normal(0, 1, (16, n_times)), axis=1)
            src -= src.mean(axis=1, keepdims=True)
            src /= src.std(axis=1, keepdims=True) + 1e-6
            noise = mixing @ src + 0.5 * rng.normal(0, 1, (n_channels, n_times))
            alpha = np.sin(2 * np.pi * rng.uniform(9, 11) * t + rng.uniform(0, 2 * np.pi))
            trial = noise_amp * 0.6 * noise + noise_amp * 0.4 * o_topo[:, None] * alpha

            vep = -4 * np.exp(-0.5 * ((t - 0.10) / 0.02) ** 2) + 3 * np.exp(-0.5 * ((t - 0.20) / 0.03) ** 2)
            trial += o_topo[:, None] * vep

            if y_s[i]:
                lat = lat_s + rng.normal(0, 0.04)
                trial += amp_s * rng.uniform(0.5, 1.5) * p_topo[:, None] * np.exp(-0.5 * ((t - lat) / 0.08) ** 2)

            trial -= trial[:, t < 0].mean(axis=1, keepdims=True)          # baseline -200..0 ms
            X[sl.start + i] = trial

    return X, y, groups


# ---------------------------------------------------------------------------
# Huấn luyện
# ---------------------------------------------------------------------------
def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int = RANDOM_SEED) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    else:
        torch.set_num_threads(8)


def _batches(n: int, batch_size: int, shuffle: bool, rng: np.random.Generator):
    idx = rng.permutation(n) if shuffle else np.arange(n)
    for i in range(0, n, batch_size):
        yield idx[i:i + batch_size]


@torch.no_grad()
def predict_proba(model: nn.Module, X: np.ndarray, scale: float = 1.0, batch_size: int = 256,
                  device: Optional[torch.device] = None) -> np.ndarray:
    device = device or next(model.parameters()).device
    model.eval()
    out = []
    for i in range(0, len(X), batch_size):
        xb = torch.from_numpy(X[i:i + batch_size] / scale).float().to(device)
        out.append(torch.softmax(model(xb), dim=1)[:, 1].cpu().numpy())
    return np.concatenate(out)


def train_eegnet(X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray,
                 epochs: int = 100, batch_size: int = 64, lr: float = 1e-3, patience: int = 15,
                 model_kwargs: Optional[dict] = None, device: Optional[torch.device] = None,
                 seed: int = RANDOM_SEED, verbose: bool = False,
                 eval_batch_size: int = 256) -> Tuple[EEGNet, Dict[str, List[float]], float]:
    """Huấn luyện một EEGNet, early stopping theo val loss.

    - Lỗi cross-entropy có trọng số lớp (bù mất cân bằng ~12% target, giống class_weight="balanced").
    - Chuẩn hóa: chia cho độ lệch chuẩn toàn cục của tập train (không rò rỉ sang val/test).
    Trả về (model tốt nhất, lịch sử, scale) — dùng lại scale khi dự đoán.
    """
    set_seed(seed)
    device = device or get_device()
    rng = np.random.default_rng(seed)

    scale = float(X_train.std()) or 1.0
    model = EEGNet(n_channels=X_train.shape[1], n_times=X_train.shape[2], **(model_kwargs or {})).to(device)

    counts = np.bincount(y_train, minlength=2).astype(np.float32)
    weights = torch.tensor(counts.sum() / (2 * np.maximum(counts, 1)), device=device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    Xtr = torch.from_numpy(X_train / scale).float()
    ytr = torch.from_numpy(y_train).long()
    # Val giữ ở CPU và đánh giá theo batch: forward cả tập val một lần trên GPU
    # (vài nghìn trial × 128 × 301) có thể vượt VRAM khi chạy đủ 39 người.
    Xva = torch.from_numpy(X_val / scale).float()
    yva = torch.from_numpy(y_val).long()

    history = {"train_loss": [], "val_loss": [], "val_auc": [], "val_bacc": []}
    best_loss, best_state, wait = np.inf, None, 0

    for ep in range(epochs):
        model.train()
        total = 0.0
        for b in _batches(len(Xtr), batch_size, True, rng):
            xb, yb = Xtr[b].to(device), ytr[b].to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            model.apply_max_norm()
            total += loss.item() * len(b)
        history["train_loss"].append(total / len(Xtr))

        model.eval()
        with torch.no_grad():
            logits = torch.cat([model(Xva[i:i + eval_batch_size].to(device)).cpu()
                                for i in range(0, len(Xva), eval_batch_size)])
            val_loss = criterion(logits.to(device), yva.to(device)).item()
            probs = torch.softmax(logits, 1)[:, 1].numpy()
        history["val_loss"].append(val_loss)
        history["val_auc"].append(roc_auc_score(y_val, probs) if len(np.unique(y_val)) > 1 else np.nan)
        history["val_bacc"].append(balanced_accuracy_score(y_val, probs >= 0.5))

        if verbose and (ep % 10 == 0 or ep == epochs - 1):
            print(f"  epoch {ep:3d}  train {history['train_loss'][-1]:.4f}  val {val_loss:.4f}  "
                  f"AUC {history['val_auc'][-1]:.3f}  BAcc {history['val_bacc'][-1]:.3f}")

        if val_loss < best_loss - 1e-4:
            best_loss, best_state, wait = val_loss, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= patience:
                if verbose:
                    print(f"  early stop ở epoch {ep} (tốt nhất: epoch {ep - patience})")
                break

    model.load_state_dict(best_state)
    return model, history, scale


# ---------------------------------------------------------------------------
# Đánh giá (cùng bộ chỉ số với V5)
# ---------------------------------------------------------------------------
def compute_metrics(y_true: np.ndarray, probs: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    preds = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
    return {
        "balanced_accuracy": balanced_accuracy_score(y_true, preds),
        "roc_auc": roc_auc_score(y_true, probs),
        "pr_auc": average_precision_score(y_true, probs),
        "f1_macro": f1_score(y_true, preds, average="macro"),
        "f1_target": f1_score(y_true, preds, pos_label=1, zero_division=0),
        "sensitivity": tp / (tp + fn) if (tp + fn) else 0.0,
        "specificity": tn / (tn + fp) if (tn + fp) else 0.0,
    }


def get_splits(y: np.ndarray, groups: np.ndarray, n_splits: int = 5):
    if len(np.unique(groups)) >= n_splits:
        return list(GroupKFold(n_splits=n_splits).split(np.zeros(len(y)), y, groups)), "GroupKFold"
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    return list(skf.split(np.zeros(len(y)), y)), "StratifiedKFold"


def cross_validate(X: np.ndarray, y: np.ndarray, groups: np.ndarray, n_splits: int = 5,
                   val_size: float = 0.15, verbose: bool = True, **train_kwargs):
    """CV ngoài như V5; trong mỗi fold tách thêm val_size từ tập train để early stopping."""
    splits, splitter_name = get_splits(y, groups, n_splits)
    if verbose:
        print(f"{splitter_name}, {n_splits} fold, {len(y)} trial, {len(np.unique(groups))} người")

    oof_probs = np.zeros(len(y), dtype=float)
    fold_rows, histories = [], []
    for k, (tr, te) in enumerate(splits):
        t0 = time.time()
        tr_idx, va_idx = train_test_split(tr, test_size=val_size, stratify=y[tr],
                                          random_state=RANDOM_SEED + k)
        model, hist, scale = train_eegnet(X[tr_idx], y[tr_idx], X[va_idx], y[va_idx],
                                          seed=RANDOM_SEED + k, **train_kwargs)
        oof_probs[te] = predict_proba(model, X[te], scale)
        row = {"fold": k + 1, "n_test": len(te), "epochs_run": len(hist["train_loss"]),
               **compute_metrics(y[te], oof_probs[te])}
        fold_rows.append(row)
        histories.append(hist)
        if verbose:
            print(f"Fold {k + 1}: AUC {row['roc_auc']:.3f}  BAcc {row['balanced_accuracy']:.3f}  "
                  f"PR-AUC {row['pr_auc']:.3f}  ({row['epochs_run']} epoch, {time.time() - t0:.0f}s)")

    fold_df = pd.DataFrame(fold_rows)
    return fold_df, oof_probs, histories


def summarize(name: str, fold_df: pd.DataFrame, y: np.ndarray, oof_probs: np.ndarray) -> Dict[str, float]:
    """Một dòng cùng cột với results/tables/baseline_metrics.csv."""
    metric_cols = ["balanced_accuracy", "roc_auc", "pr_auc", "f1_macro", "f1_target", "sensitivity", "specificity"]
    row: Dict[str, float] = {"model": name}
    for m in metric_cols:
        row[f"{m}_mean"] = fold_df[m].mean()
        row[f"{m}_std"] = fold_df[m].std()
    oof = compute_metrics(y, oof_probs)
    for m in ["balanced_accuracy", "roc_auc", "pr_auc", "f1_macro", "sensitivity", "specificity"]:
        row[f"oof_{m}"] = oof[m]
    return row


# ---------------------------------------------------------------------------
# Giải thích mô hình
# ---------------------------------------------------------------------------
def saliency_map(model: nn.Module, X: np.ndarray, scale: float = 1.0, target_class: int = 1,
                 batch_size: int = 128) -> np.ndarray:
    device = next(model.parameters()).device
    model.eval()
    acc = np.zeros(X.shape[1:], dtype=np.float64)
    for i in range(0, len(X), batch_size):
        xb = torch.from_numpy(X[i:i + batch_size] / scale).float().to(device).requires_grad_(True)
        model(xb)[:, target_class].sum().backward()
        acc += xb.grad.abs().sum(0).cpu().numpy()
    return acc / len(X)


def plot_roc_pr_curves(y_true: np.ndarray, oof_probs: np.ndarray, save_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fpr, tpr, _ = roc_curve(y_true, oof_probs)
    score_roc = roc_auc_score(y_true, oof_probs)
    axes[0].plot(fpr, tpr, color="#d62728", lw=1.8, label=f"EEGNet (AUC = {score_roc:.3f})")
    axes[0].plot([0, 1], [0, 1], "--", color="#999999", label="Chance (AUC = 0.500)")
    axes[0].set_xlabel("False positive rate (1 - specificity)", fontsize=10.5)
    axes[0].set_ylabel("True positive rate (sensitivity)", fontsize=10.5)
    axes[0].set_title("Receiver operating characteristic (ROC) curve", fontsize=12, fontweight="bold")
    axes[0].legend(loc="lower right")
    axes[0].grid(True, ls=":", alpha=0.6)

    prec, rec, _ = precision_recall_curve(y_true, oof_probs)
    score_pr = average_precision_score(y_true, oof_probs)
    prevalence = float(y_true.mean())
    axes[1].plot(rec, prec, color="#d62728", lw=1.8, label=f"EEGNet (AP = {score_pr:.3f})")
    axes[1].axhline(prevalence, ls="--", color="#999999", label=f"Prevalence ({prevalence:.1%})")
    axes[1].set_xlabel("Recall (sensitivity)", fontsize=10.5)
    axes[1].set_ylabel("Precision", fontsize=10.5)
    axes[1].set_title("Precision-recall curve", fontsize=12, fontweight="bold")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, ls=":", alpha=0.6)

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_saliency_analysis(sal: np.ndarray, times: np.ndarray, diff: np.ndarray, save_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), gridspec_kw={"width_ratios": [2, 1.2, 1.2]})
    im = axes[0].imshow(sal, aspect="auto", cmap="magma", origin="lower",
                        extent=[times[0] * 1000, times[-1] * 1000, 0, sal.shape[0]])
    axes[0].set_xlabel("Time (ms)", fontsize=10)
    axes[0].set_ylabel("Channel index", fontsize=10)
    axes[0].set_title("Saliency map (target trials)", fontsize=11, fontweight="bold")
    fig.colorbar(im, ax=axes[0], fraction=0.04)

    axes[1].plot(times * 1000, sal.mean(0), color="#d62728", lw=1.8)
    axes[1].axvspan(300, 600, color="0.9", zorder=0, label="P300 window (300-600 ms)")
    axes[1].set_xlabel("Time (ms)", fontsize=10)
    axes[1].set_ylabel("Mean saliency", fontsize=10)
    axes[1].set_title("Temporal saliency profile", fontsize=11, fontweight="bold")
    axes[1].legend(loc="upper right", fontsize=8.5)
    axes[1].grid(True, ls=":", alpha=0.5)

    axes[2].plot(sal.mean(1), color="#333333", lw=1.5, label="Saliency")
    ax2 = axes[2].twinx()
    ax2.plot(diff, color="#1f77b4", alpha=0.6, lw=1.5, label="P300 effect (Target - Std)")
    axes[2].set_xlabel("Channel index", fontsize=10)
    axes[2].set_ylabel("Mean saliency", fontsize=10)
    ax2.set_ylabel("Amplitude difference (µV)", fontsize=10)
    axes[2].set_title("Spatial profile vs ERP effect", fontsize=11, fontweight="bold")
    axes[2].grid(True, ls=":", alpha=0.5)

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_saliency_topomaps(sal: np.ndarray, times: np.ndarray, diff: np.ndarray, info,
                           save_path: Path) -> None:
    """Topomap saliency (300-600 ms) cạnh topomap hiệu ứng Target - Standard, kèm hồ sơ thời gian theo vùng."""
    import mne

    win = (times >= 0.3) & (times <= 0.6)
    regions = {"Left central (C3, C1, CCP3h, FCC3h)": ["C3", "C1", "CCP3h", "FCC3h"],
               "Right central (C4, C2, CCP4h, FCC4h)": ["C4", "C2", "CCP4h", "FCC4h"],
               "Parietal (Pz, CPz, P1, P2)": ["Pz", "CPz", "P1", "P2"]}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), gridspec_kw={"width_ratios": [1, 1, 1.6]})
    im, _ = mne.viz.plot_topomap(sal[:, win].mean(1), info, axes=axes[0], show=False, cmap="magma")
    fig.colorbar(im, ax=axes[0], fraction=0.046, label="Mean saliency")
    axes[0].set_title("EEGNet saliency, 300-600 ms", fontsize=11, fontweight="bold")
    im, _ = mne.viz.plot_topomap(diff, info, axes=axes[1], show=False, cmap="RdBu_r")
    fig.colorbar(im, ax=axes[1], fraction=0.046, label="µV")
    axes[1].set_title("P300 effect (target - standard), 300-600 ms", fontsize=11, fontweight="bold")
    idx = {n: i for i, n in enumerate(info.ch_names)}
    for (name, chs), color in zip(regions.items(), ["#d62728", "#ff9896", "#1f77b4"]):
        axes[2].plot(times * 1000, sal[[idx[c] for c in chs]].mean(0), color=color, lw=1.8, label=name)
    axes[2].axvspan(300, 600, color="0.92", zorder=0)
    axes[2].set_xlabel("Time (ms)", fontsize=10)
    axes[2].set_ylabel("Mean saliency", fontsize=10)
    axes[2].set_title("Saliency time course by region", fontsize=11, fontweight="bold")
    axes[2].legend(fontsize=8.5, loc="upper right")
    axes[2].grid(True, ls=":", alpha=0.5)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def compare_ml_vs_dl(eegnet_row: Dict[str, float]) -> pd.DataFrame:
    baseline_csv = TABLE_DIR / "baseline_metrics.csv"
    if not baseline_csv.exists():
        return pd.DataFrame([eegnet_row])

    df_base = pd.read_csv(baseline_csv)
    df_eegnet = pd.DataFrame([eegnet_row])
    merged = pd.concat([df_base, df_eegnet], ignore_index=True)

    out_csv = TABLE_DIR / "ml_vs_dl_comparison.csv"
    merged.to_csv(out_csv, index=False)

    fig, ax = plt.subplots(figsize=(9.5, 5))
    models_list = merged["model"].tolist()
    x = np.arange(len(models_list))
    width = 0.26

    ax.bar(x - width, merged["balanced_accuracy_mean"], width, yerr=merged.get("balanced_accuracy_std", 0),
           label="Balanced accuracy", color="#4c72b0", capsize=3)
    ax.bar(x, merged["roc_auc_mean"], width, yerr=merged.get("roc_auc_std", 0),
           label="ROC-AUC", color="#55a868", capsize=3)
    ax.bar(x + width, merged["pr_auc_mean"], width, yerr=merged.get("pr_auc_std", 0),
           label="PR-AUC", color="#c44e52", capsize=3)

    ax.set_xticks(x)
    ax.set_xticklabels(models_list, fontsize=9.5)
    ax.set_ylabel("Score", fontsize=10.5)
    ax.set_title("Performance comparison: standard ML baselines vs deep learning (EEGNet)", fontsize=12, fontweight="bold")
    ax.set_ylim([0, 1.15])
    ax.axhline(0.5, linestyle="--", color="#999999", lw=1, label="Chance level (0.50)")
    ax.grid(True, linestyle=":", alpha=0.5, axis="y")
    ax.legend(loc="upper center", ncol=4, fontsize=9, frameon=False)
    fig.tight_layout()

    comp_plot = FIG_DIR / "ml_vs_dl_comparison.png"
    fig.savefig(comp_plot, dpi=200, bbox_inches="tight")

    report_fig_dir = ROOT / "report" / "figures"
    if report_fig_dir.exists():
        fig.savefig(report_fig_dir / "ml_vs_dl_comparison.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description="V6: Huấn luyện và đánh giá EEGNet.")
    parser.add_argument("--synthetic", action="store_true", help="Dùng dữ liệu mô phỏng")
    parser.add_argument("--subjects", nargs="*", default=None, help="Ví dụ: sub-01 sub-03")
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--save", action="store_true", help="Lưu bảng kết quả và biểu đồ")
    args = parser.parse_args()

    if args.synthetic:
        X, y, groups = make_synthetic_dataset()
        tag = "synthetic"
    else:
        X, y, groups = load_real_dataset(args.subjects)
        tag = "real"
    print(f"Dataset ({tag}): X {X.shape}, target {y.mean():.1%}, device {get_device()}")

    fold_df, oof_probs, _ = cross_validate(
        X, y, groups, n_splits=args.splits, epochs=args.epochs, batch_size=args.batch_size
    )
    row = summarize("EEGNet", fold_df, y, oof_probs)
    print("\nEEGNet Cross-Validation Summary:")
    print(pd.Series(row).to_string())

    times = TMIN + np.arange(N_TIMES) / SFREQ
    win = (times >= 0.3) & (times <= 0.6)
    diff = X[y == 1][:, :, win].mean((0, 2)) - X[y == 0][:, :, win].mean((0, 2))

    final_tr, final_va = train_test_split(
        np.arange(len(y)), test_size=0.15, stratify=y, random_state=RANDOM_SEED
    )
    ref_model, _, scale = train_eegnet(
        X[final_tr], y[final_tr], X[final_va], y[final_va], epochs=args.epochs, batch_size=args.batch_size
    )
    sal = saliency_map(ref_model, X[y == 1], scale=scale)

    peak_ms = times[np.argmax(sal.mean(0))] * 1000
    r = np.corrcoef(sal.mean(1), diff)[0, 1]
    print(f"\nSaliency Analysis:")
    print(f"Peak saliency latency: {peak_ms:.1f} ms")
    print(f"Spatial saliency correlation with ERP P300 effect: r = {r:.3f}")
    print(f"Saliency mass in 300-600 ms window: {sal[:, win].sum() / sal.sum():.1%} "
          f"(window covers {win.mean():.1%} of the epoch)")

    ch_names = None
    if not args.synthetic:
        from src.utils.io import epoch_path
        import mne
        ch_names = mne.read_epochs(epoch_path("sub-01"), preload=False, verbose="error").ch_names
        top = np.argsort(sal.mean(1))[::-1][:10]
        print("Top-10 channels by saliency: " + ", ".join(ch_names[i] for i in top))
        top_diff = np.argsort(diff)[::-1][:10]
        print("Top-10 channels by P300 effect: " + ", ".join(ch_names[i] for i in top_diff))

    if args.save:
        TABLE_DIR.mkdir(parents=True, exist_ok=True)
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        MODEL_DIR.mkdir(parents=True, exist_ok=True)

        metrics_csv = TABLE_DIR / f"eegnet_metrics_{tag}.csv"
        pd.DataFrame([row]).to_csv(metrics_csv, index=False)
        print(f"Saved metrics table: {metrics_csv}")

        roc_pr_plot = FIG_DIR / f"eegnet_roc_pr_{tag}.png"
        plot_roc_pr_curves(y, oof_probs, roc_pr_plot)
        print(f"Saved ROC/PR curve: {roc_pr_plot}")

        sal_plot = FIG_DIR / f"eegnet_saliency_{tag}.png"
        plot_saliency_analysis(sal, times, diff, sal_plot)
        print(f"Saved saliency analysis: {sal_plot}")

        np.savez_compressed(TABLE_DIR / f"eegnet_saliency_{tag}.npz", saliency=sal, times=times,
                            p300_effect=diff, ch_names=np.array(ch_names or []))
        fold_df.to_csv(TABLE_DIR / f"eegnet_folds_{tag}.csv", index=False)
        np.savez_compressed(TABLE_DIR / f"eegnet_oof_{tag}.npz", y=y, oof_probs=oof_probs, groups=groups)

        torch.save(ref_model.state_dict(), MODEL_DIR / f"eegnet_{tag}.pt")
        print(f"Saved model checkpoint: {MODEL_DIR / f'eegnet_{tag}.pt'}")

        report_fig_dir = ROOT / "report" / "figures"
        if report_fig_dir.exists():
            plot_roc_pr_curves(y, oof_probs, report_fig_dir / f"eegnet_roc_pr_{tag}.png")
            plot_saliency_analysis(sal, times, diff, report_fig_dir / f"eegnet_saliency_{tag}.png")

        if ch_names:
            import mne
            from src.utils.io import epoch_path
            info = mne.read_epochs(epoch_path("sub-01"), preload=False, verbose="error").info
            for d in (FIG_DIR, report_fig_dir):
                if d.exists():
                    plot_saliency_topomaps(sal, times, diff, info, d / f"eegnet_saliency_topomap_{tag}.png")

        comp_df = compare_ml_vs_dl(row)
        print("\nML vs DL Comparison:")
        comp_cols = ["model", "balanced_accuracy_mean", "roc_auc_mean", "pr_auc_mean", "sensitivity_mean", "specificity_mean"]
        valid_cols = [c for c in comp_cols if c in comp_df.columns]
        print(comp_df[valid_cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
