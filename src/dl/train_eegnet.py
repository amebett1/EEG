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

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, roc_auc_score)
from sklearn.model_selection import (GroupKFold, StratifiedKFold,
                                     train_test_split)

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import (MODEL_DIR, N_CHANNELS, RANDOM_SEED, SFREQ, TABLE_DIR, TMAX,
                    TMIN)

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
    torch.cuda.manual_seed_all(seed)


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
                 seed: int = RANDOM_SEED, verbose: bool = False) -> Tuple[EEGNet, Dict[str, List[float]], float]:
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
    Xva = torch.from_numpy(X_val / scale).float().to(device)
    yva = torch.from_numpy(y_val).long().to(device)

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
            logits = model(Xva)
            val_loss = criterion(logits, yva).item()
            probs = torch.softmax(logits, 1)[:, 1].cpu().numpy()
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
    """Gradient |d logit_target / d input| trung bình trên các trial -> (kênh, thời gian)."""
    device = next(model.parameters()).device
    model.eval()
    acc = np.zeros(X.shape[1:], dtype=np.float64)
    for i in range(0, len(X), batch_size):
        xb = torch.from_numpy(X[i:i + batch_size] / scale).float().to(device).requires_grad_(True)
        model(xb)[:, target_class].sum().backward()
        acc += xb.grad.abs().sum(0).cpu().numpy()
    return acc / len(X)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="V6: Huấn luyện và đánh giá EEGNet.")
    parser.add_argument("--synthetic", action="store_true", help="Dùng dữ liệu mô phỏng")
    parser.add_argument("--subjects", nargs="*", default=None, help="Ví dụ: sub-01 sub-03")
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--save", action="store_true", help="Lưu bảng kết quả vào results/tables")
    args = parser.parse_args()

    if args.synthetic:
        X, y, groups = make_synthetic_dataset()
        tag = "synthetic"
    else:
        X, y, groups = load_real_dataset(args.subjects)
        tag = "real"
    print(f"Dữ liệu ({tag}): X {X.shape}, target {y.mean():.1%}, thiết bị {get_device()}")

    fold_df, oof_probs, _ = cross_validate(X, y, groups, n_splits=args.splits, epochs=args.epochs)
    row = summarize("EEGNet", fold_df, y, oof_probs)
    print(pd.Series(row).to_string())

    if args.save:
        TABLE_DIR.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([row]).to_csv(TABLE_DIR / f"eegnet_metrics_{tag}.csv", index=False)


if __name__ == "__main__":
    main()
