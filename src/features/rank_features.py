"""V4: Xếp hạng đặc trưng theo mức phân biệt target/standard.

AUC được tính riêng trong từng người rồi lấy trung bình, nên khác biệt giữa các người
không làm sai lệch kết quả. Bảng chỉ để tham khảo và đưa vào báo cáo, không dùng để
loại đặc trưng trước khi train: theo Laton et al. (2014), đặc trưng yếu khi đứng riêng
vẫn có thể hữu ích khi kết hợp với đặc trưng khác.

Chạy: python src/features/rank_features.py  (sau khi đã chạy extract_features.py --all)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import FEATURE_DIR, TABLE_DIR

META_COLS = ["subject", "trial_idx", "label", "condition", "rt_ms"]


def feature_auc(df: pd.DataFrame) -> pd.DataFrame:
    """AUC từng đặc trưng (0.5 = không phân biệt, >0.5 = target cao hơn, <0.5 = target thấp hơn)."""
    feats = [c for c in df.columns if c not in META_COLS]
    per_subject = []
    for _, g in df.groupby("subject"):
        y = g["label"].to_numpy()
        x = g[feats].to_numpy(dtype=float)
        u = mannwhitneyu(x[y == 1], x[y == 0], axis=0).statistic
        per_subject.append(u / ((y == 1).sum() * (y == 0).sum()))
    per_subject = np.array(per_subject)
    auc = per_subject.mean(axis=0)

    table = pd.DataFrame({
        "feature": feats,
        "auc": auc.round(3),
        "strength": (np.abs(auc - 0.5) + 0.5).round(3),
        "target_higher": auc > 0.5,
        "auc_sd_across_subjects": per_subject.std(axis=0).round(3),
    })
    return table.sort_values("strength", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    df = pd.read_csv(FEATURE_DIR / "all_subjects_features.csv")
    table = feature_auc(df)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    out_file = TABLE_DIR / "v4_feature_auc.csv"
    table.to_csv(out_file, index=False)
    print(table.head(20).to_string(index=False))
    print(f"\n[V4] Đã lưu xếp hạng {len(table)} đặc trưng ({df['subject'].nunique()} người) -> {out_file}")
