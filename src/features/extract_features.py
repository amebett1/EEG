from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import mne
import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.stats import entropy

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import (BINARY_CONDITIONS, EVENT_ID, FEATURE_DIR, N_CHANNELS,
                    SFREQ, SUBJECTS, TASK_MISUNDERSTOOD)
from src.utils.io import load_epochs

KEY_CHANNELS = ["Fz", "Cz", "Pz", "Oz", "FCz", "CPz", "F3", "F4", "C3", "C4", "P3", "P4"]
WINDOWS = {
    "early": (0.1, 0.2),
    "p3a": (0.25, 0.4),
    "p3b": (0.4, 0.6),
    "p3_full": (0.3, 0.6),
    "late": (0.6, 0.8)
}
BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0)
}


def hjorth_parameters(signal: np.ndarray) -> Tuple[float, float, float]:
    d1 = np.diff(signal)
    d2 = np.diff(d1)
    var_zero = float(np.var(signal))
    var_d1 = float(np.var(d1))
    var_d2 = float(np.var(d2))

    if var_zero < 1e-12:
        return 0.0, 0.0, 0.0

    activity = var_zero
    mobility = np.sqrt(var_d1 / var_zero) if var_d1 > 0 else 0.0
    complexity = (np.sqrt(var_d2 / var_d1) / mobility) if (var_d1 > 0 and mobility > 0) else 0.0
    return float(activity), float(mobility), float(complexity)


def spectral_features(signal: np.ndarray, sfreq: float) -> Dict[str, float]:
    freqs, psd = welch(signal, fs=sfreq, nperseg=min(len(signal), int(sfreq)))
    total_power = np.sum(psd) + 1e-12

    feats: Dict[str, float] = {}
    for band_name, (fmin, fmax) in BANDS.items():
        mask = (freqs >= fmin) & (freqs <= fmax)
        band_power = float(np.sum(psd[mask]))
        feats[f"bp_{band_name}"] = band_power
        feats[f"rel_bp_{band_name}"] = band_power / total_power

    theta = feats.get("bp_theta", 0.0)
    alpha = feats.get("bp_alpha", 1e-12)
    delta = feats.get("bp_delta", 0.0)
    feats["ratio_theta_alpha"] = theta / (alpha + 1e-12)
    feats["ratio_delta_theta"] = delta / (theta + 1e-12)

    psd_norm = psd / total_power
    feats["spectral_entropy"] = float(entropy(psd_norm + 1e-12))
    return feats


def extract_epoch_features(epoch_data: np.ndarray, times: np.ndarray, ch_names: List[str]) -> Dict[str, float]:
    feats: Dict[str, float] = {}
    sfreq = float(1.0 / (times[1] - times[0]))

    for ch in KEY_CHANNELS:
        if ch not in ch_names:
            continue
        ch_idx = ch_names.index(ch)
        sig = epoch_data[ch_idx] * 1e6

        for w_name, (tmin, tmax) in WINDOWS.items():
            mask = (times >= tmin) & (times <= tmax)
            if np.any(mask):
                w_sig = sig[mask]
                feats[f"{ch}_{w_name}_mean"] = float(np.mean(w_sig))
                feats[f"{ch}_{w_name}_std"] = float(np.std(w_sig))

        p3_mask = (times >= 0.3) & (times <= 0.6)
        if np.any(p3_mask):
            p3_sig = sig[p3_mask]
            p3_times = times[p3_mask]
            max_idx = int(np.argmax(p3_sig))
            feats[f"{ch}_p3_peak_amp"] = float(p3_sig[max_idx])
            feats[f"{ch}_p3_peak_lat"] = float(p3_times[max_idx] * 1000.0)
            feats[f"{ch}_p3_auc"] = float(np.trapezoid(p3_sig, p3_times) if hasattr(np, "trapezoid") else np.trapz(p3_sig, p3_times))

        resp_mask = (times >= 0.2) & (times <= 0.7)
        if np.any(resp_mask):
            r_sig = sig[resp_mask]
            feats[f"{ch}_ptp"] = float(np.ptp(r_sig))

        act, mob, comp = hjorth_parameters(sig)
        feats[f"{ch}_hjorth_activity"] = act
        feats[f"{ch}_hjorth_mobility"] = mob
        feats[f"{ch}_hjorth_complexity"] = comp

        spec_f = spectral_features(sig, sfreq)
        for k, v in spec_f.items():
            feats[f"{ch}_{k}"] = v

    return feats


def extract_subject_features(
    sub: str,
    epochs: Optional[mne.Epochs] = None,
    conditions: Tuple[str, ...] = BINARY_CONDITIONS,
    save: bool = True
) -> pd.DataFrame:
    if epochs is None:
        epochs = load_epochs(sub, conditions=conditions, correct_only=True)

    data = epochs.get_data()
    times = epochs.times
    ch_names = epochs.ch_names
    events = epochs.events
    meta = epochs.metadata if epochs.metadata is not None else pd.DataFrame()

    target_code = EVENT_ID["target"]
    rows: List[Dict[str, Any]] = []

    for i in range(len(epochs)):
        f_dict = extract_epoch_features(data[i], times, ch_names)
        is_target = int(events[i, 2] == target_code)

        f_dict["subject"] = sub
        f_dict["trial_idx"] = i
        f_dict["label"] = is_target

        if not meta.empty and i < len(meta):
            f_dict["condition"] = meta["condition"].iloc[i]
            if "rt_ms" in meta.columns:
                f_dict["rt_ms"] = meta["rt_ms"].iloc[i]
        else:
            f_dict["condition"] = "target" if is_target == 1 else "standard"

        rows.append(f_dict)

    df = pd.DataFrame(rows)

    if save:
        FEATURE_DIR.mkdir(parents=True, exist_ok=True)
        out_file = FEATURE_DIR / f"{sub}_features.csv"
        df.to_csv(out_file, index=False)
        print(f"[{sub}] Đã lưu {len(df)} trials x {df.shape[1]} đặc trưng -> {out_file}")

    return df


def extract_all_subjects(subjects: Optional[List[str]] = None, save_merged: bool = True) -> pd.DataFrame:
    if subjects is None:
        subjects = [s for s in SUBJECTS if s not in TASK_MISUNDERSTOOD]

    dfs: List[pd.DataFrame] = []
    for sub in subjects:
        try:
            df_sub = extract_subject_features(sub, save=True)
            dfs.append(df_sub)
        except Exception as e:
            print(f"[{sub}] Bỏ qua do lỗi: {e}")

    if not dfs:
        raise RuntimeError("Không có dữ liệu đặc trưng nào được trích xuất thành công.")

    merged_df = pd.concat(dfs, ignore_index=True)
    if save_merged:
        FEATURE_DIR.mkdir(parents=True, exist_ok=True)
        merged_file = FEATURE_DIR / "all_subjects_features.csv"
        merged_df.to_csv(merged_file, index=False)
        print(f"\n[V4] Đã gộp toàn bộ đặc trưng: {merged_df.shape} -> {merged_file}")

    return merged_df


def load_features(feature_file: Optional[Path] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    if feature_file is None:
        feature_file = FEATURE_DIR / "all_subjects_features.csv"
    feature_file = Path(feature_file)

    if not feature_file.exists():
        raise FileNotFoundError(f"Chưa có file đặc trưng: {feature_file}")

    df = pd.read_csv(feature_file)
    meta_cols = ["subject", "trial_idx", "label", "condition", "rt_ms"]
    feature_cols = [c for c in df.columns if c not in meta_cols]

    X = df[feature_cols].values
    y = df["label"].values.astype(int)

    unique_subs = {s: i for i, s in enumerate(df["subject"].unique())}
    groups = df["subject"].map(unique_subs).values

    return X, y, groups, feature_cols


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="V4: Feature Extraction for P300 classification.")
    parser.add_argument("--sub", type=str, default="sub-01", help="Mã người cần trích xuất (mặc định: sub-01)")
    parser.add_argument("--all", action="store_true", help="Trích xuất toàn bộ các người có epoch")
    args = parser.parse_args()

    if args.all:
        extract_all_subjects()
    else:
        extract_subject_features(sub=args.sub, save=True)
