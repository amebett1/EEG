from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.stats import entropy

try:
    import mne
except ImportError:
    mne = None

try:
    import pywt
except ImportError:
    pywt = None

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
    "p3_window": (0.3, 0.6),  # theo paper dataset: P3 đạt đỉnh trong 300-600 ms
    "late": (0.6, 0.8)
}
BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0)
}
# Wavelet db4 tách 4 mức ở 250 Hz. Thứ tự thành phần trả về từ pywt.mra:
# A4 ≈ 0-7.8 Hz (delta + theta), D4 ≈ 7.8-15.6 Hz (alpha), D3 ≈ 15.6-31 Hz (beta).
WAVELET = "db4"
WAVELET_LEVEL = 4
WAVELET_BANDS = {"delta_theta": 0, "alpha": 1, "beta": 2}
# Các đỉnh ERP sớm theo Laton et al. (2014), Bảng 2 (oddball thị giác, target):
# tên -> (cửa sổ tìm đỉnh, hướng: -1 = đỉnh âm, +1 = đỉnh dương). Đo ở Fz, Cz, Pz như paper.
ERP_PEAKS = {
    "n1": ((0.05, 0.2), -1),
    "p2": ((0.15, 0.3), +1),
    "n2": ((0.2, 0.4), -1),
}
PEAK_CHANNELS = ["Fz", "Cz", "Pz"]


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


def detect_peak(signal: np.ndarray, times: np.ndarray, window: Tuple[float, float],
                direction: int) -> Tuple[float, float]:
    """Tìm đỉnh theo cách của Laton et al. (2014). Trả về (biên độ µV, latency ms).

    Đỉnh là điểm cao hơn (direction=+1) hoặc thấp hơn (direction=-1) hai điểm kề bên,
    nên một điểm nằm trên sườn dốc ở mép cửa sổ không bị tính là đỉnh. Có nhiều đỉnh
    thì lấy đỉnh cực trị nhất. Không có đỉnh nào (cửa sổ chỉ là một sườn dốc) thì lấy
    điểm có đạo hàm gần 0 nhất, tương ứng tùy chọn "force" trong paper.
    """
    s = direction * signal
    in_win = (times >= window[0]) & (times <= window[1])
    is_peak = np.zeros(len(s), dtype=bool)
    is_peak[1:-1] = (s[1:-1] > s[:-2]) & (s[1:-1] >= s[2:])
    candidates = np.flatnonzero(is_peak & in_win)
    if candidates.size:
        idx = candidates[np.argmax(s[candidates])]
    else:
        win_idx = np.flatnonzero(in_win)
        idx = win_idx[np.argmin(np.abs(np.gradient(signal)[win_idx]))]
    return float(signal[idx]), float(times[idx] * 1000.0)


def wavelet_features(signal: np.ndarray, p3_mask: np.ndarray) -> Dict[str, float]:
    """Tách tín hiệu thành các dải tần bằng wavelet, rồi đo từng dải trong cửa sổ P3.

    Khác band power (tính trên cả epoch), wavelet giữ được thông tin thời gian,
    nên đo được năng lượng của từng dải đúng trong khoảng P300.
    """
    comps = pywt.mra(signal, WAVELET, level=WAVELET_LEVEL, transform="dwt")
    feats: Dict[str, float] = {}
    for band_name, idx in WAVELET_BANDS.items():
        feats[f"wav_{band_name}_p3_energy"] = float(np.mean(comps[idx][p3_mask] ** 2))
    feats["wav_delta_theta_p3_mean"] = float(np.mean(comps[WAVELET_BANDS["delta_theta"]][p3_mask]))
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

        p3_start, p3_end = WINDOWS["p3_window"]
        p3_mask = (times >= p3_start) & (times <= p3_end)
        if np.any(p3_mask):
            p3_sig = sig[p3_mask]
            p3_times = times[p3_mask]
            amp, lat = detect_peak(sig, times, (p3_start, p3_end), +1)
            feats[f"{ch}_p3_peak_amp"] = amp
            feats[f"{ch}_p3_peak_lat"] = lat
            feats[f"{ch}_p3_auc"] = float(np.trapezoid(p3_sig, p3_times) if hasattr(np, "trapezoid") else np.trapz(p3_sig, p3_times))

        if ch in PEAK_CHANNELS:
            for peak_name, (window, direction) in ERP_PEAKS.items():
                amp, lat = detect_peak(sig, times, window, direction)
                feats[f"{ch}_{peak_name}_amp"] = amp
                feats[f"{ch}_{peak_name}_lat"] = lat

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

        for k, v in wavelet_features(sig, p3_mask).items():
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
