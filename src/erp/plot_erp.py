from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import mne
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import (BINARY_CONDITIONS, EPOCH_DIR, EPOCH_FILE, EVENT_ID,
                    FIG_DIR, N_CHANNELS, QC_SUMMARY, SUBJECTS,
                    TABLE_DIR, TASK_MISUNDERSTOOD)
from src.utils.io import epoch_path

COLOR_MAP = {
    "standard": "#4d4d4d",
    "target": "#d55e00",
    "distractor": "#0072b2",
    "diff_target": "#785ef0",
    "diff_distractor": "#009e73"
}

MIDLINE_CHANNELS = ("Fz", "Cz", "Pz")
P3_WINDOW = (0.3, 0.5)


def load_subject_epochs(
    sub: str,
    conditions: Optional[Tuple[str, ...]] = None,
    correct_only: bool = True
) -> mne.Epochs:
    path = epoch_path(sub)
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file epoch: {path}")

    ep = mne.read_epochs(path, preload=True, verbose="error")
    if correct_only and ep.metadata is not None and "correct" in ep.metadata:
        ep = ep[ep.metadata["correct"].values]

    if conditions is None:
        conditions = tuple(EVENT_ID.keys())

    available_conds = [c for c in conditions if c in ep.event_id]
    return ep[available_conds]


def compute_evokeds(epochs: mne.Epochs) -> Dict[str, mne.Evoked]:
    evokeds: Dict[str, mne.Evoked] = {}
    for cond in EVENT_ID:
        if cond in epochs.event_id and len(epochs[cond]) > 0:
            evokeds[cond] = epochs[cond].average()

    if "target" in evokeds and "standard" in evokeds:
        evokeds["diff_target"] = mne.combine_evoked(
            [evokeds["target"], evokeds["standard"]], weights=[1, -1]
        )
        evokeds["diff_target"].comment = "Target - Standard (P3b)"

    if "distractor" in evokeds and "standard" in evokeds:
        evokeds["diff_distractor"] = mne.combine_evoked(
            [evokeds["distractor"], evokeds["standard"]], weights=[1, -1]
        )
        evokeds["diff_distractor"].comment = "Distractor - Standard (P3a)"

    return evokeds


def plot_midline_erp(
    epochs: mne.Epochs,
    sub: str,
    channels: Tuple[str, ...] = MIDLINE_CHANNELS,
    window: Tuple[float, float] = P3_WINDOW,
    show_sem: bool = True,
    show_diff: bool = True,
    save_path: Optional[Path] = None
) -> plt.Figure:
    evokeds = compute_evokeds(epochs)
    times_ms = epochs.times * 1000.0
    w_start_ms, w_end_ms = window[0] * 1000.0, window[1] * 1000.0
    w_mask = (times_ms >= w_start_ms) & (times_ms <= w_end_ms)

    n_chan = len(channels)
    fig, axes = plt.subplots(1, n_chan, figsize=(5.2 * n_chan, 4.5), sharey=True)
    if n_chan == 1:
        axes = [axes]

    for ax, ch in zip(axes, channels):
        if ch not in epochs.ch_names:
            ax.set_title(f"{ch} (N/A)")
            continue

        ax.axvspan(w_start_ms, w_end_ms, color="#efefef", alpha=0.9, label=f"P300 Window ({int(w_start_ms)}–{int(w_end_ms)} ms)")

        for cond in ("standard", "target", "distractor"):
            if cond not in epochs.event_id or len(epochs[cond]) == 0:
                continue
            data_trials = epochs[cond].get_data(picks=ch)[:, 0, :] * 1e6
            mean_uv = np.mean(data_trials, axis=0)
            sem_uv = stats.sem(data_trials, axis=0) if len(data_trials) > 1 else np.zeros_like(mean_uv)

            color = COLOR_MAP.get(cond, "#333333")
            n_t = len(data_trials)
            ax.plot(times_ms, mean_uv, color=color, lw=2.0, label=f"{cond.capitalize()} (n={n_t})")

            if show_sem:
                ax.fill_between(times_ms, mean_uv - sem_uv, mean_uv + sem_uv, color=color, alpha=0.15)

            if cond == "target" and np.any(w_mask):
                w_times = times_ms[w_mask]
                w_data = mean_uv[w_mask]
                peak_idx = np.argmax(w_data)
                peak_lat = w_times[peak_idx]
                peak_amp = w_data[peak_idx]
                ax.plot(peak_lat, peak_amp, marker="o", markersize=6, color="#b20000")
                ax.annotate(
                    f"{peak_amp:.1f} µV\n{peak_lat:.0f} ms",
                    xy=(peak_lat, peak_amp),
                    xytext=(peak_lat + 20, peak_amp + 0.8),
                    fontsize=8.5,
                    fontweight="bold",
                    color="#b20000",
                    arrowprops=dict(arrowstyle="->", color="#b20000", lw=0.8)
                )

        if show_diff and "diff_target" in evokeds:
            diff_data = evokeds["diff_target"].copy().pick([ch]).data[0] * 1e6
            ax.plot(times_ms, diff_data, color=COLOR_MAP["diff_target"], lw=1.6, ls="--", label="Target - Standard")

        ax.axvline(0, color="k", lw=0.8, ls=":")
        ax.axhline(0, color="k", lw=0.8, ls=":")
        ax.set_title(f"{sub} – {ch}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Time (ms)", fontsize=10.5)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.set_xlim(-150, 900)

    axes[0].set_ylabel("Amplitude (µV)", fontsize=11)
    axes[-1].legend(loc="upper right", frameon=True, fontsize=8.5)

    fig.suptitle(
        f"Midline ERP waveforms ({', '.join(channels)}) – {sub}\n"
        f"Polich (2007) hypothesis verification: parietal P3b maximum (Pz > Cz > Fz)",
        fontsize=13, fontweight="bold", y=1.03
    )
    fig.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"  -> Saved Midline ERP figure: {save_path}")

    return fig


def plot_fz_pz_erp(
    epochs: mne.Epochs,
    sub: str,
    window: Tuple[float, float] = P3_WINDOW,
    save_path: Optional[Path] = None
) -> plt.Figure:
    return plot_midline_erp(
        epochs=epochs,
        sub=sub,
        channels=("Fz", "Pz"),
        window=window,
        show_sem=True,
        show_diff=True,
        save_path=save_path
    )


def plot_difference_waves(
    epochs: mne.Epochs,
    sub: str,
    channels: Tuple[str, ...] = MIDLINE_CHANNELS,
    window: Tuple[float, float] = P3_WINDOW,
    save_path: Optional[Path] = None
) -> plt.Figure:
    evokeds = compute_evokeds(epochs)
    if "diff_target" not in evokeds:
        raise ValueError("Cần cả Target và Standard để vẽ sóng hiệu số.")

    times_ms = epochs.times * 1000.0
    w_start_ms, w_end_ms = window[0] * 1000.0, window[1] * 1000.0

    n_chan = len(channels)
    fig, axes = plt.subplots(1, n_chan, figsize=(5.0 * n_chan, 4.0), sharey=True)
    if n_chan == 1:
        axes = [axes]

    for ax, ch in zip(axes, channels):
        if ch not in epochs.ch_names:
            continue

        ax.axvspan(w_start_ms, w_end_ms, color="#f5f5f5", alpha=0.8)

        t_diff = evokeds["diff_target"].copy().pick([ch]).data[0] * 1e6
        ax.plot(times_ms, t_diff, color=COLOR_MAP["diff_target"], lw=2.2, label="Target - Standard (P3b)")

        if "diff_distractor" in evokeds:
            d_diff = evokeds["diff_distractor"].copy().pick([ch]).data[0] * 1e6
            ax.plot(times_ms, d_diff, color=COLOR_MAP["diff_distractor"], lw=2.0, ls="-.", label="Distractor - Standard (P3a)")

        ax.axvline(0, color="k", lw=0.7, ls=":")
        ax.axhline(0, color="k", lw=0.7, ls=":")
        ax.set_title(f"Difference wave – {ch}", fontsize=11.5, fontweight="bold")
        ax.set_xlabel("Time (ms)", fontsize=10.5)
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.set_xlim(-100, 850)

    axes[0].set_ylabel("Difference amplitude (µV)", fontsize=11)
    axes[-1].legend(loc="upper right", frameon=True, fontsize=9)

    fig.suptitle(
        f"Difference waves – {sub}: isolating frontal P3a vs. parietal P3b",
        fontsize=12.5, fontweight="bold", y=1.02
    )
    fig.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"  -> Saved Difference Waves figure: {save_path}")

    return fig


def plot_erp_topomap(
    epochs: mne.Epochs,
    sub: str,
    times: Tuple[float, ...] = (0.15, 0.25, 0.35, 0.45, 0.55),
    vlim: Optional[Tuple[float, float]] = None,
    save_path: Optional[Path] = None
) -> plt.Figure:
    evokeds = compute_evokeds(epochs)
    conditions = ["standard", "target", "distractor", "diff_target"]
    available_conds = [c for c in conditions if c in evokeds]
    n_rows = len(available_conds)
    n_cols = len(times)

    if vlim is None:
        max_val = 0.0
        for c in available_conds:
            ev = evokeds[c]
            t_indices = [np.argmin(np.abs(ev.times - t)) for t in times]
            sample_data = np.abs(ev.data[:, t_indices]) * 1e6
            max_val = max(max_val, float(np.percentile(sample_data, 98)))
        vlim = (-max_val, max_val)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.8 * n_cols, 2.5 * n_rows))
    if n_rows == 1:
        axes = np.expand_dims(axes, 0)
    if n_cols == 1:
        axes = np.expand_dims(axes, 1)

    cond_labels = {
        "standard": "Standard",
        "target": "Target",
        "distractor": "Distractor",
        "diff_target": "Target - Standard\n(Difference)"
    }

    last_im = None
    for r_idx, cond in enumerate(available_conds):
        ev = evokeds[cond]
        for c_idx, t in enumerate(times):
            ax = axes[r_idx, c_idx]
            im, _ = mne.viz.plot_topomap(
                ev.data[:, np.argmin(np.abs(ev.times - t))] * 1e6,
                ev.info,
                axes=ax,
                show=False,
                vlim=vlim,
                cmap="RdBu_r"
            )
            last_im = im

            if r_idx == 0:
                ax.set_title(f"{int(t * 1000)} ms", fontsize=11, fontweight="bold")
            if c_idx == 0:
                ax.set_ylabel(cond_labels.get(cond, cond), fontsize=10, fontweight="bold")

    cbar_ax = fig.add_axes([0.92, 0.25, 0.018, 0.5])
    cbar = fig.colorbar(last_im, cax=cbar_ax)
    cbar.set_label("Amplitude (µV)", fontsize=10)

    fig.suptitle(
        f"Scalp potential topomap over time – {sub}\n"
        f"128-channel montage (Colin27) – scale: [{vlim[0]:.1f}, {vlim[1]:.1f}] µV",
        fontsize=13, fontweight="bold", y=0.99
    )
    plt.subplots_adjust(right=0.90, top=0.92, hspace=0.25, wspace=0.15)

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"  -> Saved Scalp Topomap figure: {save_path}")

    return fig


def plot_p300_summary_topomap(
    epochs: mne.Epochs,
    sub: str,
    window: Tuple[float, float] = P3_WINDOW,
    save_path: Optional[Path] = None
) -> plt.Figure:
    evokeds = compute_evokeds(epochs)
    conditions = [c for c in ("standard", "target", "distractor", "diff_target") if c in evokeds]
    n_cond = len(conditions)

    fig, axes = plt.subplots(1, n_cond, figsize=(3.3 * n_cond, 3.4))
    if n_cond == 1:
        axes = [axes]

    titles = {
        "standard": "Standard",
        "target": "Target (P3b)",
        "distractor": "Distractor (P3a)",
        "diff_target": "Difference (Target - Standard)"
    }

    window_data = {}
    max_val = 0.0
    for cond in conditions:
        cropped = evokeds[cond].copy().crop(*window)
        mean_spatial = cropped.data.mean(axis=1) * 1e6
        window_data[cond] = mean_spatial
        max_val = max(max_val, float(np.percentile(np.abs(mean_spatial), 98)))

    vlim = (-max_val, max_val)

    last_im = None
    for ax, cond in zip(axes, conditions):
        im, _ = mne.viz.plot_topomap(
            window_data[cond],
            evokeds[cond].info,
            axes=ax,
            show=False,
            vlim=vlim,
            cmap="RdBu_r"
        )
        last_im = im
        ax.set_title(titles.get(cond, cond), fontsize=11, fontweight="bold")

    cbar_ax = fig.add_axes([0.92, 0.2, 0.016, 0.6])
    cbar = fig.colorbar(last_im, cax=cbar_ax)
    cbar.set_label("Mean amplitude (µV)", fontsize=9.5)

    fig.suptitle(
        f"Mean P300 window topomap ({int(window[0]*1000)}–{int(window[1]*1000)} ms) – {sub}\n"
        f"Parietal positivity hotspot (P3b component)",
        fontsize=12, fontweight="bold", y=1.04
    )
    plt.subplots_adjust(right=0.90)

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"  -> Saved Summary Topomap figure: {save_path}")

    return fig


def verify_p300(
    epochs: mne.Epochs,
    sub: str,
    window: Tuple[float, float] = P3_WINDOW
) -> Dict[str, Any]:
    evokeds = compute_evokeds(epochs)
    times_ms = epochs.times * 1000.0
    w_start_ms, w_end_ms = window[0] * 1000.0, window[1] * 1000.0
    w_mask = (times_ms >= w_start_ms) & (times_ms <= w_end_ms)

    report: Dict[str, Any] = {
        "subject": sub,
        "n_channels": len(epochs.ch_names),
        "n_trials": {c: len(epochs[c]) for c in epochs.event_id},
        "metrics": {},
        "polich_rules": {}
    }

    for cond in ("standard", "target", "distractor"):
        if cond not in evokeds:
            continue
        ev = evokeds[cond]
        for ch in MIDLINE_CHANNELS:
            if ch not in ev.ch_names:
                continue
            ch_data = ev.copy().pick([ch]).data[0] * 1e6
            mean_amp = float(np.mean(ch_data[w_mask]))
            peak_idx = int(np.argmax(ch_data[w_mask]))
            peak_amp = float(ch_data[w_mask][peak_idx])
            peak_lat = float(times_ms[w_mask][peak_idx])

            report["metrics"][f"{cond}_{ch.lower()}_mean_uv"] = round(mean_amp, 2)
            report["metrics"][f"{cond}_{ch.lower()}_peak_uv"] = round(peak_amp, 2)
            report["metrics"][f"{cond}_{ch.lower()}_latency_ms"] = round(peak_lat, 1)

    m = report["metrics"]

    t_pz = m.get("target_pz_mean_uv", 0.0)
    t_cz = m.get("target_cz_mean_uv", 0.0)
    t_fz = m.get("target_fz_mean_uv", 0.0)
    parietal_gradient = t_pz > t_fz

    s_pz = m.get("standard_pz_mean_uv", 0.0)
    oddball_effect = (t_pz - s_pz) >= 1.5

    t_pz_lat = m.get("target_pz_latency_ms", 0.0)
    latency_valid = w_start_ms <= t_pz_lat <= w_end_ms

    d_fz = m.get("distractor_fz_mean_uv", 0.0)
    d_pz = m.get("distractor_pz_mean_uv", 0.0)
    distractor_more_frontal = (d_fz - d_pz) > (t_fz - t_pz)

    report["polich_rules"] = {
        "rule1_parietal_gradient_pz_gt_fz": bool(parietal_gradient),
        "rule1_gradient_values": f"Pz ({t_pz:.2f} µV) > Fz ({t_fz:.2f} µV)" if parietal_gradient else f"Fz ({t_fz:.2f} µV) >= Pz ({t_pz:.2f} µV)",
        "rule2_oddball_effect_target_gt_std": bool(oddball_effect),
        "rule2_diff_uv": round(t_pz - s_pz, 2),
        "rule3_target_pz_latency_valid": bool(latency_valid),
        "rule3_latency_ms": t_pz_lat,
        "rule4_distractor_frontal_tendency": bool(distractor_more_frontal),
        "p300_detected": bool(parietal_gradient and oddball_effect and latency_valid),
        "window_ms": (int(w_start_ms), int(w_end_ms))
    }

    return report


def print_verification_summary(report: Dict[str, Any]) -> None:
    sub = report["subject"]
    m = report["metrics"]
    r = report["polich_rules"]
    w_start, w_end = r.get("window_ms", (300, 500))

    print("=" * 68)
    print(f"       KẾT QUẢ KIỂM CHỨNG SÓNG P300 (POLICH 2007) – {sub}")
    print("=" * 68)
    print(f"Số lượng trial: {report['n_trials']}")
    print("-" * 68)
    print(f"1. Biên độ trung bình cửa sổ {w_start}-{w_end} ms (µV):")
    print(f"   Kênh  |    Standard    |     Target     |   Distractor   | Hiệu (T - S)")
    for ch in ("Fz", "Cz", "Pz"):
        chl = ch.lower()
        s_val = m.get(f"standard_{chl}_mean_uv", np.nan)
        t_val = m.get(f"target_{chl}_mean_uv", np.nan)
        d_val = m.get(f"distractor_{chl}_mean_uv", np.nan)
        diff = t_val - s_val
        print(f"   {ch:<5} | {s_val:>12.2f} µV | {t_val:>12.2f} µV | {d_val:>12.2f} µV | {diff:>10.2f} µV")

    print("-" * 68)
    print(f"2. Đỉnh Target P3b (Peak in {w_start}-{w_end} ms):")
    for ch in ("Fz", "Cz", "Pz"):
        chl = ch.lower()
        p_val = m.get(f"target_{chl}_peak_uv", np.nan)
        l_val = m.get(f"target_{chl}_latency_ms", np.nan)
        print(f"   {ch}: Biên độ đỉnh = {p_val:>5.2f} µV | Độ trễ đỉnh = {l_val:>5.1f} ms")

    print("-" * 68)
    print("3. Kiểm tra các tiêu chuẩn Polich (2007):")
    c1 = "[PASS]" if r["rule1_parietal_gradient_pz_gt_fz"] else "[FAIL]"
    print(f"   {c1} Gradient vùng đỉnh (Pz > Fz) : {r['rule1_gradient_values']}")

    c2 = "[PASS]" if r["rule2_oddball_effect_target_gt_std"] else "[FAIL]"
    print(f"   {c2} Hiệu ứng Oddball (Target > Std): Chênh lệch tại Pz = {r['rule2_diff_uv']:+.2f} µV")

    c3 = "[PASS]" if r["rule3_target_pz_latency_valid"] else "[FAIL]"
    print(f"   {c3} Độ trễ sinh lý P3b ({w_start}-{w_end}ms) : Đạt đỉnh tại {r['rule3_latency_ms']:.1f} ms")

    c4 = "[PASS]" if r["rule4_distractor_frontal_tendency"] else "[INFO]"
    print(f"   {c4} Xu hướng trán của Distractor (P3a) : {r['rule4_distractor_frontal_tendency']}")

    print("-" * 68)
    verdict = "[XÁC NHẬN CÓ P300]" if r["p300_detected"] else "[CHƯA ĐẠT CHUẨN P300]"
    print(f"   -> KẾT LUẬN CHUNG CHO {sub}: {verdict}")
    print("=" * 68 + "\n")


def run_subject_erp_analysis(
    sub: str = "sub-01",
    save_dir: Optional[Path] = None,
    show_plots: bool = False
) -> Dict[str, Any]:
    if save_dir is None:
        save_dir = FIG_DIR
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[V3] Đang phân tích ERP & Topomap cho: {sub}...")
    epochs = load_subject_epochs(sub)

    report = verify_p300(epochs, sub)
    print_verification_summary(report)

    f_midline = save_dir / f"{sub}_midline_erp.png"
    fig_midline = plot_midline_erp(epochs, sub, save_path=f_midline)

    f_fzpz = save_dir / f"{sub}_fz_pz_erp.png"
    fig_fzpz = plot_fz_pz_erp(epochs, sub, save_path=f_fzpz)

    f_diff = save_dir / f"{sub}_difference_waves.png"
    fig_diff = plot_difference_waves(epochs, sub, save_path=f_diff)

    f_topo = save_dir / f"{sub}_scalp_topomap.png"
    fig_topo = plot_erp_topomap(epochs, sub, save_path=f_topo)

    f_summary = save_dir / f"{sub}_p300_summary_topomap.png"
    fig_summary = plot_p300_summary_topomap(epochs, sub, save_path=f_summary)

    if show_plots:
        plt.show()
    else:
        plt.close(fig_midline)
        plt.close(fig_fzpz)
        plt.close(fig_diff)
        plt.close(fig_topo)
        plt.close(fig_summary)

    print(f"[V3] Hoàn tất toàn bộ hình ảnh và phân tích cho {sub} tại: {save_dir}\n")
    return report


def plot_grand_average_erp(
    channels: Tuple[str, ...] = ("Fz", "Pz"),
    window: Tuple[float, float] = P3_WINDOW,
    save_path: Optional[Path] = None
) -> plt.Figure:
    epoch_files = sorted(list(EPOCH_DIR.glob("*_task-oddball_v1-epo.fif")))
    if not epoch_files:
        raise FileNotFoundError(f"Chưa có file epoch nào trong {EPOCH_DIR}")

    subs_included = []
    per_condition: Dict[str, List[np.ndarray]] = {name: [] for name in EVENT_ID}
    times_ms = None

    for p in epoch_files:
        sub_id = p.name.split("_")[0]
        if sub_id in TASK_MISUNDERSTOOD:
            continue
        try:
            ep = mne.read_epochs(p, preload=True, verbose="error")
            if ep.metadata is not None and "correct" in ep.metadata:
                ep = ep[ep.metadata["correct"].values]
            if times_ms is None:
                times_ms = ep.times * 1000.0
            for name in EVENT_ID:
                if name in ep.event_id and len(ep[name]) > 0:
                    per_condition[name].append(ep[name].average(picks=list(channels)).data * 1e6)
            subs_included.append(sub_id)
        except Exception:
            continue

    if not subs_included or not per_condition["target"]:
        raise RuntimeError("Không đủ dữ liệu để tính Grand Average.")

    n_subs = len(subs_included)
    w_start_ms, w_end_ms = window[0] * 1000.0, window[1] * 1000.0
    w_mask = (times_ms >= w_start_ms) & (times_ms <= w_end_ms)

    n_chan = len(channels)
    fig, axes = plt.subplots(1, n_chan, figsize=(5.6 * n_chan, 4.6), sharey=True)
    if n_chan == 1:
        axes = [axes]

    for i, (ax, ch) in enumerate(zip(axes, channels)):
        ax.axvspan(w_start_ms, w_end_ms, color="#efefef", alpha=0.9, label=f"P300 ({int(w_start_ms)}-{int(w_end_ms)}ms)")
        for name in ("standard", "target", "distractor"):
            if not per_condition[name]:
                continue
            cond_stack = np.array(per_condition[name])[:, i, :]
            mean_uv = np.mean(cond_stack, axis=0)
            sem_uv = stats.sem(cond_stack, axis=0) if len(cond_stack) > 1 else np.zeros_like(mean_uv)
            color = COLOR_MAP.get(name, "#333333")
            ax.plot(times_ms, mean_uv, color=color, lw=2.2, label=f"{name.capitalize()}")
            if len(cond_stack) > 1:
                ax.fill_between(times_ms, mean_uv - sem_uv, mean_uv + sem_uv, color=color, alpha=0.18)

            if name == "target" and np.any(w_mask):
                w_times = times_ms[w_mask]
                w_data = mean_uv[w_mask]
                peak_idx = np.argmax(w_data)
                peak_lat = w_times[peak_idx]
                peak_amp = w_data[peak_idx]
                ax.plot(peak_lat, peak_amp, marker="o", markersize=6, color="#b20000")
                ax.annotate(
                    f"{peak_amp:.1f} µV\n{peak_lat:.0f} ms",
                    xy=(peak_lat, peak_amp),
                    xytext=(peak_lat + 15, peak_amp + 0.8),
                    fontsize=9,
                    fontweight="bold",
                    color="#b20000",
                    arrowprops=dict(arrowstyle="->", color="#b20000", lw=0.8)
                )

        if len(per_condition["target"]) > 0 and len(per_condition["standard"]) > 0:
            diff_mean = np.mean(np.array(per_condition["target"])[:, i, :], axis=0) - np.mean(np.array(per_condition["standard"])[:, i, :], axis=0)
            ax.plot(times_ms, diff_mean, color=COLOR_MAP["diff_target"], lw=1.6, ls="--", label="Target - Standard")

        ax.axvline(0, color="k", lw=0.8, ls=":")
        ax.axhline(0, color="k", lw=0.8, ls=":")
        sub_desc = f"N={n_subs}" if n_subs > 1 else f"sub-01 (N=1)"
        ax.set_title(f"Grand average – {ch} ({sub_desc})", fontsize=12.5, fontweight="bold")
        ax.set_xlabel("Time (ms)", fontsize=10.5)
        ax.grid(True, linestyle=":", alpha=0.5)

    axes[0].set_ylabel("Amplitude (µV)", fontsize=10.5)
    axes[0].legend(loc="upper left", frameon=False, fontsize=8.5)
    fig.suptitle(f"Grand average ERP (Fz, Pz) – Oddball paradigm\nP300 analysis window: [{int(w_start_ms)}, {int(w_end_ms)}] ms", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"  -> Saved Grand Average figure: {save_path}")

    return fig


def plot_population_summary(
    qc_csv_path: Optional[Path] = None,
    save_path: Optional[Path] = None
) -> plt.Figure:
    if qc_csv_path is None:
        qc_csv_path = QC_SUMMARY
    qc_csv_path = Path(qc_csv_path)

    if not qc_csv_path.exists():
        raise FileNotFoundError(f"File not found: {qc_csv_path}")

    df = pd.read_csv(qc_csv_path)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    ax1 = axes[0]
    data_amps = [
        df["pz_p3_standard_uv"].dropna(),
        df["pz_p3_distractor_uv"].dropna(),
        df["pz_p3_target_uv"].dropna(),
    ]
    box = ax1.boxplot(
        data_amps,
        tick_labels=["Standard", "Distractor (P3a)", "Target (P3b)"],
        patch_artist=True,
        medianprops=dict(color="black", lw=1.8),
        widths=0.55
    )
    colors = [COLOR_MAP["standard"], COLOR_MAP["distractor"], COLOR_MAP["target"]]
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.65)

    for idx, (vals, color) in enumerate(zip(data_amps, colors), start=1):
        x = np.random.normal(idx, 0.04, size=len(vals))
        ax1.scatter(x, vals, alpha=0.8, color=color, edgecolors="k", s=30, zorder=3)

    ax1.set_ylabel("Pz mean amplitude in 300–600 ms (µV)", fontsize=11)
    ax1.set_title("A. Pz amplitude distribution across conditions (N=42)", fontsize=11.5, fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.6)

    ax2 = axes[1]
    normal_mask = ~df["task_misunderstood"]
    mis_mask = df["task_misunderstood"]

    group_data = [
        df.loc[normal_mask, "pz_p3_target_uv"].dropna(),
        df.loc[mis_mask, "pz_p3_target_uv"].dropna(),
        df.loc[normal_mask, "false_alarm_distractor_pct"].dropna(),
        df.loc[mis_mask, "false_alarm_distractor_pct"].dropna()
    ]
    x_positions = [1, 2, 4, 5]
    bplot2 = ax2.boxplot(
        group_data,
        positions=x_positions,
        patch_artist=True,
        medianprops=dict(color="black", lw=1.8),
        widths=0.6
    )
    b_colors = ["#2b83ba", "#d7191c", "#2b83ba", "#d7191c"]
    for patch, col in zip(bplot2["boxes"], b_colors):
        patch.set_facecolor(col)
        patch.set_alpha(0.65)

    ax2.set_xticks([1.5, 4.5])
    ax2.set_xticklabels(["Pz target (µV)", "False alarm rate (%)"], fontsize=10.5)
    ax2.set_title("B. Impact of task misunderstanding (N=39 vs. N=3)", fontsize=11.5, fontweight="bold")
    ax2.grid(True, linestyle=":", alpha=0.6)

    legend_elements = [
        Patch(facecolor="#2b83ba", alpha=0.65, label="Normal responders (N=39)"),
        Patch(facecolor="#d7191c", alpha=0.65, label="Misunderstood task (sub-02/14/17)")
    ]
    ax2.legend(handles=legend_elements, loc="upper right", fontsize=8.5)

    ax3 = axes[2]
    clean_df = df[normal_mask].dropna(subset=["target_rt_ms", "pz_p3_target_uv"])
    x_rt = clean_df["target_rt_ms"]
    y_p3 = clean_df["pz_p3_target_uv"]

    ax3.scatter(x_rt, y_p3, color=COLOR_MAP["target"], edgecolors="k", s=45, alpha=0.85, zorder=3)

    if len(x_rt) > 2:
        slope, intercept, r_value, p_value, std_err = stats.linregress(x_rt, y_p3)
        x_line = np.linspace(x_rt.min(), x_rt.max(), 50)
        y_line = slope * x_line + intercept
        ax3.plot(x_line, y_line, color="#b20000", lw=2, ls="--",
                 label=f"Linear fit: r = {r_value:.2f} (p = {p_value:.3f})")
        ax3.legend(loc="upper right", fontsize=9)

    ax3.set_xlabel("Target reaction time (ms)", fontsize=10.5)
    ax3.set_ylabel("Target Pz amplitude (µV)", fontsize=10.5)
    ax3.set_title("C. Reaction time vs. P3b amplitude (Polich, 2007)", fontsize=11.5, fontweight="bold")
    ax3.grid(True, linestyle=":", alpha=0.6)

    fig.suptitle(
        "Population statistical analysis (42 subjects, GigaDB 100990)\n"
        "Oddball effect and cognitive segregation",
        fontsize=13, fontweight="bold", y=1.03
    )
    fig.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"  -> Saved Population Summary figure: {save_path}")

    return fig


def export_group_p300_table(
    qc_csv_path: Optional[Path] = None,
    save_path: Optional[Path] = None
) -> pd.DataFrame:
    if qc_csv_path is None:
        qc_csv_path = QC_SUMMARY
    qc_csv_path = Path(qc_csv_path)
    df = pd.read_csv(qc_csv_path)

    table_df = pd.DataFrame()
    table_df["subject"] = df["subject"]
    table_df["group"] = df["task_misunderstood"].apply(lambda x: "Task-Misunderstood" if x else "Normal")
    table_df["pz_target_uv"] = df["pz_p3_target_uv"]
    table_df["pz_standard_uv"] = df["pz_p3_standard_uv"]
    table_df["pz_distractor_uv"] = df["pz_p3_distractor_uv"]
    table_df["oddball_diff_uv"] = (df["pz_p3_target_uv"] - df["pz_p3_standard_uv"]).round(2)
    table_df["oddball_pass"] = table_df["oddball_diff_uv"] >= 1.5
    table_df["target_rt_ms"] = df["target_rt_ms"]
    table_df["false_alarm_distractor_pct"] = df["false_alarm_distractor_pct"]
    table_df["target_omission_pct"] = df["target_omission_pct"]
    table_df["p300_verified"] = (~df["task_misunderstood"]) & table_df["oddball_pass"]

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        table_df.to_csv(save_path, index=False)
        print(f"  -> Đã lưu bảng thống kê nhóm P300: {save_path}")

    return table_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="V3: ERP & Topomap analysis based on Polich (2007).")
    parser.add_argument("--sub", type=str, default="sub-01", help="Mã người tham gia (mặc định: sub-01)")
    parser.add_argument("--out-dir", type=str, default=str(FIG_DIR), help="Thư mục xuất ảnh kết quả")
    parser.add_argument("--population", action="store_true", help="Vẽ thống kê quần thể 42 người từ qc_summary.csv")
    parser.add_argument("--grand-average", action="store_true", help="Vẽ Grand Average ERP (Fz, Pz)")
    parser.add_argument("--table", action="store_true", help="Xuất bảng số liệu kiểm định P300 nhóm 42 người")
    parser.add_argument("--show", action="store_true", help="Hiển thị đồ họa tương tác")
    args = parser.parse_args()

    if args.population:
        plot_population_summary(save_path=Path(args.out_dir) / "population_p300_summary.png")
    elif args.grand_average:
        plot_grand_average_erp(save_path=Path(args.out_dir) / "grand_average_fz_pz.png")
    elif args.table:
        export_group_p300_table(save_path=TABLE_DIR / "v3_p300_group_results.csv")
    else:
        run_subject_erp_analysis(sub=args.sub, save_dir=Path(args.out_dir), show_plots=args.show)
