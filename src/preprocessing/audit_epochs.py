"""Audit the V2 hand-off without downloading or rerunning raw-data preprocessing.

Run from the project root: python -m src.preprocessing.audit_epochs
"""
from __future__ import annotations

import json
from pathlib import Path

import mne
import numpy as np
import pandas as pd

from config import (BASELINE, EVENT_ID, FEATURE_DIR, N_CHANNELS, QC_SUMMARY,
                    SFREQ, SUBJECTS, TABLE_DIR, TASK_MISUNDERSTOOD, TMAX, TMIN)
from src.utils.io import epoch_path


def audit_epochs(save_path: Path | None = None) -> dict:
    qc = pd.read_csv(QC_SUMMARY).set_index("subject")
    if not qc.index.is_unique or set(qc.index) != set(SUBJECTS):
        raise ValueError("QC subjects do not match config.SUBJECTS")
    features = pd.read_csv(FEATURE_DIR / "all_subjects_features.csv")
    feature_names = [c for c in features if c not in (
        "subject", "trial_idx", "label", "condition", "rt_ms")]
    if (len(feature_names) != 402 or not np.isfinite(features[feature_names].to_numpy()).all()
            or features.duplicated(["subject", "trial_idx"]).any()
            or set(features["subject"]) != set(SUBJECTS) - TASK_MISUNDERSTOOD):
        raise ValueError("Feature matrix has invalid columns, values, trials or subjects")
    reference_channels = None
    reference_positions = None
    rows = []
    for sub in SUBJECTS:
        ep = mne.read_epochs(epoch_path(sub), preload=True, verbose="error")
        data = ep.get_data(copy=False)
        meta = ep.metadata
        checks = {
            "finite_data": bool(np.isfinite(data).all()),
            "shape": data.shape[1:] == (N_CHANNELS, round((TMAX - TMIN) * SFREQ) + 1),
            "sampling_rate": ep.info["sfreq"] == SFREQ,
            "time_axis": bool(np.allclose(ep.times, TMIN + np.arange(301) / SFREQ)),
            "filter_info": np.isclose(ep.info["highpass"], 0.5) and np.isclose(ep.info["lowpass"], 40),
            "baseline_info": ep.baseline is not None and np.allclose(ep.baseline, BASELINE),
            "event_id": ep.event_id == EVENT_ID,
            "no_unresolved_bad_channels": not ep.info["bads"],
            "metadata": meta is not None and {
                "subject", "trial", "condition", "responded", "rt_ms", "correct"
            }.issubset(meta.columns),
        }
        if not checks["metadata"]:
            raise ValueError(f"{sub}: metadata columns missing")
        positions = np.array([ch["loc"][:3] for ch in ep.info["chs"]])
        if reference_channels is None:
            reference_channels, reference_positions = ep.ch_names, positions
        checks["channel_order"] = ep.ch_names == reference_channels
        checks["montage"] = bool(np.isfinite(positions).all()
                                 and (np.linalg.norm(positions, axis=1) > 0).all()
                                 and np.allclose(positions, reference_positions))
        checks["metadata_subject"] = bool((meta["subject"] == sub).all())
        checks["metadata_trial"] = not meta["trial"].duplicated().any()
        checks["event_labels"] = bool(np.array_equal(
            meta["condition"].map(EVENT_ID).to_numpy(), ep.events[:, 2]))
        checks["boolean_metadata"] = meta["correct"].dtype == bool and meta["responded"].dtype == bool
        target = meta["condition"].to_numpy() == "target"
        responded = meta["responded"].to_numpy()
        rt = meta["rt_ms"].to_numpy()
        expected_correct = np.where(target, responded & (rt >= 100) & (rt <= 1400), ~responded)
        checks["correctness"] = bool(np.array_equal(meta["correct"], expected_correct))
        baseline_mask = (ep.times >= BASELINE[0]) & (ep.times <= BASELINE[1])
        max_baseline_uv = float(np.abs(data[:, :, baseline_mask].mean(axis=2)).max() * 1e6)
        max_ptp_uv = float(np.ptp(data, axis=2).max() * 1e6)
        # MNE corrects the baseline at 1000 Hz before decim=4. The retained
        # 250 Hz sample mean need not be zero; record it without rebaselining
        # the supplied data (which would invalidate the existing features).
        checks["rejection_threshold"] = max_ptp_uv <= 150.001
        checks["average_reference"] = float(np.abs(data.mean(axis=1)).max() * 1e6) < 0.001
        checks["qc_counts"] = all(
            int((ep.events[:, 2] == code).sum()) == int(qc.loc[sub, f"n_{name}_kept"])
            for name, code in EVENT_ID.items())
        qc_mask = (ep.times >= 0.3) & (ep.times <= 0.6)
        pz = ep.ch_names.index("Pz")
        checks["qc_amplitudes"] = all(
            np.isclose(data[ep.events[:, 2] == code, pz, :][:, qc_mask].mean() * 1e6,
                       qc.loc[sub, f"pz_p3_{name}_uv"], atol=0.0051, rtol=0)
            for name, code in EVENT_ID.items())
        binary = meta["correct"].to_numpy() & np.isin(ep.events[:, 2], [5, 6])
        selected = features[features["subject"] == sub].sort_values("trial_idx")
        if sub not in TASK_MISUNDERSTOOD:
            checks["feature_labels"] = bool(
                np.array_equal(selected["trial_idx"], np.arange(binary.sum()))
                and np.array_equal(selected["label"], (ep.events[binary, 2] == 6).astype(int))
                and np.array_equal(selected["condition"], meta.loc[binary, "condition"]))
            checks["feature_rt"] = bool(np.allclose(selected["rt_ms"], rt[binary], equal_nan=True))
            mask = (ep.times >= 0.4) & (ep.times <= 0.6)
            recomputed = data[binary, pz, :][:, mask].mean(axis=1) * 1e6
            checks["feature_pz_p3b_mean"] = bool(np.allclose(
                selected["Pz_p3b_mean"], recomputed, rtol=1e-5, atol=1e-6))
        checks = {name: bool(value) for name, value in checks.items()}
        rows.append({"subject": sub, "n_epochs": len(ep), "n_correct_binary": int(binary.sum()),
                     "max_baseline_residual_uv": max_baseline_uv, "max_channel_ptp_uv": max_ptp_uv,
                     "checks": checks, "passed": all(checks.values())})
        print(f"[{sub}] {'PASS' if rows[-1]['passed'] else 'FAIL'}: "
              f"{[name for name, ok in checks.items() if not ok]}", flush=True)
    result = {
        "passed": all(row["passed"] for row in rows),
        "n_subjects": len(rows),
        "n_epochs": sum(row["n_epochs"] for row in rows),
        "n_classification_trials": len(features),
        "scope": "Epoch/feature hand-off audit; raw recordings and ICA solutions were not rerun.",
        "baseline_note": "Baseline metadata checked; residual 250 Hz means are descriptive because correction preceded decimation at 1000 Hz.",
        "subjects": rows,
    }
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    result = audit_epochs(TABLE_DIR / "v2_epoch_audit.json")
    print(f"V2 audit: {result['passed']}; subjects={result['n_subjects']}; epochs={result['n_epochs']}")
    raise SystemExit(0 if result["passed"] else 1)
