from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import FEATURE_DIR, FIG_DIR, MODEL_DIR, RANDOM_SEED, TABLE_DIR
from src.features.extract_features import load_features

COLOR_PALETTE = {
    "Dummy": "#7f7f7f",
    "LDA": "#1f77b4",
    "Logistic Regression": "#2ca02c",
    "RBF SVM": "#ff7f0e",
    "Random Forest": "#9467bd",
}


def get_models() -> Dict[str, Pipeline]:
    return {
        "Dummy": Pipeline([
            ("clf", DummyClassifier(strategy="stratified", random_state=RANDOM_SEED))
        ]),
        "LDA": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"))
        ]),
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RANDOM_SEED))
        ]),
        "RBF SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", class_weight="balanced", probability=True, cache_size=1000, random_state=RANDOM_SEED))
        ]),
        "Random Forest": Pipeline([
            ("clf", RandomForestClassifier(n_estimators=100, class_weight="balanced_subsample", random_state=RANDOM_SEED, n_jobs=-1))
        ]),
    }


def evaluate_model(
    name: str,
    pipeline: Pipeline,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int = 5,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    n_unique_groups = len(np.unique(groups))
    if n_unique_groups >= n_splits:
        splitter = GroupKFold(n_splits=n_splits)
        splits = list(splitter.split(X, y, groups))
    else:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
        splits = list(splitter.split(X, y))

    oof_preds = np.zeros(len(y), dtype=int)
    oof_probs = np.zeros(len(y), dtype=float)

    fold_metrics: List[Dict[str, float]] = []

    for train_idx, test_idx in splits:
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        pipeline.fit(X_train, y_train)

        if hasattr(pipeline, "predict_proba"):
            probs = pipeline.predict_proba(X_test)[:, 1]
        elif hasattr(pipeline, "decision_function"):
            decision = pipeline.decision_function(X_test)
            probs = 1.0 / (1.0 + np.exp(-decision))
        else:
            probs = pipeline.predict(X_test).astype(float)

        preds = (probs >= 0.5).astype(int)

        oof_preds[test_idx] = preds
        oof_probs[test_idx] = probs

        tn, fp, fn, tp = confusion_matrix(y_test, preds, labels=[0, 1]).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        fold_metrics.append({
            "balanced_accuracy": balanced_accuracy_score(y_test, preds),
            "roc_auc": roc_auc_score(y_test, probs),
            "pr_auc": average_precision_score(y_test, probs),
            "f1_macro": f1_score(y_test, preds, average="macro"),
            "f1_target": f1_score(y_test, preds, pos_label=1, zero_division=0),
            "sensitivity": sensitivity,
            "specificity": specificity,
        })

    fold_df = pd.DataFrame(fold_metrics)
    mean_metrics = fold_df.mean().to_dict()
    std_metrics = fold_df.std().to_dict()

    tn, fp, fn, tp = confusion_matrix(y, oof_preds, labels=[0, 1]).ravel()
    overall_sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    overall_specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    summary = {
        "model": name,
        "balanced_accuracy_mean": mean_metrics["balanced_accuracy"],
        "balanced_accuracy_std": std_metrics["balanced_accuracy"],
        "roc_auc_mean": mean_metrics["roc_auc"],
        "roc_auc_std": std_metrics["roc_auc"],
        "pr_auc_mean": mean_metrics["pr_auc"],
        "pr_auc_std": std_metrics["pr_auc"],
        "f1_macro_mean": mean_metrics["f1_macro"],
        "f1_macro_std": std_metrics["f1_macro"],
        "f1_target_mean": mean_metrics["f1_target"],
        "f1_target_std": std_metrics["f1_target"],
        "sensitivity_mean": mean_metrics["sensitivity"],
        "sensitivity_std": std_metrics["sensitivity"],
        "specificity_mean": mean_metrics["specificity"],
        "specificity_std": std_metrics["specificity"],
        "oof_balanced_accuracy": balanced_accuracy_score(y, oof_preds),
        "oof_roc_auc": roc_auc_score(y, oof_probs),
        "oof_pr_auc": average_precision_score(y, oof_probs),
        "oof_f1_macro": f1_score(y, oof_preds, average="macro"),
        "oof_sensitivity": overall_sensitivity,
        "oof_specificity": overall_specificity,
    }

    return summary, oof_preds, oof_probs


def plot_roc_curves(
    y_true: np.ndarray,
    oof_probs_dict: Dict[str, np.ndarray],
    save_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5.5))

    for name, probs in oof_probs_dict.items():
        fpr, tpr, _ = roc_curve(y_true, probs)
        score = roc_auc_score(y_true, probs)
        color = COLOR_PALETTE.get(name, "#333333")
        ax.plot(fpr, tpr, lw=1.8, color=color, label=f"{name} (AUC = {score:.3f})")

    ax.plot([0, 1], [0, 1], linestyle="--", lw=1.2, color="#999999", label="Chance (AUC = 0.500)")
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel("False positive rate (1 - specificity)", fontsize=10.5)
    ax.set_ylabel("True positive rate (sensitivity)", fontsize=10.5)
    ax.set_title("Receiver operating characteristic (ROC) curves", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="lower right", frameon=True, fontsize=9)
    fig.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_pr_curves(
    y_true: np.ndarray,
    oof_probs_dict: Dict[str, np.ndarray],
    save_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    prevalence = float((y_true == 1).mean())

    for name, probs in oof_probs_dict.items():
        precision, recall, _ = precision_recall_curve(y_true, probs)
        score = average_precision_score(y_true, probs)
        color = COLOR_PALETTE.get(name, "#333333")
        ax.plot(recall, precision, lw=1.8, color=color, label=f"{name} (AP = {score:.3f})")

    ax.axhline(prevalence, linestyle="--", lw=1.2, color="#999999", label=f"Prevalence ({prevalence:.1%})")
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel("Recall (sensitivity)", fontsize=10.5)
    ax.set_ylabel("Precision (positive predictive value)", fontsize=10.5)
    ax.set_title("Precision-recall curves", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", frameon=True, fontsize=9)
    fig.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_model_comparison(summary_df: pd.DataFrame, save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    models_list = summary_df["model"].tolist()
    x = np.arange(len(models_list))
    width = 0.35

    ax.bar(
        x - width / 2,
        summary_df["balanced_accuracy_mean"],
        width,
        yerr=summary_df["balanced_accuracy_std"],
        label="Balanced accuracy",
        color="#4c72b0",
        capsize=3,
    )
    ax.bar(
        x + width / 2,
        summary_df["roc_auc_mean"],
        width,
        yerr=summary_df["roc_auc_std"],
        label="ROC-AUC",
        color="#55a868",
        capsize=3,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(models_list, fontsize=10)
    ax.set_ylabel("Score", fontsize=10.5)
    ax.set_title("Baseline model comparison (mean ± std across folds)", fontsize=12, fontweight="bold")
    ax.set_ylim([0, 1.05])
    ax.axhline(0.5, linestyle="--", color="#999999", lw=1, label="Chance level (0.50)")
    ax.grid(True, linestyle=":", alpha=0.5, axis="y")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def resolve_data_path(custom_path: str | None = None) -> Path:
    if custom_path:
        p = Path(custom_path)
        if p.exists():
            return p
        raise FileNotFoundError(f"Feature file not found: {p}")

    default_all = FEATURE_DIR / "all_subjects_features.csv"
    if default_all.exists():
        return default_all

    sub01 = FEATURE_DIR / "sub-01_features.csv"
    if sub01.exists():
        return sub01

    raise FileNotFoundError(
        f"No feature files found in {FEATURE_DIR}. Expected all_subjects_features.csv or sub-01_features.csv"
    )


def run_pipeline(
    data_path: Path | None = None,
    n_splits: int = 5,
    save_models: bool = False,
) -> pd.DataFrame:
    resolved_path = resolve_data_path(str(data_path) if data_path else None)
    print(f"Loading features from: {resolved_path}")

    X, y, groups, feature_names = load_features(resolved_path)
    print(f"Dataset: {X.shape[0]} trials, {X.shape[1]} features, {len(np.unique(groups))} subjects")
    print(f"Target prevalence: {(y == 1).mean():.1%} ({int((y == 1).sum())} target, {int((y == 0).sum())} standard)")

    models = get_models()
    results: List[Dict[str, float]] = []
    oof_probs_dict: Dict[str, np.ndarray] = {}

    for name, pipeline in models.items():
        print(f"Evaluating {name}...")
        summary, _, oof_probs = evaluate_model(name, pipeline, X, y, groups, n_splits=n_splits)
        results.append(summary)
        oof_probs_dict[name] = oof_probs

        if save_models:
            pipeline.fit(X, y)
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            joblib.dump(pipeline, MODEL_DIR / f"{name.lower().replace(' ', '_')}.joblib")

    summary_df = pd.DataFrame(results)

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = TABLE_DIR / "baseline_metrics.csv"
    summary_df.to_csv(metrics_path, index=False)
    np.savez_compressed(TABLE_DIR / "baseline_oof.npz", y=y, groups=groups,
                        **{name.lower().replace(" ", "_"): probs
                           for name, probs in oof_probs_dict.items()})
    print(f"Saved metrics table: {metrics_path}")

    roc_path = FIG_DIR / "baseline_roc_curves.png"
    pr_path = FIG_DIR / "baseline_pr_curves.png"
    comp_path = FIG_DIR / "baseline_model_comparison.png"
    plot_roc_curves(y, oof_probs_dict, roc_path)
    plot_pr_curves(y, oof_probs_dict, pr_path)
    plot_model_comparison(summary_df, comp_path)
    print(f"Saved ROC curve: {roc_path}")
    print(f"Saved PR curve: {pr_path}")
    print(f"Saved model comparison: {comp_path}")

    report_fig_dir = ROOT / "report" / "figures"
    if report_fig_dir.exists():
        plot_roc_curves(y, oof_probs_dict, report_fig_dir / "baseline_roc_curves.png")
        plot_pr_curves(y, oof_probs_dict, report_fig_dir / "baseline_pr_curves.png")
        plot_model_comparison(summary_df, report_fig_dir / "baseline_model_comparison.png")

    display_cols = [
        "model",
        "balanced_accuracy_mean",
        "roc_auc_mean",
        "pr_auc_mean",
        "f1_macro_mean",
        "sensitivity_mean",
        "specificity_mean",
    ]
    print("\nBenchmark Summary:")
    print(summary_df[display_cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    return summary_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="V5: Train and evaluate standard machine learning baselines.")
    parser.add_argument("--data", type=str, default=None, help="Path to features CSV file")
    parser.add_argument("--splits", type=int, default=5, help="Number of cross-validation splits")
    parser.add_argument("--save-models", action="store_true", help="Fit and save trained models to disk")
    args = parser.parse_args()

    run_pipeline(data_path=Path(args.data) if args.data else None, n_splits=args.splits, save_models=args.save_models)
