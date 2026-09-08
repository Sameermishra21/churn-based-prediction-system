"""
evaluate_models.py

Phase 6: Model evaluation.

Computes Accuracy, Precision, Recall, F1, ROC-AUC for every trained
model against the SAME held-out test split, plus confusion matrices,
ROC curves, and Precision-Recall curves. Every number here comes from
sklearn.metrics against real predictions -- nothing is hardcoded.
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGURES_DIR = os.path.join(PROJECT_ROOT, "outputs", "figures")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "outputs", "reports")


def evaluate_model(model, X_test, y_test) -> dict:
    """Compute the required metric set for a single fitted model against
    the test split. Returns a flat dict suitable for one row of the
    comparison table.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_proba),
    }


def build_comparison_table(fitted_models: dict, X_test, y_test) -> pd.DataFrame:
    """Evaluate every model and return the Model comparison table
    (section 13), sorted by ROC-AUC descending -- ROC-AUC is threshold-
    independent, which is why it's used for ranking here rather than
    accuracy or F1 alone.
    """
    rows = {}
    for name, model in fitted_models.items():
        rows[name] = evaluate_model(model, X_test, y_test)
    table = pd.DataFrame(rows).T
    table = table.sort_values("ROC-AUC", ascending=False)
    return table


def get_classification_reports(fitted_models: dict, X_test, y_test) -> dict:
    reports = {}
    for name, model in fitted_models.items():
        y_pred = model.predict(X_test)
        reports[name] = classification_report(y_test, y_pred, target_names=["No Churn", "Churn"])
    return reports


def plot_confusion_matrices(fitted_models: dict, X_test, y_test, save_path: str = None):
    n = len(fitted_models)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]
    for ax, (name, model) in zip(axes, fitted_models.items()):
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(cm, display_labels=["No Churn", "Churn"])
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(name)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    return fig


def plot_roc_curves(fitted_models: dict, X_test, y_test, save_path: str = None):
    fig, ax = plt.subplots(figsize=(6, 6))
    for name, model in fitted_models.items():
        RocCurveDisplay.from_estimator(model, X_test, y_test, name=name, ax=ax)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    ax.set_title("ROC Curves — Model Comparison")
    ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    return fig


def plot_precision_recall_curves(fitted_models: dict, X_test, y_test, save_path: str = None):
    fig, ax = plt.subplots(figsize=(6, 6))
    for name, model in fitted_models.items():
        PrecisionRecallDisplay.from_estimator(model, X_test, y_test, name=name, ax=ax)
    ax.set_title("Precision-Recall Curves — Model Comparison")
    ax.legend(loc="lower left", fontsize=8)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    return fig


if __name__ == "__main__":
    import joblib

    from src.data_preprocessing import load_and_validate_data, preprocess_data
    from src.train_models import MODELS_DIR, train_all_models

    df, _ = load_and_validate_data()
    data = preprocess_data(df)
    fitted_models = train_all_models(data.X_train, data.y_train)

    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    table = build_comparison_table(fitted_models, data.X_test, data.y_test)
    print("=== Model Comparison Table (sorted by ROC-AUC) ===")
    print(table.round(4).to_string())
    table.round(4).to_csv(os.path.join(REPORTS_DIR, "model_comparison.csv"))

    print("\n=== Classification Reports ===")
    reports = get_classification_reports(fitted_models, data.X_test, data.y_test)
    for name, report in reports.items():
        print(f"\n--- {name} ---")
        print(report)

    plot_confusion_matrices(
        fitted_models, data.X_test, data.y_test,
        save_path=os.path.join(FIGURES_DIR, "confusion_matrices.png"),
    )
    plot_roc_curves(
        fitted_models, data.X_test, data.y_test,
        save_path=os.path.join(FIGURES_DIR, "roc_curves.png"),
    )
    plot_precision_recall_curves(
        fitted_models, data.X_test, data.y_test,
        save_path=os.path.join(FIGURES_DIR, "precision_recall_curves.png"),
    )
    print(f"\nFigures saved to {FIGURES_DIR}")
    print(f"Comparison table saved to {REPORTS_DIR}/model_comparison.csv")
