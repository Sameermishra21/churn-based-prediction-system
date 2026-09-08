"""
explainability.py

Phase 9: Explainable AI (SHAP).

Provides:
  - GLOBAL explanation: which features matter most across the whole
    test set (SHAP summary plot, bar plot, ranked importance table).
  - LOCAL explanation: for one customer, which specific factors pushed
    their prediction up (toward churn) or down (toward retention).

Everything here is computed from the actual saved model's SHAP values
-- no explanation text is templated/fabricated independent of the
values. TreeExplainer is used since the final model (XGBoost) and the
baseline tree models are all tree-based; this module is written
generically enough to also work with Random Forest / Decision Tree if
the selected model changes.
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGURES_DIR = os.path.join(PROJECT_ROOT, "outputs", "figures")


def build_explainer(model, X_train_sample: np.ndarray = None) -> shap.TreeExplainer:
    """Build a SHAP TreeExplainer for a tree-based model.

    Uses feature_perturbation='tree_path_dependent', which does not
    require (or accept) a background dataset -- passing one forces
    SHAP's 'interventional' mode, which hit
    'NotImplementedError: Categorical split is not yet supported' for
    this XGBoost version during testing. tree_path_dependent estimates
    the expected value from the training data statistics baked into
    the trees themselves and works reliably across XGBoost/sklearn
    tree models alike, which is why it's used here instead.
    """
    return shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")


def compute_shap_values(explainer: shap.TreeExplainer, X: np.ndarray) -> np.ndarray:
    """Return SHAP values for X, normalized to a single 2D array of
    shape (n_samples, n_features) for the POSITIVE (churn) class,
    regardless of whether the underlying explainer returns a list
    (older SHAP/multiclass convention) or a single array (XGBoost
    binary convention) or an Explanation object.
    """
    raw = explainer.shap_values(X)
    if isinstance(raw, list):
        # [class_0_values, class_1_values] convention
        return np.asarray(raw[1])
    return np.asarray(raw)


def global_feature_importance(shap_values: np.ndarray, feature_names: list[str]) -> pd.DataFrame:
    """Rank features by mean |SHAP value| across all samples -- the
    standard global importance measure."""
    mean_abs = np.abs(shap_values).mean(axis=0)
    importance = pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs})
    importance = importance.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    return importance


def plot_shap_summary(shap_values: np.ndarray, X: np.ndarray, feature_names: list[str], save_path: str = None):
    fig = plt.figure(figsize=(8, 10))
    shap.summary_plot(shap_values, X, feature_names=feature_names, show=False)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    return fig


def plot_shap_bar(shap_values: np.ndarray, feature_names: list[str], save_path: str = None, top_n: int = 15):
    importance = global_feature_importance(shap_values, feature_names).head(top_n)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(importance["feature"][::-1], importance["mean_abs_shap"][::-1], color="#1565c0")
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(f"Top {top_n} Features by Global SHAP Importance")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    return fig


def local_explanation(
    shap_values_row: np.ndarray,
    feature_names: list[str],
    base_value: float,
    predicted_proba: float,
    top_n: int = 5,
) -> dict:
    """Build a structured local explanation for ONE customer: the top
    features pushing risk up and the top features pushing risk down,
    derived directly from that row's SHAP values (no fabrication).
    """
    contributions = pd.DataFrame({
        "feature": feature_names,
        "shap_value": shap_values_row,
    }).sort_values("shap_value", ascending=False)

    increasing = contributions[contributions["shap_value"] > 0].head(top_n)
    decreasing = contributions[contributions["shap_value"] < 0].tail(top_n).iloc[::-1]

    return {
        "predicted_probability": float(predicted_proba),
        "base_value": float(base_value),
        "factors_increasing_risk": list(
            zip(increasing["feature"], increasing["shap_value"].round(4))
        ),
        "factors_decreasing_risk": list(
            zip(decreasing["feature"], decreasing["shap_value"].round(4))
        ),
    }


def format_local_explanation(explanation: dict) -> str:
    lines = [f"Churn Probability: {explanation['predicted_probability'] * 100:.1f}%", ""]
    lines.append("Factors increasing churn risk:")
    for feat, val in explanation["factors_increasing_risk"]:
        lines.append(f"  + {feat} (impact: +{val:.4f})")
    lines.append("")
    lines.append("Factors reducing churn risk:")
    for feat, val in explanation["factors_decreasing_risk"]:
        lines.append(f"  - {feat} (impact: {val:.4f})")
    return "\n".join(lines)


if __name__ == "__main__":
    import joblib

    from src.data_preprocessing import load_and_validate_data, preprocess_data

    df, _ = load_and_validate_data()
    data = preprocess_data(df)
    bundle = joblib.load(os.path.join(PROJECT_ROOT, "models", "best_model.pkl"))
    model, feature_names = bundle["model"], bundle["feature_names"]

    print(f"Building SHAP explainer for {type(model).__name__}...")
    explainer = build_explainer(model)

    shap_values = compute_shap_values(explainer, data.X_test)
    print(f"SHAP values shape: {shap_values.shape} (expect {data.X_test.shape})")

    os.makedirs(FIGURES_DIR, exist_ok=True)

    print("\n=== Global Feature Importance (top 15) ===")
    importance = global_feature_importance(shap_values, feature_names)
    print(importance.head(15).to_string(index=False))

    plot_shap_bar(shap_values, feature_names, save_path=os.path.join(FIGURES_DIR, "shap_bar.png"))
    plot_shap_summary(shap_values, data.X_test, feature_names, save_path=os.path.join(FIGURES_DIR, "shap_summary.png"))
    print(f"\nSaved shap_bar.png and shap_summary.png to {FIGURES_DIR}")

    print("\n=== Local explanation for test customer #1 ===")
    proba = model.predict_proba(data.X_test[[1]])[0, 1]
    base_value = explainer.expected_value
    if isinstance(base_value, (list, np.ndarray)):
        base_value = base_value[1] if len(np.atleast_1d(base_value)) > 1 else base_value[0]
    explanation = local_explanation(shap_values[1], feature_names, base_value, proba)
    print(format_local_explanation(explanation))
