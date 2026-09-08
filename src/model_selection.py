"""
model_selection.py

Phase 8: Select and save the final model.

Trains all 4 baselines (Phase 5) + tunes RF and XGBoost (Phase 7),
builds one combined comparison table across all 6 candidates, selects
a final model using an explicit, documented rule (not just "highest
accuracy"), and saves it as models/best_model.pkl plus
models/model_metadata.pkl.

Selection rule (per section 14's guidance to weigh Recall/F1/ROC-AUC
for churn prediction, not accuracy):
  1. Rank candidates by ROC-AUC (threshold-independent, so it's the
     fairest single comparison across models with different default
     operating points).
  2. Among candidates within 0.01 ROC-AUC of the top score, prefer the
     one with higher Recall -- for retention, missing an actual
     churner (false negative) is assumed costlier than an unnecessary
     retention offer to a loyal customer (false positive).
"""

from __future__ import annotations

import os

import joblib
import pandas as pd

from src.data_preprocessing import load_and_validate_data, preprocess_data
from src.evaluate_models import build_comparison_table, evaluate_model
from src.train_models import MODELS_DIR, train_all_models
from src.tune_models import tune_random_forest, tune_xgboost

REPORTS_DIR = os.path.join(os.path.dirname(MODELS_DIR), "outputs", "reports")


def select_final_model(comparison_table: pd.DataFrame, roc_auc_tolerance: float = 0.01) -> str:
    """Apply the documented selection rule to a comparison table indexed
    by model name with an 'ROC-AUC' and 'Recall' column. Returns the
    chosen model name.
    """
    top_auc = comparison_table["ROC-AUC"].max()
    contenders = comparison_table[comparison_table["ROC-AUC"] >= top_auc - roc_auc_tolerance]
    chosen = contenders["Recall"].idxmax()
    return chosen


def run_full_selection():
    df, _ = load_and_validate_data()
    data = preprocess_data(df)

    print("Training 4 baseline models...")
    baseline_models = train_all_models(data.X_train, data.y_train)

    print("Tuning Random Forest...")
    rf_search = tune_random_forest(data.X_train, data.y_train)
    print("Tuning XGBoost...")
    xgb_search = tune_xgboost(data.X_train, data.y_train)

    all_candidates = dict(baseline_models)
    all_candidates["Random Forest (Tuned)"] = rf_search.best_estimator_
    all_candidates["XGBoost (Tuned)"] = xgb_search.best_estimator_

    comparison_table = build_comparison_table(all_candidates, data.X_test, data.y_test)
    print("\n=== Full Comparison Table (4 baselines + 2 tuned) ===")
    print(comparison_table.round(4).to_string())

    chosen_name = select_final_model(comparison_table)
    chosen_model = all_candidates[chosen_name]
    chosen_metrics = evaluate_model(chosen_model, data.X_test, data.y_test)

    print(f"\nSelected final model: {chosen_name}")
    print(f"Selection rule: highest ROC-AUC (within 0.01 of top) then highest Recall")
    print(f"Final model test metrics: {chosen_metrics}")

    os.makedirs(REPORTS_DIR, exist_ok=True)
    comparison_table.round(4).to_csv(os.path.join(REPORTS_DIR, "full_model_comparison.csv"))

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(
        {
            "model": chosen_model,
            "preprocessor": data.preprocessor,
            "feature_names": data.feature_names,
        },
        os.path.join(MODELS_DIR, "best_model.pkl"),
    )
    joblib.dump(data.preprocessor, os.path.join(MODELS_DIR, "preprocessor.pkl"))
    joblib.dump(
        {
            "chosen_model_name": chosen_name,
            "selection_rule": "highest ROC-AUC (within 0.01 of top), tie-broken by Recall",
            "test_metrics": chosen_metrics,
            "comparison_table": comparison_table.to_dict(orient="index"),
            "numeric_features": data.numeric_features,
            "categorical_features": data.categorical_features,
            "feature_names": data.feature_names,
        },
        os.path.join(MODELS_DIR, "model_metadata.pkl"),
    )
    print(f"\nSaved: {MODELS_DIR}/best_model.pkl, preprocessor.pkl, model_metadata.pkl")
    return chosen_name, chosen_model, data, comparison_table


if __name__ == "__main__":
    run_full_selection()
