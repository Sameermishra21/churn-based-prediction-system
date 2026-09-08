"""
inference.py

Single entry point tying together the saved model, SHAP explainer, and
recommendation engine for ONE customer -- this is what app.py (Phase 11)
calls directly, so the prediction page, customer analysis page, and
explainability page all go through the same, tested code path instead
of three separate re-implementations.
"""

from __future__ import annotations

import os

import joblib
import numpy as np
import pandas as pd
import shap

from src.data_preprocessing import clean_total_charges
from src.explainability import build_explainer, compute_shap_values, local_explanation
from src.feature_engineering import engineer_features
from src.recommendations import generate_recommendations, get_risk_level

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

_CACHE = {}


def load_model_bundle():
    """Load (and cache in-process) the saved best_model.pkl bundle."""
    if "bundle" not in _CACHE:
        path = os.path.join(MODELS_DIR, "best_model.pkl")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No trained model found at {path}. Run "
                f"`python -m src.model_selection` first to train and save one."
            )
        _CACHE["bundle"] = joblib.load(path)
    return _CACHE["bundle"]


def get_shap_explainer():
    if "explainer" not in _CACHE:
        bundle = load_model_bundle()
        _CACHE["explainer"] = build_explainer(bundle["model"])
    return _CACHE["explainer"]


def predict_customer(customer_raw: dict, top_n_factors: int = 5) -> dict:
    """Run the full pipeline for one customer, given as a dict of RAW
    (pre-encoding) attribute values matching the original CSV columns
    (e.g. {"gender": "Female", "tenure": 5, "Contract": "Month-to-month",
    "MonthlyCharges": 85.0, "TotalCharges": 425.0, ...}).

    Returns a dict with churn_probability, risk_level, local SHAP
    explanation, and rule-based recommendations -- everything the
    Prediction and Customer Analysis pages need in one call.

    Missing fields are NOT silently defaulted with fabricated values;
    a KeyError from pandas/sklearn on a genuinely missing required
    column is allowed to propagate so the caller (the Streamlit form)
    surfaces it as a validation error, per section 31's error-handling
    requirement.
    """
    bundle = load_model_bundle()
    model, preprocessor, feature_names = (
        bundle["model"], bundle["preprocessor"], bundle["feature_names"]
    )

    df_one = pd.DataFrame([customer_raw])
    # clean_total_charges expects TotalCharges/tenure columns; a brand
    # new customer with tenure 0 legitimately may not have TotalCharges
    # set yet -- default it to 0 only in that specific, documented case.
    if "TotalCharges" not in df_one.columns or pd.isna(df_one.loc[0, "TotalCharges"]):
        df_one["TotalCharges"] = 0.0 if df_one.loc[0, "tenure"] == 0 else np.nan

    df_clean = clean_total_charges(df_one)
    df_engineered = engineer_features(df_clean)

    X = preprocessor.transform(df_engineered)
    proba = float(model.predict_proba(X)[0, 1])
    risk_level = get_risk_level(proba)

    explainer = get_shap_explainer()
    shap_values = compute_shap_values(explainer, X)
    base_value = explainer.expected_value
    if isinstance(base_value, (list, np.ndarray)):
        base_value = base_value[1] if len(np.atleast_1d(base_value)) > 1 else base_value[0]

    explanation = local_explanation(
        shap_values[0], feature_names, base_value, proba, top_n=top_n_factors,
    )

    recommendations = generate_recommendations(customer_raw, proba)

    return {
        "churn_probability": proba,
        "risk_level": risk_level,
        "explanation": explanation,
        "recommendations": [{"action": r.action, "reason": r.reason} for r in recommendations],
    }


if __name__ == "__main__":
    example_customer = {
        "gender": "Female", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No",
        "tenure": 2, "PhoneService": "Yes", "MultipleLines": "No",
        "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
        "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "Yes",
        "StreamingMovies": "Yes", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check", "MonthlyCharges": 95.0, "TotalCharges": 190.0,
    }

    result = predict_customer(example_customer)
    print(f"Churn Probability: {result['churn_probability'] * 100:.1f}%")
    print(f"Risk Level: {result['risk_level']}")
    print("\nTop factors increasing risk:")
    for feat, val in result["explanation"]["factors_increasing_risk"]:
        print(f"  + {feat}: {val:+.4f}")
    print("Top factors decreasing risk:")
    for feat, val in result["explanation"]["factors_decreasing_risk"]:
        print(f"  - {feat}: {val:+.4f}")
    print("\nRecommendations:")
    for rec in result["recommendations"]:
        print(f"  - {rec['action']}")
