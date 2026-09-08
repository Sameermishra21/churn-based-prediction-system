"""
train_models.py

Phase 5: Train baseline models.
Phase 7: Hyperparameter optimization (added later, once baselines exist
to compare against).

Trains four models on the SAME preprocessed train split from
data_preprocessing.preprocess_data(), so the Phase 6 comparison table
is apples-to-apples. No metrics are hardcoded here -- everything is
computed by evaluate_models.py against the actual fitted models.
"""

from __future__ import annotations

import os

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

RANDOM_STATE = 42


def get_baseline_models() -> dict:
    """Return the four required baseline models, each with class_weight
    (or scale_pos_weight for XGBoost) set to account for the ~26.5%/73.5%
    class imbalance found in Phase 1/2 -- inspected first per the
    project's class-imbalance handling rule, rather than assumed.
    """
    # Imbalance ratio for XGBoost's scale_pos_weight: negative/positive
    # count from the training split. Computed at train time in
    # train_all_models() and passed in, not hardcoded here.
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE,
        ),
        "Decision Tree": DecisionTreeClassifier(
            class_weight="balanced", random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, class_weight="balanced", random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200, random_state=RANDOM_STATE, eval_metric="logloss",
            n_jobs=-1,
        ),
    }
    return models


def train_all_models(X_train, y_train) -> dict:
    """Fit all four baseline models on the given (already-preprocessed)
    training data. Returns {model_name: fitted_model}.
    """
    models = get_baseline_models()

    # XGBoost doesn't support class_weight='balanced' directly; set
    # scale_pos_weight = (negative count / positive count) instead,
    # computed from the actual training labels, not assumed.
    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    models["XGBoost"].set_params(scale_pos_weight=neg / pos)

    fitted = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        fitted[name] = model
    return fitted


def save_model(model, preprocessor, feature_names: list[str], name: str) -> str:
    """Save a single fitted model + the preprocessor + feature metadata
    together, so the Streamlit app can load one artifact per model
    without re-deriving feature order at inference time.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    path = os.path.join(MODELS_DIR, f"{name.lower().replace(' ', '_')}.pkl")
    joblib.dump(
        {"model": model, "preprocessor": preprocessor, "feature_names": feature_names},
        path,
    )
    return path


if __name__ == "__main__":
    from src.data_preprocessing import load_and_validate_data, preprocess_data

    df, report = load_and_validate_data()
    print(report.summary())
    print()

    data = preprocess_data(df)
    print(f"Training on {data.X_train.shape[0]} samples, {data.X_train.shape[1]} features")
    print(f"Class balance in y_train: {data.y_train.value_counts().to_dict()}")
    print()

    fitted_models = train_all_models(data.X_train, data.y_train)

    for model_name, fitted_model in fitted_models.items():
        train_acc = fitted_model.score(data.X_train, data.y_train)
        test_acc = fitted_model.score(data.X_test, data.y_test)
        print(f"{model_name:22s} train_acc={train_acc:.4f}  test_acc={test_acc:.4f}")
        save_model(fitted_model, data.preprocessor, data.feature_names, model_name)

    print(f"\nAll models saved to {MODELS_DIR}")
