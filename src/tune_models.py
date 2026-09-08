"""
tune_models.py

Phase 7: Hyperparameter optimization for Random Forest and XGBoost
(the two models the project brief specifies tuning parameters for).

Uses RandomizedSearchCV with 5-fold CV, scored on ROC-AUC (not
accuracy) since Phase 6 established that's the metric that matters
here. Logistic Regression and Decision Tree are left as Phase 5
baselines -- Decision Tree's overfitting is a reason to prefer the
ensembles, not to spend the tuning budget constraining it into
mediocrity.
"""

from __future__ import annotations

import os

import numpy as np
from scipy.stats import randint, uniform
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier

RANDOM_STATE = 42

RF_PARAM_DIST = {
    "n_estimators": randint(100, 400),
    "max_depth": randint(4, 20),
    "min_samples_split": randint(2, 20),
    "min_samples_leaf": randint(1, 10),
    "max_features": ["sqrt", "log2", None],
}

XGB_PARAM_DIST = {
    "n_estimators": randint(100, 400),
    "max_depth": randint(2, 10),
    "learning_rate": uniform(0.01, 0.29),  # 0.01-0.30
    "subsample": uniform(0.6, 0.4),         # 0.6-1.0
    "colsample_bytree": uniform(0.6, 0.4),  # 0.6-1.0
}


def tune_random_forest(X_train, y_train, n_iter: int = 25, cv: int = 5) -> RandomForestClassifier:
    base = RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1)
    search = RandomizedSearchCV(
        base, RF_PARAM_DIST, n_iter=n_iter, cv=cv, scoring="roc_auc",
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    search.fit(X_train, y_train)
    return search


def tune_xgboost(X_train, y_train, n_iter: int = 25, cv: int = 5) -> XGBClassifier:
    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    base = XGBClassifier(
        random_state=RANDOM_STATE, eval_metric="logloss",
        scale_pos_weight=neg / pos, n_jobs=-1,
    )
    search = RandomizedSearchCV(
        base, XGB_PARAM_DIST, n_iter=n_iter, cv=cv, scoring="roc_auc",
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    search.fit(X_train, y_train)
    return search


if __name__ == "__main__":
    import time

    from src.data_preprocessing import load_and_validate_data, preprocess_data
    from src.evaluate_models import evaluate_model

    df, _ = load_and_validate_data()
    data = preprocess_data(df)

    print("Tuning Random Forest (RandomizedSearchCV, 25 iters, 5-fold CV, scoring=roc_auc)...")
    t0 = time.time()
    rf_search = tune_random_forest(data.X_train, data.y_train)
    print(f"  done in {time.time() - t0:.1f}s")
    print(f"  best CV ROC-AUC: {rf_search.best_score_:.4f}")
    print(f"  best params: {rf_search.best_params_}")
    rf_test_metrics = evaluate_model(rf_search.best_estimator_, data.X_test, data.y_test)
    print(f"  test metrics: {rf_test_metrics}")

    print("\nTuning XGBoost (RandomizedSearchCV, 25 iters, 5-fold CV, scoring=roc_auc)...")
    t0 = time.time()
    xgb_search = tune_xgboost(data.X_train, data.y_train)
    print(f"  done in {time.time() - t0:.1f}s")
    print(f"  best CV ROC-AUC: {xgb_search.best_score_:.4f}")
    print(f"  best params: {xgb_search.best_params_}")
    xgb_test_metrics = evaluate_model(xgb_search.best_estimator_, data.X_test, data.y_test)
    print(f"  test metrics: {xgb_test_metrics}")
