"""
data_preprocessing.py

Phase 1: Dataset loading and validation.
Phase 3: Preprocessing pipeline (ColumnTransformer-based).

This module is deliberately split into two responsibilities:
  1. load_and_validate_data()  -> raw ingestion + sanity checks (Phase 1)
  2. build_preprocessing_pipeline() / preprocess_data() -> Phase 3, added later

Only Phase 1 is implemented for now. Phase 3 functions are added in the
next development phase per the project's phased build order.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Anchor to the project root (parent of src/), not the process's current
# working directory. Without this, the default path silently breaks the
# moment the app/script is launched from anywhere else (VS Code debugger,
# a different terminal cwd, etc.) -- this bug was caught during Phase 1
# review and fixed here rather than left for Streamlit integration to surface.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "customer_churn.csv")

# Columns we expect to see in the Telco Customer Churn dataset.
EXPECTED_COLUMNS = [
    "customerID", "gender", "SeniorCitizen", "Partner", "Dependents",
    "tenure", "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling",
    "PaymentMethod", "MonthlyCharges", "TotalCharges", "Churn",
]

TARGET_COLUMN = "Churn"
ID_COLUMN = "customerID"


@dataclass
class ValidationReport:
    """Structured result of load_and_validate_data(), used by the app
    and the EDA notebook to display data-quality information without
    re-scanning the dataframe."""

    n_rows: int
    n_cols: int
    columns: list[str]
    missing_expected_columns: list[str]
    dtypes: dict[str, str]
    missing_value_counts: dict[str, int]
    duplicate_row_count: int
    duplicate_customer_id_count: int
    non_numeric_totalcharges_count: int
    target_value_counts: dict[str, int]
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"Rows: {self.n_rows}, Columns: {self.n_cols}",
            f"Duplicate rows: {self.duplicate_row_count}",
            f"Duplicate customerIDs: {self.duplicate_customer_id_count}",
            f"Non-numeric TotalCharges values: {self.non_numeric_totalcharges_count}",
            f"Target distribution: {self.target_value_counts}",
        ]
        if self.missing_expected_columns:
            lines.append(f"MISSING EXPECTED COLUMNS: {self.missing_expected_columns}")
        cols_with_na = {k: v for k, v in self.missing_value_counts.items() if v > 0}
        if cols_with_na:
            lines.append(f"Columns with missing values: {cols_with_na}")
        if self.warnings:
            lines.append("Warnings:")
            lines.extend(f"  - {w}" for w in self.warnings)
        return "\n".join(lines)


def load_raw_data(path: str = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the raw customer churn CSV from disk.

    Raises FileNotFoundError with a clear, user-facing message if the
    dataset has not been placed in data/customer_churn.csv, per the
    project's error-handling requirement (no dataset -> no traceback
    dumped on the user).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at '{path}'. Place the Telco Customer "
            f"Churn CSV at this path before running the pipeline."
        )
    df = pd.read_csv(path)
    return df


def validate_data(df: pd.DataFrame) -> ValidationReport:
    """Run data-quality checks on the raw dataframe and return a
    ValidationReport. Does not mutate df. Does not fabricate or infer
    values -- every field is computed directly from the data.
    """
    warnings: list[str] = []

    missing_expected_columns = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing_expected_columns:
        warnings.append(
            f"Dataset is missing expected columns: {missing_expected_columns}. "
            f"Downstream preprocessing steps that reference these columns will fail."
        )

    dtypes = {c: str(t) for c, t in df.dtypes.items()}
    missing_value_counts = df.isna().sum().to_dict()

    duplicate_row_count = int(df.duplicated().sum())
    duplicate_customer_id_count = (
        int(df[ID_COLUMN].duplicated().sum()) if ID_COLUMN in df.columns else 0
    )

    # TotalCharges is notoriously stored as an object dtype in this dataset
    # because a handful of rows contain blank strings (new customers with
    # tenure == 0). Detect this without altering the dataframe.
    non_numeric_totalcharges_count = 0
    if "TotalCharges" in df.columns:
        coerced = pd.to_numeric(df["TotalCharges"], errors="coerce")
        non_numeric_totalcharges_count = int(coerced.isna().sum() - df["TotalCharges"].isna().sum())
        if non_numeric_totalcharges_count > 0:
            warnings.append(
                f"'TotalCharges' has {non_numeric_totalcharges_count} non-numeric "
                f"values (likely blank strings for customers with tenure == 0). "
                f"These will need explicit handling in preprocessing, not silent coercion."
            )

    target_value_counts: dict[str, int] = {}
    if TARGET_COLUMN in df.columns:
        target_value_counts = df[TARGET_COLUMN].value_counts(dropna=False).to_dict()
        target_value_counts = {str(k): int(v) for k, v in target_value_counts.items()}
    else:
        warnings.append(f"Target column '{TARGET_COLUMN}' not found in dataset.")

    return ValidationReport(
        n_rows=df.shape[0],
        n_cols=df.shape[1],
        columns=list(df.columns),
        missing_expected_columns=missing_expected_columns,
        dtypes=dtypes,
        missing_value_counts={k: int(v) for k, v in missing_value_counts.items()},
        duplicate_row_count=duplicate_row_count,
        duplicate_customer_id_count=duplicate_customer_id_count,
        non_numeric_totalcharges_count=non_numeric_totalcharges_count,
        target_value_counts=target_value_counts,
        warnings=warnings,
    )


def load_and_validate_data(path: str = RAW_DATA_PATH) -> tuple[pd.DataFrame, ValidationReport]:
    """Convenience wrapper: load the raw CSV and validate it in one call.
    This is the single entry point Phase 1 exposes to the rest of the
    project (notebook, training script, Streamlit app).
    """
    df = load_raw_data(path)
    report = validate_data(df)
    return df, report


# ---------------------------------------------------------------------------
# Phase 3: Preprocessing pipeline
# ---------------------------------------------------------------------------
#
# Design decisions (documented so training/inference stay consistent):
#   - customerID is dropped: it's a record identifier, not a predictive
#     feature, and one-hot/leave-as-is would either leak nothing useful or
#     (worse) create 7043 useless dummy columns.
#   - TotalCharges blank strings (11 rows, all tenure == 0) are filled with
#     0.0 -- this is a real value (a brand-new customer has billed $0 total
#     so far), not a missing value to impute from other rows' statistics.
#   - SeniorCitizen is already numeric 0/1 in the raw data (see Phase 1
#     review) and is treated as a numeric feature, not one-hot encoded.
#   - All remaining object-dtype columns are treated as categorical and
#     one-hot encoded. The "No internet service" / "No phone service"
#     sub-categories are left as-is here; collapsing them into the parent
#     "No" category is a Phase 4 feature-engineering decision, not a
#     preprocessing one -- keeping Phase 3 encoding faithful to the raw
#     categories avoids silently discarding information before feature
#     engineering has had a chance to decide what to do with it.
#   - The ColumnTransformer is fit ONLY on the training split. The same
#     fitted object then transforms the test split (and, later, any new
#     customer at inference time). This is the single mechanism that
#     prevents train/test leakage in this module -- callers must not
#     re-fit the preprocessor on anything but the training data.

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.feature_engineering import ENGINEERED_NUMERIC_FEATURES, engineer_features

NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"] + ENGINEERED_NUMERIC_FEATURES
# Every remaining feature column (everything except id, target, and the
# numeric features above) is treated as categorical. Computed dynamically
# in build_preprocessor()/get_feature_columns() rather than hardcoded twice,
# so adding/removing a raw column can't silently desync the two lists.


def clean_total_charges(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with TotalCharges coerced to float and the
    known blank-string rows (tenure == 0) filled with 0.0.

    Any *other* non-numeric TotalCharges value (i.e. not explained by
    tenure == 0) is left as NaN and surfaced via an assertion, rather than
    silently filled -- a new failure mode here should be investigated,
    not masked.
    """
    df = df.copy()
    coerced = pd.to_numeric(df["TotalCharges"], errors="coerce")
    unexplained_mask = coerced.isna() & (df["tenure"] != 0)
    if unexplained_mask.any():
        raise ValueError(
            f"Found {int(unexplained_mask.sum())} row(s) with non-numeric "
            f"TotalCharges that do NOT have tenure == 0. The known/handled "
            f"failure mode is tenure-0 blank strings only -- investigate "
            f"these rows before proceeding."
        )
    df["TotalCharges"] = coerced.fillna(0.0)
    return df


def get_feature_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return (numeric_features, categorical_features) present in df,
    excluding the id and target columns.
    """
    numeric_features = [c for c in NUMERIC_FEATURES if c in df.columns]
    categorical_features = [
        c for c in df.columns
        if c not in numeric_features and c not in (ID_COLUMN, TARGET_COLUMN)
    ]
    return numeric_features, categorical_features


def build_preprocessor(numeric_features: list[str], categorical_features: list[str]) -> ColumnTransformer:
    """Build the ColumnTransformer used for both training and inference.

    StandardScaler is included even though tree models (Random Forest,
    XGBoost) don't need it, because Logistic Regression (Phase 11) does --
    a single shared preprocessor keeps all four models' inputs identical
    and comparable, which the project's evaluation section requires.
    """
    numeric_pipeline = Pipeline(steps=[("scaler", StandardScaler())])
    categorical_pipeline = Pipeline(steps=[
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )
    return preprocessor


def get_feature_target_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split a cleaned dataframe into feature matrix X and encoded target y.
    Encodes Churn: Yes -> 1, No -> 0. Drops customerID from X.
    """
    y = (df[TARGET_COLUMN] == "Yes").astype(int)
    X = df.drop(columns=[TARGET_COLUMN, ID_COLUMN])
    return X, y


@dataclass
class PreprocessedData:
    """Everything downstream phases need, bundled together so training,
    evaluation, and the Streamlit app all consume the same artifact shape."""

    X_train: np.ndarray
    X_test: np.ndarray
    y_train: pd.Series
    y_test: pd.Series
    preprocessor: ColumnTransformer
    feature_names: list[str]
    numeric_features: list[str]
    categorical_features: list[str]


def preprocess_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> PreprocessedData:
    """Full Phase 3 pipeline: clean -> split features/target -> train/test
    split (stratified, before any fitting) -> fit preprocessor on train
    only -> transform both splits.

    This ordering is the leakage-prevention contract: nothing about the
    test split is allowed to influence the fitted preprocessor. Feature
    engineering (Phase 4) runs after cleaning and before the split, since
    every engineered feature here is computed per-row from fixed rules
    or fixed cutoffs (not fit statistics) -- see feature_engineering.py
    for why each one is leakage-safe to compute before splitting.
    """
    df_clean = clean_total_charges(df)
    df_engineered = engineer_features(df_clean)
    X, y = get_feature_target_split(df_engineered)
    numeric_features, categorical_features = get_feature_columns(df_engineered)

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y,
    )

    preprocessor = build_preprocessor(numeric_features, categorical_features)
    X_train = preprocessor.fit_transform(X_train_raw)
    X_test = preprocessor.transform(X_test_raw)

    feature_names = list(preprocessor.get_feature_names_out())

    return PreprocessedData(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train.reset_index(drop=True),
        y_test=y_test.reset_index(drop=True),
        preprocessor=preprocessor,
        feature_names=feature_names,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )


if __name__ == "__main__":
    dataframe, validation_report = load_and_validate_data()
    print(validation_report.summary())
    print()
    result = preprocess_data(dataframe)
    print(f"X_train shape: {result.X_train.shape}")
    print(f"X_test shape: {result.X_test.shape}")
    print(f"y_train churn rate: {result.y_train.mean():.4f}")
    print(f"y_test churn rate: {result.y_test.mean():.4f}")
    print(f"Numeric features: {result.numeric_features}")
    print(f"Categorical features: {result.categorical_features}")
    print(f"Total engineered feature count: {len(result.feature_names)}")
