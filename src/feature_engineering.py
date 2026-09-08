"""
feature_engineering.py

Phase 4: Feature Engineering.

Every engineered feature is documented inline (why it exists, how it's
computed, what EDA finding motivated it) per the project's requirement
to document every engineered feature and avoid target leakage.

None of these features use the Churn column -- all are derivable from a
single customer's raw attributes alone, which also means they can be
computed identically at training time and at Streamlit prediction time
for a brand-new customer.
"""

from __future__ import annotations

import pandas as pd

# Columns where the raw dataset encodes "customer doesn't have this
# service because they have no internet/phone at all" as a THIRD category
# value, redundant with InternetService/PhoneService. Collapsing these to
# "No" removes duplicated information and cuts one-hot dimensionality --
# flagged during the Phase 1 review of unique category values.
INTERNET_DEPENDENT_COLUMNS = [
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
]
PHONE_DEPENDENT_COLUMNS = ["MultipleLines"]

# Numeric engineered features, appended to NUMERIC_FEATURES in
# data_preprocessing.py so the preprocessor picks them up automatically.
ENGINEERED_NUMERIC_FEATURES = [
    "TotalChargesPerMonth", "ServiceCount", "HasSecurityService",
    "HasStreamingService", "SupportRisk", "ContractRisk",
]
# Categorical engineered features -- no registration needed, they're
# picked up automatically by get_feature_columns() since it treats
# every non-numeric, non-id, non-target column as categorical.
ENGINEERED_CATEGORICAL_FEATURES = ["TenureGroup", "MonthlyChargeGroup"]


def _tenure_group(tenure: pd.Series) -> pd.Series:
    """TenureGroup: bucket tenure (0-72 months) into 5 bands.
    Fixed, business-meaningful cutoffs (not derived from data
    statistics), so this is leakage-safe regardless of the train/test
    split and stays interpretable for the retention team (e.g. "0-12
    months" reads as "first-year customer" on the dashboard).
    """
    bins = [-1, 12, 24, 48, 60, 72]
    labels = ["0-12", "13-24", "25-48", "49-60", "61-72"]
    return pd.cut(tenure, bins=bins, labels=labels).astype(str)


def _monthly_charge_group(monthly_charges: pd.Series) -> pd.Series:
    """MonthlyChargeGroup: bucket MonthlyCharges into Low/Medium/High
    using fixed dollar cutoffs (not quantiles of the current split),
    chosen from the Phase 2 EDA describe() output (~25th/75th percentile
    landed near $35/$70/$90). Fixed cutoffs keep this leakage-safe and
    stable across train/test/inference rather than shifting if the
    split changes.
    """
    bins = [0, 35, 70, 1000]
    labels = ["Low", "Medium", "High"]
    return pd.cut(monthly_charges, bins=bins, labels=labels).astype(str)


def _service_count(df: pd.DataFrame) -> pd.Series:
    """ServiceCount: number of subscribed services out of 9 possible
    (phone, multiple lines, internet, online security, online backup,
    device protection, tech support, streaming TV, streaming movies).
    Computed from the ORIGINAL (pre-collapse) category values so
    'No internet service' correctly contributes 0, not a miscount.
    """
    count = pd.Series(0, index=df.index)
    count += (df["PhoneService"] == "Yes").astype(int)
    count += (df["MultipleLines"] == "Yes").astype(int)
    count += (df["InternetService"] != "No").astype(int)
    for col in ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
                "TechSupport", "StreamingTV", "StreamingMovies"]:
        count += (df[col] == "Yes").astype(int)
    return count


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered features to a cleaned dataframe (i.e. one that has
    already been through clean_total_charges()). Returns a new dataframe;
    does not mutate the input.

    Order matters internally: risk/count features that need to
    distinguish "No" from "No internet/phone service" are computed
    BEFORE those categories are collapsed to "No" for the encoder.
    """
    out = df.copy()

    # --- risk & count features computed from ORIGINAL 3-way categories ---

    # SupportRisk: EDA showed TechSupport == 'No' customers churn at 41.64%,
    # vs 15.17% for 'Yes' and 7.40% for 'No internet service' (i.e. the
    # 3rd category is actually the LOWEST-risk group, not a risk at all --
    # collapsing it into a binary "no support = risk" would misclassify
    # it). So SupportRisk is 1 only for the genuine "declined support
    # despite having internet" case.
    out["SupportRisk"] = (out["TechSupport"] == "No").astype(int)

    # ContractRisk: EDA showed month-to-month churns at 42.71% vs 11.27%
    # (one year) / 2.83% (two year) -- by far the single strongest
    # categorical signal found in Phase 2.
    out["ContractRisk"] = (out["Contract"] == "Month-to-month").astype(int)

    out["HasSecurityService"] = (
        (out["OnlineSecurity"] == "Yes") | (out["DeviceProtection"] == "Yes")
    ).astype(int)
    out["HasStreamingService"] = (
        (out["StreamingTV"] == "Yes") | (out["StreamingMovies"] == "Yes")
    ).astype(int)

    out["ServiceCount"] = _service_count(out)

    # TotalChargesPerMonth: normalizes cumulative spend by tenure so a
    # long-tenured and a brand-new customer are comparable on a
    # "typical monthly spend" basis. Guards div-by-zero for tenure == 0
    # by flooring the denominator at 1 (matches how TotalCharges == 0 was
    # already handled for those same rows in Phase 3).
    out["TotalChargesPerMonth"] = out["TotalCharges"] / out["tenure"].clip(lower=1)

    # --- bucketed features ---
    out["TenureGroup"] = _tenure_group(out["tenure"])
    out["MonthlyChargeGroup"] = _monthly_charge_group(out["MonthlyCharges"])

    # --- collapse redundant 3-way categories to 2-way, AFTER the above ---
    for col in INTERNET_DEPENDENT_COLUMNS:
        out[col] = out[col].replace("No internet service", "No")
    for col in PHONE_DEPENDENT_COLUMNS:
        out[col] = out[col].replace("No phone service", "No")

    return out


if __name__ == "__main__":
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.data_preprocessing import load_and_validate_data, clean_total_charges

    df, _ = load_and_validate_data()
    df_clean = clean_total_charges(df)
    df_engineered = engineer_features(df_clean)

    new_cols = [c for c in df_engineered.columns if c not in df_clean.columns]
    print(f"Engineered {len(new_cols)} new columns: {new_cols}")
    print()
    print(df_engineered[new_cols].describe(include="all").T)
    print()
    print("Collapsed category check (TechSupport unique values now):",
          sorted(df_engineered["TechSupport"].unique()))
    print("Collapsed category check (MultipleLines unique values now):",
          sorted(df_engineered["MultipleLines"].unique()))
