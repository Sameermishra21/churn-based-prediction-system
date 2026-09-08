"""
app.py

Phase 11: Streamlit application.

Multi-page app (sidebar navigation) covering all 7 pages from the
project brief: Dashboard, Prediction, Customer Analysis, Model
Performance, Explainability, Recommendations, About.

Loads the saved model/preprocessor/metadata from models/ -- does NOT
retrain on every page load or refresh (section 29/36.11). All figures
and tables are computed from the actual dataset/model at runtime
(cached with st.cache_data/st.cache_resource), never hardcoded.
"""

import os
import sys

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_preprocessing import (
    TARGET_COLUMN,
    clean_total_charges,
    load_and_validate_data,
    preprocess_data,
)
from src.evaluate_models import evaluate_model
from src.explainability import (
    build_explainer,
    compute_shap_values,
    global_feature_importance,
    local_explanation,
)
from src.feature_engineering import engineer_features
from src.recommendations import DISCLAIMER, generate_recommendations, get_risk_level

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "customer_churn.csv")

st.set_page_config(
    page_title="Customer Churn Prediction & Retention System",
    page_icon="📊",
    layout="wide",
)

RAW_FORM_COLUMNS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges",
]


# ---------------------------------------------------------------------------
# Cached data / model loading -- computed once per session, not per rerun.
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def get_raw_data():
    if not os.path.exists(DATA_PATH):
        return None, None
    df, report = load_and_validate_data(DATA_PATH)
    return df, report


@st.cache_resource(show_spinner=False)
def get_model_bundle():
    path = os.path.join(MODELS_DIR, "best_model.pkl")
    if not os.path.exists(path):
        return None
    return joblib.load(path)


@st.cache_resource(show_spinner=False)
def get_model_metadata():
    path = os.path.join(MODELS_DIR, "model_metadata.pkl")
    if not os.path.exists(path):
        return None
    return joblib.load(path)


@st.cache_resource(show_spinner=False)
def get_all_saved_models():
    """Load every individually-saved baseline model (for the Model
    Performance page's full comparison), skipping the combined
    best_model.pkl bundle."""
    models = {}
    if not os.path.isdir(MODELS_DIR):
        return models
    for fname in os.listdir(MODELS_DIR):
        if fname in ("best_model.pkl", "preprocessor.pkl", "model_metadata.pkl", ".gitkeep"):
            continue
        if fname.endswith(".pkl"):
            bundle = joblib.load(os.path.join(MODELS_DIR, fname))
            name = fname.replace(".pkl", "").replace("_", " ").title()
            models[name] = bundle
    return models


@st.cache_resource(show_spinner=False)
def get_shap_explainer(_model):
    return build_explainer(_model)


@st.cache_data(show_spinner=False)
def score_full_dataset(_df: pd.DataFrame) -> pd.DataFrame:
    """Run the saved model over every customer in the dataset once, and
    cache the result -- every page that needs "all customers scored"
    (Dashboard, Customer Analysis, Recommendations) shares this instead
    of re-scoring 7000+ rows per page load.
    """
    bundle = get_model_bundle()
    model, preprocessor = bundle["model"], bundle["preprocessor"]

    df_clean = clean_total_charges(_df)
    df_engineered = engineer_features(df_clean)
    X = preprocessor.transform(df_engineered)
    proba = model.predict_proba(X)[:, 1]

    result = _df.copy()
    result["ChurnProbability"] = proba
    result["RiskLevel"] = [get_risk_level(p) for p in proba]
    result["Prediction"] = np.where(proba >= 0.5, "Likely to Churn", "Likely to Stay")
    return result


# ---------------------------------------------------------------------------
# Page: Dashboard
# ---------------------------------------------------------------------------

def page_dashboard(df, scored_df):
    st.title("📊 Churn Overview Dashboard")

    total = len(df)
    churned = int((df[TARGET_COLUMN] == "Yes").sum())
    churn_rate = churned / total * 100
    avg_charges = df["MonthlyCharges"].mean()
    avg_tenure = df["tenure"].mean()
    high_risk = int((scored_df["RiskLevel"] == "HIGH").sum()) if scored_df is not None else None

    cols = st.columns(6)
    cols[0].metric("Total Customers", f"{total:,}")
    cols[1].metric("Churned Customers", f"{churned:,}")
    cols[2].metric("Churn Rate", f"{churn_rate:.2f}%")
    cols[3].metric("Avg. Monthly Charges", f"${avg_charges:.2f}")
    cols[4].metric("Avg. Tenure", f"{avg_tenure:.1f} mo")
    cols[5].metric("High-Risk Customers", f"{high_risk:,}" if high_risk is not None else "N/A")

    st.divider()

    c1, c2 = st.columns(2)
    with c1:
        fig = px.pie(
            df, names=TARGET_COLUMN, title="Churn Distribution",
            color=TARGET_COLUMN, color_discrete_map={"No": "#2e7d32", "Yes": "#c62828"},
            hole=0.4,
        )
        st.plotly_chart(fig, width='stretch')
    with c2:
        contract_churn = (
            df.groupby("Contract")[TARGET_COLUMN]
            .apply(lambda s: (s == "Yes").mean() * 100)
            .reset_index(name="ChurnRate")
        )
        fig = px.bar(
            contract_churn, x="Contract", y="ChurnRate",
            title="Churn Rate (%) by Contract Type", color="ChurnRate",
            color_continuous_scale="Reds",
        )
        st.plotly_chart(fig, width='stretch')

    c3, c4 = st.columns(2)
    with c3:
        fig = px.histogram(
            df, x="tenure", color=TARGET_COLUMN, barmode="overlay",
            title="Churn by Tenure",
            color_discrete_map={"No": "#2e7d32", "Yes": "#c62828"},
            opacity=0.6,
        )
        st.plotly_chart(fig, width='stretch')
    with c4:
        payment_churn = (
            df.groupby("PaymentMethod")[TARGET_COLUMN]
            .apply(lambda s: (s == "Yes").mean() * 100)
            .reset_index(name="ChurnRate")
            .sort_values("ChurnRate", ascending=False)
        )
        fig = px.bar(
            payment_churn, x="PaymentMethod", y="ChurnRate",
            title="Churn Rate (%) by Payment Method", color="ChurnRate",
            color_continuous_scale="Reds",
        )
        st.plotly_chart(fig, width='stretch')

    c5, c6 = st.columns(2)
    with c5:
        internet_churn = (
            df.groupby("InternetService")[TARGET_COLUMN]
            .apply(lambda s: (s == "Yes").mean() * 100)
            .reset_index(name="ChurnRate")
        )
        fig = px.bar(
            internet_churn, x="InternetService", y="ChurnRate",
            title="Churn Rate (%) by Internet Service", color="ChurnRate",
            color_continuous_scale="Reds",
        )
        st.plotly_chart(fig, width='stretch')
    with c6:
        if scored_df is not None:
            bundle = get_model_bundle()
            explainer = get_shap_explainer(bundle["model"])
            sample = scored_df.sample(min(300, len(scored_df)), random_state=42)
            df_clean = clean_total_charges(sample[RAW_FORM_COLUMNS + [TARGET_COLUMN]])
            df_eng = engineer_features(df_clean)
            X_sample = bundle["preprocessor"].transform(df_eng)
            shap_vals = compute_shap_values(explainer, X_sample)
            importance = global_feature_importance(shap_vals, bundle["feature_names"]).head(10)
            fig = px.bar(
                importance[::-1], x="mean_abs_shap", y="feature", orientation="h",
                title="Top 10 Features (Global SHAP Importance, 300-row sample)",
            )
            st.plotly_chart(fig, width='stretch')


# ---------------------------------------------------------------------------
# Page: Prediction
# ---------------------------------------------------------------------------

def page_prediction(df):
    st.title("🔮 Predict Churn for a Customer")
    st.caption("Enter customer information to get a churn prediction, explanation, and retention recommendations.")

    with st.form("prediction_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            gender = st.selectbox("Gender", ["Female", "Male"])
            senior = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x else "No")
            partner = st.selectbox("Partner", ["Yes", "No"])
            dependents = st.selectbox("Dependents", ["Yes", "No"])
            tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12)
            phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        with c2:
            multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
            internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
            online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
            device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
            tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
        with c3:
            streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
            streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
            payment_method = st.selectbox("Payment Method", [
                "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)",
            ])
            monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=70.0)

        total_charges = st.number_input(
            "Total Charges ($) — leave as tenure×monthly if unsure",
            min_value=0.0, value=float(round(tenure * monthly_charges, 2)),
        )

        submitted = st.form_submit_button("Predict Churn", type="primary")

    if submitted:
        bundle = get_model_bundle()
        if bundle is None:
            st.error("No trained model found. Run the training pipeline first (see README).")
            return

        customer = {
            "gender": gender, "SeniorCitizen": senior, "Partner": partner,
            "Dependents": dependents, "tenure": tenure, "PhoneService": phone_service,
            "MultipleLines": multiple_lines, "InternetService": internet_service,
            "OnlineSecurity": online_security, "OnlineBackup": online_backup,
            "DeviceProtection": device_protection, "TechSupport": tech_support,
            "StreamingTV": streaming_tv, "StreamingMovies": streaming_movies,
            "Contract": contract, "PaperlessBilling": paperless,
            "PaymentMethod": payment_method, "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
        }

        try:
            from src.inference import predict_customer
            result = predict_customer(customer)
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            return

        proba = result["churn_probability"]
        risk = result["risk_level"]
        risk_color = {"LOW": "green", "MEDIUM": "orange", "HIGH": "red"}[risk]

        st.divider()
        c1, c2, c3 = st.columns(3)
        c1.metric("Churn Probability", f"{proba * 100:.1f}%")
        c2.metric("Prediction", "LIKELY TO CHURN" if proba >= 0.5 else "LIKELY TO STAY")
        c3.markdown(f"**Risk Level:** :{risk_color}[{risk}]")
        st.progress(min(max(proba, 0.0), 1.0))

        c4, c5 = st.columns(2)
        with c4:
            st.subheader("Top Contributing Factors")
            st.markdown("**Increasing churn risk:**")
            for feat, val in result["explanation"]["factors_increasing_risk"]:
                st.markdown(f"- `{feat}` (impact: +{val:.3f})")
            st.markdown("**Reducing churn risk:**")
            for feat, val in result["explanation"]["factors_decreasing_risk"]:
                st.markdown(f"- `{feat}` (impact: {val:.3f})")
        with c5:
            st.subheader("Recommended Retention Actions")
            for rec in result["recommendations"]:
                st.markdown(f"- **{rec['action']}**")
                st.caption(rec["reason"])
            st.info(DISCLAIMER)


# ---------------------------------------------------------------------------
# Page: Customer Analysis
# ---------------------------------------------------------------------------

def page_customer_analysis(scored_df):
    st.title("🔍 Customer Analysis")

    if scored_df is None:
        st.warning("No dataset/model available.")
        return

    risk_filter = st.multiselect(
        "Filter by Risk Level", ["LOW", "MEDIUM", "HIGH"], default=["HIGH", "MEDIUM", "LOW"],
    )
    filtered = scored_df[scored_df["RiskLevel"].isin(risk_filter)]
    st.caption(f"Showing {len(filtered):,} of {len(scored_df):,} customers")

    st.dataframe(
        filtered[["customerID", "Contract", "tenure", "MonthlyCharges", "ChurnProbability", "RiskLevel", "Prediction"]]
        .sort_values("ChurnProbability", ascending=False)
        .style.format({"ChurnProbability": "{:.2%}", "MonthlyCharges": "${:.2f}"}),
        width='stretch', height=400,
    )

    st.divider()
    st.subheader("Inspect One Customer")
    customer_id = st.selectbox("Select customerID", filtered["customerID"].tolist())
    if customer_id:
        row = scored_df[scored_df["customerID"] == customer_id].iloc[0]
        customer_raw = {col: row[col] for col in RAW_FORM_COLUMNS}

        from src.inference import predict_customer
        result = predict_customer(customer_raw)

        c1, c2, c3 = st.columns(3)
        c1.metric("Churn Probability", f"{result['churn_probability'] * 100:.1f}%")
        c2.metric("Risk Level", result["risk_level"])
        c3.metric("Contract", row["Contract"])

        c4, c5 = st.columns(2)
        with c4:
            st.markdown("**Top factors increasing risk:**")
            for feat, val in result["explanation"]["factors_increasing_risk"]:
                st.markdown(f"- `{feat}` (+{val:.3f})")
        with c5:
            st.markdown("**Recommendations:**")
            for rec in result["recommendations"]:
                st.markdown(f"- {rec['action']}")


# ---------------------------------------------------------------------------
# Page: Explainability
# ---------------------------------------------------------------------------

def page_explainability(df):
    st.title("🧠 Model Explainability (SHAP)")

    bundle = get_model_bundle()
    if bundle is None:
        st.warning("No trained model available.")
        return
    model, preprocessor, feature_names = bundle["model"], bundle["preprocessor"], bundle["feature_names"]

    st.subheader("Global Feature Importance")
    sample_n = st.slider("Sample size for global SHAP (larger = slower)", 100, min(2000, len(df)), 500, step=100)
    sample = df.sample(sample_n, random_state=42)
    df_clean = clean_total_charges(sample)
    df_eng = engineer_features(df_clean)
    X_sample = preprocessor.transform(df_eng)

    explainer = get_shap_explainer(model)
    shap_vals = compute_shap_values(explainer, X_sample)
    importance = global_feature_importance(shap_vals, feature_names).head(15)

    fig = px.bar(
        importance[::-1], x="mean_abs_shap", y="feature", orientation="h",
        title=f"Top 15 Features by Global SHAP Importance (n={sample_n})",
    )
    st.plotly_chart(fig, width='stretch')

    st.divider()
    st.subheader("Local Explanation for a Selected Customer")
    sample_reset = sample.reset_index(drop=True)
    customer_id = st.selectbox("Select customerID", sample_reset["customerID"].tolist())
    if customer_id:
        idx = sample_reset.index[sample_reset["customerID"] == customer_id][0]
        proba = model.predict_proba(X_sample[[idx]])[0, 1]
        base_value = explainer.expected_value
        if isinstance(base_value, (list, np.ndarray)):
            base_value = base_value[1] if len(np.atleast_1d(base_value)) > 1 else base_value[0]
        explanation = local_explanation(shap_vals[idx], feature_names, base_value, proba)

        st.metric("Churn Probability", f"{proba * 100:.1f}%")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Increasing risk:**")
            for feat, val in explanation["factors_increasing_risk"]:
                st.markdown(f"- `{feat}`: +{val:.3f}")
        with c2:
            st.markdown("**Decreasing risk:**")
            for feat, val in explanation["factors_decreasing_risk"]:
                st.markdown(f"- `{feat}`: {val:.3f}")


# ---------------------------------------------------------------------------
# Page: Model Performance
# ---------------------------------------------------------------------------

def page_model_performance(df):
    st.title("📈 Model Performance")

    metadata = get_model_metadata()
    all_models = get_all_saved_models()

    if not all_models:
        st.warning("No saved models found.")
        return

    data = preprocess_data(df)

    comparison_rows = {}
    for name, bundle in all_models.items():
        comparison_rows[name] = evaluate_model(bundle["model"], data.X_test, data.y_test)
    comparison_df = pd.DataFrame(comparison_rows).T.sort_values("ROC-AUC", ascending=False)

    st.subheader("Model Comparison")
    st.dataframe(comparison_df.style.format("{:.4f}").background_gradient(cmap="Greens", subset=["ROC-AUC"]),
                 width='stretch')

    if metadata:
        st.success(
            f"**Selected final model: {metadata['chosen_model_name']}** — "
            f"{metadata['selection_rule']}"
        )

    st.divider()
    model_choice = st.selectbox("Inspect a model", list(all_models.keys()))
    chosen_bundle = all_models[model_choice]
    model = chosen_bundle["model"]
    y_pred = model.predict(data.X_test)
    y_proba = model.predict_proba(data.X_test)[:, 1]

    c1, c2 = st.columns(2)
    with c1:
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(data.y_test, y_pred)
        fig = px.imshow(
            cm, text_auto=True, x=["No Churn", "Churn"], y=["No Churn", "Churn"],
            title=f"Confusion Matrix — {model_choice}", color_continuous_scale="Blues",
        )
        st.plotly_chart(fig, width='stretch')
    with c2:
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(data.y_test, y_proba)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, name=model_choice))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], line=dict(dash="dash"), name="Chance"))
        fig.update_layout(title=f"ROC Curve — {model_choice}", xaxis_title="FPR", yaxis_title="TPR")
        st.plotly_chart(fig, width='stretch')

    with st.expander("Classification Report"):
        from sklearn.metrics import classification_report
        st.text(classification_report(data.y_test, y_pred, target_names=["No Churn", "Churn"]))


# ---------------------------------------------------------------------------
# Page: Recommendations
# ---------------------------------------------------------------------------

def page_recommendations(scored_df):
    st.title("🎯 Retention Recommendations")

    if scored_df is None:
        st.warning("No dataset/model available.")
        return

    counts = scored_df["RiskLevel"].value_counts().reindex(["LOW", "MEDIUM", "HIGH"]).fillna(0).astype(int)
    c1, c2, c3 = st.columns(3)
    c1.metric("Low Risk", f"{counts['LOW']:,}")
    c2.metric("Medium Risk", f"{counts['MEDIUM']:,}")
    c3.metric("High Risk", f"{counts['HIGH']:,}")

    st.divider()
    st.subheader("High-Risk Customers")
    high_risk = scored_df[scored_df["RiskLevel"] == "HIGH"].sort_values("ChurnProbability", ascending=False)

    rows = []
    for _, row in high_risk.head(200).iterrows():
        customer_raw = {col: row[col] for col in RAW_FORM_COLUMNS}
        recs = generate_recommendations(customer_raw, row["ChurnProbability"])
        top_action = recs[0].action if recs else "N/A"
        rows.append({
            "customerID": row["customerID"],
            "ChurnProbability": row["ChurnProbability"],
            "RiskLevel": row["RiskLevel"],
            "MainRiskFactor": "Contract" if customer_raw.get("Contract") == "Month-to-month" else
                              ("TechSupport" if customer_raw.get("TechSupport") == "No" else "Other"),
            "RecommendedAction": top_action,
        })
    display_df = pd.DataFrame(rows)
    st.caption(f"Showing top {len(display_df)} of {len(high_risk):,} high-risk customers (sorted by probability)")
    st.dataframe(
        display_df.style.format({"ChurnProbability": "{:.2%}"}),
        width='stretch', height=400,
    )

    csv = display_df.to_csv(index=False).encode("utf-8")
    st.download_button("Export High-Risk Customer List (CSV)", csv, "high_risk_customers.csv", "text/csv")
    st.info(DISCLAIMER)


# ---------------------------------------------------------------------------
# Page: About
# ---------------------------------------------------------------------------

def page_about():
    st.title("ℹ️ About This Project")
    st.markdown("""
### Explainable Customer Churn Prediction and Retention Recommendation System

An end-to-end machine learning application that predicts customer churn risk,
explains *why* using SHAP, and generates rule-based retention recommendations.

**Tech Stack:** Python, Pandas, Scikit-learn, XGBoost, SHAP, Streamlit, Plotly

**Pipeline:** Data validation → EDA → Preprocessing → Feature engineering →
Model training (Logistic Regression, Decision Tree, Random Forest, XGBoost) →
Hyperparameter tuning → Model selection → SHAP explainability →
Retention recommendation engine → This dashboard.

**Important:** Predictions and recommendations are decision-support tools
based on historical patterns. They describe correlation, not causation, and
are not guarantees of individual customer behavior or intervention outcomes.

**Dataset:** Public IBM Telco Customer Churn dataset (7,043 customers).
""")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to",
        ["Dashboard", "Prediction", "Customer Analysis", "Explainability",
         "Model Performance", "Recommendations", "About"],
    )

    df, report = get_raw_data()
    if df is None:
        st.error(
            "Dataset not found at `data/customer_churn.csv`. Place the Telco "
            "Customer Churn CSV there, then restart the app."
        )
        return

    bundle = get_model_bundle()
    scored_df = score_full_dataset(df) if bundle is not None else None

    if bundle is None and page != "About":
        st.warning(
            "No trained model found in `models/`. Run "
            "`python -m src.model_selection` to train and save one, then restart the app."
        )

    if page == "Dashboard":
        page_dashboard(df, scored_df)
    elif page == "Prediction":
        page_prediction(df)
    elif page == "Customer Analysis":
        page_customer_analysis(scored_df)
    elif page == "Explainability":
        page_explainability(df)
    elif page == "Model Performance":
        page_model_performance(df)
    elif page == "Recommendations":
        page_recommendations(scored_df)
    elif page == "About":
        page_about()


if __name__ == "__main__":
    main()
