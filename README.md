# Explainable Customer Churn Prediction and Retention Recommendation System

An end-to-end machine learning application that predicts customer churn,
explains *why* using SHAP, and generates rule-based retention
recommendations — built as a final-year/portfolio-grade project with a
full Streamlit dashboard.

---

## 1. Introduction

Customer churn — a customer ending their relationship with a business —
is one of the most costly problems for subscription-based companies
(telecom, SaaS, banking, streaming, insurance). Acquiring a new customer
is typically far more expensive than retaining an existing one, so being
able to predict *which* customers are at risk, and *why*, lets a business
intervene before it's too late.

## 2. Problem Statement

Given historical customer data (demographics, account details, services
subscribed, billing information), predict the probability that a given
customer will churn, and provide actionable, explainable retention
guidance — not just a black-box score.

## 3. Objectives

1. Predict churn probability for a customer (binary classification).
2. Explain predictions with SHAP (global + per-customer local explanations).
3. Segment customers into Low / Medium / High risk.
4. Generate rule-based, explainable retention recommendations.
5. Present all of the above through an interactive dashboard.

## 4. Dataset

**Public IBM Telco Customer Churn dataset** — 7,043 customers, 21 columns
(demographics, account info, subscribed services, billing, and the
`Churn` target). Place it at `data/customer_churn.csv` (already included
in this repo for out-of-the-box deployment — see the note in
`.gitignore`).

## 5. Methodology

| Stage | What was done | Module |
|---|---|---|
| Data validation | Column/dtype checks, duplicate detection, `TotalCharges` blank-string diagnosis | `src/data_preprocessing.py` |
| EDA | Churn distribution, numerical/categorical analysis vs. churn, correlation heatmap, 6 required research questions — answered from real computed statistics | `notebooks/churn_analysis.ipynb` |
| Preprocessing | `ColumnTransformer` (StandardScaler + OneHotEncoder), fit on train split only (no leakage) | `src/data_preprocessing.py` |
| Feature engineering | 8 engineered features (`ContractRisk`, `SupportRisk`, `TenureGroup`, etc.), all leakage-safe (fixed rules/cutoffs, no fit statistics) | `src/feature_engineering.py` |
| Model training | Logistic Regression, Decision Tree, Random Forest, XGBoost — all class-imbalance-aware | `src/train_models.py` |
| Evaluation | Accuracy/Precision/Recall/F1/ROC-AUC, confusion matrices, ROC & PR curves | `src/evaluate_models.py` |
| Hyperparameter tuning | `RandomizedSearchCV` (5-fold CV, scored on ROC-AUC) for Random Forest & XGBoost | `src/tune_models.py` |
| Model selection | Explicit, programmatic rule (ROC-AUC first, Recall tie-break) — not "pick the top row" | `src/model_selection.py` |
| Explainability | SHAP `TreeExplainer`, global + local explanations | `src/explainability.py` |
| Recommendations | Rule-based retention engine tied to real EDA findings | `src/recommendations.py` |
| Integration | Single `predict_customer()` entry point used by the app | `src/inference.py` |
| Dashboard | 7-page Streamlit app | `app.py` |

## 6. Installation

```bash
git clone <your-repo-url>
cd customer-churn-prediction
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 7. Running the Project

The trained model artifacts are already committed under `models/`, so
you can run the app immediately:

```bash
streamlit run app.py
```

To retrain from scratch (optional):

```bash
python -m src.data_preprocessing   # Phase 1 + 3 sanity check
python -m src.feature_engineering  # Phase 4 sanity check
python -m src.model_selection      # Phases 5-8: trains, tunes, selects, saves best_model.pkl
python -m src.explainability       # Phase 9 sanity check
python -m src.recommendations      # Phase 10 sanity check
python -m src.inference            # End-to-end sanity check
```

## 8. Model Results

Final selected model: **XGBoost (Tuned)**, chosen by the rule *"highest
ROC-AUC, tie-broken by Recall within 0.01"* — implemented in
`select_final_model()`, not eyeballed.

Full comparison (test set, `outputs/reports/full_model_comparison.csv`):

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| **XGBoost (Tuned)** | 0.7580 | 0.5296 | **0.7888** | 0.6337 | **0.8470** |
| Random Forest (Tuned) | 0.7686 | 0.5458 | 0.7647 | **0.6370** | 0.8438 |
| Logistic Regression | 0.7374 | 0.5034 | 0.7861 | 0.6138 | 0.8414 |
| Random Forest | 0.7786 | 0.6054 | 0.4759 | 0.5329 | 0.8225 |
| XGBoost | 0.7672 | 0.5550 | 0.6203 | 0.5859 | 0.8135 |
| Decision Tree | 0.7374 | 0.5055 | 0.4947 | 0.5000 | 0.6601 |

**Note on model choice:** Random Forest (untuned) has the highest raw
accuracy but the worst recall of the ensembles — it misses over half of
actual churners, which defeats the purpose of a retention system. That's
why accuracy alone was not used for selection (see section 14 of the
original spec).

**Top global SHAP features:** `ContractRisk` (engineered), `tenure`,
`InternetService_Fiber optic`, `SupportRisk` (engineered), `PaymentMethod_Electronic check`.

## 9. Screenshots

*(Run the app locally or view the deployed link below — dashboard,
prediction, and explainability pages.)*

## 10. Deployment (Streamlit Community Cloud)

Streamlit apps cannot run on Vercel (no support for the persistent
WebSocket connection Streamlit's reactivity relies on). The standard
free host is **Streamlit Community Cloud**:

1. Push this repo to GitHub (public or private).
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with
   GitHub, click "New app".
3. Select this repo, branch `main`, main file path `app.py`.
4. Deploy. Model artifacts and the dataset are already committed, so no
   extra setup step is needed.

Alternatives that also support Streamlit: Render, Railway, Hugging Face
Spaces (all run a persistent Python process, unlike Vercel).

## 11. Future Scope

- Real-time prediction API (FastAPI) for CRM integration
- K-Means customer segmentation
- Automated retention email/SMS campaigns
- Cost-sensitive learning (weight false negatives by customer LTV)
- Time-to-churn / survival analysis instead of binary classification
- Model drift monitoring as the customer base evolves
- A/B testing of the recommended retention actions

## 12. Research Framing

**Possible contribution statement:** An explainable churn prediction
framework combining machine learning, SHAP-based local/global
explanations, and rule-based retention recommendations grounded in the
same explanations. Novelty is not claimed without a literature review —
this framing is provided for a project report / survey paper, not as a
research claim in itself.

**Research questions this project can support:**
1. Which customer attributes are most strongly associated with churn?
   (Answered empirically in section 8 above / the EDA notebook.)
2. Which ML model provides the best churn prediction for this dataset?
3. How does class-imbalance handling affect churn prediction metrics?
4. Can SHAP explanations plausibly support better retention decisions?

## 13. Important Disclaimers

- Predictions describe **correlation**, not causation.
- Retention recommendations are **decision-support suggestions**, not
  guaranteed interventions.
- Risk-level thresholds (Low <30%, Medium 30-70%, High >70%) are
  business-oriented defaults, not statistically optimized cutoffs — see
  `src/recommendations.py`.
- This project uses only public, non-identifying data.

## 14. Project Structure

```
customer-churn-prediction/
├── data/customer_churn.csv
├── notebooks/churn_analysis.ipynb      # Phase 2 EDA (executed)
├── src/
│   ├── data_preprocessing.py           # Phases 1 & 3
│   ├── feature_engineering.py          # Phase 4
│   ├── train_models.py                 # Phase 5
│   ├── evaluate_models.py              # Phase 6
│   ├── tune_models.py                  # Phase 7
│   ├── model_selection.py              # Phase 8
│   ├── explainability.py               # Phase 9
│   ├── recommendations.py              # Phase 10
│   └── inference.py                    # Integration layer
├── models/                             # Trained model artifacts (committed)
├── outputs/reports/                    # Comparison tables (CSV)
├── app.py                              # Phase 11: Streamlit app (7 pages)
├── requirements.txt                    # Pinned versions
├── .streamlit/config.toml              # Theme + server config
└── .gitignore
```
