"""
recommendations.py

Phase 10: Retention Recommendation Engine.

Rule-based (per section 18's explicit requirement -- explainable, not
another opaque model). Rules are keyed off the SAME raw customer
attributes used by feature engineering, so this module can run on a
single customer dict from the Streamlit form without needing the full
preprocessed array. Risk level thresholds are configurable, per
section 16, and NOT claimed to be statistically optimal -- they are
business-oriented defaults.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Configurable, business-oriented thresholds (section 16). Not derived
# from ROC/PR-curve optimization -- deliberately simple and explainable
# for a viva/demo setting.
LOW_RISK_MAX = 0.30
MEDIUM_RISK_MAX = 0.70


def get_risk_level(churn_probability: float) -> str:
    if churn_probability < LOW_RISK_MAX:
        return "LOW"
    elif churn_probability < MEDIUM_RISK_MAX:
        return "MEDIUM"
    return "HIGH"


@dataclass
class Recommendation:
    action: str
    reason: str


def generate_recommendations(customer: dict, churn_probability: float) -> list[Recommendation]:
    """Generate rule-based retention recommendations for one customer.

    `customer` is a dict of raw (pre-encoding) attribute values, e.g.
    {"Contract": "Month-to-month", "MonthlyCharges": 95.0,
     "TechSupport": "No", "tenure": 3, "OnlineSecurity": "No", ...}
    Only HIGH and MEDIUM risk customers get proactive recommendations;
    LOW risk customers get a no-action note -- matching section 18's
    examples, which are all framed around "High Churn + <condition>".

    These are decision-support suggestions only -- generate_recommendations
    never claims an intervention will prevent churn (section 18, 36.18).
    """
    risk_level = get_risk_level(churn_probability)
    recs: list[Recommendation] = []

    if risk_level == "LOW":
        return [Recommendation(
            action="No proactive action needed at this time.",
            reason=f"Predicted churn probability ({churn_probability * 100:.1f}%) is low.",
        )]

    if customer.get("Contract") == "Month-to-month":
        recs.append(Recommendation(
            action="Offer a discounted annual or two-year contract upgrade.",
            reason="Month-to-month contracts have the highest observed churn rate "
                   "(42.71% in this dataset) of any contract type.",
        ))

    if customer.get("MonthlyCharges", 0) >= 70:
        recs.append(Recommendation(
            action="Offer a lower-cost plan tier or a personalized loyalty discount.",
            reason=f"Monthly charges (${customer.get('MonthlyCharges', 0):.2f}) are in the "
                   f"range associated with elevated churn.",
        ))

    if customer.get("TechSupport") == "No" and customer.get("InternetService", "No") != "No":
        recs.append(Recommendation(
            action="Proactively offer a free trial of priority technical support.",
            reason="Customers without TechSupport churn at 41.64% vs. 15.17% with it.",
        ))

    if customer.get("OnlineSecurity") == "No" and customer.get("InternetService", "No") != "No":
        recs.append(Recommendation(
            action="Offer a bundled security/backup add-on package.",
            reason="Lack of security/backup add-ons is associated with higher churn "
                   "and no add-on revenue to anchor the relationship.",
        ))

    tenure = customer.get("tenure", 0)
    if tenure <= 12:
        recs.append(Recommendation(
            action="Enroll in a new-customer onboarding and first-year retention program.",
            reason=f"Tenure of {tenure} months is in the highest-churn band (0-12 months, "
                   f"47.4% churn rate in this dataset).",
        ))

    if customer.get("PaymentMethod") == "Electronic check":
        recs.append(Recommendation(
            action="Encourage switching to automatic bank transfer or credit card payment "
                   "(e.g. small incentive for enrolling).",
            reason="Electronic check payers churn at 45.29%, roughly 3x the rate of "
                   "automatic payment methods (15-17%).",
        ))

    if not recs:
        recs.append(Recommendation(
            action="Flag for manual review by the retention team.",
            reason=f"Model flags this customer as {risk_level} risk "
                   f"({churn_probability * 100:.1f}%), but no specific rule matched -- "
                   f"warrants a closer look.",
        ))

    return recs


def format_recommendations(recs: list[Recommendation]) -> str:
    return "\n".join(f"- {r.action}\n    ({r.reason})" for r in recs)


DISCLAIMER = (
    "These recommendations are decision-support suggestions generated from "
    "observed patterns in historical data. They are not guaranteed to "
    "prevent any individual customer's churn."
)


if __name__ == "__main__":
    example_customers = [
        {
            "Contract": "Month-to-month", "MonthlyCharges": 95.0, "TechSupport": "No",
            "OnlineSecurity": "No", "InternetService": "Fiber optic", "tenure": 2,
            "PaymentMethod": "Electronic check",
        },
        {
            "Contract": "Two year", "MonthlyCharges": 45.0, "TechSupport": "Yes",
            "OnlineSecurity": "Yes", "InternetService": "DSL", "tenure": 60,
            "PaymentMethod": "Bank transfer (automatic)",
        },
    ]
    example_probs = [0.87, 0.08]

    for customer, prob in zip(example_customers, example_probs):
        print(f"Churn Probability: {prob * 100:.1f}% -> Risk Level: {get_risk_level(prob)}")
        recs = generate_recommendations(customer, prob)
        print(format_recommendations(recs))
        print()
    print(DISCLAIMER)
