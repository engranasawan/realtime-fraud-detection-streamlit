import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ==============================
# 1. Load artifacts (cached)
# ==============================


@st.cache_resource
def load_artifacts():
    """
    Load all persisted models and preprocessing artifacts from the models/ folder.
    Cached so they are only loaded once per session.
    """
    models_dir = Path("models")

    def _load(name):
        path = models_dir / name
        try:
            return joblib.load(path)
        except Exception as e:
            st.error(f"❌ Error loading {name}")
            st.exception(e)
            raise

    preprocess_pipeline = _load("preprocess_pipeline.joblib")
    imputer = _load("post_preprocess_imputer.joblib")
    lgbm_model = _load("lightgbm.joblib")
    iso_model = _load("isolation_forest.joblib")

    return preprocess_pipeline, imputer, lgbm_model, iso_model


preprocess_pipeline, imputer, lgbm_model, iso_model = load_artifacts()

# ==============================
# 2. Risk thresholds & logic
# ==============================

# LightGBM thresholds from validation search
FRAUD_MED = 0.00005
FRAUD_HIGH = 0.00023328
FRAUD_CRIT = 0.01732857

# Isolation Forest thresholds (tuned)
ANOM_MED = 0.04  # mild anomaly
ANOM_HIGH = 0.05  # stronger anomaly
ANOM_CRIT = 0.08  # extreme anomaly


def risk_from_scores(fraud_prob: float, anomaly_score: float) -> str:
    """
    Combined rule:
    - CRITICAL: either model extremely suspicious
    - HIGH: fraud_prob clearly high OR (moderate fraud_prob + strong anomaly)
    - MEDIUM: any moderate signal from either model
    - LOW: everything looks calm
    """
    if fraud_prob >= FRAUD_CRIT or anomaly_score >= ANOM_CRIT:
        return "CRITICAL"
    elif fraud_prob >= FRAUD_HIGH or (
        fraud_prob >= FRAUD_MED and anomaly_score >= ANOM_HIGH
    ):
        return "HIGH"
    elif fraud_prob >= FRAUD_MED or anomaly_score >= ANOM_MED:
        return "MEDIUM"
    else:
        return "LOW"


def score_transaction(input_dict: dict):
    """
    input_dict must contain the final feature columns used in training:
    Amount, TransactionType, Location, DeviceID, Channel, hour, day_of_week, month
    """
    df = pd.DataFrame([input_dict])

    # 1) Preprocess (same pipeline as during training)
    X_prep = preprocess_pipeline.transform(df)

    # 2) Impute missing values
    X_imp = imputer.transform(X_prep).astype(np.float32)

    # 3) Supervised fraud probability (LightGBM)
    fraud_prob = float(lgbm_model.predict_proba(X_imp)[0, 1])

    # 4) Unsupervised anomaly score (Isolation Forest)
    # NOTE: we invert decision_function so that higher = more anomalous
    anomaly_score = float(-iso_model.decision_function(X_imp)[0])

    # 5) Final risk level
    risk = risk_from_scores(fraud_prob, anomaly_score)

    return fraud_prob, anomaly_score, risk


def render_risk_badge(risk: str):
    color_map = {
        "LOW": "#2e7d32",  # green
        "MEDIUM": "#f9a825",  # amber
        "HIGH": "#f57c00",  # orange
        "CRITICAL": "#c62828",  # red
    }
    risk_color = color_map.get(risk, "#607d8b")

    st.markdown(
        f"""
        <div style="padding: 0.75rem 1rem; border-radius: 0.5rem;
                    background-color: {risk_color}22; border: 1px solid {risk_color};">
            <span style="font-size: 1.1rem; font-weight: 600; color: {risk_color};">
                Risk Level: {risk}
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==============================
# 3. Streamlit UI
# ==============================

st.set_page_config(
    page_title="Real-Time Fraud Detection Demo",
    page_icon="💳",
    layout="centered",
)

st.title("💳 Real-Time Fraud Detection Prototype")
st.write(
    "This demo uses a **supervised LightGBM model** and an **unsupervised Isolation Forest** "
    "to assess the fraud risk of a single transaction in real time."
)

st.markdown("---")

st.sidebar.header("⚙️ Configuration")
st.sidebar.markdown(
    """
**How this works:**

- We apply the **same preprocessing pipeline** used during training.
- The **LightGBM model** outputs a fraud probability.
- The **Isolation Forest** outputs an anomaly score.
- A simple rule-engine combines both into a **risk level**:
  - LOW / MEDIUM / HIGH / CRITICAL
"""
)

st.header("🧾 Enter Transaction Details")

with st.form("txn_form"):
    col1, col2 = st.columns(2)

    with col1:
        amount = st.number_input(
            "Transaction Amount", min_value=0.0, value=1200.0, step=10.0
        )
        txn_type = st.selectbox(
            "Transaction Type",
            ["PAYMENT", "TRANSFER", "DEBIT", "CREDIT", "CASH_OUT", "OTHER"],
            index=0,
        )
        channel = st.selectbox(
            "Channel",
            ["Mobile", "NetBanking", "Online", "Card", "ATM", "Other"],
            index=2,
        )

    with col2:
        location = st.text_input("Location (City / Region)", value="Karachi")
        device = st.selectbox(
            "Device / OS",
            ["Android", "iOS", "Windows", "Linux", "Other"],
            index=0,
        )
        # Streamlit Cloud doesn't support st.datetime_input, so use date + time
        txn_date = st.date_input("Transaction Date", value=datetime.date.today())
        txn_time = st.time_input(
            "Transaction Time", value=datetime.datetime.now().time()
        )

    submitted = st.form_submit_button("🚀 Run Fraud Check")

if submitted:
    # Combine date & time and extract time-based features
    txn_datetime = datetime.datetime.combine(txn_date, txn_time)
    hour = txn_datetime.hour
    day_of_week = txn_datetime.weekday()  # 0 = Monday
    month = txn_datetime.month

    # Build model payload
    input_payload = {
        "Amount": amount,
        "TransactionType": txn_type,
        "Location": location,
        "DeviceID": device,
        "Channel": channel,
        "hour": hour,
        "day_of_week": day_of_week,
        "month": month,
    }

    with st.spinner("Scoring transaction..."):
        fraud_prob, anomaly_score, risk = score_transaction(input_payload)

    st.markdown("## 🔎 Results")
    render_risk_badge(risk)

    col_a, col_b = st.columns(2)
    with col_a:
        st.metric(
            "Fraud Probability (LightGBM)",
            f"{fraud_prob:.8f}",
            help="Direct output probability from the supervised LightGBM model.",
        )
    with col_b:
        st.metric(
            "Anomaly Score (Isolation Forest)",
            f"{anomaly_score:.5f}",
            help="Higher = more unusual compared to 'normal' historical patterns.",
        )

    st.markdown("### 📦 Model Input Payload")
    st.json(input_payload)

    st.markdown(
        """
        ### 🧠 How to interpret this
        
        - **Fraud Probability** is learned from historic labeled data (fraud vs non-fraud).
        - **Anomaly Score** comes from an unsupervised model trained only on normal behavior.
        - The final **Risk Level** is determined using thresholds calibrated on a validation set.
        """
    )
else:
    st.info("Fill the form above and click **Run Fraud Check** to score a transaction.")
