# app.py (updated)
import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ------------------------------
# 1. Load artifacts (cached)
# ------------------------------
@st.cache_resource
def load_artifacts():
    models_dir = Path("models")

    def _load(name: str):
        path = models_dir / name
        try:
            return joblib.load(path)
        except Exception as e:
            st.error(f"❌ Error loading {name}")
            st.exception(e)
            raise

    supervised_pipeline = _load("supervised_lgbm_pipeline.joblib")
    iforest_pipeline = _load("iforest_pipeline.joblib")

    return supervised_pipeline, iforest_pipeline


supervised_pipeline, iforest_pipeline = load_artifacts()

# ------------------------------
# 2. Risk thresholds & mapping
# ------------------------------
FRAUD_MED = 0.00005
FRAUD_HIGH = 0.00023328
FRAUD_CRIT = 0.01732857

ANOM_MED = 0.04
ANOM_HIGH = 0.05
ANOM_CRIT = 0.08


def risk_from_scores(fraud_prob: float, anomaly_score: float) -> str:
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


# ------------------------------
# 3. Rule engine
# ------------------------------
def evaluate_rules(
    payload: Dict,
    fraud_prob: float,
    anomaly_score: float,
) -> Tuple[List[Dict], str]:
    """
    Returns (triggered_rules, rules_overall_level)
    Each rule is a dict: {"name": str, "severity": "LOW|MEDIUM|HIGH|CRITICAL", "detail": str}
    The overall rules_overall_level is the highest severity among rules.
    """

    rules: List[Dict] = []
    # helper
    def add(name, severity, detail):
        rules.append({"name": name, "severity": severity, "detail": detail})

    amount = float(payload.get("Amount", 0.0))
    channel = payload.get("Channel", "").lower()
    location = payload.get("Location", "").strip().lower()
    hour = int(payload.get("hour", 0))
    device = payload.get("DeviceID", "")
    ip_country = payload.get("ip_country", None)
    declared_country = payload.get("declared_country", None)
    monthly_avg = float(payload.get("monthly_avg", 0.0) or 0.0)
    txns_24h = int(payload.get("transactions_last_24h", 0) or 0)
    failed_logins = int(payload.get("failed_login_attempts", 0) or 0)
    new_beneficiary = bool(payload.get("new_beneficiary", False))
    atm_distance_km = float(payload.get("atm_distance_km", 0.0) or 0.0)
    cvv_provided = payload.get("cvv_provided", True)
    card_country = payload.get("card_country", None)
    device_last_seen = payload.get("device_last_seen", None)
    txn_country = declared_country or location

    # CRITICAL rules
    # 1) CVV missing on online card payment (common high-risk indicator)
    if channel == "card" or channel == "online":
        if (channel == "online" or payload.get("card_used_online", False)) and not cvv_provided:
            add(
                "Missing CVV for online card transaction",
                "HIGH",
                "Card transaction attempted online without CVV verification.",
            )

    # 2) Very large single transaction vs limits
    ABSOLUTE_HIGH_AMOUNT = 10_000_000  # example absolute threshold (tune for your currency)
    if amount >= ABSOLUTE_HIGH_AMOUNT:
        add(
            "Very large amount",
            "CRITICAL",
            f"Transaction amount {amount} exceeds absolute high threshold {ABSOLUTE_HIGH_AMOUNT}.",
        )

    # 3) ATM distance large (card used far from last known location)
    if channel == "atm" and atm_distance_km and atm_distance_km > 500:
        add(
            "ATM distant from last known location",
            "HIGH",
            f"ATM distance {atm_distance_km:.1f} km from customer's last known region.",
        )

    # HIGH / MEDIUM rules
    # 4) Amount much higher than customer's monthly average (sudden spike)
    if monthly_avg > 0:
        if amount >= 5 * monthly_avg and amount > 2000:
            add(
                "Spike vs monthly average",
                "HIGH",
                f"Transaction amount {amount} is >=5x customer's monthly average {monthly_avg:.2f}.",
            )
        elif amount >= 2 * monthly_avg and amount > 1000:
            add(
                "Above usual spend",
                "MEDIUM",
                f"Transaction amount {amount} is >=2x customer's monthly average {monthly_avg:.2f}.",
            )

    # 5) Velocity: many transactions in last 24 hours
    if txns_24h >= 20:
        add(
            "High velocity (24h)",
            "HIGH",
            f"{txns_24h} transactions in the last 24 hours.",
        )
    elif txns_24h >= 6:
        add(
            "Suspicious velocity (24h)",
            "MEDIUM",
            f"{txns_24h} transactions in the last 24 hours.",
        )

    # 6) IP country vs declared country mismatch
    if ip_country and txn_country and ip_country.lower() != str(txn_country).lower():
        # Some mismatches are benign; use heuristics
        suspicious_countries = {"nigeria", "romania", "ukraine", "russia"}  # example
        severity = "HIGH" if ip_country.lower() in suspicious_countries or amount > 2000 else "MEDIUM"
        add(
            "IP country mismatch",
            severity,
            f"IP country={ip_country} differs from declared/txn country={txn_country}.",
        )

    # 7) Device has not been seen before and a high-value transaction
    if device_last_seen is None or device_last_seen == "":
        if amount > 1000:
            add(
                "New device + high amount",
                "HIGH",
                "Transaction originates from a device not seen before and amount is high.",
            )
        else:
            add(
                "New device (low amount)",
                "MEDIUM",
                "Transaction originates from a device not seen before.",
            )
    else:
        if device and device_last_seen and device.lower() != device_last_seen.lower():
            add(
                "Device mismatch",
                "MEDIUM",
                f"Current device '{device}' differs from last seen '{device_last_seen}'.",
            )

    # 8) Multiple failed logins recently
    if failed_logins >= 5:
        add(
            "Many failed login attempts",
            "HIGH",
            f"{failed_logins} failed authentication attempts recently.",
        )
    elif failed_logins >= 3:
        add(
            "Failed login attempts",
            "MEDIUM",
            f"{failed_logins} failed authentication attempts recently.",
        )

    # 9) New beneficiary + high amount
    if new_beneficiary and amount > 1000:
        add(
            "New beneficiary + high transfer",
            "HIGH",
            "Funds are being sent to a newly added beneficiary with a high amount.",
        )

    # 10) Time-of-day oddity (late-night transaction for user with low activity)
    if hour >= 0 and hour <= 5 and monthly_avg < 2000 and amount > 100:
        add(
            "Odd hour transaction",
            "MEDIUM",
            f"Transaction at hour {hour} (00:00-05:00) and user has low monthly average activity.",
        )

    # 11) Cross-border card country mismatch
    if card_country and txn_country and str(card_country).lower() != str(txn_country).lower():
        add(
            "Card country mismatch",
            "MEDIUM",
            f"Card registered in {card_country} but transaction declared in {txn_country}.",
        )

    # 12) Low anomaly but rules indicate risk — e.g., if ML says low but many rules fired
    # (we'll compute a combined level below)

    # compute aggregate highest severity
    severity_order = {"LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}
    highest = "LOW"
    for r in rules:
        if severity_order[r["severity"]] > severity_order[highest]:
            highest = r["severity"]

    return rules, highest


# ------------------------------
# 4. Combine rule results with models
# ------------------------------
def combine_decision(ml_risk: str, rule_level: str) -> str:
    """
    Combine ML-derived risk and rule-derived risk into a final label.
    Priority: if either is CRITICAL -> CRITICAL, else escalate if either is HIGH etc.
    """
    order = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    ml_idx = order.index(ml_risk)
    rule_idx = order.index(rule_level)
    final_idx = max(ml_idx, rule_idx)
    return order[final_idx]


# ------------------------------
# 5. UI
# ------------------------------
st.set_page_config(
    page_title="Real-Time Fraud Detection Prototype (Rules + ML)",
    page_icon="💳",
    layout="centered",
)

st.title("💳 Real-Time Fraud Detection Prototype (Rules + ML)")
st.write(
    "This demo combines pre-trained ML models (LightGBM + IsolationForest) with explicit rule-based checks "
    "for additional safety. Enter transaction details below."
)
st.markdown("---")

# Sidebar info
st.sidebar.header("⚙️ Configuration / Quick notes")
st.sidebar.markdown(
    """
- Start by selecting the **Channel**. The UI will adapt to collect channel-specific fields.
- Provide any available account history (monthly average, recent transaction count) to enable spending-pattern rules.
- IP country / Declared country fields allow IP-location mismatch checks.
"""
)

# Step 1: choose channel (this drives the rest of the form)
st.header("Step 1 — Choose Channel")
channel_choice = st.selectbox(
    "Which channel is this transaction originating from?",
    ["Bank", "Mobile App", "ATM", "Credit Card", "POS", "Online Purchase", "NetBanking", "Other"],
)

st.markdown("---")
st.header("Step 2 — Channel-specific Transaction Details")

# We'll collect a common block + channel-specific fields
with st.form("txn_form_v2"):
    col1, col2 = st.columns(2)
    with col1:
        amount = st.number_input("Transaction Amount", min_value=0.0, value=1200.0, step=10.0)
        txn_type = st.selectbox(
            "Transaction Type",
            ["PAYMENT", "TRANSFER", "DEBIT", "CREDIT", "CASH_OUT", "OTHER"],
            index=0,
        )
        location = st.text_input("Location (City / Region)", value="Karachi")
        declared_country = st.text_input("Declared Country (country for transaction)", value="Pakistan")
    with col2:
        txn_date = st.date_input("Transaction Date", value=datetime.date.today())
        txn_time = st.time_input("Transaction Time", value=datetime.datetime.now().time())
        hour = txn_time.hour
        device = st.text_input("Device / OS", value="Android")
        device_last_seen = st.text_input("Last known device (if any)", value="Android")

    # Account history (helps spending-pattern rules)
    st.markdown("**Account history (optional but recommended)**")
    col3, col4 = st.columns(2)
    with col3:
        monthly_avg = st.number_input("Customer monthly average spend", min_value=0.0, value=10000.0, step=100.0)
        transactions_last_24h = st.number_input("Transactions in last 24h", min_value=0, value=1, step=1)
    with col4:
        failed_login_attempts = st.number_input("Failed login attempts (recent)", min_value=0, value=0, step=1)
        new_beneficiary = st.checkbox("Is this to a newly added beneficiary?", value=False)

    # IP & geo
    st.markdown("**IP & geo details**")
    col5, col6 = st.columns(2)
    with col5:
        ip_address = st.text_input("Client IP address (optional)", value="")
        ip_country = st.text_input("IP-derived country (optional)", value="")
    with col6:
        atm_distance_km = st.number_input("ATM distance from last location (km) — if ATM use", min_value=0.0, value=0.0)
        card_country = st.text_input("Card registered country (if card used)", value="")

    # Channel specific fields
    st.markdown(f"**Fields for channel: {channel_choice}**")
    channel = channel_choice.lower()
    card_used_online = False
    cvv_provided = True
    if channel_choice == "Credit Card":
        col7, col8 = st.columns(2)
        with col7:
            card_masked = st.text_input("Card number (masked) — e.g. 4111****1111", value="")
            card_bin = st.text_input("Card BIN (first 6 digits) (optional)", value="")
            card_holder_name = st.text_input("Cardholder name", value="")
        with col8:
            card_country = st.text_input("Card country (if different)", value=card_country or "")
            cvv_provided = st.checkbox("CVV entered/verified", value=True)
            ecom = st.checkbox("Card used for e-commerce / online", value=False)
            card_used_online = ecom
    elif channel_choice == "Online Purchase":
        merchant = st.text_input("Merchant name / ID", value="")
        browser = st.text_input("Browser / UA string (short)", value="")
        ip_address = st.text_input("Client IP address (optional)", value=ip_address or "")
        ip_country = st.text_input("IP-derived country (optional)", value=ip_country or "")
        # card details if card used online
        used_card_online = st.checkbox("Payment by card", value=False)
        if used_card_online:
            card_used_online = True
            cvv_provided = st.checkbox("CVV entered/verified", value=True)
            card_country = st.text_input("Card country (if different)", value=card_country or "")
    elif channel_choice == "Mobile App":
        app_version = st.text_input("App version", value="1.0.0")
        device_fingerprint = st.text_input("Device fingerprint (optional)", value="")
        failed_login_attempts = st.number_input("Failed login attempts (recent)", min_value=0, value=failed_login_attempts, step=1)
    elif channel_choice == "ATM":
        atm_id = st.text_input("ATM ID", value="")
        atm_location = st.text_input("ATM location city/region", value=location)
        atm_distance_km = st.number_input("ATM distance (km) from last known location", min_value=0.0, value=atm_distance_km)
    elif channel_choice == "POS":
        merchant_id = st.text_input("POS Merchant ID", value="")
        pos_location = st.text_input("POS location", value=location)
    elif channel_choice == "NetBanking" or channel_choice == "Bank":
        beneficiary = st.text_input("Beneficiary (if transfer)", value="")
        new_beneficiary = st.checkbox("Is beneficiary newly added?", value=new_beneficiary)
    else:
        extra_notes = st.text_area("Any extra notes / context", value="")

    submitted = st.form_submit_button("🚀 Run Fraud Check")

# When submitted, create payload, call ML scoring and rules engine
if submitted:
    txn_datetime = datetime.datetime.combine(txn_date, txn_time)
    hour = txn_datetime.hour
    day_of_week = txn_datetime.weekday()
    month = txn_datetime.month

    # Build payload for model: keep the original model field names
    input_payload = {
        "Amount": amount,
        "TransactionType": txn_type,
        "Location": location,
        "DeviceID": device,
        "Channel": channel_choice,
        "hour": hour,
        "day_of_week": day_of_week,
        "month": month,
        # extra fields for rules
        "ip_address": ip_address,
        "ip_country": ip_country,
        "declared_country": declared_country,
        "monthly_avg": monthly_avg,
        "transactions_last_24h": transactions_last_24h,
        "failed_login_attempts": failed_login_attempts,
        "new_beneficiary": new_beneficiary,
        "atm_distance_km": atm_distance_km,
        "cvv_provided": cvv_provided,
        "card_country": card_country,
        "device_last_seen": device_last_seen,
        "card_used_online": card_used_online,
    }

    # Score with ML models
    with st.spinner("Scoring transaction with ML models..."):
        df_for_model = pd.DataFrame([{
            # The supervised pipeline expects these fields — adapt if your pipeline expects others
            "Amount": input_payload["Amount"],
            "TransactionType": input_payload["TransactionType"],
            "Location": input_payload["Location"],
            "DeviceID": input_payload["DeviceID"],
            "Channel": input_payload["Channel"],
            "hour": input_payload["hour"],
            "day_of_week": input_payload["day_of_week"],
            "month": input_payload["month"],
        }])
        # If your pipeline expects more fields, extend the dict accordingly.
        try:
            fraud_prob = float(supervised_pipeline.predict_proba(df_for_model)[0, 1])
        except Exception as e:
            st.error("Error scoring with supervised model. Check model input format.")
            st.exception(e)
            fraud_prob = 0.0
        try:
            raw_score = float(iforest_pipeline.decision_function(df_for_model)[0])
            anomaly_score = -raw_score
        except Exception as e:
            st.error("Error scoring with IsolationForest. Check model input format.")
            st.exception(e)
            anomaly_score = 0.0

    ml_risk = risk_from_scores(fraud_prob, anomaly_score)

    # Evaluate rules
    rules_triggered, rules_level = evaluate_rules(input_payload, fraud_prob, anomaly_score)

    # Combine into final decision
    final_risk = combine_decision(ml_risk, rules_level)

    # Show results
    st.markdown("## 🔎 Results")
    # risk badge
    color_map = {
        "LOW": "#2e7d32",
        "MEDIUM": "#f9a825",
        "HIGH": "#f57c00",
        "CRITICAL": "#c62828",
    }
    risk_color = color_map.get(final_risk, "#607d8b")
    st.markdown(
        f"""
        <div style="padding: 0.75rem 1rem; border-radius: 0.5rem;
                    background-color: {risk_color}22; border: 1px solid {risk_color};">
            <span style="font-size: 1.1rem; font-weight: 600; color: {risk_color};">
                Final Risk Level: {final_risk}
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ML metrics
    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("Fraud Probability (LightGBM)", f"{fraud_prob:.8f}")
    with col_b:
        st.metric("Anomaly Score (IsolationForest, higher=more anomalous)", f"{anomaly_score:.5f}")

    # Show triggered rules, if any
    st.markdown("### ⚠️ Rule-based checks triggered")
    if rules_triggered:
        for r in rules_triggered:
            sev = r["severity"]
            icon = "🔴" if sev in ("HIGH", "CRITICAL") else "🟠" if sev == "MEDIUM" else "🟢"
            st.write(f"{icon} **{r['name']}** — *{sev}*")
            st.caption(r["detail"])
    else:
        st.info("No explicit rules were triggered.")

    # Show ML-only risk and rule-level separately for transparency
    st.markdown("### 🧾 Diagnostic summary")
    st.write(f"- ML-derived risk: **{ml_risk}**")
    st.write(f"- Rule-derived highest severity: **{rules_level}**")
    st.write(f"- Final combined risk: **{final_risk}**")

    st.markdown("### 📦 Input payload (for debugging)")
    st.json(input_payload)

    st.markdown(
        """
        ### 🧠 Notes on interpretation
        - Rule checks are deterministic heuristics (IP vs declared country mismatch, velocity, ATM distance, CVV missing, etc.).
        - ML models provide probabilistic/anomaly signals. Final risk escalates if either ML or rule checks are high.
        - Tune thresholds (amount multipliers, suspicious countries, distance km) for your product and region.
        """
    )
else:
    st.info("Choose a channel and fill the form above, then click **Run Fraud Check**.")
