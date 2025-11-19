# app.py
"""
Real-Time Fraud Detection (Optimized)
- INR base currency (configurable conversion table)
- Currency, Amount, Date, Time are the only common fields present for every channel.
- Channel-specific fields are exclusive to each channel.
- Device-related rules disabled for Bank and ATM (Option A); NetBanking kept device-aware.
- Fraud confidence & anomaly score shown in 0-100 range (percent).
- Channel-specific transaction types (enforced).
- ML models integrated (supervised + isolation forest). If model expects INR, convert before scoring.
"""

import datetime
from math import radians, sin, cos, asin, sqrt
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import pandas as pd
import streamlit as st

# ======================
# CONFIG: Currency / thresholds (INR base)
# ======================
INR_PER_UNIT = {
    "INR": 1.0,
    "USD": 83.2,
    "EUR": 90.5,
    "GBP": 105.3,
    "AED": 22.7,
    "SAR": 22.2,
}
CURRENCY_OPTIONS = list(INR_PER_UNIT.keys())
DEFAULT_CURRENCY = "INR"

# Base thresholds (INR) — tune these using historical data
BASE_THRESHOLDS_INR = {
    "absolute_crit_amount": 10_000_000,  # critical single tx in INR
    "high_amount_threshold": 2_000_000,  # high
    "medium_amount_threshold": 100_000,  # medium
    "atm_high_withdrawal": 300_000,
    "card_test_small_amount_inr": 200,
}

# ML thresholds (probabilities / anomaly scores are currency-agnostic)
FRAUD_MED = 0.00005
FRAUD_HIGH = 0.00023328
FRAUD_CRIT = 0.01732857

ANOM_MED = 0.04
ANOM_HIGH = 0.05
ANOM_CRIT = 0.08

# Severity ordering
SEVERITY_ORDER = {"LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}


# ----------------------
# Helpers
# ----------------------
def escalate(a: str, b: str) -> str:
    return a if SEVERITY_ORDER[a] >= SEVERITY_ORDER[b] else b


def inr_to_currency(amount_in_inr: float, currency: str) -> float:
    """Convert INR-denominated threshold to selected currency units."""
    rate = INR_PER_UNIT.get(currency, 1.0)
    # amount_in_currency = amount_in_inr / INR_PER_UNIT[currency]
    return amount_in_inr / rate


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    if None in (lat1, lon1, lat2, lon2):
        return None
    lat1, lon1, lat2, lon2 = map(radians, (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    return 6371 * c


# ----------------------
# Load ML artifacts (cached)
# ----------------------
@st.cache_resource
def load_artifacts():
    models_dir = Path("models")
    def _load(name: str):
        path = models_dir / name
        try:
            return joblib.load(path)
        except Exception as e:
            st.error(f"Error loading model artifact: {name}")
            st.exception(e)
            raise
    supervised_pipeline = _load("supervised_lgbm_pipeline.joblib")
    iforest_pipeline = _load("iforest_pipeline.joblib")
    return supervised_pipeline, iforest_pipeline


supervised_pipeline, iforest_pipeline = load_artifacts()


# ----------------------
# ML scoring helpers
# ----------------------
def ml_risk_label(fraud_prob: float, anomaly_score: float) -> str:
    if fraud_prob >= FRAUD_CRIT or anomaly_score >= ANOM_CRIT:
        return "CRITICAL"
    elif fraud_prob >= FRAUD_HIGH or (fraud_prob >= FRAUD_MED and anomaly_score >= ANOM_HIGH):
        return "HIGH"
    elif fraud_prob >= FRAUD_MED or anomaly_score >= ANOM_MED:
        return "MEDIUM"
    else:
        return "LOW"


def score_transaction_ml(model_pipeline, iforest_pipeline, model_payload: Dict, convert_to_inr: bool = False, currency: str = "INR") -> Tuple[float, float, str]:
    """
    Score transaction with supervised and isolation forest.
    - If your models were trained on INR, set convert_to_inr=True to multiply amount by INR_PER_UNIT[currency].
    - Returns: (fraud_prob [0..1], anomaly_score [>=0], ml_label)
    """
    df = pd.DataFrame([{
        "Amount": model_payload.get("Amount", 0.0) * (INR_PER_UNIT[currency] if convert_to_inr else 1.0),
        "TransactionType": model_payload.get("TransactionType", "PAYMENT"),
        "Location": model_payload.get("Location", "Unknown"),
        "DeviceID": model_payload.get("DeviceID", "Unknown"),
        "Channel": model_payload.get("Channel", "Other"),
        "hour": model_payload.get("hour", 0),
        "day_of_week": model_payload.get("day_of_week", 0),
        "month": model_payload.get("month", 0),
    }])
    try:
        fraud_prob = float(model_pipeline.predict_proba(df)[0, 1])
    except Exception as e:
        st.error("Supervised model scoring error")
        st.exception(e)
        fraud_prob = 0.0
    try:
        raw = float(iforest_pipeline.decision_function(df)[0])
        anomaly_score = -raw
    except Exception as e:
        st.error("IsolationForest scoring error")
        st.exception(e)
        anomaly_score = 0.0
    ml_label = ml_risk_label(fraud_prob, anomaly_score)
    return fraud_prob, anomaly_score, ml_label


# ----------------------
# Rule engine
# ----------------------
def evaluate_rules(payload: Dict, currency: str) -> Tuple[List[Dict], str]:
    """
    Evaluate deterministic rules on payload.
    Device-related rules are disabled for Bank and ATM (Option A).
    NetBanking retains device checks (because it involves device login).
    """
    # Convert thresholds
    ABS_CRIT = inr_to_currency(BASE_THRESHOLDS_INR["absolute_crit_amount"], currency)
    HIGH_AMT = inr_to_currency(BASE_THRESHOLDS_INR["high_amount_threshold"], currency)
    MED_AMT = inr_to_currency(BASE_THRESHOLDS_INR["medium_amount_threshold"], currency)
    ATM_HIGH = inr_to_currency(BASE_THRESHOLDS_INR["atm_high_withdrawal"], currency)
    CARD_TEST_SMALL = inr_to_currency(BASE_THRESHOLDS_INR["card_test_small_amount_inr"], currency)

    rules = []

    # Get fields (safe defaults)
    amt = float(payload.get("Amount", 0.0))
    channel = str(payload.get("Channel", "")).lower()
    hour = int(payload.get("hour", 0))
    monthly_avg = float(payload.get("monthly_avg", 0.0) or 0.0)
    rolling_avg_7d = float(payload.get("rolling_avg_7d", 0.0) or 0.0)
    txns_1h = int(payload.get("txns_last_1h", 0) or 0)
    txns_24h = int(payload.get("txns_last_24h", 0) or 0)
    failed_logins = int(payload.get("failed_login_attempts", 0) or 0)
    new_benef = bool(payload.get("new_beneficiary", False))
    ip_country = str(payload.get("ip_country", "")).lower()
    declared_country = str(payload.get("declared_country", "")).lower()
    last_device = str(payload.get("device_last_seen", "")).lower()
    curr_device = str(payload.get("DeviceID", "")).lower()
    last_lat = payload.get("last_known_lat")
    last_lon = payload.get("last_known_lon")
    txn_lat = payload.get("txn_lat")
    txn_lon = payload.get("txn_lon")
    atm_distance_km = float(payload.get("atm_distance_km", 0.0) or 0.0)
    card_country = str(payload.get("card_country", "")).lower()
    cvv_provided = payload.get("cvv_provided", True)
    shipping_addr = payload.get("shipping_address", "")
    billing_addr = payload.get("billing_address", "")
    beneficiaries_added_24h = int(payload.get("beneficiaries_added_24h", 0) or 0)
    suspicious_ip_flag = payload.get("suspicious_ip_flag", False)
    card_small_attempts = int(payload.get("card_small_attempts_in_5min", 0) or 0)
    pos_repeat_count = int(payload.get("pos_repeat_count", 0) or 0)
    beneficiary_added_minutes = int(payload.get("beneficiary_added_minutes", 9999) or 9999)

    # Helper to add rule
    def add_rule(name: str, sev: str, detail: str):
        rules.append({"name": name, "severity": sev, "detail": detail})

    # CRITICAL rules
    if amt >= ABS_CRIT:
        add_rule("Absolute very large amount", "CRITICAL",
                 f"Amount {amt:.2f} {currency} >= critical {ABS_CRIT:.2f} {currency}.")

    # Impossible travel calculation
    impossible_travel_distance = None
    if last_lat is not None and last_lon is not None and txn_lat is not None and txn_lon is not None:
        impossible_travel_distance = haversine_km(last_lat, last_lon, txn_lat, txn_lon)

    # Device-related checks: only active when channel is NOT Bank or ATM
    device_checks_enabled = channel not in ("bank", "atm")

    # CRIT: new device + impossible travel + high amount (only when device checks enabled)
    if device_checks_enabled:
        device_new = (not last_device) or last_device == ""
        location_changed = impossible_travel_distance is not None and impossible_travel_distance > 500
        if device_new and location_changed and amt > MED_AMT:
            add_rule("New device + Impossible travel + High amount", "CRITICAL",
                     f"New device + travel {impossible_travel_distance:.1f} km; amount {amt:.2f} {currency}.")

    # CRIT: multiple beneficiaries added & high transfer (applies to bank/netbanking)
    if beneficiaries_added_24h >= 3 and amt > HIGH_AMT:
        add_rule("Multiple beneficiaries added recently + high transfer", "CRITICAL",
                 f"{beneficiaries_added_24h} added and amount {amt:.2f} {currency}.")

    # HIGH rules
    if txns_1h >= 10:
        add_rule("High velocity (1h)", "HIGH", f"{txns_1h} transactions in last 1 hour.")
    if txns_24h >= 50:
        add_rule("Very high velocity (24h)", "HIGH", f"{txns_24h} transactions in last 24h.")

    if ip_country and declared_country and ip_country != declared_country:
        sev = "HIGH" if amt > HIGH_AMT else "MEDIUM"
        add_rule("IP / Declared country mismatch", sev,
                 f"IP '{ip_country}' != declared '{declared_country}' (amount {amt:.2f}).")

    if failed_logins >= 5:
        add_rule("Multiple failed logins", "HIGH", f"{failed_logins} recent failed logins.")

    if new_benef and amt >= MED_AMT:
        add_rule("New beneficiary + significant amount", "HIGH", "Transfer to newly added beneficiary with significant amount.")

    if suspicious_ip_flag and amt > (MED_AMT / 4):
        add_rule("IP flagged by intel", "HIGH", "IP flagged by threat intel and amount is non-trivial.")

    if channel == "atm" and atm_distance_km and atm_distance_km > 300:
        add_rule("ATM distance from last location", "HIGH", f"ATM is {atm_distance_km:.1f} km from last known location.")

    if card_country and declared_country and card_country != declared_country and amt > MED_AMT:
        add_rule("Card country mismatch", "HIGH", f"Card country {card_country} != declared country {declared_country}.")

    # MEDIUM rules
    if monthly_avg > 0 and amt >= 5 * monthly_avg and amt > MED_AMT:
        add_rule("Large spike vs monthly avg", "HIGH", f"Amount {amt:.2f} >= 5x monthly avg {monthly_avg:.2f}.")
    elif rolling_avg_7d > 0 and amt >= 3 * rolling_avg_7d and amt > (MED_AMT / 2):
        add_rule("Spike vs 7-day rolling avg", "MEDIUM", f"Amount {amt:.2f} >= 3x 7-day avg {rolling_avg_7d:.2f}.")
    elif monthly_avg > 0 and amt >= 2 * monthly_avg and amt > (MED_AMT / 2):
        add_rule("Above monthly usual", "MEDIUM", f"Amount {amt:.2f} >= 2x monthly avg {monthly_avg:.2f}.")

    if txns_1h >= 5:
        add_rule("Elevated velocity (1h)", "MEDIUM", f"{txns_1h} in last 1 hour.")
    if 10 <= txns_24h < 50:
        add_rule("Elevated velocity (24h)", "MEDIUM", f"{txns_24h} in last 24h.")

    if 0 <= hour <= 5 and monthly_avg < (MED_AMT * 2) and amt > (MED_AMT / 10):
        add_rule("Late-night transaction for low-activity customer", "MEDIUM",
                 f"Transaction at hour {hour} for low-activity customer; amount {amt:.2f}.")

    # Device mismatch (only if device checks enabled and device fields present)
    if device_checks_enabled and last_device and curr_device and last_device != curr_device:
        add_rule("Device mismatch", "MEDIUM", f"Device changed from '{last_device}' to '{curr_device}'.")

    # Billing vs shipping mismatch (only for online channel)
    if channel in ("online purchase", "online"):
        if shipping_addr and billing_addr and shipping_addr.strip().lower() != billing_addr.strip().lower():
            add_rule("Billing vs shipping address mismatch", "MEDIUM", "Billing address differs from shipping address.")

    # Missing CVV for card transactions (only when card is used)
    if channel in ("credit card", "online purchase", "online") and not cvv_provided:
        add_rule("Missing CVV for card txn", "MEDIUM", "CVV not provided for card e-commerce transaction.")

    # LOW rules
    if device_checks_enabled and (not last_device or last_device == "") and amt < (MED_AMT / 10):
        add_rule("New device (low amount)", "LOW", "New device but low amount.")

    if 0 < beneficiaries_added_24h < 3:
        add_rule("Beneficiaries recently added", "LOW", f"{beneficiaries_added_24h} beneficiaries added.")

    if ip_country and ip_country in {"nigeria", "romania", "ukraine", "russia"}:
        add_rule("IP from higher-risk country", "MEDIUM", f"IP country flagged as higher-risk: {ip_country}.")

    # Channel micro rules
    if card_small_attempts >= 6 and CARD_TEST_SMALL > 0:
        add_rule("Card testing / micro-charges detected", "HIGH",
                 f"{card_small_attempts} small attempts; micro amount ~{CARD_TEST_SMALL:.2f} {currency}.")

    if channel == "atm" and amt >= ATM_HIGH:
        add_rule("Large ATM withdrawal", "HIGH", f"ATM withdrawal {amt:.2f} >= {ATM_HIGH:.2f} {currency}.")

    if pos_repeat_count >= 10:
        add_rule("POS repeat transactions", "HIGH", f"{pos_repeat_count} rapid transactions at same POS.")

    if channel in ("netbanking", "bank") and beneficiary_added_minutes < 10 and amt >= MED_AMT:
        add_rule("Immediate transfer to newly added beneficiary", "HIGH",
                 f"Beneficiary added {beneficiary_added_minutes} minutes ago and transfer amount {amt:.2f} {currency}.")

    # compute overall highest severity
    highest = "LOW"
    for r in rules:
        highest = escalate(highest, r["severity"])

    return rules, highest


# ----------------------
# Combine ML + Rules
# ----------------------
def combine_final_risk(ml_risk: str, rule_highest: str) -> str:
    return escalate(ml_risk, rule_highest)


# ----------------------
# UI: channel-specific transaction types
# ----------------------
CHANNEL_TXN_TYPES = {
    "atm": ["CASH_WITHDRAWAL", "TRANSFER"],
    "credit card": ["PAYMENT", "REFUND"],
    "mobile app": ["PAYMENT", "TRANSFER", "BILL_PAY"],
    "pos": ["PAYMENT"],
    "online purchase": ["PAYMENT"],
    "bank": ["DEPOSIT", "TRANSFER", "WITHDRAWAL"],
    "netbanking": ["TRANSFER", "BILL_PAY", "PAYMENT"],
}

# ----------------------
# Streamlit UI
# ----------------------
st.set_page_config(page_title="Fraud Detection (Optimized)", page_icon="💳", layout="centered")
st.title("💳 Optimized Real-Time Fraud Detection")

st.markdown("**Global common fields** (present for all channels): currency, amount, date, time.")

# --- Common fields shown for all channels ---
col0, col0b = st.columns([2, 1])
with col0:
    currency = st.selectbox("Currency (affects thresholds)", CURRENCY_OPTIONS, index=CURRENCY_OPTIONS.index(DEFAULT_CURRENCY))
with col0b:
    st.caption(f"Rates (INR per unit): {currency} = {INR_PER_UNIT[currency]:,.2f} INR")

col1, col2 = st.columns(2)
with col1:
    amount = st.number_input(f"Transaction amount ({currency})", min_value=0.0, value=1200.0, step=10.0)
with col2:
    txn_date = st.date_input("Transaction date", value=datetime.date.today())
    txn_time = st.time_input("Transaction time", value=datetime.datetime.now().time())

txn_dt = datetime.datetime.combine(txn_date, txn_time)
hour = txn_dt.hour
day_of_week = txn_dt.weekday()
month = txn_dt.month

st.markdown("---")

# Channel selection (after common fields)
channel = st.selectbox("Transaction Channel", ["Choose...", "Bank", "Mobile App", "ATM", "Credit Card", "POS", "Online Purchase", "NetBanking"])

# Prepare variables for exclusive fields
# NOTE: We intentionally DO NOT show device fields for Bank and ATM
# and DO show device fields for NetBanking (per your instruction).
channel_lower = channel.lower() if isinstance(channel, str) else ""

# Channel-specific UI (exclusive)
if channel != "Choose...":
    st.markdown(f"### Channel: {channel} — channel-specific fields")

    # Enforce channel-specific transaction types
    txn_type_options = CHANNEL_TXN_TYPES.get(channel_lower, ["OTHER"])
    txn_type = st.selectbox("Transaction Type", txn_type_options)

    # Reset all channel-specific variables
    # Bank: identity details (in-person). No device.
    if channel_lower == "bank":
        st.subheader("In-branch (Bank) fields — identity focused")
        id_type = st.selectbox("ID Document Type", ["Passport", "Driver License", "Government ID", "Other"])
        id_number = st.text_input("ID Document Number")
        branch = st.text_input("Branch Name / Code", value="")
        teller_id = st.text_input("Teller ID (optional)", value="")
        # No device fields here

    # ATM: ATM-specific fields, no device
    elif channel_lower == "atm":
        st.subheader("ATM fields (card + ATM info)")
        atm_id = st.text_input("ATM ID / Terminal", value="")
        atm_location = st.text_input("ATM Location", value="")
        atm_distance_km = st.number_input("ATM distance from last known location (km)", min_value=0.0, value=0.0, step=1.0)
        card_masked = st.text_input("Card masked (e.g., 4111****1111)", value="")

    # Mobile App: needs device fields
    elif channel_lower == "mobile app":
        st.subheader("Mobile App fields (device + app telemetry)")
        device = st.text_input("Device / OS (e.g., Android)", value="Android")
        device_fingerprint = st.text_input("Device fingerprint (optional)", value="")
        app_version = st.text_input("App version", value="1.0.0")
        last_device = st.text_input("Last known device (optional)", value="")

    # Credit Card: card-specific (may include device if e-commerce)
    elif channel_lower == "credit card":
        st.subheader("Credit Card fields")
        card_masked = st.text_input("Card masked (4111****1111)", value="")
        card_country = st.text_input("Card issuing country", value="")
        cvv_provided = st.checkbox("CVV provided (checked if present)", value=True)
        used_ecom = st.checkbox("Used for e-commerce?", value=False)
        # If used_ecom, capture device/browser for additional checks
        if used_ecom:
            device = st.text_input("Device / Browser (if known)", value="")
            last_device = st.text_input("Last known device (optional)", value="")

    # POS: merchant terminal related
    elif channel_lower == "pos":
        st.subheader("POS fields")
        pos_merchant_id = st.text_input("POS Merchant ID", value="")
        store_name = st.text_input("Store name", value="")
        pos_repeat_count = st.number_input("Rapid repeat transactions at same POS", min_value=0, value=0, step=1)

    # Online Purchase: shipping / billing + IP + optional device
    elif channel_lower == "online purchase":
        st.subheader("Online Purchase fields")
        merchant = st.text_input("Merchant name / ID", value="")
        shipping_address = st.text_input("Shipping address", value="")
        billing_address = st.text_input("Billing address", value=shipping_address)
        client_ip = st.text_input("Client IP (optional)", value="")
        ip_country = st.text_input("IP-derived country (optional)", value="")
        used_card_online = st.checkbox("Paid by card online?", value=False)
        if used_card_online:
            cvv_provided = st.checkbox("CVV provided (checked if present)", value=True)
            card_masked = st.text_input("Card masked", value="")
        # allow device/browser info optionally
        device = st.text_input("Device / Browser (optional)", value="")
        last_device = st.text_input("Last known device (optional)", value="")

    # NetBanking: includes device + beneficiary info
    elif channel_lower == "netbanking":
        st.subheader("NetBanking fields")
        username = st.text_input("User ID / Login", value="")
        device = st.text_input("Device / Browser (used to login)", value="")
        last_device = st.text_input("Last known device (optional)", value="")
        beneficiary = st.text_input("Beneficiary (if transfer)", value="")
        new_beneficiary = st.checkbox("Is beneficiary newly added?", value=False)
        beneficiary_added_minutes = st.number_input("Minutes since beneficiary was added (if known)", min_value=0, value=9999, step=1)

    # --- Optional telemetry shared only to evaluate rules but NOT shown across channels ---
    st.markdown("#### Optional telemetry (for rules; keep blank if unknown)")
    colT1, colT2 = st.columns(2)
    with colT1:
        monthly_avg = st.number_input(f"Customer monthly average spend ({currency})", min_value=0.0, value=10000.0, step=100.0)
        rolling_avg_7d = st.number_input(f"7-day rolling avg ({currency})", min_value=0.0, value=3000.0, step=50.0)
        txns_last_1h = st.number_input("Transactions in last 1 hour", min_value=0, value=0, step=1)
        txns_last_24h = st.number_input("Transactions in last 24 hours", min_value=0, value=0, step=1)
    with colT2:
        txns_last_7d = st.number_input("Transactions in last 7 days", min_value=0, value=7, step=1)
        beneficiaries_added_24h = st.number_input("Beneficiaries added in last 24h", min_value=0, value=0, step=1)
        failed_login_attempts = st.number_input("Failed login attempts (recent)", min_value=0, value=0, step=1)

    # IP / geo telemetry (only used by rule engine)
    st.markdown("#### Optional IP / Geo (used by rules)")
    colG1, colG2 = st.columns(2)
    with colG1:
        client_ip = st.text_input("Client IP (optional)", value="")
        ip_country = st.text_input("IP-derived country (optional)", value="")
        suspicious_ip_flag = st.checkbox("IP flagged by threat intel?", value=False)
    with colG2:
        last_known_lat = st.number_input("Last known latitude (optional)", format="%.6f", value=0.0)
        last_known_lon = st.number_input("Last known longitude (optional)", format="%.6f", value=0.0)
        txn_lat = st.number_input("Transaction latitude (optional)", format="%.6f", value=0.0)
        txn_lon = st.number_input("Transaction longitude (optional)", format="%.6f", value=0.0)

    # Submit
    submit = st.button("🚀 Run Fraud Check")

    if submit:
        # Build payload for rules & ML. Only include fields that exist for this channel.
        payload = {
            "Amount": amount,
            "Currency": currency,
            "TransactionType": txn_type,
            "Channel": channel,
            "hour": hour,
            "day_of_week": day_of_week,
            "month": month,
            # telemetry
            "monthly_avg": monthly_avg,
            "rolling_avg_7d": rolling_avg_7d,
            "txns_last_1h": int(txns_last_1h),
            "txns_last_24h": int(txns_last_24h),
            "txns_last_7d": int(txns_last_7d),
            "beneficiaries_added_24h": int(beneficiaries_added_24h),
            "failed_login_attempts": int(failed_login_attempts),
            # ip/geo
            "client_ip": client_ip,
            "ip_country": ip_country,
            "declared_country": "",  # not captured in UI per-channel, leave blank or fill if available
            "suspicious_ip_flag": suspicious_ip_flag,
            "last_known_lat": last_known_lat if last_known_lat != 0.0 else None,
            "last_known_lon": last_known_lon if last_known_lon != 0.0 else None,
            "txn_lat": txn_lat if txn_lat != 0.0 else None,
            "txn_lon": txn_lon if txn_lon != 0.0 else None,
        }

        # Attach channel-exclusive fields to the payload as present
        if channel_lower == "bank":
            payload.update({
                "id_type": id_type,
                "id_number": id_number,
                "branch": branch,
                "teller_id": teller_id,
            })
        elif channel_lower == "atm":
            payload.update({
                "atm_id": atm_id,
                "atm_location": atm_location,
                "atm_distance_km": atm_distance_km,
                "card_masked": card_masked,
            })
        elif channel_lower == "mobile app":
            payload.update({
                "DeviceID": device,
                "device_fingerprint": device_fingerprint,
                "app_version": app_version,
                "device_last_seen": last_device,
            })
        elif channel_lower == "credit card":
            payload.update({
                "card_masked": card_masked,
                "card_country": card_country,
                "cvv_provided": cvv_provided,
            })
            if used_ecom:
                payload.update({"DeviceID": device, "device_last_seen": last_device})
        elif channel_lower == "pos":
            payload.update({
                "pos_merchant_id": pos_merchant_id,
                "store_name": store_name,
                "pos_repeat_count": pos_repeat_count,
            })
        elif channel_lower == "online purchase":
            payload.update({
                "merchant": merchant,
                "shipping_address": shipping_address,
                "billing_address": billing_address,
                "DeviceID": device,
                "device_last_seen": last_device,
                "card_masked": card_masked if used_card_online else "",
                "cvv_provided": cvv_provided if used_card_online else True,
            })
        elif channel_lower == "netbanking":
            payload.update({
                "username": username,
                "DeviceID": device,
                "device_last_seen": last_device,
                "beneficiary": beneficiary,
                "new_beneficiary": new_beneficiary,
                "beneficiary_added_minutes": int(beneficiary_added_minutes),
            })

        # ML model scoring:
        # NOTE: convert_to_inr flag: if your models were trained on INR amounts set to True.
        # Otherwise set to False. Here we leave False (user must set conversion if needed).
        convert_to_inr_for_model = False
        with st.spinner("Scoring with ML models..."):
            fraud_prob_raw, anomaly_raw, ml_label = score_transaction_ml(supervised_pipeline, iforest_pipeline, payload, convert_to_inr=convert_to_inr_for_model, currency=currency)

        # Convert ML outputs to 0-100 scale for display (clamp)
        fraud_confidence_pct = max(0.0, min(100.0, fraud_prob_raw * 100.0))
        anomaly_pct = max(0.0, min(100.0, anomaly_raw * 100.0))

        # Evaluate deterministic rules (device rules disabled for Bank & ATM)
        rules_triggered, rules_highest = evaluate_rules(payload, currency)

        # Final risk label
        final_risk = combine_final_risk(ml_label, rules_highest)

        # Display results
        st.markdown("## 🔎 Results")
        color_map = {"LOW": "#2e7d32", "MEDIUM": "#f9a825", "HIGH": "#f57c00", "CRITICAL": "#c62828"}
        badge_color = color_map.get(final_risk, "#607d8b")
        st.markdown(
            f"""<div style="padding:0.75rem 1rem;border-radius:0.5rem;background-color:{badge_color}22;border:1px solid {badge_color};">
                <strong style="color:{badge_color};font-size:1.1rem;">Final Risk Level: {final_risk}</strong>
            </div>""",
            unsafe_allow_html=True,
        )

        # Show ML metrics as percentages
        colA, colB = st.columns(2)
        with colA:
            st.metric("Fraud Confidence (ML)", f"{fraud_confidence_pct:.2f}%", help="Supervised model fraud probability scaled to 0-100.")
            st.metric("ML Risk Label", ml_label)
        with colB:
            st.metric("Anomaly Score (ML)", f"{anomaly_pct:.2f}%", help="IsolationForest anomaly score scaled to 0-100 (higher = more anomalous).")
            st.metric("Rules-derived highest severity", rules_highest)

        st.markdown("### ⚠ Triggered Rules")
        if rules_triggered:
            for r in rules_triggered:
                sev = r["severity"]
                emoji = "🔴" if sev in ("HIGH", "CRITICAL") else "🟠" if sev == "MEDIUM" else "🟢"
                st.write(f"{emoji} **{r['name']}** — *{r['severity']}*")
                st.caption(r["detail"])
        else:
            st.success("No deterministic rules triggered.")

        st.markdown("### 📦 Payload (debug)")
        st.json(payload)

        st.markdown(
            """
            Notes:
            - Currency thresholds are converted from INR to selected currency. Tune BASE_THRESHOLDS_INR to your data.
            - Device-related rules are disabled for Bank and ATM (Option A). NetBanking retains device checks.
            - If your ML models were trained on INR amounts, set `convert_to_inr_for_model = True` above.
            - Fraud Confidence and Anomaly Score are scaled to 0-100 for easier interpretation.
            """
        )

else:
    st.info("First select currency, enter amount/date/time, then select the transaction channel.")
