# app.py
"""
Optimized Real-Time Fraud Detection (single-file)
- INR default currency; thresholds adaptive to selected currency (INR base)
- Channel-exclusive UI: only currency/amount/date/time are common
- Bank channel: no client IP / ip-derived country fields
- ATM channel: no device fields
- NetBanking, Mobile App, CreditCard(Mobile/Web), Online Purchase: device checks enabled
- Credit Card: two modes -> POS (physical) and Mobile/Web (device-aware)
- Time input selectable (not forced to now)
- All repeated widget labels use unique keys to avoid StreamlitDuplicateElementId
- ML outputs scaled to 0-100 for display
- Rule engine disables device rules for Bank & ATM
"""

import datetime
from math import radians, sin, cos, asin, sqrt
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import pandas as pd
import streamlit as st

# -------------------------
# CONFIGURATION
# -------------------------

# INR base conversion table (INR per 1 unit of currency)
INR_PER_UNIT = {
    "INR": 1.0,
    "USD": 83.2,
    "EUR": 90.5,
    "GBP": 105.3,
    "AED": 22.7,
    "AUD": 61.0,
    "SGD": 61.5,
}
CURRENCIES = list(INR_PER_UNIT.keys())
DEFAULT_CURRENCY = "INR"

# Base thresholds stored in INR (tune these to your data)
BASE_THRESHOLDS_INR = {
    "absolute_crit_amount": 10_000_000,  # extremely large tx (INR)
    "high_amount_threshold": 2_000_000,
    "medium_amount_threshold": 100_000,
    "atm_high_withdrawal": 300_000,
    "card_test_small_amount_inr": 200,
}

# ML thresholds (probabilities / anomaly scores) - currency agnostic
FRAUD_MED = 0.00005
FRAUD_HIGH = 0.00023328
FRAUD_CRIT = 0.01732857

ANOM_MED = 0.04
ANOM_HIGH = 0.05
ANOM_CRIT = 0.08

# Severity ordering
SEVERITY_ORDER = {"LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}

# Channel-specific allowed transaction types
CHANNEL_TXN_TYPES = {
    "atm": ["CASH_WITHDRAWAL", "TRANSFER"],
    "credit card": ["PAYMENT", "REFUND"],
    "mobile app": ["PAYMENT", "TRANSFER", "BILL_PAY"],
    "pos": ["PAYMENT"],
    "online purchase": ["PAYMENT"],
    "bank": ["DEPOSIT", "TRANSFER", "WITHDRAWAL"],
    "netbanking": ["TRANSFER", "BILL_PAY", "PAYMENT"],
}

# -------------------------
# HELPERS
# -------------------------
def inr_to_currency(amount_in_inr: float, currency: str) -> float:
    """Convert INR-denominated base threshold to selected currency units."""
    if currency not in INR_PER_UNIT or INR_PER_UNIT[currency] == 0:
        return amount_in_inr
    return amount_in_inr / INR_PER_UNIT[currency]


def clamp_pct(x: float) -> float:
    """Clamp numeric to 0..100 (percentage)."""
    try:
        v = float(x) * 100.0
    except Exception:
        v = 0.0
    if v < 0:
        return 0.0
    if v > 100:
        return 100.0
    return v


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    """Haversine distance in km."""
    if None in (lat1, lon1, lat2, lon2):
        return None
    lat1, lon1, lat2, lon2 = map(radians, (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    return 6371 * c


def escalate(a: str, b: str) -> str:
    """Return the higher severity label of two."""
    return a if SEVERITY_ORDER[a] >= SEVERITY_ORDER[b] else b


# -------------------------
# ML MODEL LOADING
# -------------------------
@st.cache_resource
def load_models():
    models_dir = Path("models")
    def _load(name: str):
        path = models_dir / name
        try:
            return joblib.load(path)
        except Exception as e:
            st.error(f"Error loading model artifact: {name}")
            st.exception(e)
            raise
    supervised = _load("supervised_lgbm_pipeline.joblib")
    iforest = _load("iforest_pipeline.joblib")
    return supervised, iforest


supervised_pipeline, iforest_pipeline = load_models()


# -------------------------
# ML SCORING WRAPPER
# -------------------------
def ml_risk_label(fraud_prob: float, anomaly_score: float) -> str:
    """Map ML numeric outputs to risk labels using preset thresholds."""
    if fraud_prob >= FRAUD_CRIT or anomaly_score >= ANOM_CRIT:
        return "CRITICAL"
    if fraud_prob >= FRAUD_HIGH or (fraud_prob >= FRAUD_MED and anomaly_score >= ANOM_HIGH):
        return "HIGH"
    if fraud_prob >= FRAUD_MED or anomaly_score >= ANOM_MED:
        return "MEDIUM"
    return "LOW"


def score_transaction_ml(model_pipeline, iforest_pipeline, model_payload: Dict, convert_to_inr: bool = False, currency: str = "INR") -> Tuple[float, float, str]:
    """
    Score using supervised and isolation forest pipelines.
    convert_to_inr: if True, convert user amount from selected currency -> INR before scoring.
    (Set True if the model was trained on INR amounts.)
    Returns (fraud_prob [0..1], anomaly_score, ml_label)
    """
    amt_for_model = model_payload.get("Amount", 0.0)
    if convert_to_inr:
        # Multiply user currency -> INR by INR_PER_UNIT[currency]
        amt_for_model = amt_for_model * INR_PER_UNIT.get(currency, 1.0)

    model_df = pd.DataFrame([{
        "Amount": amt_for_model,
        "TransactionType": model_payload.get("TransactionType", "PAYMENT"),
        "Location": model_payload.get("Location", "Unknown"),
        "DeviceID": model_payload.get("DeviceID", "Unknown"),
        "Channel": model_payload.get("Channel", "Other"),
        "hour": model_payload.get("hour", 0),
        "day_of_week": model_payload.get("day_of_week", 0),
        "month": model_payload.get("month", 0),
    }])
    try:
        fraud_prob = float(model_pipeline.predict_proba(model_df)[0, 1])
    except Exception as e:
        st.error("Supervised model scoring error - check pipeline input schema")
        st.exception(e)
        fraud_prob = 0.0
    try:
        raw = float(iforest_pipeline.decision_function(model_df)[0])
        anomaly_score = -raw
    except Exception as e:
        st.error("IsolationForest scoring error - check pipeline input schema")
        st.exception(e)
        anomaly_score = 0.0
    label = ml_risk_label(fraud_prob, anomaly_score)
    return fraud_prob, anomaly_score, label


# -------------------------
# RULE ENGINE
# -------------------------
def evaluate_rules(payload: Dict, currency: str) -> Tuple[List[Dict], str]:
    """
    Evaluate deterministic rules on payload.
    Device-related rules disabled for Bank & ATM (Option A).
    Returns (list of triggered rule dicts, highest_severity)
    """
    # Convert canonical INR thresholds into selected currency
    ABS_CRIT = inr_to_currency(BASE_THRESHOLDS_INR["absolute_crit_amount"], currency)
    HIGH_AMT = inr_to_currency(BASE_THRESHOLDS_INR["high_amount_threshold"], currency)
    MED_AMT = inr_to_currency(BASE_THRESHOLDS_INR["medium_amount_threshold"], currency)
    ATM_HIGH = inr_to_currency(BASE_THRESHOLDS_INR["atm_high_withdrawal"], currency)
    CARD_TEST_SMALL = inr_to_currency(BASE_THRESHOLDS_INR["card_test_small_amount_inr"], currency)

    rules: List[Dict] = []

    # Safe parsing
    amt = float(payload.get("Amount", 0.0) or 0.0)
    channel = str(payload.get("Channel", "")).lower()
    hour = int(payload.get("hour", 0) or 0)
    monthly_avg = float(payload.get("monthly_avg", 0.0) or 0.0)
    rolling_avg_7d = float(payload.get("rolling_avg_7d", 0.0) or 0.0)
    txns_1h = int(payload.get("txns_last_1h", 0) or 0)
    txns_24h = int(payload.get("txns_last_24h", 0) or 0)
    txns_7d = int(payload.get("txns_last_7d", 0) or 0)
    failed_logins = int(payload.get("failed_login_attempts", 0) or 0)
    new_benef = bool(payload.get("new_beneficiary", False))
    # IP fields may or may not be present in payload; bank channel shouldn't include these
    ip_country = str(payload.get("ip_country", "") or "").lower()
    declared_country = str(payload.get("declared_country", "") or "").lower()
    last_device = str(payload.get("device_last_seen", "") or "").lower()
    curr_device = str(payload.get("DeviceID", "") or "").lower()
    last_lat = payload.get("last_known_lat")
    last_lon = payload.get("last_known_lon")
    txn_lat = payload.get("txn_lat")
    txn_lon = payload.get("txn_lon")
    atm_distance_km = float(payload.get("atm_distance_km", 0.0) or 0.0)
    card_country = str(payload.get("card_country", "") or "").lower()
    cvv_provided = payload.get("cvv_provided", True)
    shipping_addr = payload.get("shipping_address", "")
    billing_addr = payload.get("billing_address", "")
    beneficiaries_added_24h = int(payload.get("beneficiaries_added_24h", 0) or 0)
    suspicious_ip_flag = payload.get("suspicious_ip_flag", False)
    card_small_attempts = int(payload.get("card_small_attempts_in_5min", 0) or 0)
    pos_repeat_count = int(payload.get("pos_repeat_count", 0) or 0)
    beneficiary_added_minutes = int(payload.get("beneficiary_added_minutes", 9999) or 9999)

    # small helper
    def add_rule(name: str, sev: str, detail: str):
        rules.append({"name": name, "severity": sev, "detail": detail})

    # CRITICAL rules
    if amt >= ABS_CRIT:
        add_rule("Absolute very large amount", "CRITICAL",
                 f"Amount {amt:.2f} {currency} >= critical {ABS_CRIT:.2f} {currency}.")

    # impossible travel
    impossible_travel_distance = None
    if last_lat is not None and last_lon is not None and txn_lat is not None and txn_lon is not None:
        impossible_travel_distance = haversine_km(last_lat, last_lon, txn_lat, txn_lon)

    # Device checks enabled for channels other than bank and atm
    device_checks_enabled = channel not in ("bank", "atm")

    # CRIT: new device + impossible travel + high amount (only if device checks enabled)
    if device_checks_enabled:
        device_new = (not last_device) or last_device == ""
        location_changed = impossible_travel_distance is not None and impossible_travel_distance > 500
        if device_new and location_changed and amt > MED_AMT:
            add_rule("New device + Impossible travel + High amount", "CRITICAL",
                     f"New device + travel {impossible_travel_distance:.1f} km; amount {amt:.2f} {currency}.")

    # CRIT: many beneficiaries added + big transfer
    if beneficiaries_added_24h >= 3 and amt > HIGH_AMT:
        add_rule("Multiple beneficiaries added + high transfer", "CRITICAL",
                 f"{beneficiaries_added_24h} beneficiaries added and amount {amt:.2f} {currency}.")

    # HIGH rules
    if txns_1h >= 10:
        add_rule("High velocity (1h)", "HIGH", f"{txns_1h} txns in last 1 hour.")
    if txns_24h >= 50:
        add_rule("Very high velocity (24h)", "HIGH", f"{txns_24h} txns in last 24h.")

    # IP/declared mismatch (only if ip info present; do NOT consider for Bank because Bank UI does not collect ip)
    if ip_country and declared_country and ip_country != declared_country:
        sev = "HIGH" if amt > HIGH_AMT else "MEDIUM"
        add_rule("IP / Declared country mismatch", sev,
                 f"IP country '{ip_country}' differs from declared '{declared_country}'.")

    if failed_logins >= 5:
        add_rule("Multiple failed login attempts", "HIGH", f"{failed_logins} failed auth attempts.")

    if new_benef and amt >= MED_AMT:
        add_rule("New beneficiary + significant amount", "HIGH",
                 "Transfer to newly added beneficiary with amount above threshold.")

    if suspicious_ip_flag and amt > (MED_AMT / 4):
        add_rule("IP flagged by threat intelligence", "HIGH", "IP flagged and non-trivial amount.")

    if channel == "atm" and atm_distance_km and atm_distance_km > 300:
        add_rule("ATM distance from last location", "HIGH", f"ATM is {atm_distance_km:.1f} km away.")

    if card_country and declared_country and card_country != declared_country and amt > MED_AMT:
        add_rule("Card country mismatch", "HIGH", f"Card country {card_country} != declared country {declared_country}.")

    # MEDIUM rules (spending spikes, velocity, time anomalies)
    if monthly_avg > 0 and amt >= 5 * monthly_avg and amt > MED_AMT:
        add_rule("Large spike vs monthly avg", "HIGH",
                 f"Amount {amt:.2f} >= 5x monthly avg {monthly_avg:.2f}.")
    elif rolling_avg_7d > 0 and amt >= 3 * rolling_avg_7d and amt > (MED_AMT / 2):
        add_rule("Spike vs 7-day avg", "MEDIUM", f"Amount {amt:.2f} >= 3x 7-day avg {rolling_avg_7d:.2f}.")
    elif monthly_avg > 0 and amt >= 2 * monthly_avg and amt > (MED_AMT / 2):
        add_rule("Above monthly usual", "MEDIUM", f"Amount {amt:.2f} >= 2x monthly avg {monthly_avg:.2f}.")

    if txns_1h >= 5:
        add_rule("Elevated velocity (1h)", "MEDIUM", f"{txns_1h} in last 1 hour.")
    if 10 <= txns_24h < 50:
        add_rule("Elevated velocity (24h)", "MEDIUM", f"{txns_24h} in last 24h.")

    if 0 <= hour <= 5 and monthly_avg < (MED_AMT * 2) and amt > (MED_AMT / 10):
        add_rule("Late-night txn for low-activity customer", "MEDIUM",
                 f"Txn at hour {hour} for low-activity customer; amt {amt:.2f}.")

    # Device mismatch check (only if device checks enabled and device fields present)
    if device_checks_enabled and last_device and curr_device and last_device != curr_device:
        add_rule("Device mismatch", "MEDIUM", f"Device changed from '{last_device}' to '{curr_device}'.")

    # Billing vs shipping mismatch (online purchases)
    if channel in ("online purchase", "online"):
        if shipping_addr and billing_addr and shipping_addr.strip().lower() != billing_addr.strip().lower():
            add_rule("Billing vs shipping mismatch", "MEDIUM", "Billing address differs from shipping address.")

    # Missing CVV for card transactions that are card-present/online
    if channel in ("credit card", "online purchase", "online"):
        if not cvv_provided:
            add_rule("Missing CVV", "MEDIUM", "CVV not provided for card txn.")

    # Low severity items
    if device_checks_enabled and ((not last_device) or last_device == "") and amt < (MED_AMT / 10):
        add_rule("New device (low amount)", "LOW", "Transaction from new device but low amount.")

    if 0 < beneficiaries_added_24h < 3:
        add_rule("Beneficiaries recently added", "LOW", f"{beneficiaries_added_24h} beneficiaries added.")

    if ip_country and ip_country in {"nigeria", "romania", "ukraine", "russia"}:
        add_rule("IP from higher-risk country", "MEDIUM", f"IP country flagged as higher-risk: {ip_country}.")

    # Channel micro rules
    if card_small_attempts >= 6 and CARD_TEST_SMALL > 0:
        add_rule("Card testing / micro-charges detected", "HIGH",
                 f"{card_small_attempts} small attempts; micro amount {CARD_TEST_SMALL:.2f} {currency}.")

    if channel == "atm" and amt >= ATM_HIGH:
        add_rule("Large ATM withdrawal", "HIGH", f"ATM withdrawal {amt:.2f} {currency} >= {ATM_HIGH:.2f}")

    if pos_repeat_count >= 10:
        add_rule("POS repeat transactions", "HIGH", f"{pos_repeat_count} rapid transactions at same POS.")

    if channel in ("netbanking", "bank") and beneficiary_added_minutes < 10 and amt >= MED_AMT:
        add_rule("Immediate transfer to newly added beneficiary", "HIGH",
                 f"Beneficiary added {beneficiary_added_minutes} minutes ago and transfer amount {amt:.2f} {currency}.")

    # compute highest severity
    highest = "LOW"
    for r in rules:
        highest = escalate(highest, r["severity"])

    return rules, highest


# -------------------------
# Combine final risk
# -------------------------
def combine_final_risk(ml_risk: str, rule_highest: str) -> str:
    return escalate(ml_risk, rule_highest)


# -------------------------
# STREAMLIT UI
# -------------------------
st.set_page_config(page_title="Real-Time Fraud Detection (Optimized)", page_icon="💳", layout="centered")
st.title("💳 Real-Time Fraud Detection — Optimized")

# --- Global common fields (present for ALL channels) ---
st.markdown("**Common fields (shown for all channels):** Currency, Amount, Date, Time")

col0, col0b = st.columns([2, 1])
with col0:
    currency = st.selectbox("Select currency", CURRENCIES, index=CURRENCIES.index(DEFAULT_CURRENCY), key="currency_select")
with col0b:
    st.caption(f"(INR per unit) {currency} = {INR_PER_UNIT[currency]:,.2f} INR")

col1, col2 = st.columns([1, 1])
with col1:
    amount = st.number_input(f"Transaction amount ({currency})", min_value=0.0, value=1200.0, step=10.0, key="amount_common")
with col2:
    # Use a selectable default time so user can pick any time
    txn_date = st.date_input("Transaction date", value=datetime.date.today(), key="txn_date")
    txn_time = st.time_input("Transaction time", value=datetime.time(12, 0), key="txn_time")

txn_dt = datetime.datetime.combine(txn_date, txn_time)
hour = txn_dt.hour
day_of_week = txn_dt.weekday()
month = txn_dt.month

st.markdown("---")

# Channel selector
channel = st.selectbox("Transaction Channel", ["Choose...", "Bank", "Mobile App", "ATM", "Credit Card", "POS", "Online Purchase", "NetBanking"], key="channel_select")
if channel and channel != "Choose...":
    channel_lower = channel.lower()
    st.markdown(f"### Channel: {channel} — channel-specific fields (exclusive)")

    # Transaction type options per channel (enforced)
    txn_options = CHANNEL_TXN_TYPES.get(channel_lower, ["OTHER"])
    txn_type = st.selectbox("Transaction type", txn_options, key=f"txn_type_{channel_lower}")

    # Setup placeholders / default values for exclusive fields
    # We'll add each channel's fields with unique keys to avoid duplicate IDs
    if channel_lower == "bank":
        # Bank is in-person: identity only (no IP, no device)
        st.subheader("In-branch (Bank) fields — Identity-focused (no IP/device)")
        id_type = st.selectbox("ID Document Type", ["Passport", "Driver License", "Government ID", "Other"], key="bank_id_type")
        id_number = st.text_input("ID Document Number", key="bank_id_number")
        branch = st.text_input("Branch Name / Code", value="", key="bank_branch")
        teller_id = st.text_input("Teller ID (optional)", value="", key="bank_teller")

    elif channel_lower == "atm":
        st.subheader("ATM fields (card + ATM info) — no device")
        atm_id = st.text_input("ATM ID / Terminal", key="atm_id")
        atm_location = st.text_input("ATM Location", key="atm_location")
        atm_distance_km = st.number_input("ATM distance from last known location (km)", min_value=0.0, value=0.0, step=1.0, key="atm_distance")
        card_masked = st.text_input("Card masked (e.g., 4111****1111)", key="atm_card_masked")

    elif channel_lower == "mobile app":
        st.subheader("Mobile App fields (device + app telemetry)")
        device = st.text_input("Device / OS (e.g., Android)", value="Android", key="mobile_device")
        device_fingerprint = st.text_input("Device fingerprint (optional)", key="mobile_device_fp")
        app_version = st.text_input("App version", value="1.0.0", key="mobile_app_ver")
        last_device = st.text_input("Last known device (optional)", key="mobile_last_device")

    elif channel_lower == "credit card":
        st.subheader("Credit Card: choose mode")
        cc_mode = st.radio("Credit Card mode", ["POS (physical)", "Mobile/Web (app or web)"], key="cc_mode")
        if cc_mode == "POS (physical)":
            st.markdown("**POS (physical) card present flow** — fewer device fields")
            card_masked = st.text_input("Card masked (4111****1111)", key="cc_pos_card")
            card_country = st.text_input("Card issuing country", key="cc_pos_country")
            cvv_provided = st.checkbox("CVV provided (checked if present)", value=True, key="cc_pos_cvv")
            pos_merchant_id = st.text_input("POS Merchant ID (optional)", key="cc_pos_mid")
        else:
            st.markdown("**Mobile/Web (app or web) card flow — device-aware**")
            card_masked = st.text_input("Card masked (4111****1111)", key="cc_web_card")
            card_country = st.text_input("Card issuing country", key="cc_web_country")
            cvv_provided = st.checkbox("CVV provided (checked if present)", value=True, key="cc_web_cvv")
            # Device fields here (only for mobile/web)
            device = st.text_input("Device / Browser (optional)", key="cc_web_device")
            device_fingerprint = st.text_input("Device fingerprint (optional)", key="cc_web_fp")
            last_device = st.text_input("Last known device (optional)", key="cc_web_last_device")
            client_ip = st.text_input("Client IP (optional)", key="cc_web_client_ip")
            ip_country = st.text_input("IP-derived country (optional)", key="cc_web_ip_country")

    elif channel_lower == "pos":
        st.subheader("POS fields")
        pos_merchant_id = st.text_input("POS Merchant ID", key="pos_mid")
        store_name = st.text_input("Store name", key="pos_store")
        pos_repeat_count = st.number_input("Rapid repeat transactions at same POS", min_value=0, value=0, step=1, key="pos_repeat")

    elif channel_lower == "online purchase":
        st.subheader("Online Purchase fields (device + IP + addresses)")
        merchant = st.text_input("Merchant name / ID", key="online_merchant")
        shipping_address = st.text_input("Shipping address", key="online_ship")
        billing_address = st.text_input("Billing address", value=shipping_address, key="online_bill")
        # Unique keys for IP fields here to avoid duplicates with other channels
        client_ip = st.text_input("Client IP (optional)", key="online_client_ip")
        ip_country = st.text_input("IP-derived country (optional)", key="online_ip_country")
        used_card_online = st.checkbox("Paid by card online?", value=False, key="online_card_used")
        if used_card_online:
            cvv_provided = st.checkbox("CVV provided (checked if present)", value=True, key="online_cvv")
            card_masked = st.text_input("Card masked", key="online_card_masked")
        # Device fields optional
        device = st.text_input("Device / Browser (optional)", key="online_device")
        last_device = st.text_input("Last known device (optional)", key="online_last_device")

    elif channel_lower == "netbanking":
        st.subheader("NetBanking fields (device-aware)")
        username = st.text_input("User ID / Login", key="nb_user")
        device = st.text_input("Device / Browser (used to login)", key="nb_device")
        last_device = st.text_input("Last known device (optional)", key="nb_last_device")
        beneficiary = st.text_input("Beneficiary (if transfer)", key="nb_beneficiary")
        new_beneficiary = st.checkbox("Is beneficiary newly added?", value=False, key="nb_new_benef")
        beneficiary_added_minutes = st.number_input("Minutes since beneficiary was added (if known)", min_value=0, value=9999, step=1, key="nb_benef_minutes")

    # Optional telemetry panel (common but shown here after exclusive channel fields)
    st.markdown("#### Optional telemetry (helps rules; provide if available)")
    colT1, colT2 = st.columns(2)
    with colT1:
        monthly_avg = st.number_input(f"Customer monthly average spend ({currency})", min_value=0.0, value=10000.0, step=100.0, key=f"monthly_avg_{channel_lower}")
        rolling_avg_7d = st.number_input(f"7-day rolling average ({currency})", min_value=0.0, value=3000.0, step=50.0, key=f"rolling_avg_{channel_lower}")
        txns_last_1h = st.number_input("Transactions in last 1 hour", min_value=0, value=0, step=1, key=f"txns1h_{channel_lower}")
        txns_last_24h = st.number_input("Transactions in last 24 hours", min_value=0, value=0, step=1, key=f"txns24h_{channel_lower}")
    with colT2:
        txns_last_7d = st.number_input("Transactions in last 7 days", min_value=0, value=7, step=1, key=f"txns7d_{channel_lower}")
        beneficiaries_added_24h = st.number_input("Beneficiaries added in last 24h", min_value=0, value=0, step=1, key=f"ben24_{channel_lower}")
        beneficiaries_added_24h = int(beneficiaries_added_24h)
        failed_login_attempts = st.number_input("Failed login attempts (recent)", min_value=0, value=0, step=1, key=f"failed_{channel_lower}")

    # IP & Geo telemetry (only include fields for channels that allow IP input; Bank excludes)
    st.markdown("#### Optional IP / Geo (used by rules)")
    if channel_lower == "bank":
        st.info("Bank channel does not collect IP / client IP fields by design (in-branch).")
        # Provide fields as None / defaults in payload
        client_ip = ""
        ip_country = ""
        suspicious_ip_flag = False
        last_known_lat = None
        last_known_lon = None
        txn_lat = None
        txn_lon = None
    else:
        # Unique keys per channel to avoid duplicate ids
        client_ip = st.text_input("Client IP (optional)", key=f"client_ip_{channel_lower}")
        ip_country = st.text_input("IP-derived country (optional)", key=f"ip_country_{channel_lower}")
        suspicious_ip_flag = st.checkbox("IP flagged by threat intel?", value=False, key=f"suspicious_{channel_lower}")
        last_known_lat = st.number_input("Last known latitude (optional)", format="%.6f", value=0.0, key=f"lastlat_{channel_lower}")
        last_known_lon = st.number_input("Last known longitude (optional)", format="%.6f", value=0.0, key=f"lastlon_{channel_lower}")
        txn_lat = st.number_input("Transaction latitude (optional)", format="%.6f", value=0.0, key=f"txnlat_{channel_lower}")
        txn_lon = st.number_input("Transaction longitude (optional)", format="%.6f", value=0.0, key=f"txnon_{channel_lower}")
        # Normalize None values: keep as None if user left defaults
        last_known_lat = last_known_lat if last_known_lat != 0.0 else None
        last_known_lon = last_known_lon if last_known_lon != 0.0 else None
        txn_lat = txn_lat if txn_lat != 0.0 else None
        txn_lon = txn_lon if txn_lon != 0.0 else None

    # Submit button
    submit = st.button("🚀 Run Fraud Check", key=f"submit_{channel_lower}")

    if submit:
        # Build payload: include only fields that exist for that channel (exclusive)
        payload: Dict = {
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
            # ip / geo
            "client_ip": client_ip,
            "ip_country": ip_country,
            "declared_country": "",  # optional, not captured in UI; can be set externally
            "suspicious_ip_flag": suspicious_ip_flag,
            "last_known_lat": last_known_lat,
            "last_known_lon": last_known_lon,
            "txn_lat": txn_lat,
            "txn_lon": txn_lon,
        }

        # Attach channel-specific keys explicitly
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
            if cc_mode == "POS (physical)":
                payload.update({
                    "card_masked": card_masked,
                    "card_country": card_country,
                    "cvv_provided": cvv_provided,
                    "pos_merchant_id": pos_merchant_id if 'pos_merchant_id' in locals() else "",
                })
            else:
                payload.update({
                    "card_masked": card_masked,
                    "card_country": card_country,
                    "cvv_provided": cvv_provided,
                    "DeviceID": device if 'device' in locals() else "",
                    "device_fingerprint": device_fingerprint if 'device_fingerprint' in locals() else "",
                    "device_last_seen": last_device if 'last_device' in locals() else "",
                    "client_ip": client_ip if 'client_ip' in locals() else "",
                    "ip_country": ip_country if 'ip_country' in locals() else "",
                })
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
                "DeviceID": device if 'device' in locals() else "",
                "device_last_seen": last_device if 'last_device' in locals() else "",
                "card_masked": card_masked if 'card_masked' in locals() else "",
                "cvv_provided": cvv_provided if 'cvv_provided' in locals() else True,
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

        # Decide whether to convert to INR before sending to model (depends on your model training)
        convert_to_inr_for_model = False  # change to True if your model was trained on INR amounts

        # ML scoring
        with st.spinner("Scoring with ML models..."):
            fraud_prob_raw, anomaly_raw, ml_label = score_transaction_ml(supervised_pipeline, iforest_pipeline, payload, convert_to_inr=convert_to_inr_for_model, currency=currency)

        # Convert ML numbers to 0-100% for display
        fraud_pct = clamp_pct(fraud_prob_raw)
        anomaly_pct = clamp_pct(anomaly_raw)

        # Evaluate rules (device rules disabled for Bank & ATM as implemented in evaluate_rules)
        rules_triggered, rules_highest = evaluate_rules(payload, currency)

        # Final combined risk
        final_risk = combine_final_risk(ml_label, rules_highest)

        # Present results
        st.markdown("## 🔎 Results")
        color_map = {"LOW": "#2e7d32", "MEDIUM": "#f9a825", "HIGH": "#f57c00", "CRITICAL": "#c62828"}
        badge_color = color_map.get(final_risk, "#607d8b")
        st.markdown(
            f"""<div style="padding:0.75rem 1rem;border-radius:0.5rem;background-color:{badge_color}22;border:1px solid {badge_color};">
                <strong style="color:{badge_color};font-size:1.1rem;">Final Risk Level: {final_risk}</strong>
            </div>""",
            unsafe_allow_html=True,
        )

        colA, colB = st.columns(2)
        with colA:
            st.metric("Fraud Confidence (ML)", f"{fraud_pct:.2f}%", help="Supervised model probability scaled to 0-100%")
            st.metric("ML Risk Label", ml_label)
        with colB:
            st.metric("Anomaly Score (ML)", f"{anomaly_pct:.2f}%", help="IsolationForest anomaly (scaled to 0-100% for display)")
            st.metric("Rules-derived highest severity", rules_highest)

        # Show triggered rules
        st.markdown("### ⚠ Triggered Rules (detailed)")
        if rules_triggered:
            for r in rules_triggered:
                sev = r["severity"]
                emoji = "🔴" if sev in ("HIGH", "CRITICAL") else "🟠" if sev == "MEDIUM" else "🟢"
                st.write(f"{emoji} **{r['name']}** — *{r['severity']}*")
                st.caption(r["detail"])
        else:
            st.success("No deterministic rules triggered.")

        # Debug payload
        st.markdown("### 📦 Payload (debug)")
        st.json(payload)

        

else:
    st.info("Select currency, enter amount/date/time, then pick a channel to show channel-specific inputs.")
