# app.py (enhanced)
# Real-Time Fraud Detection — UI & data-model improvements
# - Aligned UI layout
# - Clear Transaction Type field definitions for TRANSFER / PAYMENT / BILL_PAY
# - ATM channel removes client IP / derived IP / device fields
# - Duplicate IP fields removed (single client_ip / ip_country usage where applicable)
# - Consistent location fields for ALL transaction types: home_city, home_country, txn_location_ip, txn_city, txn_country

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

BASE_THRESHOLDS_INR = {
    "absolute_crit_amount": 10_000_000,
    "high_amount_threshold": 2_000_000,
    "medium_amount_threshold": 100_000,
    "atm_high_withdrawal": 300_000,
    "card_test_small_amount_inr": 200,
}

FRAUD_MED = 0.00005
FRAUD_HIGH = 0.00023328
FRAUD_CRIT = 0.01732857

ANOM_MED = 0.04
ANOM_HIGH = 0.05
ANOM_CRIT = 0.08

SEVERITY_ORDER = {"LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}

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
    if currency not in INR_PER_UNIT or INR_PER_UNIT[currency] == 0:
        return amount_in_inr
    return amount_in_inr / INR_PER_UNIT[currency]


def clamp_pct(x: float) -> float:
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
    if None in (lat1, lon1, lat2, lon2):
        return None
    lat1, lon1, lat2, lon2 = map(radians, (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    return 6371 * c


def escalate(a: str, b: str) -> str:
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

    supervised = None
    iforest = None
    try:
        supervised = _load("supervised_lgbm_pipeline.joblib")
    except Exception:
        supervised = None
    try:
        iforest = _load("iforest_pipeline.joblib")
    except Exception:
        iforest = None
    return supervised, iforest

supervised_pipeline, iforest_pipeline = load_models()

# -------------------------
# ML SCORING WRAPPER
# -------------------------

def ml_risk_label(fraud_prob: float, anomaly_score: float) -> str:
    if fraud_prob >= FRAUD_CRIT or anomaly_score >= ANOM_CRIT:
        return "CRITICAL"
    if fraud_prob >= FRAUD_HIGH or (fraud_prob >= FRAUD_MED and anomaly_score >= ANOM_HIGH):
        return "HIGH"
    if fraud_prob >= FRAUD_MED or anomaly_score >= ANOM_MED:
        return "MEDIUM"
    return "LOW"


def score_transaction_ml(model_pipeline, iforest_pipeline, model_payload: Dict, convert_to_inr: bool = False, currency: str = "INR") -> Tuple[float, float, str]:
    amt_for_model = model_payload.get("Amount", 0.0)
    if convert_to_inr:
        amt_for_model = amt_for_model * INR_PER_UNIT.get(currency, 1.0)

    model_df = pd.DataFrame([{
        "Amount": amt_for_model,
        "TransactionType": model_payload.get("TransactionType", "PAYMENT"),
        "Location": model_payload.get("txn_city", "Unknown"),
        "DeviceID": model_payload.get("DeviceID", "Unknown"),
        "Channel": model_payload.get("Channel", "Other"),
        "hour": model_payload.get("hour", 0),
        "day_of_week": model_payload.get("day_of_week", 0),
        "month": model_payload.get("month", 0),
    }])

    fraud_prob = 0.0
    anomaly_score = 0.0
    try:
        if model_pipeline is not None:
            fraud_prob = float(model_pipeline.predict_proba(model_df)[0, 1])
    except Exception as e:
        st.error("Supervised model scoring error - check pipeline input schema")
        st.exception(e)
    try:
        if iforest_pipeline is not None:
            raw = float(iforest_pipeline.decision_function(model_df)[0])
            anomaly_score = -raw
    except Exception as e:
        st.error("IsolationForest scoring error - check pipeline input schema")
        st.exception(e)

    label = ml_risk_label(fraud_prob, anomaly_score)
    return fraud_prob, anomaly_score, label

# -------------------------
# RULE ENGINE
# -------------------------

def evaluate_rules(payload: Dict, currency: str) -> Tuple[List[Dict], str]:
    ABS_CRIT = inr_to_currency(BASE_THRESHOLDS_INR["absolute_crit_amount"], currency)
    HIGH_AMT = inr_to_currency(BASE_THRESHOLDS_INR["high_amount_threshold"], currency)
    MED_AMT = inr_to_currency(BASE_THRESHOLDS_INR["medium_amount_threshold"], currency)
    ATM_HIGH = inr_to_currency(BASE_THRESHOLDS_INR["atm_high_withdrawal"], currency)
    CARD_TEST_SMALL = inr_to_currency(BASE_THRESHOLDS_INR["card_test_small_amount_inr"], currency)

    rules: List[Dict] = []

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

    def add_rule(name: str, sev: str, detail: str):
        rules.append({"name": name, "severity": sev, "detail": detail})

    if amt >= ABS_CRIT:
        add_rule("Absolute very large amount", "CRITICAL",
                 f"Amount {amt:.2f} {currency} >= critical {ABS_CRIT:.2f} {currency}.")

    impossible_travel_distance = None
    if last_lat is not None and last_lon is not None and txn_lat is not None and txn_lon is not None:
        impossible_travel_distance = haversine_km(last_lat, last_lon, txn_lat, txn_lon)

    device_checks_enabled = channel not in ("bank", "atm")

    if device_checks_enabled:
        device_new = (not last_device) or last_device == ""
        location_changed = impossible_travel_distance is not None and impossible_travel_distance > 500
        if device_new and location_changed and amt > MED_AMT:
            add_rule("New device + Impossible travel + High amount", "CRITICAL",
                     f"New device + travel {impossible_travel_distance:.1f} km; amount {amt:.2f} {currency}.")

    if beneficiaries_added_24h >= 3 and amt > HIGH_AMT:
        add_rule("Multiple beneficiaries added + high transfer", "CRITICAL",
                 f"{beneficiaries_added_24h} beneficiaries added and amount {amt:.2f} {currency}.")

    if txns_1h >= 10:
        add_rule("High velocity (1h)", "HIGH", f"{txns_1h} txns in last 1 hour.")
    if txns_24h >= 50:
        add_rule("Very high velocity (24h)", "HIGH", f"{txns_24h} txns in last 24h.")

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

    if device_checks_enabled and ((not last_device) or last_device == "") and amt < (MED_AMT / 10):
        add_rule("New device (low amount)", "LOW", "Transaction from new device but low amount.")

    if 0 < beneficiaries_added_24h < 3:
        add_rule("Beneficiaries recently added", "LOW", f"{beneficiaries_added_24h} beneficiaries added.")

    if ip_country and ip_country in {"nigeria", "romania", "ukraine", "russia"}:
        add_rule("IP from higher-risk country", "MEDIUM", f"IP country flagged as higher-risk: {ip_country}.")

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

    # Transfer-specific checks: ensure from/to accounts present when TransactionType is TRANSFER
    if str(payload.get("TransactionType", "")).upper() == "TRANSFER":
        from_acc = payload.get("from_account_number")
        to_acc = payload.get("to_account_number")
        if not from_acc or not to_acc:
            add_rule("Missing transfer account data", "HIGH", "Transfer missing source or destination account details.")

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
st.title("💳 Real-Time Fraud Detection — Enhanced UI & Data Model")

st.markdown("**Common fields (shown for ALL transactions):** Currency, Amount, Date, Time, Home Address (city/country), Transaction origin location (IP/city/country)")

# Top row: currency and INR rate
col0, col0b = st.columns([3, 1])
with col0:
    currency = st.selectbox("Select currency", CURRENCIES, index=CURRENCIES.index(DEFAULT_CURRENCY), key="currency_select")
with col0b:
    st.caption(f"(INR per unit) {currency} = {INR_PER_UNIT[currency]:,.2f} INR")

# Amount + date/time aligned
col1, col2 = st.columns([1, 1])
with col1:
    amount = st.number_input(f"Transaction amount ({currency})", min_value=0.0, value=1200.0, step=10.0, key="amount_common")
with col2:
    txn_date = st.date_input("Transaction date", value=datetime.date.today(), key="txn_date")
    txn_time = st.time_input("Transaction time", value=datetime.time(12, 0), key="txn_time")

txn_dt = datetime.datetime.combine(txn_date, txn_time)
hour = txn_dt.hour
day_of_week = txn_dt.weekday()
month = txn_dt.month

st.markdown("---")

channel = st.selectbox("Transaction Channel", ["Choose...", "Bank", "Mobile App", "ATM", "Credit Card", "POS", "Online Purchase", "NetBanking"], key="channel_select")
if channel and channel != "Choose...":
    channel_lower = channel.lower()
    st.markdown(f"### Channel: {channel}")

    txn_options = CHANNEL_TXN_TYPES.get(channel_lower, ["OTHER"])
    txn_type = st.selectbox("Transaction type", txn_options, key=f"txn_type_{channel_lower}")

    # Transaction-type specific panels: TRANSFER / PAYMENT / BILL_PAY
    st.markdown("#### Transaction type details")
    # Initialize all tx-type variables with defaults to avoid missing locals
    transfer_fields = {}
    payment_fields = {}
    billpay_fields = {}

    # Transfer UI (when selected)
    if str(txn_type).upper() == "TRANSFER":
        st.subheader("TRANSFER — source and destination account details")
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            from_account_number = st.text_input("From account number", key=f"from_acc_{channel_lower}")
            from_account_holder_name = st.text_input("From account holder name", key=f"from_name_{channel_lower}")
        with col_f2:
            to_account_number = st.text_input("To account number", key=f"to_acc_{channel_lower}")
            to_account_holder_name = st.text_input("To account holder name", key=f"to_name_{channel_lower}")
        col_tf = st.columns([2, 1])[0]
        beneficiary_flag = st.checkbox("Is this to a known beneficiary?", value=False, key=f"benef_flag_{channel_lower}")
        new_beneficiary = st.checkbox("Is this a newly added beneficiary?", value=False, key=f"new_benef_{channel_lower}")
        beneficiary_added_minutes = st.number_input("Minutes since beneficiary added (if applicable)", min_value=0, value=9999, step=1, key=f"ben_min_{channel_lower}")
        reason = st.text_area("Reason / notes (optional)", key=f"transfer_reason_{channel_lower}")
        transfer_fields.update({
            "from_account_number": from_account_number,
            "from_account_holder_name": from_account_holder_name,
            "to_account_number": to_account_number,
            "to_account_holder_name": to_account_holder_name,
            "beneficiary_flag": beneficiary_flag,
            "new_beneficiary": new_beneficiary,
            "beneficiary_added_minutes": int(beneficiary_added_minutes),
            "reason": reason,
        })

    # Payment UI (when selected)
    if str(txn_type).upper() == "PAYMENT":
        st.subheader("PAYMENT — merchant / payment details")
        payment_category = st.selectbox("Payment category", ["ecommerce", "utilities", "subscription", "pos", "other"], key=f"pay_cat_{channel_lower}")
        merchant_id = st.text_input("Merchant name / ID", key=f"merchant_{channel_lower}")
        card_used = st.checkbox("Paid using card?", value=False, key=f"card_used_{channel_lower}")
        card_masked = ""
        cvv_provided = True
        device_for_payment = ""
        if card_used:
            card_masked = st.text_input("Card masked (e.g., 4111****1111)", key=f"pay_card_{channel_lower}")
            cvv_provided = st.checkbox("CVV provided (checked if present)", value=True, key=f"pay_cvv_{channel_lower}")
        # Device info optional for digital channels
        if channel_lower in ("mobile app", "online purchase", "netbanking", "credit card"):
            device_for_payment = st.text_input("Device / Browser (optional)", key=f"pay_device_{channel_lower}")
        payment_fields.update({
            "payment_category": payment_category,
            "merchant_id": merchant_id,
            "card_used": bool(card_used),
            "card_masked": card_masked,
            "cvv_provided": bool(cvv_provided),
            "device_info": device_for_payment,
        })

    # Bill Pay UI (when selected)
    if str(txn_type).upper() == "BILL_PAY":
        st.subheader("BILL PAY — structured biller details")
        biller_category = st.selectbox("Biller category", ["electricity", "water", "telecom", "internet", "insurance", "other"], key=f"biller_cat_{channel_lower}")
        biller_id = st.text_input("Biller ID", key=f"biller_id_{channel_lower}")
        bill_reference_number = st.text_input("Bill/reference number", key=f"bill_ref_{channel_lower}")
        due_date = st.date_input("Bill due date (optional)", value=None, key=f"bill_due_{channel_lower}")
        col_bp = st.columns(2)
        bill_period_start = st.date_input("Bill period start (optional)", value=None, key=f"bill_start_{channel_lower}")
        bill_period_end = st.date_input("Bill period end (optional)", value=None, key=f"bill_end_{channel_lower}")
        billpay_fields.update({
            "biller_category": biller_category,
            "biller_id": biller_id,
            "bill_reference_number": bill_reference_number,
            "due_date": str(due_date) if due_date else "",
            "bill_period_start": str(bill_period_start) if bill_period_start else "",
            "bill_period_end": str(bill_period_end) if bill_period_end else "",
        })

    st.markdown("---")
    st.markdown("#### Channel-specific fields (aligned and grouped)")

    # Start channel-specific fields with explicit, non-duplicated keys
    # Bank
    bank_fields = {}
    if channel_lower == "bank":
        st.subheader("In-branch (Bank) fields — Identity only")
        id_type = st.selectbox("ID Document Type", ["Passport", "Driver License", "Government ID", "Other"], key="bank_id_type")
        id_number = st.text_input("ID Document Number", key="bank_id_number")
        branch = st.text_input("Branch Name / Code", value="", key="bank_branch")
        teller_id = st.text_input("Teller ID (optional)", value="", key="bank_teller")
        bank_fields.update({"id_type": id_type, "id_number": id_number, "branch": branch, "teller_id": teller_id})

    # ATM
    atm_fields = {}
    if channel_lower == "atm":
        st.subheader("ATM fields — card + ATM info (no IP/device)")
        atm_id = st.text_input("ATM ID / Terminal", key="atm_id")
        atm_location = st.text_input("ATM Location (free text)", key="atm_location")
        atm_distance_km = st.number_input("ATM distance from last known location (km)", min_value=0.0, value=0.0, step=1.0, key="atm_distance")
        card_masked_atm = st.text_input("Card masked (e.g., 4111****1111)", key="atm_card_masked")
        atm_fields.update({"atm_id": atm_id, "atm_location": atm_location, "atm_distance_km": atm_distance_km, "card_masked": card_masked_atm})

    # Mobile App
    mobile_fields = {}
    if channel_lower == "mobile app":
        st.subheader("Mobile App fields — device + app telemetry")
        device = st.text_input("Device / OS (e.g., Android)", value="Android", key="mobile_device")
        device_fingerprint = st.text_input("Device fingerprint (optional)", key="mobile_device_fp")
        app_version = st.text_input("App version", value="1.0.0", key="mobile_app_ver")
        last_device = st.text_input("Last known device (optional)", key="mobile_last_device")
        mobile_fields.update({"DeviceID": device, "device_fingerprint": device_fingerprint, "app_version": app_version, "device_last_seen": last_device})

    # Credit Card
    cc_fields = {}
    if channel_lower == "credit card":
        st.subheader("Credit Card: choose mode")
        cc_mode = st.radio("Credit Card mode", ["POS (physical)", "Mobile/Web (app or web)"], key="cc_mode")
        if cc_mode == "POS (physical)":
            card_masked_cc = st.text_input("Card masked (4111****1111)", key="cc_pos_card")
            card_country = st.text_input("Card issuing country", key="cc_pos_country")
            cvv_provided = st.checkbox("CVV provided (checked if present)", value=True, key="cc_pos_cvv")
            pos_merchant_id = st.text_input("POS Merchant ID (optional)", key="cc_pos_mid")
            cc_fields.update({"card_masked": card_masked_cc, "card_country": card_country, "cvv_provided": cvv_provided, "pos_merchant_id": pos_merchant_id})
        else:
            card_masked_cc = st.text_input("Card masked (4111****1111)", key="cc_web_card")
            card_country = st.text_input("Card issuing country", key="cc_web_country")
            cvv_provided = st.checkbox("CVV provided (checked if present)", value=True, key="cc_web_cvv")
            device_cc = st.text_input("Device / Browser (optional)", key="cc_web_device")
            device_fingerprint_cc = st.text_input("Device fingerprint (optional)", key="cc_web_fp")
            last_device_cc = st.text_input("Last known device (optional)", key="cc_web_last_device")
            cc_fields.update({"card_masked": card_masked_cc, "card_country": card_country, "cvv_provided": cvv_provided, "DeviceID": device_cc, "device_fingerprint": device_fingerprint_cc, "device_last_seen": last_device_cc})

    # POS
    pos_fields = {}
    if channel_lower == "pos":
        st.subheader("POS fields")
        pos_merchant_id = st.text_input("POS Merchant ID", key="pos_mid")
        store_name = st.text_input("Store name", key="pos_store")
        pos_repeat_count = st.number_input("Rapid repeat transactions at same POS", min_value=0, value=0, step=1, key="pos_repeat")
        pos_fields.update({"pos_merchant_id": pos_merchant_id, "store_name": store_name, "pos_repeat_count": pos_repeat_count})

    # Online Purchase
    online_fields = {}
    if channel_lower == "online purchase":
        st.subheader("Online Purchase fields (device + addresses)")
        merchant = st.text_input("Merchant name / ID", key="online_merchant")
        shipping_address = st.text_input("Shipping address", key="online_ship")
        billing_address = st.text_input("Billing address", value=shipping_address, key="online_bill")
        used_card_online = st.checkbox("Paid by card online?", value=False, key="online_card_used")
        card_masked_online = ""
        cvv_provided_online = True
        if used_card_online:
            cvv_provided_online = st.checkbox("CVV provided (checked if present)", value=True, key="online_cvv")
            card_masked_online = st.text_input("Card masked", key="online_card_masked")
        device_online = st.text_input("Device / Browser (optional)", key="online_device")
        last_device_online = st.text_input("Last known device (optional)", key="online_last_device")
        online_fields.update({"merchant": merchant, "shipping_address": shipping_address, "billing_address": billing_address, "card_masked": card_masked_online, "cvv_provided": cvv_provided_online, "DeviceID": device_online, "device_last_seen": last_device_online})

    # NetBanking
    netbanking_fields = {}
    if channel_lower == "netbanking":
        st.subheader("NetBanking fields (device-aware)")
        username = st.text_input("User ID / Login", key="nb_user")
        device = st.text_input("Device / Browser (used to login)", key="nb_device")
        last_device = st.text_input("Last known device (optional)", key="nb_last_device")
        beneficiary = st.text_input("Beneficiary (if transfer)", key="nb_beneficiary")
        new_beneficiary = st.checkbox("Is beneficiary newly added?", value=False, key="nb_new_benef")
        beneficiary_added_minutes = st.number_input("Minutes since beneficiary was added (if known)", min_value=0, value=9999, step=1, key="nb_benef_minutes")
        netbanking_fields.update({"username": username, "DeviceID": device, "device_last_seen": last_device, "beneficiary": beneficiary, "new_beneficiary": new_beneficiary, "beneficiary_added_minutes": int(beneficiary_added_minutes)})

    # Optional telemetry panel
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

    # IP & Geo telemetry (centralized): Bank and ATM suppress client IP
    st.markdown("#### IP / Geo (centralized fields)")
    # Home address (required for fraud context)
    home_city = st.text_input("Customer home city", key=f"home_city_{channel_lower}")
    home_country = st.text_input("Customer home country", key=f"home_country_{channel_lower}")

    if channel_lower in ("bank", "atm"):
        st.info("Bank/ATM: client IP is not collected by design for in-branch and ATM flows.")
        client_ip = ""
        ip_country = ""
        suspicious_ip_flag = False
    else:
        client_ip = st.text_input("Client IP (optional)", key=f"client_ip_{channel_lower}")
        ip_country = st.text_input("IP-derived country (optional)", key=f"ip_country_{channel_lower}")
        suspicious_ip_flag = st.checkbox("IP flagged by threat intel?", value=False, key=f"suspicious_{channel_lower}")

    # Transaction origin location fields (always present)
    txn_location_ip = st.text_input("Transaction origin IP (txn_location_ip) (optional)", key=f"txn_loc_ip_{channel_lower}")
    txn_city = st.text_input("Transaction city (txn_city)", key=f"txn_city_{channel_lower}")
    txn_country = st.text_input("Transaction country (txn_country)", key=f"txn_country_{channel_lower}")

    # Lat/long optional for distance/impossible-travel checks
    last_known_lat = st.number_input("Last known latitude (optional)", format="%.6f", value=0.0, key=f"lastlat_{channel_lower}")
    last_known_lon = st.number_input("Last known longitude (optional)", format="%.6f", value=0.0, key=f"lastlon_{channel_lower}")
    txn_lat = st.number_input("Transaction latitude (optional)", format="%.6f", value=0.0, key=f"txnlat_{channel_lower}")
    txn_lon = st.number_input("Transaction longitude (optional)", format="%.6f", value=0.0, key=f"txnon_{channel_lower}")

    last_known_lat = last_known_lat if last_known_lat != 0.0 else None
    last_known_lon = last_known_lon if last_known_lon != 0.0 else None
    txn_lat = txn_lat if txn_lat != 0.0 else None
    txn_lon = txn_lon if txn_lon != 0.0 else None

    # Submit
    submit = st.button("🚀 Run Fraud Check", key=f"submit_{channel_lower}")

    if submit:
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
            # centralized ip/geo
            "client_ip": client_ip,
            "ip_country": ip_country,
            "txn_location_ip": txn_location_ip,
            "txn_city": txn_city,
            "txn_country": txn_country,
            "declared_country": "",
            "suspicious_ip_flag": suspicious_ip_flag,
            "last_known_lat": last_known_lat,
            "last_known_lon": last_known_lon,
            "txn_lat": txn_lat,
            "txn_lon": txn_lon,
        }

        # attach transaction-type specific details
        if str(txn_type).upper() == "TRANSFER":
            payload.update(transfer_fields)
        elif str(txn_type).upper() == "PAYMENT":
            payload.update(payment_fields)
        elif str(txn_type).upper() == "BILL_PAY":
            payload.update(billpay_fields)

        # attach channel-specific details
        if channel_lower == "bank":
            payload.update(bank_fields)
        elif channel_lower == "atm":
            payload.update(atm_fields)
        elif channel_lower == "mobile app":
            payload.update(mobile_fields)
        elif channel_lower == "credit card":
            payload.update(cc_fields)
        elif channel_lower == "pos":
            payload.update(pos_fields)
        elif channel_lower == "online purchase":
            payload.update(online_fields)
        elif channel_lower == "netbanking":
            payload.update(netbanking_fields)

        convert_to_inr_for_model = False

        with st.spinner("Scoring with ML models..."):
            fraud_prob_raw, anomaly_raw, ml_label = score_transaction_ml(supervised_pipeline, iforest_pipeline, payload, convert_to_inr=convert_to_inr_for_model, currency=currency)

        fraud_pct = clamp_pct(fraud_prob_raw)
        anomaly_pct = clamp_pct(anomaly_raw)

        rules_triggered, rules_highest = evaluate_rules(payload, currency)

        final_risk = combine_final_risk(ml_label, rules_highest)

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

        st.markdown("### ⚠ Triggered Rules (detailed)")
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

else:
    st.info("Select currency, enter amount/date/time, then pick a channel to show channel-specific inputs.")
