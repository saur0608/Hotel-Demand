"""
Streamlit Decision Dashboard with JWT Authentication, Registration & RBAC.

Features:
1. Two-tab Login & Registration:
   - Tab 1 (Sign In): Email/Password authentication, status verification (approved vs pending/rejected),
     and 1-click demo logins (Manager, Front Desk, Pending Trainee).
   - Tab 2 (Register): Public self-registration (enforces password >= 8 chars, email validation,
     confirm password match, and forces role='front_desk', status='pending').
2. Role-Based Access Control (RBAC):
   - 'manager': Full access to Forecasting, Newsvendor Decision Layer, Model Performance/SHAP,
     and the Access Requests management page to Approve/Reject pending accounts.
   - 'front_desk': Restricted access to Demand Forecasting only; Decision Layer and Admin pages blocked.
"""

from datetime import date, datetime, timedelta
from pathlib import Path
import sys

# Ensure root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import HTTPException
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.auth import (
    authenticate_user,
    create_access_token,
    get_all_users,
    init_user_db,
    register_user,
    update_user_status,
)
from src.config import (
    ARTIFACTS_DIR,
    CONFORMAL_ARTIFACT_PATH,
    DEMO_FRONT_DESK_EMAIL,
    DEMO_FRONT_DESK_NAME,
    DEMO_FRONT_DESK_PASSWORD,
    DEMO_MANAGER_EMAIL,
    DEMO_MANAGER_NAME,
    DEMO_MANAGER_PASSWORD,
    DEMO_PENDING_EMAIL,
    DEMO_PENDING_NAME,
    DEMO_PENDING_PASSWORD,
    MODEL_ARTIFACT_PATH,
    ROOM_CATEGORIES,
)
from src.data.generate_data import get_festival_multiplier
from src.decision.optimizer import NewsvendorDecisionLayer
from src.models.conformal import SplitConformalCalibrator

# Page setup
st.set_page_config(
    page_title="Hotel Demand & Inventory Optimizer",
    page_icon="🏨",
    layout="wide",
)

# Initialize database schema
init_user_db()


# ==============================================================================
# 1. AUTHENTICATION STATE & LOGIN / REGISTRATION SCREEN
# ==============================================================================
if "auth_token" not in st.session_state:
    st.session_state["auth_token"] = None
    st.session_state["email"] = None
    st.session_state["role"] = None
    st.session_state["name"] = None
    st.session_state["user_id"] = None


def render_auth_page():
    """Render combined Sign In & Registration screen with demo shortcuts."""
    st.markdown("<h2 style='text-align: center;'>🏨 Hotel Demand Forecasting & Inventory Optimizer</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: gray;'>Secure Role-Based Access Control (RBAC) System with Manager Approvals</p>", unsafe_allow_html=True)

    col_l, col_center, col_r = st.columns([1, 2, 1])
    with col_center:
        auth_tab1, auth_tab2 = st.tabs(["🔐 Sign In", "📝 Register New Account"])

        # ----------------------------------------------------------------------
        # TAB 1: SIGN IN
        # ----------------------------------------------------------------------
        with auth_tab1:
            st.subheader("Sign In to Your Account")
            with st.form("login_form"):
                email_input = st.text_input("Work Email", placeholder="e.g. manager@hotel.com")
                password_input = st.text_input("Password", type="password", placeholder="Enter your password")
                submit_btn = st.form_submit_button("Sign In", use_container_width=True)

                if submit_btn:
                    if not email_input or not password_input:
                        st.error("Please enter both email and password.")
                    else:
                        user, failure_reason = authenticate_user(email_input.strip().lower(), password_input)
                        if failure_reason == "pending":
                            st.warning("⏳ **Access Pending**: Your account registration has been submitted and is currently awaiting approval from a hotel manager.")
                        elif failure_reason == "rejected":
                            st.error("❌ **Access Denied**: Your account registration was rejected by an administrator.")
                        elif failure_reason == "invalid_credentials" or not user:
                            st.error("❌ Invalid email or password. Please try again.")
                        else:
                            token = create_access_token(data={
                                "sub": user["email"],
                                "role": user["role"],
                                "name": user["name"],
                                "user_id": user["id"],
                            })
                            st.session_state["auth_token"] = token
                            st.session_state["email"] = user["email"]
                            st.session_state["role"] = user["role"]
                            st.session_state["name"] = user["name"]
                            st.session_state["user_id"] = user["id"]
                            st.success(f"Welcome, {user['name']}! Logging in...")
                            st.rerun()

            st.markdown("---")
            st.markdown("#### ⚡ 1-Click Demo Profiles (For Walkthrough & Testing)")
            cd1, cd2, cd3 = st.columns(3)
            with cd1:
                if st.button("👑 Manager", use_container_width=True, help="Full access: Forecast, Decisions, SHAP, Admin"):
                    user, _ = authenticate_user(DEMO_MANAGER_EMAIL, DEMO_MANAGER_PASSWORD)
                    token = create_access_token(data={"sub": user["email"], "role": user["role"], "name": user["name"], "user_id": user["id"]})
                    st.session_state["auth_token"] = token
                    st.session_state["email"] = user["email"]
                    st.session_state["role"] = user["role"]
                    st.session_state["name"] = user["name"]
                    st.session_state["user_id"] = user["id"]
                    st.rerun()
                st.caption(f"Role: `manager`<br>`{DEMO_MANAGER_EMAIL}`", unsafe_allow_html=True)

            with cd2:
                if st.button("🛎️ Front Desk", use_container_width=True, help="Restricted access: Forecast only"):
                    user, _ = authenticate_user(DEMO_FRONT_DESK_EMAIL, DEMO_FRONT_DESK_PASSWORD)
                    token = create_access_token(data={"sub": user["email"], "role": user["role"], "name": user["name"], "user_id": user["id"]})
                    st.session_state["auth_token"] = token
                    st.session_state["email"] = user["email"]
                    st.session_state["role"] = user["role"]
                    st.session_state["name"] = user["name"]
                    st.session_state["user_id"] = user["id"]
                    st.rerun()
                st.caption(f"Role: `front_desk`<br>`{DEMO_FRONT_DESK_EMAIL}`", unsafe_allow_html=True)

            with cd3:
                if st.button("⏳ Trainee (Pending)", use_container_width=True, help="Shows pending approval lock"):
                    user, failure_reason = authenticate_user(DEMO_PENDING_EMAIL, DEMO_PENDING_PASSWORD)
                    if failure_reason == "pending":
                        st.warning("⏳ **Access Pending**: Trainee account is pending approval by the General Manager!")
                    else:
                        st.info(f"Account status: {failure_reason}")
                st.caption(f"Role: `front_desk`<br>Status: `pending`", unsafe_allow_html=True)

        # ----------------------------------------------------------------------
        # TAB 2: REGISTER NEW ACCOUNT
        # ----------------------------------------------------------------------
        with auth_tab2:
            st.subheader("Request Staff Access")
            st.markdown(
                "Submit your details to request access to the hotel system. "
                "New accounts are assigned the **Front Desk** role with **Pending** status until approved by a Revenue Manager."
            )
            with st.form("register_form"):
                reg_name = st.text_input("Full Name", placeholder="e.g. Priya Sharma")
                reg_email = st.text_input("Work Email", placeholder="e.g. priya@hotel.com")
                reg_password = st.text_input("Password (min 8 chars)", type="password", placeholder="At least 8 characters")
                reg_confirm = st.text_input("Confirm Password", type="password", placeholder="Repeat password")
                reg_submit = st.form_submit_button("Submit Registration", use_container_width=True)

                if reg_submit:
                    try:
                        result = register_user(
                            name=reg_name,
                            email=reg_email,
                            password=reg_password,
                            confirm_password=reg_confirm,
                        )
                        st.success(
                            f"🎉 **Registration Successful!** Account for **{result['name']}** created with role `{result['role']}`. "
                            f"Status is **Pending Manager Approval**. Please inform your Revenue Manager."
                        )
                    except HTTPException as e:
                        st.error(f"❌ Registration Error: {e.detail}")
                    except Exception as ex:
                        st.error(f"❌ Registration Error: {str(ex)}")


if not st.session_state["auth_token"]:
    render_auth_page()
    st.stop()


# ==============================================================================
# 2. LOAD SAVED MODEL ARTIFACTS
# ==============================================================================
@st.cache_resource
def load_forecasting_artifacts():
    """Load trained model, conformal calibrator, and pickup lookup."""
    if not MODEL_ARTIFACT_PATH.exists() or not CONFORMAL_ARTIFACT_PATH.exists():
        st.error(f"Model artifacts missing in {ARTIFACTS_DIR}. Please run `bash run.sh train` first.")
        st.stop()

    model_payload = joblib.load(MODEL_ARTIFACT_PATH)
    pipeline = model_payload["pipeline"]
    feature_cols = model_payload["feature_cols"]
    model_name = model_payload.get("model_name", "GradientBoosting_Pickup_Forecaster")

    conformal = SplitConformalCalibrator.load(CONFORMAL_ARTIFACT_PATH)

    lookup_path = ARTIFACTS_DIR / "pickup_lookup.joblib"
    pickup_lookup = joblib.load(lookup_path) if lookup_path.exists() else {}
    optimizer = NewsvendorDecisionLayer()

    return pipeline, feature_cols, model_name, conformal, pickup_lookup, optimizer


pipeline, feature_cols, model_name, conformal, pickup_lookup, optimizer = load_forecasting_artifacts()


# ==============================================================================
# 3. SIDEBAR WITH USER PROFILE & LOGOUT
# ==============================================================================
st.sidebar.title("🏨 Navigation")

# User profile badge
role_badge = "👑 Manager" if st.session_state["role"] == "manager" else "🛎️ Front Desk"
st.sidebar.info(
    f"**Logged in as**: {st.session_state['name']}\n\n"
    f"**Email**: `{st.session_state['email']}`\n\n"
    f"**Role**: `{role_badge}`"
)

if st.sidebar.button("🚪 Log Out", use_container_width=True):
    st.session_state["auth_token"] = None
    st.session_state["email"] = None
    st.session_state["role"] = None
    st.session_state["name"] = None
    st.session_state["user_id"] = None
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.subheader("Forecast Query Inputs")

room_cat = st.sidebar.selectbox("Room Category", list(ROOM_CATEGORIES.keys()), index=0)
spec = ROOM_CATEGORIES[room_cat]
capacity = spec["capacity"]
base_adr = spec["base_adr"]

default_date = date(2025, 11, 15)  # Mid-November wedding/festival peak
stay_date = st.sidebar.date_input("Target Arrival Date (Stay Date)", value=default_date)
horizon = st.sidebar.slider("Forecast Horizon (Lead Time in Days)", min_value=1, max_value=14, value=7)
as_of_date = stay_date - timedelta(days=horizon)
st.sidebar.markdown(f"**Information Cutoff (T_as_of)**: `{as_of_date}` (23:59)")

stay_dt = datetime.combine(stay_date, datetime.min.time())
stay_dow = stay_dt.weekday()
hist_pickup = pickup_lookup.get((room_cat, stay_dow, horizon), float(horizon * 0.8))
default_otb = max(1, int(capacity * 0.78 - hist_pickup))

otb_input = st.sidebar.slider("Current Active On-The-Books (OTB) Rooms", min_value=0, max_value=int(capacity * 1.2), value=min(default_otb, capacity))

with st.sidebar.expander("Advanced Booking Pace Velocity", expanded=False):
    p3 = st.number_input("Bookings in Last 3 Days", min_value=0, value=max(0, int(otb_input * 0.15)))
    p7 = st.number_input("Bookings in Last 7 Days", min_value=0, value=max(0, int(otb_input * 0.30)))
    canc_td = st.number_input("Cancellations Recorded to Date", min_value=0, value=max(0, int(otb_input * 0.12)))


# ==============================================================================
# 4. COMPUTE FORECAST
# ==============================================================================
stay_month = stay_dt.month
stay_quarter = (stay_month - 1) // 3 + 1
doy = stay_dt.timetuple().tm_yday
sin_doy = float(np.sin(2 * np.pi * doy / 365.25))
cos_doy = float(np.cos(2 * np.pi * doy / 365.25))
fest_mult, fest_name = get_festival_multiplier(stay_date)
is_fest = int(fest_mult > 1.0)
mean_cap = capacity * 0.78
canc_rate = canc_td / (otb_input + canc_td + 1e-5)

feat_row = pd.DataFrame([{
    "room_category": room_cat,
    "horizon": horizon,
    "otb_rooms": otb_input,
    "otb_occupancy_ratio": otb_input / capacity,
    "pickup_last_3d": p3,
    "pickup_last_7d": p7,
    "cancellations_to_date": canc_td,
    "cancellation_rate_otb": canc_rate,
    "capacity": capacity,
    "base_adr": base_adr,
    "stay_dow": stay_dow,
    "stay_is_weekend": int(stay_dow in [4, 5]),
    "stay_month": stay_month,
    "stay_quarter": stay_quarter,
    "stay_day_of_month": stay_dt.day,
    "stay_sin_doy": sin_doy,
    "stay_cos_doy": cos_doy,
    "festival_uplift_factor": fest_mult,
    "is_festival": is_fest,
    "lag_T0": mean_cap,
    "lag_T1": mean_cap,
    "lag_T7": mean_cap,
    "lag_T14": mean_cap,
    "roll_mean_7d": mean_cap,
    "roll_mean_14d": mean_cap,
    "roll_std_7d": 2.0,
    "hist_avg_pickup": hist_pickup,
}])

pred_raw = float(pipeline.predict(feat_row[feature_cols])[0])
pred = float(np.clip(pred_raw, 0.0, capacity * 1.25))
low_arr, up_arr = conformal.predict_interval(feat_row, np.array([pred]))
lower, upper = round(float(low_arr[0]), 1), round(float(up_arr[0]), 1)
width = upper - lower

uncertainty_note = (
    f"80% Conformal Interval: [{lower} to {upper} rooms] (Width: {width:.1f} rooms). "
    f"Based on empirical validation residuals at {horizon}-day lead time, there is an 80% probability "
    f"that final realized demand will fall within this range."
)

decision_dict = optimizer.optimize_decision(
    room_category=room_cat,
    point_forecast=pred,
    conformal_lower=lower,
    conformal_upper=upper,
    otb_rooms=otb_input,
    horizon=horizon,
)


# ==============================================================================
# 5. ROLE-BASED TABS
# ==============================================================================
st.title("🏨 Hotel Demand & Inventory Management")
st.markdown(
    f"**Arrival**: `{stay_date}` ({stay_dt.strftime('%A')}) | "
    f"**Category**: `{room_cat}` (Cap: **{capacity}**, ADR: **₹{base_adr:,.0f}**) | "
    f"**Horizon**: **{horizon} days** out"
)

tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Demand Forecast (All Roles)",
    "⚖️ Asymmetric Decision Layer (Manager Only)",
    "📈 Model Performance & SHAP (Manager Only)",
    "👥 Access Requests (Manager Only)",
])

# ------------------------------------------------------------------------------
# TAB 1: DEMAND FORECAST (Accessible by: manager & front_desk)
# ------------------------------------------------------------------------------
with tab1:
    st.subheader("Room Demand Forecast & 80% Conformal Interval")
    c1, c2, c3 = st.columns(3)
    c1.metric("🎯 Point Demand Forecast", f"{pred:.1f} rooms", f"{(pred / capacity) * 100:.0f}% Occupancy")
    c2.metric("🛡️ 80% Conformal Interval", f"[{lower:.0f} - {upper:.0f}] rooms", f"Width: {width:.1f} rms")
    c3.metric("📋 Current Active OTB", f"{otb_input} rooms", f"{otb_input / capacity * 100:.0f}% Cap Booked")

    st.info(f"**Plain-Language Uncertainty Note**: {uncertainty_note}")

    # Inventory Gauge Chart
    fig, ax = plt.subplots(figsize=(10, 2.5))
    ax.barh([0], [otb_input], color="#1f77b4", label=f"Current OTB ({otb_input} rms)", alpha=0.9, height=0.45)
    expected_pickup = max(0, pred - otb_input)
    ax.barh([0], [expected_pickup], left=[otb_input], color="#aec7e8", label=f"Expected Pickup (+{expected_pickup:.1f} rms)", height=0.45)
    ax.errorbar([pred], [0], xerr=[[pred - lower], [upper - pred]], fmt="o", color="black", elinewidth=3, capsize=8, label="80% Conformal Interval")
    ax.axvline(capacity, color="darkred", linestyle="--", linewidth=2.2, label=f"Physical Capacity ({capacity} rms)")

    ax.set_xlim(0, max(capacity * 1.3, upper + 5))
    ax.set_yticks([])
    ax.set_xlabel("Rooms")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.45), ncol=3, frameon=True)
    st.pyplot(fig)
    plt.close(fig)

    # 14-day projection curve
    st.markdown("#### 14-Day Booking Pace Projection Curve")
    pace_records = []
    for h_sim in range(14, 0, -1):
        sim_hist = pickup_lookup.get((room_cat, stay_dow, h_sim), float(h_sim * 0.8))
        sim_otb = max(1, int(pred - sim_hist))
        pace_records.append({
            "Horizon (Days Out)": h_sim,
            "Simulated OTB": sim_otb,
            "Final Target Forecast": pred,
            "Physical Capacity": capacity,
        })
    pace_chart_df = pd.DataFrame(pace_records).set_index("Horizon (Days Out)")
    st.line_chart(pace_chart_df)

# ------------------------------------------------------------------------------
# TAB 2: ASYMMETRIC DECISION LAYER (Manager ONLY)
# ------------------------------------------------------------------------------
with tab2:
    if st.session_state["role"] != "manager":
        st.error(
            "🔒 **Access Denied**: The Newsvendor Decision Layer (Overbooking authorizations and inventory protection) "
            "is restricted to **Manager** roles only. Front desk accounts cannot modify inventory allocation policies."
        )
    else:
        st.subheader("⚖️ Asymmetric Newsvendor Decision Controls")
        d1, d2, d3 = st.columns(3)
        d1.metric("🚪 Authorized Overbooking", f"{decision_dict['authorized_overbooking_rooms']} rooms", f"Cap: {decision_dict['authorized_booking_limit']}")
        d2.metric("🔒 Protected Inventory (Hold)", f"{decision_dict['rooms_to_hold']} rooms", f"Open: {decision_dict['available_discount_rooms']}")
        d3.metric("📊 Critical Fractile (CR)", f"{decision_dict['critical_ratio']:.3f}", f"Walk: {spec['walk_penalty_ratio']}x ADR")

        if decision_dict["authorized_overbooking_rooms"] > 0:
            st.error(f"**Manager Action**: {decision_dict['action_text']}")
        elif decision_dict["rooms_to_hold"] > 0:
            st.warning(f"**Manager Action**: {decision_dict['action_text']}")
        else:
            st.success(f"**Manager Action**: {decision_dict['action_text']}")

        st.markdown("#### Economic Risk Structure")
        st.markdown(f"""
        - **Underage Cost ($C_u$)**: Lost margin if room sits empty = **₹{decision_dict['underage_cost_inr']:,.0f}**
        - **Overage Cost ($C_o$)**: Penalty to walk/re-accommodate a confirmed guest ({spec['walk_penalty_ratio']}x ADR) = **₹{decision_dict['overage_cost_inr']:,.0f}**
        - **Critical Ratio ($CR = \\frac{{C_u}}{{C_u + C_o}}$)**: **{decision_dict['critical_ratio']:.3f}**
        - **Financial Protection**: Operating at the {decision_dict['critical_ratio']*100:.1f}th percentile prevents costly walk penalties while maximizing revenue.
        """)

# ------------------------------------------------------------------------------
# TAB 3: MODEL PERFORMANCE & SHAP (Manager ONLY)
# ------------------------------------------------------------------------------
with tab3:
    if st.session_state["role"] != "manager":
        st.error("🔒 **Access Denied**: Performance metrics and explainability reports require Manager role.")
    else:
        st.subheader("Model Performance & Backtesting Summary")
        st.markdown("""
        | Model | MAE | RMSE | WAPE (%) |
        | :--- | :---: | :---: | :---: |
        | **1. Naive (Same Weekday)** | 7.475 | 10.315 | 24.02% |
        | **2. Seasonal Naive (4-Wk)** | 6.308 | 9.205 | 20.27% |
        | **3. Historical Pickup Baseline** | 3.111 | 5.179 | 10.00% |
        | **4. Ridge Regression** | 3.069 | 4.748 | 9.86% |
        | **5. Gradient Boosted Trees (Winning)** | **2.792** | **4.275** | **8.97%** |
        """)

        shap_img = ARTIFACTS_DIR / "eda_plots" / "08_shap_summary_beeswarm.png"
        if shap_img.exists():
            st.image(str(shap_img), caption="TreeSHAP Global Feature Importances", use_column_width=True)

# ------------------------------------------------------------------------------
# TAB 4: ACCESS REQUESTS & USER APPROVAL (Manager ONLY)
# ------------------------------------------------------------------------------
with tab4:
    if st.session_state["role"] != "manager":
        st.error("🔒 **Access Denied**: User administration and account approvals are restricted to **Manager** roles only.")
    else:
        st.subheader("👥 User Access Requests & Account Management")
        st.markdown("Review and approve/reject staff registration requests. Approved users can immediately sign in.")

        users = get_all_users()
        total_users = len(users)
        pending_users = [u for u in users if u["status"] == "pending"]
        approved_users = [u for u in users if u["status"] == "approved"]
        rejected_users = [u for u in users if u["status"] == "rejected"]

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Accounts", total_users)
        m2.metric("🟡 Pending Requests", len(pending_users))
        m3.metric("🟢 Approved Accounts", len(approved_users))
        m4.metric("🔴 Rejected Accounts", len(rejected_users))

        st.markdown("---")
        st.markdown("#### Registered Staff & Requests")

        for u in users:
            status_color = "🟡" if u["status"] == "pending" else ("🟢" if u["status"] == "approved" else "🔴")
            with st.container():
                uc1, uc2, uc3, uc4, uc5 = st.columns([2.5, 2.5, 1.5, 1.5, 2])
                with uc1:
                    st.markdown(f"**{u['name']}**")
                    st.caption(f"ID: #{u['id']} | `{u['email']}`")
                with uc2:
                    st.markdown(f"Role: `{u['role']}`")
                    st.caption(f"Created: {u.get('created_at', 'N/A')}")
                with uc3:
                    st.markdown(f"Status: **{status_color} {u['status'].upper()}**")
                with uc4:
                    if u["status"] != "approved":
                        if st.button("✅ Approve", key=f"app_{u['id']}", use_container_width=True):
                            update_user_status(u["id"], "approved")
                            st.success(f"Approved {u['email']}!")
                            st.rerun()
                with uc5:
                    if u["status"] != "rejected" and u["email"] != DEMO_MANAGER_EMAIL:
                        if st.button("❌ Reject", key=f"rej_{u['id']}", use_container_width=True):
                            update_user_status(u["id"], "rejected")
                            st.warning(f"Rejected {u['email']}!")
                            st.rerun()
                st.markdown("<hr style='margin: 4px 0;' />", unsafe_allow_html=True)
