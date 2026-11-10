"""
FastAPI Demand Forecasting & Inventory Decision API with JWT Auth, Registration & RBAC.

Endpoints:
- POST /auth/register: Public registration (role='front_desk', status='pending').
- POST /auth/login: Rate-limited JWT authentication (status='approved' required).
- GET  /auth/me: Current user profile.
- GET  /admin/requests: Manager-only view of access requests.
- POST /admin/requests/{id}/approve: Manager-only approval of pending accounts.
- POST /admin/requests/{id}/reject: Manager-only rejection of access requests.
- GET  /health: Service and artifact status check.
- GET  /hotel/info: Physical room capacities, ADRs, and cost parameters.
- POST /predict: Multi-horizon room demand forecast + 80% conformal interval (Roles: manager, front_desk).
- POST /recommend: Asymmetric Newsvendor inventory controls (Role: manager ONLY; front_desk -> 403).
"""

from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
import joblib
import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.auth import (
    authenticate_user,
    check_rate_limit,
    create_access_token,
    get_all_users,
    get_current_user,
    get_user_by_id,
    init_user_db,
    record_failed_attempt,
    register_user,
    require_role,
    reset_failed_attempts,
    update_user_status,
)
from src.config import (
    ARTIFACTS_DIR,
    CONFORMAL_ARTIFACT_PATH,
    MODEL_ARTIFACT_PATH,
    ROOM_CATEGORIES,
)
from src.data.generate_data import get_festival_multiplier
from src.decision.optimizer import NewsvendorDecisionLayer
from src.models.conformal import SplitConformalCalibrator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ml_state = {}


def ensure_artifacts_loaded():
    """Ensure artifacts and user DB are loaded into memory."""
    init_user_db()
    if "pipeline" in ml_state:
        return

    logger.info("Loading artifacts into ml_state...")
    if not MODEL_ARTIFACT_PATH.exists():
        raise FileNotFoundError(f"Model artifact not found at {MODEL_ARTIFACT_PATH}. Run training first.")
    if not CONFORMAL_ARTIFACT_PATH.exists():
        raise FileNotFoundError(f"Conformal artifact not found at {CONFORMAL_ARTIFACT_PATH}.")

    model_payload = joblib.load(MODEL_ARTIFACT_PATH)
    ml_state["pipeline"] = model_payload["pipeline"]
    ml_state["feature_cols"] = model_payload["feature_cols"]
    ml_state["model_name"] = model_payload["model_name"]
    ml_state["conformal"] = SplitConformalCalibrator.load(CONFORMAL_ARTIFACT_PATH)

    lookup_path = ARTIFACTS_DIR / "pickup_lookup.joblib"
    if lookup_path.exists():
        ml_state["pickup_lookup"] = joblib.load(lookup_path)
    else:
        ml_state["pickup_lookup"] = {}

    ml_state["optimizer"] = NewsvendorDecisionLayer()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle handler for startup and shutdown."""
    ensure_artifacts_loaded()
    logger.info("FastAPI service successfully loaded %s and conformal calibrator.", ml_state["model_name"])
    yield
    ml_state.clear()


app = FastAPI(
    title="Hotel Demand Forecasting & Inventory Optimization API",
    description="Multi-horizon demand forecasting with JWT Authentication, Public Registration, and Manager-Admin Approvals.",
    version="2.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==============================================================================
# SCHEMAS
# ==============================================================================
class RegisterRequest(BaseModel):
    name: str = Field(..., json_schema_extra={"example": "Jane Doe"}, description="Full Name")
    email: str = Field(..., json_schema_extra={"example": "jane@hotel.com"}, description="Work Email")
    password: str = Field(..., min_length=8, json_schema_extra={"example": "StrongPass@123"}, description="Password (>= 8 chars)")
    confirm_password: str = Field(..., min_length=8, json_schema_extra={"example": "StrongPass@123"}, description="Confirm Password")


class RegisterResponse(BaseModel):
    id: int
    name: str
    email: str
    role: str
    status: str
    message: str


class LoginRequest(BaseModel):
    email: str = Field(..., json_schema_extra={"example": "manager@hotel.com"}, description="User email")
    password: str = Field(..., json_schema_extra={"example": "ManagerPass@123"}, description="User password")


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    role: str
    email: str
    name: str
    user_id: int
    expires_in_minutes: int = 60


class UserProfileResponse(BaseModel):
    id: int
    name: str
    email: str
    role: str
    status: str


class UserAdminView(BaseModel):
    id: int
    name: str
    email: str
    role: str
    status: str
    created_at: Optional[str] = None


class PredictRequest(BaseModel):
    stay_date: str = Field(..., json_schema_extra={"example": "2025-11-15"}, description="Target arrival date (YYYY-MM-DD)")
    room_category: str = Field(..., json_schema_extra={"example": "Standard"}, description="Category: Standard, Deluxe, or Suite")
    horizon: int = Field(7, ge=1, le=14, description="Forecast horizon in days prior to arrival")
    otb_rooms: Optional[int] = Field(None, ge=0, description="Current confirmed On-The-Books rooms")
    pickup_last_3d: Optional[int] = Field(None, ge=0, description="New reservations booked in past 3 days")
    pickup_last_7d: Optional[int] = Field(None, ge=0, description="New reservations booked in past 7 days")
    cancellations_to_date: Optional[int] = Field(None, ge=0, description="Cumulative cancellations recorded")


ForecastRequest = PredictRequest


class PredictResponse(BaseModel):
    stay_date: str
    as_of_date: str
    room_category: str
    horizon_days: int
    physical_capacity: int
    otb_rooms: int
    predicted_demand: float
    conformal_interval_80: Tuple[float, float]
    uncertainty_note: str


class RecommendResponse(BaseModel):
    stay_date: str
    as_of_date: str
    room_category: str
    horizon_days: int
    physical_capacity: int
    predicted_demand: float
    conformal_interval_80: Tuple[float, float]
    critical_ratio: float
    underage_cost_inr: float
    overage_cost_inr: float
    authorized_booking_limit: int
    authorized_overbooking_rooms: int
    rooms_to_hold: int
    available_discount_rooms: int
    action_text: str


# ==============================================================================
# HELPER: FEATURE VECTOR CONSTRUCTION
# ==============================================================================
def construct_feature_row(req: PredictRequest) -> Tuple[pd.DataFrame, datetime, int]:
    """Build single-row DataFrame matching the exact training feature schema."""
    ensure_artifacts_loaded()
    if req.room_category not in ROOM_CATEGORIES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid room_category '{req.room_category}'. Allowed: {list(ROOM_CATEGORIES.keys())}"
        )

    try:
        stay_dt = datetime.strptime(req.stay_date, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid stay_date format. Must be YYYY-MM-DD.")

    as_of_dt = stay_dt - pd.Timedelta(days=req.horizon)
    spec = ROOM_CATEGORIES[req.room_category]
    capacity = spec["capacity"]
    base_adr = spec["base_adr"]

    stay_dow = stay_dt.weekday()
    stay_is_weekend = int(stay_dow in [4, 5])
    stay_month = stay_dt.month
    stay_quarter = (stay_month - 1) // 3 + 1
    stay_day = stay_dt.day

    day_of_year = stay_dt.timetuple().tm_yday
    stay_sin_doy = float(np.sin(2 * np.pi * day_of_year / 365.25))
    stay_cos_doy = float(np.cos(2 * np.pi * day_of_year / 365.25))

    fest_mult, fest_name = get_festival_multiplier(stay_dt.date())
    is_festival = int(fest_mult > 1.0)

    lookup_key = (req.room_category, stay_dow, req.horizon)
    hist_avg_pickup = ml_state["pickup_lookup"].get(lookup_key, float(req.horizon * 0.8))

    if req.otb_rooms is not None:
        otb = req.otb_rooms
    else:
        otb = max(1, int(capacity * 0.78 - hist_avg_pickup))

    otb_ratio = otb / capacity
    p3 = req.pickup_last_3d if req.pickup_last_3d is not None else max(0, int(otb * 0.15))
    p7 = req.pickup_last_7d if req.pickup_last_7d is not None else max(0, int(otb * 0.30))
    canc_td = req.cancellations_to_date if req.cancellations_to_date is not None else max(0, int(otb * 0.12))
    canc_rate = canc_td / (otb + canc_td + 1e-5)

    mean_cap = capacity * 0.78
    row_dict = {
        "room_category": req.room_category,
        "horizon": req.horizon,
        "otb_rooms": otb,
        "otb_occupancy_ratio": otb_ratio,
        "pickup_last_3d": p3,
        "pickup_last_7d": p7,
        "cancellations_to_date": canc_td,
        "cancellation_rate_otb": canc_rate,
        "capacity": capacity,
        "base_adr": base_adr,
        "stay_dow": stay_dow,
        "stay_is_weekend": stay_is_weekend,
        "stay_month": stay_month,
        "stay_quarter": stay_quarter,
        "stay_day_of_month": stay_day,
        "stay_sin_doy": stay_sin_doy,
        "stay_cos_doy": stay_cos_doy,
        "festival_uplift_factor": fest_mult,
        "is_festival": is_festival,
        "lag_T0": mean_cap,
        "lag_T1": mean_cap,
        "lag_T7": mean_cap,
        "lag_T14": mean_cap,
        "roll_mean_7d": mean_cap,
        "roll_mean_14d": mean_cap,
        "roll_std_7d": 2.0,
        "hist_avg_pickup": hist_avg_pickup,
    }

    return pd.DataFrame([row_dict]), as_of_dt, otb


# ==============================================================================
# AUTHENTICATION & REGISTRATION ENDPOINTS
# ==============================================================================
@app.post("/auth/register", response_model=RegisterResponse, status_code=status.HTTP_201_CREATED)
def register(req: RegisterRequest):
    """
    Public registration endpoint.
    Creates account with role='front_desk' and status='pending'.
    Client can never choose or elevate role.
    """
    ensure_artifacts_loaded()
    result = register_user(
        name=req.name,
        email=req.email,
        password=req.password,
        confirm_password=req.confirm_password,
    )
    return RegisterResponse(**result)


@app.post("/auth/login", response_model=TokenResponse)
def login(login_req: LoginRequest, request: Request):
    """
    Rate-limited login endpoint (5 failed attempts per minute per email/IP).
    Only status='approved' users receive JWT tokens.
    """
    ensure_artifacts_loaded()
    # Rate limit key combines client IP and email
    client_ip = request.client.host if request.client else "unknown"
    rate_key = f"{client_ip}:{login_req.email.strip().lower()}"

    # Check sliding window rate limit
    check_rate_limit(rate_key, max_attempts=5, window_seconds=60)

    user, failure_reason = authenticate_user(login_req.email, login_req.password)

    if failure_reason == "invalid_credentials":
        record_failed_attempt(rate_key)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if failure_reason == "pending":
        # Clear password, but reject with clear message
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account registration is pending approval by a hotel manager.",
        )

    if failure_reason == "rejected":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account registration was rejected. Please contact your administrator.",
        )

    # Success: reset rate limiter
    reset_failed_attempts(rate_key)
    token = create_access_token(data={
        "sub": user["email"],
        "role": user["role"],
        "name": user["name"],
        "user_id": user["id"],
    })

    return TokenResponse(
        access_token=token,
        token_type="bearer",
        role=user["role"],
        email=user["email"],
        name=user["name"],
        user_id=user["id"],
        expires_in_minutes=60,
    )


@app.get("/auth/me", response_model=UserProfileResponse)
def get_my_profile(current_user: Dict = Depends(get_current_user)):
    """Return profile and role of currently authenticated user."""
    return UserProfileResponse(**current_user)


# ==============================================================================
# MANAGER-ONLY ADMIN ENDPOINTS (ACCESS REQUEST APPROVALS)
# ==============================================================================
@app.get("/admin/requests", response_model=List[UserAdminView])
def list_access_requests(current_user: Dict = Depends(require_role(["manager"]))):
    """
    Manager-only endpoint: view all user registration requests and current approval statuses.
    """
    users = get_all_users()
    return [UserAdminView(**u) for u in users]


@app.post("/admin/requests/{user_id}/approve", response_model=UserAdminView)
def approve_access_request(
    user_id: int,
    current_user: Dict = Depends(require_role(["manager"]))
):
    """
    Manager-only endpoint: approve a pending user account, granting login access.
    """
    user = get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404, detail=f"User with ID {user_id} not found.")

    updated = update_user_status(user_id, "approved")
    logger.info("Manager '%s' APPROVED user id=%d (%s)", current_user["email"], user_id, user["email"])
    return UserAdminView(**updated)


@app.post("/admin/requests/{user_id}/reject", response_model=UserAdminView)
def reject_access_request(
    user_id: int,
    current_user: Dict = Depends(require_role(["manager"]))
):
    """
    Manager-only endpoint: reject a user access request.
    """
    user = get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404, detail=f"User with ID {user_id} not found.")

    updated = update_user_status(user_id, "rejected")
    logger.info("Manager '%s' REJECTED user id=%d (%s)", current_user["email"], user_id, user["email"])
    return UserAdminView(**updated)


# ==============================================================================
# PUBLIC INFORMATION ENDPOINTS
# ==============================================================================
@app.get("/health")
def health_check():
    """Service health and loaded artifact confirmation."""
    return {
        "status": "healthy",
        "model_loaded": "pipeline" in ml_state,
        "model_name": ml_state.get("model_name", "None"),
        "conformal_calibrator_loaded": "conformal" in ml_state,
        "hotel_total_capacity": 110,
    }


@app.get("/hotel/info")
def get_hotel_info():
    """Hotel room inventory, base pricing, and cost parameters."""
    return {
        "total_capacity": 110,
        "room_categories": ROOM_CATEGORIES,
    }


# ==============================================================================
# PROTECTED FORECASTING & DECISION ENDPOINTS
# ==============================================================================
@app.post("/predict", response_model=PredictResponse)
def predict_demand(
    req: PredictRequest,
    current_user: Dict = Depends(require_role(["manager", "front_desk"]))
):
    """
    Generate room demand point forecast and 80% conformal uncertainty interval.
    Accessible by: manager and front_desk.
    """
    feature_df, as_of_dt, otb = construct_feature_row(req)
    pipeline = ml_state["pipeline"]
    calibrator: SplitConformalCalibrator = ml_state["conformal"]

    pred_raw = float(pipeline.predict(feature_df[ml_state["feature_cols"]])[0])
    capacity = ROOM_CATEGORIES[req.room_category]["capacity"]
    pred = float(np.clip(pred_raw, 0.0, capacity * 1.25))

    lower_arr, upper_arr = calibrator.predict_interval(feature_df, np.array([pred]))
    lower, upper = round(float(lower_arr[0]), 1), round(float(upper_arr[0]), 1)
    width = upper - lower

    uncertainty_note = (
        f"80% Conformal Interval: [{lower} to {upper} rooms] (Width: {width:.1f} rooms). "
        f"Based on empirical validation residuals at {req.horizon}-day lead time, there is an 80% probability "
        f"that final realized demand will fall within this range."
    )

    return PredictResponse(
        stay_date=req.stay_date,
        as_of_date=as_of_dt.strftime("%Y-%m-%d"),
        room_category=req.room_category,
        horizon_days=req.horizon,
        physical_capacity=capacity,
        otb_rooms=otb,
        predicted_demand=round(pred, 1),
        conformal_interval_80=(lower, upper),
        uncertainty_note=uncertainty_note,
    )


@app.post("/recommend", response_model=RecommendResponse)
def recommend_inventory_action(
    req: PredictRequest,
    current_user: Dict = Depends(require_role(["manager"]))
):
    """
    Generate asymmetric Newsvendor inventory controls (rooms to hold / overbook).
    Accessible by: manager ONLY.
    Attempting access as front_desk raises HTTP 403 Forbidden.
    """
    feature_df, as_of_dt, otb = construct_feature_row(req)
    pipeline = ml_state["pipeline"]
    calibrator: SplitConformalCalibrator = ml_state["conformal"]
    optimizer: NewsvendorDecisionLayer = ml_state["optimizer"]

    pred_raw = float(pipeline.predict(feature_df[ml_state["feature_cols"]])[0])
    capacity = ROOM_CATEGORIES[req.room_category]["capacity"]
    pred = float(np.clip(pred_raw, 0.0, capacity * 1.25))

    lower_arr, upper_arr = calibrator.predict_interval(feature_df, np.array([pred]))
    lower, upper = round(float(lower_arr[0]), 1), round(float(upper_arr[0]), 1)

    decision_dict = optimizer.optimize_decision(
        room_category=req.room_category,
        point_forecast=pred,
        conformal_lower=lower,
        conformal_upper=upper,
        otb_rooms=otb,
        horizon=req.horizon,
    )

    clean_dec = {k: v for k, v in decision_dict.items() if k not in ["capacity", "point_forecast", "conformal_interval", "room_category"]}

    return RecommendResponse(
        stay_date=req.stay_date,
        as_of_date=as_of_dt.strftime("%Y-%m-%d"),
        room_category=req.room_category,
        horizon_days=req.horizon,
        physical_capacity=capacity,
        predicted_demand=round(pred, 1),
        conformal_interval_80=(lower, upper),
        **clean_dec,
    )
