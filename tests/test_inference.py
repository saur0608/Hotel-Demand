"""
Inference & API Endpoint Integration Test Suite.

Verifies:
1. Model artifact loading from disk (joblib).
2. Dynamic inference: predictions respond to input changes (no hardcoding).
3. Conformal prediction interval mathematical validity (lower <= upper, lower >= 0).
4. Decision layer Newsvendor action properties.
5. FastAPI endpoints (/health, /hotel/info, /forecast) via TestClient.
"""

import joblib
import numpy as np
import pandas as pd
import pytest
from starlette.testclient import TestClient

from app.api import app as test_app, construct_feature_row, ForecastRequest

test_app.__test__ = False
from src.config import (
    CONFORMAL_ARTIFACT_PATH,
    MODEL_ARTIFACT_PATH,
    ROOM_CATEGORIES,
)
from src.decision.optimizer import NewsvendorDecisionLayer
from src.models.conformal import SplitConformalCalibrator


@pytest.fixture(scope="module")
def model_artifacts():
    """Load saved artifacts from disk."""
    assert MODEL_ARTIFACT_PATH.exists(), f"Model missing: {MODEL_ARTIFACT_PATH}"
    assert CONFORMAL_ARTIFACT_PATH.exists(), f"Conformal calibrator missing: {CONFORMAL_ARTIFACT_PATH}"

    payload = joblib.load(MODEL_ARTIFACT_PATH)
    calibrator = SplitConformalCalibrator.load(CONFORMAL_ARTIFACT_PATH)
    return payload, calibrator


def test_artifacts_loaded_and_predict_dynamically(model_artifacts):
    """Verify that predictions dynamically respond to different input values."""
    payload, calibrator = model_artifacts
    pipeline = payload["pipeline"]

    req_low = ForecastRequest(
        stay_date="2025-11-15",
        room_category="Standard",
        horizon=7,
        otb_rooms=10,
    )
    req_high = ForecastRequest(
        stay_date="2025-11-15",
        room_category="Standard",
        horizon=7,
        otb_rooms=55,
    )

    df_low, _, _ = construct_feature_row(req_low)
    df_high, _, _ = construct_feature_row(req_high)

    pred_low = float(pipeline.predict(df_low[payload["feature_cols"]])[0])
    pred_high = float(pipeline.predict(df_high[payload["feature_cols"]])[0])

    assert pred_high > pred_low, f"Model not responding dynamically: low={pred_low}, high={pred_high}"


def test_conformal_interval_validity(model_artifacts):
    """Verify conformal intervals: lower <= upper, lower >= 0."""
    payload, calibrator = model_artifacts
    pipeline = payload["pipeline"]

    req = ForecastRequest(
        stay_date="2025-11-15",
        room_category="Deluxe",
        horizon=7,
        otb_rooms=25,
    )
    df, _, _ = construct_feature_row(req)
    pred = float(pipeline.predict(df[payload["feature_cols"]])[0])

    lower_arr, upper_arr = calibrator.predict_interval(df, np.array([pred]))
    lower, upper = float(lower_arr[0]), float(upper_arr[0])

    assert lower >= 0.0, f"Conformal lower bound negative: {lower}"
    assert lower <= upper, f"Conformal lower ({lower}) > upper ({upper})"
    assert upper <= ROOM_CATEGORIES["Deluxe"]["capacity"] * 1.3


def test_decision_layer_outputs():
    """Verify that Newsvendor decision layer produces mathematically sound actions."""
    optimizer = NewsvendorDecisionLayer()
    dec = optimizer.optimize_decision(
        room_category="Standard",
        point_forecast=58.0,
        conformal_lower=52.0,
        conformal_upper=64.0,
        otb_rooms=45,
        horizon=7,
    )

    assert dec["authorized_overbooking_rooms"] >= 0
    assert dec["rooms_to_hold"] >= 0
    assert dec["authorized_booking_limit"] >= ROOM_CATEGORIES["Standard"]["capacity"]
    assert "action_text" in dec and len(dec["action_text"]) > 10


def test_fastapi_endpoints():
    """Test FastAPI service endpoints using Starlette TestClient."""
    from src.config import DEMO_MANAGER_EMAIL, DEMO_MANAGER_PASSWORD

    with TestClient(test_app) as client:
        # 1. Health check
        res_health = client.get("/health")
        assert res_health.status_code == 200
        assert res_health.json()["status"] == "healthy"

        # 2. Hotel Info
        res_info = client.get("/hotel/info")
        assert res_info.status_code == 200
        assert "Standard" in res_info.json()["room_categories"]

        # 3. Login to get token
        login_res = client.post("/auth/login", json={
            "email": DEMO_MANAGER_EMAIL,
            "password": DEMO_MANAGER_PASSWORD,
        })
        assert login_res.status_code == 200
        token = login_res.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        # 4. Valid Predict Request
        payload = {
            "stay_date": "2025-11-15",
            "room_category": "Standard",
            "horizon": 7,
            "otb_rooms": 40,
        }
        res_pred = client.post("/predict", json=payload, headers=headers)
        assert res_pred.status_code == 200
        body_pred = res_pred.json()
        assert "predicted_demand" in body_pred
        assert "conformal_interval_80" in body_pred
        assert len(body_pred["conformal_interval_80"]) == 2

        # 5. Valid Recommend Request (manager)
        res_rec = client.post("/recommend", json=payload, headers=headers)
        assert res_rec.status_code == 200
        body_rec = res_rec.json()
        assert "authorized_overbooking_rooms" in body_rec
        assert "rooms_to_hold" in body_rec

        # 6. Invalid Room Category -> 400 Bad Request
        bad_payload = {
            "stay_date": "2025-11-15",
            "room_category": "PenthouseNonExistent",
            "horizon": 7,
        }
        res_bad = client.post("/predict", json=bad_payload, headers=headers)
        assert res_bad.status_code == 400
