"""
Candidate Forecasting Models: Ridge Regression & Gradient Boosting.

Modular scikit-learn pipelines with column transformers, standard scaling,
and hyperparameter tuning strictly on the validation partition.
Supports LightGBM if libomp is available, with seamless fallback to
HistGradientBoostingRegressor (scikit-learn's native histogram-based LightGBM algorithm).
"""

from typing import Dict, List, Tuple
import logging
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Try importing LightGBM; fallback to HistGradientBoostingRegressor if libomp is missing
USE_LIGHTGBM = False
try:
    from lightgbm import LGBMRegressor
    USE_LIGHTGBM = True
except (ImportError, OSError):
    from sklearn.ensemble import HistGradientBoostingRegressor

from src.config import RANDOM_SEED
from src.models.metrics import compute_wape

logger = logging.getLogger(__name__)

# Feature definitions
CATEGORICAL_FEATURES = ["room_category"]
NUMERICAL_FEATURES = [
    "horizon",
    "otb_rooms",
    "otb_occupancy_ratio",
    "pickup_last_3d",
    "pickup_last_7d",
    "cancellations_to_date",
    "cancellation_rate_otb",
    "capacity",
    "base_adr",
    "stay_dow",
    "stay_is_weekend",
    "stay_month",
    "stay_quarter",
    "stay_day_of_month",
    "stay_sin_doy",
    "stay_cos_doy",
    "festival_uplift_factor",
    "is_festival",
    "lag_T0",
    "lag_T1",
    "lag_T7",
    "lag_T14",
    "roll_mean_7d",
    "roll_mean_14d",
    "roll_std_7d",
    "hist_avg_pickup",
]

ALL_FEATURE_COLS = CATEGORICAL_FEATURES + NUMERICAL_FEATURES
TARGET_COL = "target_net_demand"


def get_preprocessor() -> ColumnTransformer:
    """Build column transformer for linear model scaling and encoding."""
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERICAL_FEATURES),
            ("cat", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )


def train_and_tune_ridge(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
) -> Tuple[Pipeline, Dict]:
    """
    Train Ridge regression and tune regularization parameter alpha on validation WAPE.
    """
    logger.info("Tuning Ridge regression on validation partition...")
    alphas = [0.01, 0.1, 1.0, 10.0, 50.0, 100.0, 500.0]
    best_alpha = 1.0
    best_wape = float("inf")
    best_pipe = None

    for alpha in alphas:
        pipe = Pipeline([
            ("preprocessor", get_preprocessor()),
            ("regressor", Ridge(alpha=alpha, random_state=RANDOM_SEED)),
        ])
        pipe.fit(X_train[ALL_FEATURE_COLS], y_train)
        val_preds = np.maximum(0, pipe.predict(X_val[ALL_FEATURE_COLS]))
        wape = compute_wape(y_val, val_preds)

        if wape < best_wape:
            best_wape = wape
            best_alpha = alpha
            best_pipe = pipe

    logger.info("Best Ridge alpha: %s (Val WAPE: %.2f%%)", best_alpha, best_wape * 100)
    return best_pipe, {"alpha": best_alpha, "val_wape": best_wape}


def train_and_tune_lightgbm(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
) -> Tuple[Pipeline, Dict]:
    """
    Train Gradient Boosting Regressor (LightGBM or HistGradientBoosting)
    and select best hyperparameter configuration on validation WAPE.
    """
    global USE_LIGHTGBM
    logger.info("Tuning Gradient Boosting (LightGBM: %s) on validation partition...", USE_LIGHTGBM)

    if USE_LIGHTGBM:
        param_grid = [
            {"n_estimators": 100, "learning_rate": 0.05, "num_leaves": 31, "min_child_samples": 20},
            {"n_estimators": 150, "learning_rate": 0.05, "num_leaves": 45, "min_child_samples": 15},
            {"n_estimators": 200, "learning_rate": 0.03, "num_leaves": 31, "min_child_samples": 25},
            {"n_estimators": 150, "learning_rate": 0.08, "num_leaves": 25, "min_child_samples": 30},
        ]
    else:
        from sklearn.ensemble import HistGradientBoostingRegressor
        param_grid = [
            {"max_iter": 100, "learning_rate": 0.05, "max_leaf_nodes": 31, "min_samples_leaf": 20},
            {"max_iter": 150, "learning_rate": 0.05, "max_leaf_nodes": 45, "min_samples_leaf": 15},
            {"max_iter": 200, "learning_rate": 0.03, "max_leaf_nodes": 31, "min_samples_leaf": 25},
            {"max_iter": 150, "learning_rate": 0.08, "max_leaf_nodes": 25, "min_samples_leaf": 30},
        ]

    best_params = param_grid[0]
    best_wape = float("inf")
    best_pipe = None

    for params in param_grid:
        if USE_LIGHTGBM:
            regressor = LGBMRegressor(
                **params,
                objective="regression",
                random_state=RANDOM_SEED,
                verbose=-1,
                n_jobs=-1,
            )
        else:
            regressor = HistGradientBoostingRegressor(
                **params,
                random_state=RANDOM_SEED,
            )

        pipe = Pipeline([
            ("preprocessor", get_preprocessor()),
            ("regressor", regressor),
        ])
        pipe.fit(X_train[ALL_FEATURE_COLS], y_train)
        val_preds = np.maximum(0, pipe.predict(X_val[ALL_FEATURE_COLS]))
        wape = compute_wape(y_val, val_preds)

        if wape < best_wape:
            best_wape = wape
            best_params = params
            best_pipe = pipe

    model_engine = "LightGBM" if USE_LIGHTGBM else "HistGradientBoosting"
    logger.info("Best %s params: %s (Val WAPE: %.2f%%)", model_engine, best_params, best_wape * 100)
    return best_pipe, {**best_params, "engine": model_engine, "val_wape": best_wape}
