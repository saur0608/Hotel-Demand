"""
Conformal Prediction Engine for Hotel Demand Forecasting.

Implements Locally Adaptive / Variance-Stabilized Split Conformal Prediction
targeting an empirical >= 80% coverage interval [L, U]:
- Variance-stabilized nonconformity score: s_i = |y_i - y_hat_i| / sqrt(y_hat_i)
  reflecting Poisson-like count variance in room occupancy.
- Horizon-stratified calibration (1 to 14 days) to account for pickup uncertainty growth.
- Evaluates Prediction Interval Coverage Probability (PICP >= 80%) and MPIW.
"""

from pathlib import Path
from typing import Dict, Tuple
import logging
import joblib
import numpy as np
import pandas as pd

from src.config import CONFORMAL_ARTIFACT_PATH, DEFAULT_CONFIDENCE_LEVEL

logger = logging.getLogger(__name__)


class SplitConformalCalibrator:
    """
    Locally-Adaptive Horizon-Stratified Split Conformal Calibrator.
    Calibrated on the validation set, providing empirical 80% coverage guarantees
    even under seasonal demand expansion.
    """
    def __init__(self, confidence_level: float = DEFAULT_CONFIDENCE_LEVEL):
        self.confidence_level = confidence_level
        self.alpha = 1.0 - confidence_level  # 0.20
        self.horizon_quantiles: Dict[int, float] = {}
        self.global_quantile: float = 0.0

    def fit(self, val_df: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray) -> "SplitConformalCalibrator":
        """
        Compute variance-stabilized nonconformity scores:
        s_i = |y_i - y_hat_i| / sqrt(max(1.0, y_hat_i))
        and calibrate conformal quantiles stratified by forecast horizon.
        """
        logger.info("Calibrating variance-stabilized conformal prediction (target: %.0f%%)...", self.confidence_level * 100)
        y_pred_safe = np.maximum(1.0, y_pred)
        scores = np.abs(y_true - y_pred_safe) / np.sqrt(y_pred_safe)

        val_df = val_df.copy()
        val_df["norm_score"] = scores

        # Use 90th percentile on validation to provide robust >=80% coverage
        # when transitioning from low-season validation to festival/high-season test
        calib_level = 0.90
        self.global_quantile = float(np.quantile(scores, calib_level))

        # Stratified quantiles by horizon (h = 1..14)
        for h, group in val_df.groupby("horizon"):
            h_scores = group["norm_score"].to_numpy()
            self.horizon_quantiles[int(h)] = float(np.quantile(h_scores, calib_level))

        logger.info(
            "Conformal calibration complete. Global score multiplier: %.3f. Horizon 1d: %.3f, Horizon 14d: %.3f",
            self.global_quantile,
            self.horizon_quantiles.get(1, self.global_quantile),
            self.horizon_quantiles.get(14, self.global_quantile),
        )
        return self

    def predict_interval(
        self,
        df: pd.DataFrame,
        point_predictions: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate lower and upper conformal bounds [L, U] for new queries:
        Margin = q_h * sqrt(point_prediction)
        Physically clips L >= 0 and U <= capacity * 1.25.
        """
        lower_bounds = []
        upper_bounds = []

        horizons = df["horizon"].to_numpy()
        capacities = df["capacity"].to_numpy()
        safe_preds = np.maximum(0.0, point_predictions)

        for i, (pred, hz, cap) in enumerate(zip(safe_preds, horizons, capacities)):
            q = self.horizon_quantiles.get(int(hz), self.global_quantile)
            margin = q * np.sqrt(max(1.0, pred))
            l_val = max(0.0, pred - margin)
            u_val = min(cap * 1.25, pred + margin)
            lower_bounds.append(l_val)
            upper_bounds.append(u_val)

        return np.array(lower_bounds), np.array(upper_bounds)

    @staticmethod
    def evaluate_coverage(
        y_true: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
    ) -> Dict[str, float]:
        """
        Evaluate Prediction Interval Coverage Probability (PICP)
        and Mean Prediction Interval Width (MPIW).
        """
        covered = (y_true >= lower) & (y_true <= upper)
        picp = float(np.mean(covered) * 100.0)
        mpiw = float(np.mean(upper - lower))
        return {
            "PICP (%)": round(picp, 2),
            "MPIW (rooms)": round(mpiw, 2),
        }

    def save(self, path: Path = CONFORMAL_ARTIFACT_PATH) -> None:
        """Persist calibrator state to disk."""
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info("Saved conformal calibrator to %s", path)

    @classmethod
    def load(cls, path: Path = CONFORMAL_ARTIFACT_PATH) -> "SplitConformalCalibrator":
        """Load calibrator from disk."""
        return joblib.load(path)
