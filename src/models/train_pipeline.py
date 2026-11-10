"""
End-to-End Training, Validation Tuning, Conformal Uncertainty, Backtesting & Explainability Pipeline.

Workflow:
1. Load leakage-safe feature matrix.
2. Train & tune candidate models (Ridge & LightGBM) strictly on Validation split.
3. Calibrate 80% Split-Conformal Prediction intervals on Validation split.
4. Backtest all baselines, candidates, and conformal intervals on untouched Test split.
5. Compute SHAP explainability, single-prediction breakdown, and failure post-mortem.
6. Persist winning model artifact and conformal calibrator to artifacts/.
"""

from pathlib import Path
from typing import Dict, Tuple
import logging
import joblib
import pandas as pd

from src.config import (
    CONFORMAL_ARTIFACT_PATH,
    DAILY_AGGREGATED_PARQUET,
    DEFAULT_CONFIDENCE_LEVEL,
    MODEL_ARTIFACT_PATH,
    RAW_BOOKINGS_CSV,
)
from src.explainability.shap_explain import (
    explain_large_error_example,
    explain_model_shap,
    explain_single_prediction_example,
    write_explainability_report,
)
from src.features.build_features import build_feature_dataset, compute_daily_realized_demand
from src.models.backtest import evaluate_all_models_on_test, plot_backtest_results
from src.models.candidate_models import (
    ALL_FEATURE_COLS,
    TARGET_COL,
    train_and_tune_lightgbm,
    train_and_tune_ridge,
)
from src.models.conformal import SplitConformalCalibrator
from src.models.metrics import format_metrics_table

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_training_pipeline() -> Tuple[pd.DataFrame, Dict]:
    """Execute complete model training, conformal calibration, and comparative evaluation."""
    logger.info("Starting model training pipeline...")

    # 1. Load feature dataset (or build if not present)
    if DAILY_AGGREGATED_PARQUET.exists():
        df = pd.read_parquet(DAILY_AGGREGATED_PARQUET)
    elif (DAILY_AGGREGATED_PARQUET.parent / "daily_demand_as_of.csv").exists():
        df = pd.read_csv(DAILY_AGGREGATED_PARQUET.parent / "daily_demand_as_of.csv")
    else:
        df = build_feature_dataset()

    # Load raw data to compute historical realized matrix for baseline lookups
    df_raw = pd.read_csv(RAW_BOOKINGS_CSV)
    daily_realized = compute_daily_realized_demand(df_raw)

    # 2. Strict Chronological Splits
    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()
    test_df = df[df["split"] == "test"].copy()

    logger.info(
        "Chronological splits loaded: Train=%d, Val=%d, Test=%d",
        len(train_df), len(val_df), len(test_df)
    )

    X_train, y_train = train_df[ALL_FEATURE_COLS], train_df[TARGET_COL].to_numpy()
    X_val, y_val = val_df[ALL_FEATURE_COLS], val_df[TARGET_COL].to_numpy()

    # 3. Model 1: Ridge Regression (tuned on validation)
    ridge_pipeline, ridge_info = train_and_tune_ridge(X_train, y_train, X_val, y_val)

    # 4. Model 2: Gradient Boosting (LightGBM/HistGradientBoosting, tuned on validation)
    lgbm_pipeline, lgbm_info = train_and_tune_lightgbm(X_train, y_train, X_val, y_val)

    # 5. Conformal Prediction Calibration on Validation Set
    val_lgbm_preds = lgbm_pipeline.predict(X_val)
    conformal_calibrator = SplitConformalCalibrator(confidence_level=DEFAULT_CONFIDENCE_LEVEL)
    conformal_calibrator.fit(val_df, y_val, val_lgbm_preds)
    conformal_calibrator.save(CONFORMAL_ARTIFACT_PATH)

    # 6. Out-of-Time Backtest Evaluation on Untouched Test Set
    evaluated_test_df, summary_tables = evaluate_all_models_on_test(
        df=df,
        daily_realized=daily_realized,
        trained_ridge_pipeline=ridge_pipeline,
        trained_lgbm_pipeline=lgbm_pipeline,
    )

    # 7. Apply Conformal Prediction Intervals to Test Set
    lower_bounds, upper_bounds = conformal_calibrator.predict_interval(
        evaluated_test_df,
        evaluated_test_df["pred_lightgbm"].to_numpy()
    )
    evaluated_test_df["conformal_lower"] = lower_bounds
    evaluated_test_df["conformal_upper"] = upper_bounds

    coverage_metrics = conformal_calibrator.evaluate_coverage(
        evaluated_test_df[TARGET_COL].to_numpy(),
        lower_bounds,
        upper_bounds
    )
    coverage_df = pd.DataFrame([coverage_metrics])

    # 8. Print Markdown Tables
    print(format_metrics_table(summary_tables["overall"], "Overall Model Performance on Untouched Test Set (Oct-Dec 2025)"))
    print(format_metrics_table(coverage_df, "80% Conformal Prediction Interval Calibration on Test Set"))
    print(format_metrics_table(summary_tables["by_category"], "Performance by Room Category (WAPE %)"))
    print(format_metrics_table(summary_tables["by_horizon"], "Performance by Forecast Horizon (WAPE %)"))

    # 9. Asymmetric Decision Layer Financial Evaluation
    from src.decision.optimizer import NewsvendorDecisionLayer
    optimizer = NewsvendorDecisionLayer()
    financial_eval_df = optimizer.evaluate_financial_loss(evaluated_test_df)
    print(format_metrics_table(financial_eval_df, "Decision Layer: Financial Loss & Risk Mitigation Under Asymmetric Costs"))

    # 10. Generate Comparison Plots
    plot_backtest_results(evaluated_test_df, summary_tables["by_horizon"])

    # 10. Explainability: SHAP + Worked Example + Failure Post-Mortem
    explain_model_shap(lgbm_pipeline, evaluated_test_df)
    worked_example = explain_single_prediction_example(lgbm_pipeline, evaluated_test_df)
    error_example = explain_large_error_example(lgbm_pipeline, evaluated_test_df)
    write_explainability_report(worked_example, error_example, coverage_metrics)

    # 11. Persist Winning Model Artifact
    MODEL_ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    artifact_payload = {
        "model_name": "GradientBoosting_Pickup_Forecaster",
        "pipeline": lgbm_pipeline,
        "feature_cols": ALL_FEATURE_COLS,
        "target_col": TARGET_COL,
        "tuning_info": lgbm_info,
        "ridge_tuning_info": ridge_info,
        "metrics_summary": summary_tables["overall"].to_dict(orient="records"),
        "conformal_metrics": coverage_metrics,
    }
    joblib.dump(artifact_payload, MODEL_ARTIFACT_PATH)
    logger.info("Persisted winning model pipeline artifact to %s", MODEL_ARTIFACT_PATH)

    return evaluated_test_df, summary_tables


if __name__ == "__main__":
    run_training_pipeline()
