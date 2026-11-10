"""
Rolling-Origin Backtesting and Comparative Model Evaluation.

Evaluates baselines, regularized linear models, and gradient boosting trees
over the untouched out-of-time test period with zero data leakage.
"""

from pathlib import Path
from typing import Dict, List, Tuple
import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import ARTIFACTS_DIR, ROOM_CATEGORIES
from src.models.baselines import (
    HistoricalPickupBaseline,
    NaiveSameWeekdayBaseline,
    SeasonalNaiveRollingBaseline,
)
from src.models.candidate_models import ALL_FEATURE_COLS, TARGET_COL
from src.models.metrics import calculate_metrics_dict, compute_sliced_metrics, format_metrics_table

logger = logging.getLogger(__name__)


def evaluate_all_models_on_test(
    df: pd.DataFrame,
    daily_realized: pd.DataFrame,
    trained_ridge_pipeline,
    trained_lgbm_pipeline,
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Run evaluation on the untouched test partition across all 5 models:
    1. Naive Same Weekday
    2. Seasonal Naive 4-Week
    3. Historical Pickup Baseline
    4. Ridge Regression
    5. LightGBM Regressor

    Returns:
        test_df with all model prediction columns, and dictionary of summary comparison tables.
    """
    logger.info("Executing comprehensive model evaluation on untouched test set...")
    test_df = df[df["split"] == "test"].copy()

    # 1. Baseline Predictions
    naive_model = NaiveSameWeekdayBaseline(daily_realized)
    test_df["pred_naive"] = np.maximum(0, naive_model.predict(test_df))

    snaive_model = SeasonalNaiveRollingBaseline(daily_realized)
    test_df["pred_seasonal_naive"] = np.maximum(0, snaive_model.predict(test_df))

    pickup_model = HistoricalPickupBaseline()
    test_df["pred_historical_pickup"] = np.maximum(0, pickup_model.predict(test_df))

    # 2. ML Candidate Model Predictions
    test_df["pred_ridge"] = np.maximum(0, trained_ridge_pipeline.predict(test_df[ALL_FEATURE_COLS]))
    test_df["pred_lightgbm"] = np.maximum(0, trained_lgbm_pipeline.predict(test_df[ALL_FEATURE_COLS]))

    # Physical capacity capping (predictions cannot physically exceed capacity by more than 20% overbooking)
    for model_col in ["pred_naive", "pred_seasonal_naive", "pred_historical_pickup", "pred_ridge", "pred_lightgbm"]:
        test_df[model_col] = np.clip(test_df[model_col], 0, test_df["capacity"] * 1.2)

    # 3. Compute Comparative Metrics Summary
    models = {
        "1. Naive (Same Weekday)": "pred_naive",
        "2. Seasonal Naive (4-Wk)": "pred_seasonal_naive",
        "3. Historical Pickup Baseline": "pred_historical_pickup",
        "4. Ridge Regression": "pred_ridge",
        "5. LightGBM Regressor": "pred_lightgbm",
    }

    # Overall table
    overall_rows = []
    y_true = test_df[TARGET_COL].to_numpy()

    for name, pred_col in models.items():
        m = calculate_metrics_dict(y_true, test_df[pred_col].to_numpy())
        m["Model"] = name
        overall_rows.append(m)

    overall_df = pd.DataFrame(overall_rows)[["Model", "MAE", "RMSE", "WAPE (%)"]]

    # Category breakdown table (WAPE %)
    cat_rows = []
    for cat in ROOM_CATEGORIES.keys():
        row = {"Room Category": cat}
        sub = test_df[test_df["room_category"] == cat]
        y_cat = sub[TARGET_COL].to_numpy()
        for name, pred_col in models.items():
            short_name = name.split(". ")[-1].split(" ")[0]
            m = calculate_metrics_dict(y_cat, sub[pred_col].to_numpy())
            row[f"{short_name} WAPE (%)"] = m["WAPE (%)"]
        cat_rows.append(row)
    category_df = pd.DataFrame(cat_rows)

    # Horizon breakdown table (WAPE %)
    hz_rows = []
    for hz in range(1, 15):
        row = {"Horizon (Days)": hz}
        sub = test_df[test_df["horizon"] == hz]
        y_hz = sub[TARGET_COL].to_numpy()
        for name, pred_col in models.items():
            short_name = name.split(". ")[-1].split(" ")[0]
            m = calculate_metrics_dict(y_hz, sub[pred_col].to_numpy())
            row[f"{short_name} WAPE (%)"] = m["WAPE (%)"]
        hz_rows.append(row)
    horizon_df = pd.DataFrame(hz_rows)

    summary_tables = {
        "overall": overall_df,
        "by_category": category_df,
        "by_horizon": horizon_df,
    }

    return test_df, summary_tables


def plot_backtest_results(test_df: pd.DataFrame, horizon_summary_df: pd.DataFrame) -> None:
    """Generate and save publication-grade backtesting comparison plots."""
    plots_dir = ARTIFACTS_DIR / "eda_plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Plot 1: WAPE vs Forecast Horizon for all models
    fig, ax = plt.subplots(figsize=(10, 5))
    hz = horizon_summary_df["Horizon (Days)"]
    cols_to_plot = [c for c in horizon_summary_df.columns if c != "Horizon (Days)"]

    palette = {
        "Naive WAPE (%)": "#8c564b",
        "Seasonal WAPE (%)": "#e377c2",
        "Historical WAPE (%)": "#2ca02c",
        "Ridge WAPE (%)": "#ff7f0e",
        "LightGBM WAPE (%)": "#1f77b4",
    }

    for col in cols_to_plot:
        label = col.replace(" WAPE (%)", "")
        color = palette.get(col, "gray")
        style = "--" if "Naive" in col or "Seasonal" in col else "-"
        width = 2.5 if "LightGBM" in col else 1.8
        ax.plot(hz, horizon_summary_df[col], marker="o", linestyle=style, color=color, linewidth=width, label=label)

    ax.set_title("Forecast Accuracy across Horizons: WAPE (%) on Untouched Test Set", fontsize=12, weight="bold")
    ax.set_xlabel("Forecast Horizon (Days ahead: 1 to 14)", fontsize=11)
    ax.set_ylabel("WAPE (%) [Lower is Better]", fontsize=11)
    ax.set_xticks(range(1, 15))
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(title="Model", frameon=True)
    fig.savefig(plots_dir / "06_model_comparison_by_horizon.png", dpi=200)
    plt.close(fig)

    # Plot 2: Actual vs Predicted Demand Trajectory on Test Period (7-day horizon)
    h7_df = test_df[test_df["horizon"] == 7].copy()
    h7_df = h7_df.sort_values("stay_date")

    # Aggregate total hotel occupied rooms per stay date
    daily_actual = h7_df.groupby("stay_date")[TARGET_COL].sum()
    daily_lgbm = h7_df.groupby("stay_date")["pred_lightgbm"].sum()
    daily_ridge = h7_df.groupby("stay_date")["pred_ridge"].sum()
    daily_naive = h7_df.groupby("stay_date")["pred_naive"].sum()

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(daily_actual.index, daily_actual.values, color="black", linewidth=2.2, label="Actual Occupied Rooms")
    ax.plot(daily_lgbm.index, daily_lgbm.values, color="#1f77b4", linewidth=2.0, label="LightGBM (h=7d)")
    ax.plot(daily_ridge.index, daily_ridge.values, color="#ff7f0e", linestyle=":", linewidth=1.8, label="Ridge (h=7d)")
    ax.plot(daily_naive.index, daily_naive.values, color="gray", linestyle="--", alpha=0.7, label="Naive Baseline")

    ax.set_title("Test Set Actual vs Forecasted Total Hotel Occupancy (7-Day Horizon: Oct - Dec 2025)", fontsize=12, weight="bold")
    ax.set_xlabel("Stay Date", fontsize=11)
    ax.set_ylabel("Total Occupied Rooms (Capacity: 110)", fontsize=11)
    ax.tick_params(axis="x", rotation=30)
    ax.axhline(110, color="darkred", linestyle="--", alpha=0.7, label="100% Hotel Capacity")
    ax.legend(loc="upper left")
    fig.savefig(plots_dir / "07_actual_vs_predicted_test_trajectory.png", dpi=200)
    plt.close(fig)

    logger.info("Saved backtest comparison plots to %s", plots_dir)
