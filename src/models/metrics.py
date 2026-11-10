"""
Evaluation Metrics for Hotel Room-Demand Forecasting.

Implements standard regression metrics and domain-specific hospitality metrics:
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- WAPE (Weighted Absolute Percentage Error = sum(|y - y_hat|) / sum(y))
Supports slicing by room category and forecast horizon (1 to 14 days).
"""

from typing import Dict, List
import numpy as np
import pandas as pd
from tabulate import tabulate


def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(y_true - y_pred)))


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def compute_wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Weighted Absolute Percentage Error (WAPE).
    WAPE = sum(|y - y_hat|) / sum(y).
    Preferred in hospitality over MAPE because it avoids division-by-zero on low-demand days.
    """
    total_actual = np.sum(y_true)
    if total_actual == 0:
        return 0.0
    return float(np.sum(np.abs(y_true - y_pred)) / total_actual)


def calculate_metrics_dict(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate MAE, RMSE, and WAPE as a dictionary."""
    return {
        "MAE": round(compute_mae(y_true, y_pred), 3),
        "RMSE": round(compute_rmse(y_true, y_pred), 3),
        "WAPE (%)": round(compute_wape(y_true, y_pred) * 100.0, 2),
    }


def compute_sliced_metrics(
    eval_df: pd.DataFrame,
    y_true_col: str = "target_net_demand",
    y_pred_col: str = "prediction",
) -> Dict[str, pd.DataFrame]:
    """
    Compute metrics broken down by:
    1. Overall
    2. Room Category (Standard, Deluxe, Suite)
    3. Forecast Horizon (1 to 14 days)
    4. Room Category x Horizon matrix
    """
    results = {}

    # 1. Overall
    overall = calculate_metrics_dict(eval_df[y_true_col].to_numpy(), eval_df[y_pred_col].to_numpy())
    results["overall"] = pd.DataFrame([overall])

    # 2. By Room Category
    cat_rows = []
    for cat, group in eval_df.groupby("room_category"):
        m = calculate_metrics_dict(group[y_true_col].to_numpy(), group[y_pred_col].to_numpy())
        m["room_category"] = cat
        cat_rows.append(m)
    results["by_category"] = pd.DataFrame(cat_rows)[["room_category", "MAE", "RMSE", "WAPE (%)"]]

    # 3. By Horizon
    hz_rows = []
    for hz, group in eval_df.groupby("horizon"):
        m = calculate_metrics_dict(group[y_true_col].to_numpy(), group[y_pred_col].to_numpy())
        m["horizon"] = hz
        hz_rows.append(m)
    results["by_horizon"] = pd.DataFrame(hz_rows)[["horizon", "MAE", "RMSE", "WAPE (%)"]]

    return results


def format_metrics_table(metrics_df: pd.DataFrame, title: str = "") -> str:
    """Format DataFrame as a clean Markdown table."""
    table_str = tabulate(metrics_df, headers="keys", tablefmt="github", showindex=False)
    if title:
        return f"\n### {title}\n{table_str}\n"
    return table_str
