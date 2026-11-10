"""
Hospitality Forecasting Baseline Models.

Provides non-ML operational benchmarks:
1. Naive Same Weekday Last Week (most recent same weekday <= T)
2. Seasonal Naive 4-Week Rolling Same Weekday Average
3. Historical Pickup Benchmark (OTB + train-set average pickup pace)
"""

from typing import Dict
import numpy as np
import pandas as pd


class NaiveSameWeekdayBaseline:
    """
    Predicts demand using the most recent observation of the same day-of-week
    that occurred on or before prediction date T.
    """
    def __init__(self, realized_daily: pd.DataFrame):
        self.realized_daily = realized_daily

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        preds = []
        for _, row in df.iterrows():
            stay_date = pd.Timestamp(row["stay_date"])
            horizon = int(row["horizon"])
            cat = row["room_category"]

            # Number of weeks to step back so reference date <= as_of_date T (where T = S - h)
            # If h in 1..7, step back 7 days (S - 7 <= T). If h in 8..14, step back 14 days (S - 14 <= T)
            weeks_back = int(np.ceil(horizon / 7.0))
            ref_date = stay_date - pd.Timedelta(days=7 * weeks_back)

            val = self.realized_daily.loc[ref_date, cat] if ref_date in self.realized_daily.index else row["capacity"] * 0.78
            preds.append(float(val))

        return np.array(preds)


class SeasonalNaiveRollingBaseline:
    """
    Predicts demand using the average of the 4 most recent observations
    of the same day-of-week occurring on or before date T.
    """
    def __init__(self, realized_daily: pd.DataFrame):
        self.realized_daily = realized_daily

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        preds = []
        for _, row in df.iterrows():
            stay_date = pd.Timestamp(row["stay_date"])
            horizon = int(row["horizon"])
            cat = row["room_category"]

            weeks_back = int(np.ceil(horizon / 7.0))
            past_vals = []
            for k in range(4):
                ref_date = stay_date - pd.Timedelta(days=7 * (weeks_back + k))
                if ref_date in self.realized_daily.index:
                    past_vals.append(self.realized_daily.loc[ref_date, cat])

            if past_vals:
                preds.append(float(np.mean(past_vals)))
            else:
                preds.append(float(row["capacity"] * 0.78))

        return np.array(preds)


class HistoricalPickupBaseline:
    """
    Domain-specific revenue management benchmark:
    Prediction = OTB(T, S) + Historical_Average_Pickup(Category, DOW, Horizon).
    """
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if "baseline_pickup_forecast" in df.columns:
            return df["baseline_pickup_forecast"].to_numpy()
        return (df["otb_rooms"] + df.get("hist_avg_pickup", 0)).to_numpy()
