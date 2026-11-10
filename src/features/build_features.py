"""
Leakage-Safe Feature Engineering Pipeline for Hotel Demand Forecasting.

Constructs multi-horizon (1 to 14 days) feature matrices as-of cutoff date T:
1. Standout Booking-Pace Features: On-The-Books (OTB) reservations, pace velocity (3d/7d),
   cancellations to date, and historical average pickup curve.
2. Calendar & Seasonality: Day-of-week, weekend indicator, month, cyclical day of year,
   and Indian festival/holiday multipliers.
3. As-Of Historical Lags: Realized demand lags strictly relative to prediction date T
   (T0, T-1, T-7, T-14, 7-day rolling mean/std) — avoiding look-ahead leakage.
4. Strict Out-Of-Time Separation: Learned lookup tables (historical pace) are fit ONLY
   on the training split and mapped onto validation and test splits.
"""

from datetime import date, datetime
from pathlib import Path
from typing import Dict, Tuple
import logging
import numpy as np
import pandas as pd

from src.config import (
    RAW_BOOKINGS_CSV,
    DAILY_AGGREGATED_PARQUET,
    PROCESSED_DATA_DIR,
    ROOM_CATEGORIES,
    TRAIN_START_DATE,
    TRAIN_END_DATE,
    VAL_START_DATE,
    VAL_END_DATE,
    TEST_START_DATE,
    TEST_END_DATE,
)
from src.data.generate_data import get_festival_multiplier

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ==============================================================================
# 1. REALIZED DAILY DEMAND AGGREGATION
# ==============================================================================
def compute_daily_realized_demand(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Compute actual final net occupied rooms per stay_date and room_category.
    Realized demand = guests who booked, did not cancel, and showed up.
    """
    realized = df_raw[(df_raw["is_cancelled"] == 0) & (df_raw["is_no_show"] == 0)].copy()
    realized["checkin_date"] = pd.to_datetime(realized["checkin_date"])

    daily = (
        realized.groupby(["checkin_date", "room_category"])
        .size()
        .unstack(fill_value=0)
        .sort_index()
    )

    # Ensure all room categories are represented
    for cat in ROOM_CATEGORIES.keys():
        if cat not in daily.columns:
            daily[cat] = 0

    return daily


# ==============================================================================
# 2. BOOKING-PACE (PICKUP) FEATURE EXTRACTION AS-OF DATE T
# ==============================================================================
def extract_as_of_pickup_features(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    For each (stay_date, room_category) and horizon h in 1..14 days:
    Evaluate the reservation ledger strictly as of T = stay_date - h days.
    
    Returns:
        DataFrame with columns:
        [as_of_date, stay_date, room_category, horizon, otb_rooms,
         pickup_last_3d, pickup_last_7d, cancellations_to_date, target_net_demand]
    """
    logger.info("Extracting booking pace (pickup) features across horizons 1-14...")
    df = df_raw.copy()
    df["booking_date"] = pd.to_datetime(df["booking_date"])
    df["checkin_date"] = pd.to_datetime(df["checkin_date"])
    # If not cancelled, treat cancellation as far in the future
    df["cancellation_date"] = pd.to_datetime(df["cancellation_date"]).fillna(pd.Timestamp("2099-01-01"))

    records = []
    grouped = df.groupby(["checkin_date", "room_category"])

    for (stay_date, category), group in grouped:
        b_dates = group["booking_date"].to_numpy()
        c_dates = group["cancellation_date"].to_numpy()
        is_canc = group["is_cancelled"].to_numpy().astype(bool)
        is_ns = group["is_no_show"].to_numpy().astype(bool)

        # Ground truth target: realized net occupied rooms on stay_date
        realized_net_demand = int(np.sum(~is_canc & ~is_ns))

        for h in range(1, 15):
            as_of_date = stay_date - pd.Timedelta(days=h)
            as_of_np = np.datetime64(as_of_date)
            as_of_minus_3d = np.datetime64(as_of_date - pd.Timedelta(days=3))
            as_of_minus_7d = np.datetime64(as_of_date - pd.Timedelta(days=7))

            # Active on the books at date T:
            # Booked on or before T AND (not cancelled OR cancelled AFTER T)
            active_mask = (b_dates <= as_of_np) & (~is_canc | (c_dates > as_of_np))
            otb_rooms = int(np.sum(active_mask))

            # Recent pickup velocity (new bookings created in past 3 / 7 days of T)
            pickup_3d = int(np.sum(active_mask & (b_dates > as_of_minus_3d)))
            pickup_7d = int(np.sum(active_mask & (b_dates > as_of_minus_7d)))

            # Cancellations officially recorded on or before T
            cancellations_td = int(np.sum(is_canc & (c_dates <= as_of_np)))

            records.append({
                "as_of_date": as_of_date,
                "stay_date": stay_date,
                "room_category": category,
                "horizon": h,
                "otb_rooms": otb_rooms,
                "pickup_last_3d": pickup_3d,
                "pickup_last_7d": pickup_7d,
                "cancellations_to_date": cancellations_td,
                "target_net_demand": realized_net_demand,
            })

    feat_df = pd.DataFrame(records)
    
    # Capacity and ratios
    cap_map = {k: v["capacity"] for k, v in ROOM_CATEGORIES.items()}
    adr_map = {k: v["base_adr"] for k, v in ROOM_CATEGORIES.items()}
    feat_df["capacity"] = feat_df["room_category"].map(cap_map)
    feat_df["base_adr"] = feat_df["room_category"].map(adr_map)
    feat_df["otb_occupancy_ratio"] = feat_df["otb_rooms"] / feat_df["capacity"]
    feat_df["cancellation_rate_otb"] = (
        feat_df["cancellations_to_date"] / (feat_df["otb_rooms"] + feat_df["cancellations_to_date"] + 1e-5)
    )
    feat_df["target_net_pickup"] = feat_df["target_net_demand"] - feat_df["otb_rooms"]

    return feat_df


# ==============================================================================
# 3. CALENDAR & FESTIVAL FEATURES FOR STAY DATE
# ==============================================================================
def add_calendar_and_holiday_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add calendar features for the target stay_date."""
    logger.info("Adding calendar and festival features...")
    df = df.copy()
    stay_dt = pd.to_datetime(df["stay_date"])

    df["stay_dow"] = stay_dt.dt.dayofweek  # 0=Monday, 6=Sunday
    df["stay_is_weekend"] = df["stay_dow"].isin([4, 5]).astype(int)  # Fri & Sat
    df["stay_month"] = stay_dt.dt.month
    df["stay_quarter"] = stay_dt.dt.quarter
    df["stay_day_of_month"] = stay_dt.dt.day

    # Cyclical day of year encoding
    day_of_year = stay_dt.dt.dayofyear
    df["stay_sin_doy"] = np.sin(2 * np.pi * day_of_year / 365.25)
    df["stay_cos_doy"] = np.cos(2 * np.pi * day_of_year / 365.25)

    # Festival & holiday multipliers
    fest_info = [get_festival_multiplier(d.date()) for d in stay_dt]
    df["festival_uplift_factor"] = [info[0] for info in fest_info]
    df["is_festival"] = (df["festival_uplift_factor"] > 1.0).astype(int)

    return df


# ==============================================================================
# 4. AS-OF HISTORICAL DEMAND LAGS (RELATIVE TO T)
# ==============================================================================
def add_as_of_lag_features(feat_df: pd.DataFrame, daily_realized: pd.DataFrame) -> pd.DataFrame:
    """
    Attach historical realized demand lags.
    CRITICAL: All lags are relative to as_of_date T (NOT stay_date S) to guarantee zero leakage!
    """
    logger.info("Attaching leakage-safe historical demand lags relative to as-of date...")
    feat_df = feat_df.copy()

    # Pre-compute lag and rolling series for each room category indexed by date
    lag_dicts: Dict[str, Dict] = {}

    for cat in ROOM_CATEGORIES.keys():
        series = daily_realized[cat].copy()
        lag_dicts[cat] = {
            "lag_T0": series,                                # Realized demand on as_of_date T
            "lag_T1": series.shift(1),                       # Realized demand on T - 1
            "lag_T7": series.shift(7),                       # Realized demand on T - 7 (same weekday as T)
            "lag_T14": series.shift(14),                     # Realized demand on T - 14
            "roll_mean_7d": series.rolling(7, min_periods=1).mean(),
            "roll_mean_14d": series.rolling(14, min_periods=1).mean(),
            "roll_std_7d": series.rolling(7, min_periods=1).std().fillna(0.0),
        }

    # Vectorized mapping via index lookup
    for lag_name in ["lag_T0", "lag_T1", "lag_T7", "lag_T14", "roll_mean_7d", "roll_mean_14d", "roll_std_7d"]:
        # Map (category, as_of_date) to precomputed value
        feat_df[lag_name] = feat_df.apply(
            lambda row: lag_dicts[row["room_category"]][lag_name].get(pd.Timestamp(row["as_of_date"]), np.nan),
            axis=1
        )

    # Fill any warmup boundary NaNs (first 14 days of dataset) with category mean
    for cat, spec in ROOM_CATEGORIES.items():
        cat_mask = feat_df["room_category"] == cat
        mean_cap = spec["capacity"] * 0.78
        for col in ["lag_T0", "lag_T1", "lag_T7", "lag_T14", "roll_mean_7d", "roll_mean_14d"]:
            feat_df.loc[cat_mask, col] = feat_df.loc[cat_mask, col].fillna(mean_cap)
        feat_df.loc[cat_mask, "roll_std_7d"] = feat_df.loc[cat_mask, "roll_std_7d"].fillna(2.0)

    return feat_df


# ==============================================================================
# 5. HISTORICAL AVERAGE PICKUP PACE (LEARNED STRICTLY ON TRAIN SPLIT)
# ==============================================================================
def add_historical_pickup_pace_feature(
    feat_df: pd.DataFrame,
    train_end_date: str = TRAIN_END_DATE
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Learn average historical pickup: delta_pickup(category, stay_dow, horizon)
    strictly from the training partition. Then map onto the full dataset.
    """
    logger.info("Computing historical pickup pace lookup from training data (<= %s)...", train_end_date)
    feat_df = feat_df.copy()

    # Mask training split
    train_mask = pd.to_datetime(feat_df["stay_date"]) <= pd.Timestamp(train_end_date)
    train_data = feat_df[train_mask]

    # Historical average pickup = realized demand minus OTB at horizon h
    pickup_lookup = (
        train_data.groupby(["room_category", "stay_dow", "horizon"])["target_net_pickup"]
        .mean()
        .reset_index(name="hist_avg_pickup")
    )

    # Merge lookup table onto full dataframe
    feat_df = feat_df.merge(
        pickup_lookup,
        on=["room_category", "stay_dow", "horizon"],
        how="left"
    )

    # Baseline pickup forecast = OTB + historical average pickup
    feat_df["baseline_pickup_forecast"] = np.maximum(0, feat_df["otb_rooms"] + feat_df["hist_avg_pickup"])

    return feat_df, pickup_lookup


# ==============================================================================
# 6. END-TO-END FEATURE BUILDER & PIPELINE
# ==============================================================================
def build_feature_dataset() -> pd.DataFrame:
    """Execute complete leakage-safe feature engineering pipeline and persist."""
    logger.info("Starting feature engineering pipeline...")
    df_raw = pd.read_csv(RAW_BOOKINGS_CSV)

    # 1. Realized daily demand
    daily_realized = compute_daily_realized_demand(df_raw)

    # 2. As-of pickup features
    feat_df = extract_as_of_pickup_features(df_raw)

    # 3. Calendar & festivals
    feat_df = add_calendar_and_holiday_features(feat_df)

    # 4. As-of historical lags
    feat_df = add_as_of_lag_features(feat_df, daily_realized)

    # 5. Historical pace curve lookup (strictly train-fit)
    feat_df, pickup_lookup = add_historical_pickup_pace_feature(feat_df, TRAIN_END_DATE)

    # 6. Assign temporal split labels
    stay_dt = pd.to_datetime(feat_df["stay_date"])
    feat_df["split"] = "train"
    feat_df.loc[
        (stay_dt >= pd.Timestamp(VAL_START_DATE)) & (stay_dt <= pd.Timestamp(VAL_END_DATE)),
        "split"
    ] = "val"
    feat_df.loc[
        (stay_dt >= pd.Timestamp(TEST_START_DATE)) & (stay_dt <= pd.Timestamp(TEST_END_DATE)),
        "split"
    ] = "test"

    # Sort chronologically by stay_date, then horizon, then category
    feat_df = feat_df.sort_values(by=["stay_date", "horizon", "room_category"]).reset_index(drop=True)

    # Persist to parquet and CSV
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    try:
        feat_df.to_parquet(DAILY_AGGREGATED_PARQUET, index=False)
        logger.info("Persisted feature matrix to Parquet: %s", DAILY_AGGREGATED_PARQUET)
    except Exception as e:
        logger.warning("Parquet save failed (%s), saving CSV fallback.", e)
        csv_path = PROCESSED_DATA_DIR / "daily_demand_as_of.csv"
        feat_df.to_csv(csv_path, index=False)
        logger.info("Persisted feature matrix to CSV: %s", csv_path)

    # Also save CSV for simple inspection
    feat_df.to_csv(PROCESSED_DATA_DIR / "daily_demand_as_of.csv", index=False)

    logger.info(
        "Feature engineering complete. Total rows: %d. Splits: Train=%d, Val=%d, Test=%d",
        len(feat_df),
        (feat_df["split"] == "train").sum(),
        (feat_df["split"] == "val").sum(),
        (feat_df["split"] == "test").sum(),
    )
    return feat_df


if __name__ == "__main__":
    build_feature_dataset()
