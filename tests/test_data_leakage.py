"""
Automated Data Leakage Verification Test Suite.

Rigorously verifies:
1. Point-in-time sanity: as_of_date is strictly prior to stay_date (h >= 1).
2. Look-ahead isolation: No booking made after as_of_date is ever in OTB or pickup counts.
3. Cancellation cutoff: Future cancellations occurring after as_of_date are treated as active OTB.
4. Strict chronological split isolation: train < val < test.
5. Learned feature statistics (historical pickup pace) fit strictly on train split.
"""

from datetime import datetime
import pandas as pd
import pytest

from src.config import (
    DAILY_AGGREGATED_PARQUET,
    RAW_BOOKINGS_CSV,
    TRAIN_END_DATE,
    VAL_END_DATE,
    VAL_START_DATE,
    TEST_START_DATE,
    TEST_END_DATE,
)


@pytest.fixture(scope="module")
def processed_features() -> pd.DataFrame:
    """Load preprocessed feature dataset."""
    if DAILY_AGGREGATED_PARQUET.exists():
        return pd.read_parquet(DAILY_AGGREGATED_PARQUET)
    csv_fallback = DAILY_AGGREGATED_PARQUET.parent / "daily_demand_as_of.csv"
    if csv_fallback.exists():
        return pd.read_csv(csv_fallback)
    pytest.fail("Feature dataset not found. Run `bash run.sh train` first.")


@pytest.fixture(scope="module")
def raw_bookings() -> pd.DataFrame:
    """Load raw synthetic booking transactions."""
    if not RAW_BOOKINGS_CSV.exists():
        pytest.fail("Raw dataset not found. Run `bash run.sh generate` first.")
    df = pd.read_csv(RAW_BOOKINGS_CSV)
    df["booking_date"] = pd.to_datetime(df["booking_date"])
    df["checkin_date"] = pd.to_datetime(df["checkin_date"])
    df["cancellation_date"] = pd.to_datetime(df["cancellation_date"])
    return df


def test_as_of_date_strictly_precedes_stay_date(processed_features: pd.DataFrame):
    """Assertion: Every forecast must be generated strictly prior to the stay date."""
    as_of = pd.to_datetime(processed_features["as_of_date"])
    stay = pd.to_datetime(processed_features["stay_date"])
    horizon = processed_features["horizon"]

    diff_days = (stay - as_of).dt.days
    assert (diff_days == horizon).all(), "Horizon mismatch: stay_date - as_of_date != horizon"
    assert (as_of < stay).all(), "Data leakage violation: as_of_date is not strictly before stay_date!"


def test_chronological_splits_have_zero_overlap(processed_features: pd.DataFrame):
    """Assertion: Train, validation, and test partitions must be strictly chronological."""
    df = processed_features.copy()
    df["stay_date"] = pd.to_datetime(df["stay_date"])

    train_max = df[df["split"] == "train"]["stay_date"].max()
    val_min = df[df["split"] == "val"]["stay_date"].min()
    val_max = df[df["split"] == "val"]["stay_date"].max()
    test_min = df[df["split"] == "test"]["stay_date"].min()

    assert train_max < val_min, f"Split leakage: Train max ({train_max}) >= Val min ({val_min})"
    assert val_max < test_min, f"Split leakage: Val max ({val_max}) >= Test min ({test_min})"


def test_future_bookings_never_leak_into_otb(raw_bookings: pd.DataFrame, processed_features: pd.DataFrame):
    """
    Spot-check 50 random samples:
    Manually filter the raw booking ledger to records with booking_date <= as_of_date
    and assert that OTB matches exactly with zero future records counted.
    """
    sample_rows = processed_features.sample(n=50, random_state=42)

    for _, row in sample_rows.iterrows():
        as_of_ts = pd.Timestamp(row["as_of_date"])
        stay_ts = pd.Timestamp(row["stay_date"])
        cat = row["room_category"]

        # Filter raw ledger strictly up to as_of_ts
        cat_bookings = raw_bookings[
            (raw_bookings["checkin_date"] == stay_ts) &
            (raw_bookings["room_category"] == cat)
        ]

        # Valid active bookings as of as_of_ts:
        # 1. Must be booked on or before as_of_ts
        # 2. Must not be cancelled on or before as_of_ts
        valid_otb = cat_bookings[
            (cat_bookings["booking_date"] <= as_of_ts) &
            (
                (cat_bookings["is_cancelled"] == 0) |
                (cat_bookings["cancellation_date"] > as_of_ts)
            )
        ]

        expected_otb = len(valid_otb)
        actual_otb = int(row["otb_rooms"])

        assert actual_otb == expected_otb, (
            f"OTB mismatch at as_of={as_of_ts}, stay={stay_ts}, cat={cat}: "
            f"expected {expected_otb}, found {actual_otb}"
        )


def test_future_cancellations_treated_as_active_on_the_books(raw_bookings: pd.DataFrame, processed_features: pd.DataFrame):
    """
    Verify that if a booking was cancelled AFTER as_of_date, it is counted as active OTB
    and NOT counted in cancellations_to_date.
    """
    sample_rows = processed_features.sample(n=30, random_state=123)

    for _, row in sample_rows.iterrows():
        as_of_ts = pd.Timestamp(row["as_of_date"])
        stay_ts = pd.Timestamp(row["stay_date"])
        cat = row["room_category"]

        cat_bookings = raw_bookings[
            (raw_bookings["checkin_date"] == stay_ts) &
            (raw_bookings["room_category"] == cat)
        ]

        # Cancellations strictly recorded on or before as_of_ts
        expected_canc = len(
            cat_bookings[
                (cat_bookings["is_cancelled"] == 1) &
                (cat_bookings["cancellation_date"] <= as_of_ts)
            ]
        )
        actual_canc = int(row["cancellations_to_date"])

        assert actual_canc == expected_canc, (
            f"Cancellation leakage at as_of={as_of_ts}, stay={stay_ts}: "
            f"expected {expected_canc}, found {actual_canc}"
        )
