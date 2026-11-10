"""
Feature Preprocessing & Quality Assurance Test Suite.

Verifies:
1. Raw booking data constraints (record counts, date validity, price positivity).
2. Processed feature matrix completeness (zero nulls, expected schema).
3. Capacity consistency across categories.
4. Mathematical bounds on cyclical calendar encodings.
"""

import numpy as np
import pandas as pd
import pytest

from src.config import (
    DAILY_AGGREGATED_PARQUET,
    RAW_BOOKINGS_CSV,
    ROOM_CATEGORIES,
)
from src.data.generate_data import validate_booking_dataset
from src.models.candidate_models import ALL_FEATURE_COLS, TARGET_COL


def test_raw_dataset_quality_validator():
    """Verify that the raw booking ledger satisfies all quality constraints."""
    df_raw = pd.read_csv(RAW_BOOKINGS_CSV)
    results = validate_booking_dataset(df_raw)
    assert all(results.values()), f"Raw data quality checks failed: {results}"


def test_processed_feature_matrix_schema():
    """Verify that processed features contain all required columns with zero missing values."""
    if DAILY_AGGREGATED_PARQUET.exists():
        df = pd.read_parquet(DAILY_AGGREGATED_PARQUET)
    else:
        df = pd.read_csv(DAILY_AGGREGATED_PARQUET.parent / "daily_demand_as_of.csv")

    # Check required feature columns
    for col in ALL_FEATURE_COLS + [TARGET_COL, "split"]:
        assert col in df.columns, f"Missing required column in feature matrix: {col}"

    # Check null values in feature columns
    null_counts = df[ALL_FEATURE_COLS + [TARGET_COL]].isnull().sum().sum()
    assert null_counts == 0, f"Found {null_counts} null values in feature matrix!"


def test_cyclical_calendar_encodings_bounded():
    """Verify that sine and cosine day-of-year encodings fall strictly in [-1.0, 1.0]."""
    if DAILY_AGGREGATED_PARQUET.exists():
        df = pd.read_parquet(DAILY_AGGREGATED_PARQUET)
    else:
        df = pd.read_csv(DAILY_AGGREGATED_PARQUET.parent / "daily_demand_as_of.csv")

    assert (df["stay_sin_doy"] >= -1.0001).all() and (df["stay_sin_doy"] <= 1.0001).all()
    assert (df["stay_cos_doy"] >= -1.0001).all() and (df["stay_cos_doy"] <= 1.0001).all()


def test_room_category_capacities_match_config():
    """Verify that room capacities match the single source of truth in src/config.py."""
    if DAILY_AGGREGATED_PARQUET.exists():
        df = pd.read_parquet(DAILY_AGGREGATED_PARQUET)
    else:
        df = pd.read_csv(DAILY_AGGREGATED_PARQUET.parent / "daily_demand_as_of.csv")

    for cat, spec in ROOM_CATEGORIES.items():
        subset = df[df["room_category"] == cat]
        assert (subset["capacity"] == spec["capacity"]).all(), f"Capacity mismatch for {cat}"
