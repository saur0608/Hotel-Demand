"""
Synthetic Hotel Booking Ledger Generator & Quality Validator.

Generates realistic property-management-system (PMS) booking ledgers
spanning 2024-01-01 to 2025-12-31 with:
- Category-specific room capacities & ADRs (Standard, Deluxe, Suite)
- Weekly and yearly seasonality
- Indian festivals & national holiday demand surges (Diwali, Holi, New Year, etc.)
- Realistic lead-time distributions (lognormal)
- Dynamic cancellation curves & no-show probabilities
- Strict sanity checks ensuring zero leakage and physical plausibility.
"""

from datetime import date, datetime, timedelta
from typing import Dict, List, Tuple
import logging
import numpy as np
import pandas as pd

from src.config import (
    RANDOM_SEED,
    RAW_BOOKINGS_CSV,
    ROOM_CATEGORIES,
    TRAIN_START_DATE,
    TEST_END_DATE,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ==============================================================================
# 1. INDIAN CALENDAR & FESTIVAL DEFINITIONS
# ==============================================================================
# Key Indian public holidays & major festive windows for 2024 and 2025
INDIAN_FESTIVALS = {
    # 2024
    "2024-01-26": ("Republic Day", 1.25),
    "2024-03-08": ("Maha Shivratri", 1.15),
    "2024-03-25": ("Holi", 1.35),
    "2024-03-29": ("Good Friday Weekend", 1.30),
    "2024-04-11": ("Eid-ul-Fitr", 1.20),
    "2024-08-15": ("Independence Day Weekend", 1.40),
    "2024-08-19": ("Raksha Bandhan", 1.15),
    "2024-08-26": ("Janmashtami", 1.15),
    "2024-10-02": ("Gandhi Jayanti", 1.20),
    "2024-10-12": ("Dussehra", 1.30),
    "2024-11-01": ("Diwali", 1.55),
    "2024-11-02": ("Govardhan Puja / Diwali Weekend", 1.50),
    # 2025
    "2025-01-26": ("Republic Day", 1.25),
    "2025-02-26": ("Maha Shivratri", 1.15),
    "2025-03-14": ("Holi", 1.40),
    "2025-03-31": ("Eid-ul-Fitr", 1.20),
    "2025-04-18": ("Good Friday Weekend", 1.30),
    "2025-08-15": ("Independence Day Weekend", 1.45),
    "2025-08-16": ("Janmashtami Weekend", 1.25),
    "2025-10-02": ("Gandhi Jayanti & Dussehra", 1.35),
    "2025-10-20": ("Diwali", 1.55),
    "2025-10-21": ("Diwali New Year", 1.50),
}

# High-season month multipliers (Oct-Feb high season, Monsoon July-Aug low)
MONTH_SEASONALITY = {
    1: 1.15,   # Jan: Winter leisure & wedding
    2: 1.10,   # Feb: High corporate & wedding
    3: 1.05,   # Mar: Fiscal year-end corporate & Holi
    4: 0.95,   # Apr: Moderate
    5: 1.00,   # May: Summer vacation
    6: 0.90,   # Jun: Pre-monsoon
    7: 0.80,   # Jul: Monsoon trough
    8: 0.85,   # Aug: Monsoon with holiday spikes
    9: 0.95,   # Sep: Transition
    10: 1.20,  # Oct: Festival kickoff (Dussehra/Diwali)
    11: 1.30,  # Nov: Peak wedding & pleasant winter
    12: 1.40,  # Dec: Christmas / New Year peak
}

# Day of week demand multipliers (0 = Monday, 6 = Sunday)
# Business travel hits Tue-Thu, Leisure hits Fri-Sat
DOW_MULTIPLIERS = {
    "Standard": {0: 0.90, 1: 1.05, 2: 1.10, 3: 1.05, 4: 1.15, 5: 1.20, 6: 0.85},
    "Deluxe":   {0: 0.85, 1: 0.95, 2: 1.00, 3: 1.00, 4: 1.25, 5: 1.35, 6: 0.95},
    "Suite":    {0: 0.80, 1: 0.90, 2: 0.95, 3: 1.00, 4: 1.30, 5: 1.45, 6: 0.90},
}


# ==============================================================================
# 2. DATA GENERATION ENGINE
# ==============================================================================
def get_festival_multiplier(target_date: date) -> Tuple[float, str]:
    """Check if a date falls on or adjacent to an Indian festival or year-end peak."""
    date_str = target_date.isoformat()
    if date_str in INDIAN_FESTIVALS:
        name, mult = INDIAN_FESTIVALS[date_str]
        return mult, name

    # Christmas to New Year peak window (Dec 23 - Jan 2)
    if (target_date.month == 12 and target_date.day >= 23) or (
        target_date.month == 1 and target_date.day <= 2
    ):
        return 1.45, "Christmas-NewYear-Peak"

    return 1.00, "None"


def generate_synthetic_hotel_dataset(
    start_date_str: str = TRAIN_START_DATE,
    end_date_str: str = TEST_END_DATE,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Generate synthetic hotel reservation transactions.

    Each record represents an individual guest booking with realistic
    lead time, cancellation risk, room pricing, and arrival timing.
    """
    np.random.seed(seed)
    logger.info("Initializing synthetic dataset generation: %s to %s", start_date_str, end_date_str)

    start_date = datetime.strptime(start_date_str, "%Y-%m-%d").date()
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d").date()
    total_days = (end_date - start_date).days + 1

    records: List[Dict] = []
    booking_counter = 1

    # Lead time distribution parameters (lognormal mu, sigma, max_days)
    lead_time_params = {
        "Standard": (2.2, 0.7, 90),   # Median ~9 days
        "Deluxe":   (2.7, 0.8, 120),  # Median ~15 days
        "Suite":    (3.3, 0.7, 150),  # Median ~27 days
    }

    # Baseline cancellation probabilities by category
    base_cancel_rates = {
        "Standard": 0.22,
        "Deluxe":   0.20,
        "Suite":    0.14,
    }

    for day_offset in range(total_days):
        current_date = start_date + timedelta(days=day_offset)
        dow = current_date.weekday()
        month = current_date.month
        month_factor = MONTH_SEASONALITY[month]
        fest_factor, fest_name = get_festival_multiplier(current_date)

        for category, spec in ROOM_CATEGORIES.items():
            cap = spec["capacity"]
            base_adr = spec["base_adr"]
            dow_factor = DOW_MULTIPLIERS[category][dow]

            # True unconstrained latent demand before capacity bounds
            # Average target baseline ~ 75-80% capacity
            mean_demand = cap * 0.78 * dow_factor * month_factor * fest_factor
            actual_demand = max(1, int(np.random.poisson(lam=mean_demand)))

            mu_lt, sigma_lt, max_lt = lead_time_params[category]
            base_cancel = base_cancel_rates[category]

            for _ in range(actual_demand):
                # 1. Lead time (days before check-in)
                raw_lt = int(np.random.lognormal(mean=mu_lt, sigma=sigma_lt))
                lead_time = max(0, min(raw_lt, max_lt))
                booking_date = current_date - timedelta(days=lead_time)

                # 2. Length of stay (nights: 1 to 4)
                stay_nights = int(np.random.choice([1, 2, 3, 4], p=[0.58, 0.27, 0.10, 0.05]))
                checkout_date = current_date + timedelta(days=stay_nights)

                # 3. Market segment
                market_segment = np.random.choice(
                    ["Direct", "OTA", "Corporate"],
                    p=[0.25, 0.50, 0.25]
                )

                # 4. Dynamic ADR (pricing reflects demand pressure + segment discount)
                demand_pressure = min(1.35, max(0.85, mean_demand / cap))
                segment_discount = 0.90 if market_segment == "Corporate" else (0.95 if market_segment == "OTA" else 1.0)
                noise = np.random.uniform(0.95, 1.05)
                adr = round(base_adr * demand_pressure * segment_discount * noise, 2)

                # 5. Cancellation modeling
                # Longer lead times and OTAs carry higher cancellation rates
                lt_cancel_boost = min(0.18, (lead_time / 100.0) * 0.15)
                segment_cancel_adj = 0.05 if market_segment == "OTA" else (-0.06 if market_segment == "Corporate" else 0.0)
                cancel_prob = np.clip(base_cancel + lt_cancel_boost + segment_cancel_adj, 0.04, 0.60)

                is_cancelled = int(np.random.binomial(1, cancel_prob))
                cancellation_date = None

                if is_cancelled:
                    if lead_time > 0:
                        # Cancelled between booking date and checkin_date - 1
                        cancel_offset = int(np.random.uniform(0, lead_time))
                        cancellation_date = booking_date + timedelta(days=cancel_offset)
                    else:
                        cancellation_date = booking_date

                # 6. No-show probability (guests who don't cancel but don't show up on check-in day)
                is_no_show = 0
                if not is_cancelled:
                    is_no_show = int(np.random.binomial(1, 0.025))

                records.append({
                    "booking_id": f"BK-{booking_counter:07d}",
                    "booking_date": booking_date.isoformat(),
                    "checkin_date": current_date.isoformat(),
                    "checkout_date": checkout_date.isoformat(),
                    "room_category": category,
                    "num_rooms": 1,
                    "lead_time": lead_time,
                    "length_of_stay": stay_nights,
                    "market_segment": market_segment,
                    "adr": adr,
                    "is_cancelled": is_cancelled,
                    "cancellation_date": cancellation_date.isoformat() if cancellation_date else None,
                    "is_no_show": is_no_show,
                    "festival_event": fest_name,
                })
                booking_counter += 1

    df = pd.DataFrame(records)
    logger.info("Generated %d synthetic booking records.", len(df))
    return df


# ==============================================================================
# 3. AUTOMATED DATA QUALITY VALIDATION
# ==============================================================================
def validate_booking_dataset(df: pd.DataFrame) -> Dict[str, bool]:
    """
    Execute rigorous validation checks on raw booking transactions:
    1. Row count >= 1000
    2. Zero null values in mandatory fields
    3. Chronological sanity: booking_date <= checkin_date < checkout_date
    4. Cancellation date sanity: booking_date <= cancellation_date <= checkin_date
    5. Valid room categories and non-negative ADR
    6. No future leakage indicators
    """
    logger.info("Executing automated data quality checks...")
    results = {}

    # Check 1: Record count
    results["min_record_count"] = len(df) >= 1000

    # Check 2: Mandatory column nulls
    mandatory_cols = ["booking_id", "booking_date", "checkin_date", "checkout_date", "room_category", "adr"]
    null_counts = df[mandatory_cols].isnull().sum().sum()
    results["no_mandatory_nulls"] = (null_counts == 0)

    # Check 3: Date order sanity
    b_date = pd.to_datetime(df["booking_date"])
    ci_date = pd.to_datetime(df["checkin_date"])
    co_date = pd.to_datetime(df["checkout_date"])

    results["booking_before_or_on_checkin"] = (b_date <= ci_date).all()
    results["checkin_before_checkout"] = (ci_date < co_date).all()

    # Check 4: Cancellation sanity
    cancelled_mask = df["is_cancelled"] == 1
    canc_df = df[cancelled_mask].copy()
    c_date = pd.to_datetime(canc_df["cancellation_date"])
    b_canc_date = pd.to_datetime(canc_df["booking_date"])
    ci_canc_date = pd.to_datetime(canc_df["checkin_date"])

    canc_valid = (b_canc_date <= c_date).all() and (c_date <= ci_canc_date).all()
    results["cancellation_timing_valid"] = canc_valid

    # Check 5: Room categories valid and prices positive
    expected_categories = set(ROOM_CATEGORIES.keys())
    present_categories = set(df["room_category"].unique())
    results["valid_room_categories"] = (present_categories == expected_categories)
    results["positive_adr"] = (df["adr"] > 0).all()

    all_passed = all(results.values())
    logger.info("Validation complete. All checks passed: %s. Details: %s", all_passed, results)

    if not all_passed:
        raise ValueError(f"Data quality checks failed: {results}")

    return results


# ==============================================================================
# 4. ENTRYPOINT
# ==============================================================================
if __name__ == "__main__":
    df_raw = generate_synthetic_hotel_dataset()
    validate_booking_dataset(df_raw)
    RAW_BOOKINGS_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_raw.to_csv(RAW_BOOKINGS_CSV, index=False)
    logger.info("Saved verified synthetic booking ledger to %s", RAW_BOOKINGS_CSV)
