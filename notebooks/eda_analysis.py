"""
Exploratory Data Analysis (EDA) on Hotel Booking Ledger.

Generates key hospitality analytics:
1. Booking Pace / Pickup Curves by Room Category
2. Weekly and Monthly Seasonality Patterns
3. Cancellation Rate Dynamics by Lead-Time and Segment
4. Indian Festival & Event Uplift
5. Target Occupancy Distribution vs Physical Capacity
"""

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import (
    RAW_BOOKINGS_CSV,
    ARTIFACTS_DIR,
    ROOM_CATEGORIES,
)

# Output directory for plots
PLOTS_DIR = ARTIFACTS_DIR / "eda_plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Visual styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({"font.size": 11, "figure.autolayout": True})


def load_and_preprocess_raw_data() -> pd.DataFrame:
    """Load raw booking transactions and attach datetime features."""
    df = pd.read_csv(RAW_BOOKINGS_CSV)
    df["booking_date"] = pd.to_datetime(df["booking_date"])
    df["checkin_date"] = pd.to_datetime(df["checkin_date"])
    df["checkout_date"] = pd.to_datetime(df["checkout_date"])
    df["cancellation_date"] = pd.to_datetime(df["cancellation_date"])
    return df


def plot_booking_pace_curves(df: pd.DataFrame) -> None:
    """
    Plot cumulative bookings On-The-Books as a function of lead-time (days before arrival).
    Highlights the standout idea: distinct pickup velocity profiles across room types.
    """
    print("Generating Plot 1: Booking Pace Curves...")
    # Calculate for each booking how many days prior to arrival it was on the books
    # Only non-cancelled or cancelled AFTER lead time
    horizons = np.arange(0, 61)  # 60 days down to 0 (arrival)

    pace_records = []
    # Sample 100 random stay dates to compute average pace curve
    sample_dates = df["checkin_date"].drop_duplicates().sample(n=120, random_state=42)
    sample_df = df[df["checkin_date"].isin(sample_dates)]

    for cat in ROOM_CATEGORIES.keys():
        cat_df = sample_df[sample_df["room_category"] == cat]
        for h in horizons:
            # On-The-Books h days prior:
            # Booked on or before (checkin_date - h days)
            # AND not cancelled before (checkin_date - h days)
            otb_mask = (cat_df["lead_time"] >= h) & (
                (cat_df["is_cancelled"] == 0) |
                ((cat_df["checkin_date"] - cat_df["cancellation_date"]).dt.days < h)
            )
            avg_otb = otb_mask.sum() / len(sample_dates)
            pace_records.append({"horizon_days_out": h, "room_category": cat, "avg_otb": avg_otb})

    pace_df = pd.DataFrame(pace_records)

    fig, ax = plt.subplots(figsize=(9, 5))
    palette = {"Standard": "#1f77b4", "Deluxe": "#ff7f0e", "Suite": "#2ca02c"}
    sns.lineplot(
        data=pace_df,
        x="horizon_days_out",
        y="avg_otb",
        hue="room_category",
        palette=palette,
        linewidth=2.5,
        ax=ax
    )
    ax.invert_xaxis()  # 60 days out -> 0 days (day of arrival)
    ax.set_title("Standout Idea: Booking-Pace (Pickup) Curves by Room Category", fontsize=13, weight="bold")
    ax.set_xlabel("Days Before Arrival (Lead Time Countdown: 60d → Arrival)", fontsize=11)
    ax.set_ylabel("Average Active Bookings On-The-Books", fontsize=11)
    ax.axvline(x=14, color="gray", linestyle="--", alpha=0.7, label="14-Day Horizon")
    ax.axvline(x=7, color="darkred", linestyle="--", alpha=0.7, label="7-Day Horizon")
    ax.legend(title="Room Category")
    fig.savefig(PLOTS_DIR / "01_booking_pace_curves.png", dpi=200)
    plt.close(fig)


def plot_seasonality_patterns(df: pd.DataFrame) -> None:
    """Plot weekly and monthly occupancy rates."""
    print("Generating Plot 2: Seasonality Patterns...")
    # Calculate daily realized occupied rooms per category
    realized = df[(df["is_cancelled"] == 0) & (df["is_no_show"] == 0)]
    daily_occ = realized.groupby(["checkin_date", "room_category"]).size().reset_index(name="occupied_rooms")

    # Map capacity to get occupancy percentage
    cap_map = {k: v["capacity"] for k, v in ROOM_CATEGORIES.items()}
    daily_occ["capacity"] = daily_occ["room_category"].map(cap_map)
    daily_occ["occupancy_rate"] = daily_occ["occupied_rooms"] / daily_occ["capacity"] * 100.0

    daily_occ["dow"] = daily_occ["checkin_date"].dt.day_name()
    daily_occ["month"] = daily_occ["checkin_date"].dt.strftime("%b")

    dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Day of week
    sns.barplot(
        data=daily_occ,
        x="dow",
        y="occupancy_rate",
        hue="room_category",
        order=dow_order,
        errorbar=None,
        ax=axes[0]
    )
    axes[0].set_title("Weekly Pattern: Day of Week Occupancy Rate (%)", fontsize=12, weight="bold")
    axes[0].set_xlabel("Day of Week")
    axes[0].set_ylabel("Occupancy Rate (%)")
    axes[0].tick_params(axis="x", rotation=30)
    axes[0].set_ylim(0, 100)

    # Monthly
    sns.barplot(
        data=daily_occ,
        x="month",
        y="occupancy_rate",
        order=month_order,
        color="#3470a3",
        errorbar=None,
        ax=axes[1]
    )
    axes[1].set_title("Yearly Pattern: Monthly Occupancy Rate (%)", fontsize=12, weight="bold")
    axes[1].set_xlabel("Month")
    axes[1].set_ylabel("Average Occupancy Rate (%)")
    axes[1].set_ylim(0, 100)

    fig.savefig(PLOTS_DIR / "02_seasonality_patterns.png", dpi=200)
    plt.close(fig)


def plot_cancellation_rates(df: pd.DataFrame) -> None:
    """Plot cancellation rates by lead-time bucket and market segment."""
    print("Generating Plot 3: Cancellation Rates...")
    bins = [-1, 7, 14, 30, 60, 200]
    labels = ["0-7d (Last Minute)", "8-14d (1-2 Wks)", "15-30d (2-4 Wks)", "31-60d (1-2 Mos)", ">60d (Long Lead)"]
    df["lead_time_bucket"] = pd.cut(df["lead_time"], bins=bins, labels=labels)

    lt_canc = df.groupby(["lead_time_bucket", "market_segment"], observed=False)["is_cancelled"].mean().reset_index()
    lt_canc["cancellation_rate_pct"] = lt_canc["is_cancelled"] * 100.0

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(
        data=lt_canc,
        x="lead_time_bucket",
        y="cancellation_rate_pct",
        hue="market_segment",
        palette="muted",
        ax=ax
    )
    ax.set_title("Cancellation Rate (%) by Lead-Time Bucket & Market Segment", fontsize=12, weight="bold")
    ax.set_xlabel("Lead Time Bucket")
    ax.set_ylabel("Cancellation Rate (%)")
    ax.set_ylim(0, 50)
    fig.savefig(PLOTS_DIR / "03_cancellation_rates.png", dpi=200)
    plt.close(fig)


def plot_festival_demand_uplift(df: pd.DataFrame) -> None:
    """Compare occupancy and ADR during Indian festivals vs non-festival days."""
    print("Generating Plot 4: Festival Demand Uplift...")
    realized = df[(df["is_cancelled"] == 0) & (df["is_no_show"] == 0)].copy()
    realized["is_festival"] = realized["festival_event"] != "None"

    summary = realized.groupby(["is_festival", "room_category"]).agg(
        avg_adr=("adr", "mean"),
        total_rooms_sold=("num_rooms", "sum")
    ).reset_index()

    summary["Event Type"] = summary["is_festival"].map({True: "Festival / Holiday", False: "Normal Days"})

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sns.barplot(
        data=summary,
        x="room_category",
        y="avg_adr",
        hue="Event Type",
        palette={"Festival / Holiday": "#E1974C", "Normal Days": "#7293CB"},
        ax=axes[0]
    )
    axes[0].set_title("Average Daily Rate (ADR in INR): Festival vs Normal", fontsize=12, weight="bold")
    axes[0].set_ylabel("ADR (INR)")
    axes[0].set_xlabel("Room Category")

    # Occupancy per day comparison
    daily_realized = realized.groupby(["checkin_date", "is_festival"])["num_rooms"].sum().reset_index()
    daily_realized["Event Type"] = daily_realized["is_festival"].map({True: "Festival / Holiday", False: "Normal Days"})

    sns.boxplot(
        data=daily_realized,
        x="Event Type",
        y="num_rooms",
        hue="Event Type",
        palette={"Festival / Holiday": "#E1974C", "Normal Days": "#7293CB"},
        legend=False,
        ax=axes[1]
    )
    axes[1].set_title("Total Hotel Occupied Rooms per Day", fontsize=12, weight="bold")
    axes[1].set_ylabel("Occupied Rooms (out of 110 total)")
    axes[1].axhline(y=110, color="red", linestyle="--", label="100% Hotel Capacity")
    axes[1].legend()

    fig.savefig(PLOTS_DIR / "04_festival_demand_uplift.png", dpi=200)
    plt.close(fig)


def plot_demand_distribution(df: pd.DataFrame) -> None:
    """Plot target variable distribution per room category against physical capacity."""
    print("Generating Plot 5: Target Demand Distribution...")
    realized = df[(df["is_cancelled"] == 0) & (df["is_no_show"] == 0)]
    daily = realized.groupby(["checkin_date", "room_category"]).size().reset_index(name="net_demand")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=False)

    colors = {"Standard": "#1f77b4", "Deluxe": "#ff7f0e", "Suite": "#2ca02c"}

    for idx, (cat, spec) in enumerate(ROOM_CATEGORIES.items()):
        ax = axes[idx]
        sub = daily[daily["room_category"] == cat]
        sns.histplot(sub["net_demand"], kde=True, color=colors[cat], bins=20, ax=ax)
        ax.axvline(spec["capacity"], color="darkred", linestyle="--", linewidth=2, label=f"Max Cap ({spec['capacity']})")
        ax.set_title(f"{cat} Room Demand Distribution\n(Mean: {sub['net_demand'].mean():.1f} rooms)", weight="bold")
        ax.set_xlabel("Daily Occupied Rooms")
        ax.set_ylabel("Count of Days")
        ax.legend()

    fig.savefig(PLOTS_DIR / "05_demand_distribution.png", dpi=200)
    plt.close(fig)


def run_full_eda() -> None:
    """Run all EDA routines and print statistical summary."""
    df = load_and_preprocess_raw_data()
    print(f"Loaded {len(df):,} booking transactions.")
    print("Dataset Date Range:", df["checkin_date"].min().date(), "to", df["checkin_date"].max().date())
    print("\nSummary by Room Category:")
    summary = df.groupby("room_category").agg(
        total_bookings=("booking_id", "count"),
        cancellation_rate=("is_cancelled", lambda x: f"{x.mean()*100:.1f}%"),
        median_lead_time=("lead_time", "median"),
        mean_adr=("adr", "mean")
    )
    print(summary)

    plot_booking_pace_curves(df)
    plot_seasonality_patterns(df)
    plot_cancellation_rates(df)
    plot_festival_demand_uplift(df)
    plot_demand_distribution(df)
    print(f"\nAll 5 EDA visualizations saved successfully to {PLOTS_DIR}/")


if __name__ == "__main__":
    run_full_eda()
