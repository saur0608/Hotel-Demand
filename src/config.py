"""
Central configuration for Hotel Occupancy & Room-Demand Forecasting.

All project-wide paths, random seeds, business assumptions (room inventory,
costs, pricing), and forecasting horizon definitions are centralized here.
"""

from pathlib import Path
import os

# ==============================================================================
# 1. REPRODUCIBILITY & PATHS
# ==============================================================================
RANDOM_SEED: int = 42

# Base paths
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
ARTIFACTS_DIR: Path = PROJECT_ROOT / "artifacts"
NOTEBOOKS_DIR: Path = PROJECT_ROOT / "notebooks"

# Ensure runtime directories exist
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# Data file locations
RAW_BOOKINGS_CSV: Path = RAW_DATA_DIR / "hotel_bookings_synthetic.csv"
DAILY_AGGREGATED_PARQUET: Path = PROCESSED_DATA_DIR / "daily_demand_as_of.parquet"
MODEL_ARTIFACT_PATH: Path = ARTIFACTS_DIR / "best_forecasting_pipeline.joblib"
CONFORMAL_ARTIFACT_PATH: Path = ARTIFACTS_DIR / "conformal_calibrator.joblib"

# ==============================================================================
# 2. HOTEL INVENTORY & PRICING SETUP
# ==============================================================================
# We model a mid-to-upscale 110-room hotel in India (e.g., business & leisure hub)
ROOM_CATEGORIES = {
    "Standard": {
        "capacity": 60,
        "base_adr": 3500.0,       # Average Daily Rate in INR
        "marginal_cost": 500.0,    # Housekeeping / laundry / consumables
        "walk_penalty_ratio": 1.75 # Multiplier of ADR to walk/re-accommodate guest
    },
    "Deluxe": {
        "capacity": 35,
        "base_adr": 6000.0,
        "marginal_cost": 800.0,
        "walk_penalty_ratio": 2.0
    },
    "Suite": {
        "capacity": 15,
        "base_adr": 12000.0,
        "marginal_cost": 1500.0,
        "walk_penalty_ratio": 2.5
    }
}

TOTAL_CAPACITY: int = sum(spec["capacity"] for spec in ROOM_CATEGORIES.values())

# ==============================================================================
# 3. FORECASTING TIMELINES & HORIZONS
# ==============================================================================
# Forecast is generated at cutoff_hour (23:59) on date T for arrivals at T + h
CUTOFF_HOUR: int = 23
MIN_HORIZON_DAYS: int = 1
MAX_HORIZON_DAYS: int = 14
KEY_HORIZONS: list[int] = [7, 14]

# Training / Validation / Test temporal cutoffs (Chronological splits)
# Data spans 2 years: 2024-01-01 to 2025-12-31 (~730 days)
TRAIN_START_DATE: str = "2024-01-01"
TRAIN_END_DATE: str = "2025-06-30"    # 18 months (~546 days)
VAL_START_DATE: str = "2025-07-01"
VAL_END_DATE: str = "2025-09-30"      # 3 months (~92 days) - Model selection & tuning
TEST_START_DATE: str = "2025-10-01"
TEST_END_DATE: str = "2025-12-31"     # 3 months (~92 days) - Untouched final test

# ==============================================================================
# 4. DECISION LAYER & ASYMMETRIC COSTS (Newsvendor)
# ==============================================================================
# Underage cost (Cu): Lost margin if demand > held capacity (or room sold out early)
# Overage cost (Co): Holding an empty room when demand was lower (or walk cost if overbooked)
DEFAULT_CONFIDENCE_LEVEL: float = 0.80  # 80% prediction interval target

# ==============================================================================
# 5. ENVIRONMENT & AUTHENTICATION (JWT & RBAC)
# ==============================================================================
API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
API_PORT: int = int(os.getenv("API_PORT", "8000"))
STREAMLIT_PORT: int = int(os.getenv("STREAMLIT_PORT", "8501"))

# JWT & Password Hashing
JWT_SECRET_KEY: str = os.getenv("JWT_SECRET_KEY", os.getenv("SECRET_KEY", "hotel-forecasting-super-secret-key-change-in-production-2026"))
JWT_ALGORITHM: str = os.getenv("JWT_ALGORITHM", "HS256")
JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv("JWT_ACCESS_TOKEN_EXPIRE_MINUTES", "60"))

# SQLite Application Database (users, auth status, audit)
APP_DB_PATH: Path = DATA_DIR / "app.db"

# Demo user credentials for scripts/seed_users.py
DEMO_MANAGER_NAME: str = os.getenv("DEMO_MANAGER_NAME", "Hotel General Manager")
DEMO_MANAGER_EMAIL: str = os.getenv("DEMO_MANAGER_EMAIL", "manager@hotel.com")
DEMO_MANAGER_PASSWORD: str = os.getenv("DEMO_MANAGER_PASSWORD", os.getenv("DEMO_MANAGER_PW", "ManagerPass@123"))

DEMO_FRONT_DESK_NAME: str = os.getenv("DEMO_FRONT_DESK_NAME", "Front Desk Receptionist")
DEMO_FRONT_DESK_EMAIL: str = os.getenv("DEMO_FRONT_DESK_EMAIL", "frontdesk@hotel.com")
DEMO_FRONT_DESK_PASSWORD: str = os.getenv("DEMO_FRONT_DESK_PASSWORD", "FrontDeskPass@123")

DEMO_PENDING_NAME: str = os.getenv("DEMO_PENDING_NAME", "New Trainee Receptionist")
DEMO_PENDING_EMAIL: str = os.getenv("DEMO_PENDING_EMAIL", "pending_trainee@hotel.com")
DEMO_PENDING_PASSWORD: str = os.getenv("DEMO_PENDING_PASSWORD", "TraineePass@123")
