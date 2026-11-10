#!/usr/bin/env bash
# ==============================================================================
# Pipeline Execution Script: Hotel Occupancy & Room-Demand Forecasting
# Usage:
#   bash run.sh generate   # Generate synthetic data & run quality checks
#   bash run.sh train      # Run feature engineering, backtesting, & model tuning
#   bash run.sh test       # Execute test suite (leakage, preprocessing, inference)
#   bash run.sh serve-api  # Start FastAPI server on port 8000
#   bash run.sh serve-ui   # Start Streamlit dashboard on port 8501
#   bash run.sh all        # End-to-end run
# ==============================================================================

set -euo pipefail

# Locate virtualenv python if present, else fall back to python3
if [ -f "./.venv/bin/python" ]; then
    PYTHON="./.venv/bin/python"
    PYTEST="./.venv/bin/pytest"
    UVICORN="./.venv/bin/uvicorn"
    STREAMLIT="./.venv/bin/streamlit"
else
    PYTHON="python3"
    PYTEST="pytest"
    UVICORN="uvicorn"
    STREAMLIT="streamlit"
fi

STEP=${1:-all}

case "$STEP" in
  generate)
    echo "==> [Stage 2] Generating synthetic bookings ledger & quality checks..."
    $PYTHON -m src.data.generate_data
    ;;
  seed)
    echo "==> [Auth] Seeding demo users with bcrypt passwords..."
    $PYTHON scripts/seed_users.py
    ;;
  train)
    echo "==> [Stages 3-5] Training models, rolling backtest, uncertainty & artifacts..."
    $PYTHON -m src.models.train_pipeline
    ;;
  test)
    echo "==> [Stage 8] Running automated tests (leakage, preprocessing, inference, auth)..."
    $PYTEST tests/ -v
    ;;
  serve-api)
    echo "==> [Stage 7] Starting FastAPI server on port 8000..."
    $UVICORN app.api:app --host 0.0.0.0 --port 8000 --reload
    ;;
  serve-ui)
    echo "==> [Stage 7] Starting Streamlit dashboard on port 8501..."
    $STREAMLIT run app/streamlit_app.py --server.port 8501
    ;;
  all)
    echo "==> Running full end-to-end pipeline..."
    $PYTHON -m src.data.generate_data
    $PYTHON scripts/seed_users.py
    $PYTHON -m src.models.train_pipeline
    $PYTEST tests/ -v
    echo "==> Pipeline complete. Artifacts saved in artifacts/"
    ;;
  *)
    echo "Usage: $0 {generate|seed|train|test|serve-api|serve-ui|all}"
    exit 1
    ;;
esac
