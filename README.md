# 🏨 Hotel Occupancy & Room-Demand Forecasting (7/14-Day Horizon)

An enterprise-grade, leakage-safe machine learning system for hotel room-demand forecasting and inventory optimization. Built for operational revenue management under asymmetric risk.

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-27%20passed-brightgreen.svg)]()
[![License](https://img.shields.io/badge/license-MIT-green.svg)]()

---

## 🌟 The Standout Idea: Booking-Pace (Pickup) Modeling

Traditional time-series models (e.g. ARIMA, Prophet) treat hotel demand as a 1D sequence of past total stays, completely ignoring the most valuable operational signal: **the reservation ledger**.

In hotel revenue management:
- $h$ days before arrival date $T+h$, a substantial portion of guests have already booked (**On-The-Books, OTB**).
- What needs forecasting is the **net pickup** (future bookings minus cancellations) between day $T$ and arrival day $T+h$.
- This project models booking-pace velocity curves across lead-times, pairs them with calendar dynamics and Indian holiday shocks, produces **80% conformal prediction intervals**, and feeds them into an asymmetric **Newsvendor decision layer** (recommending rooms to protect or authorize for overbooking).

---

## 📁 Repository Structure

```
hotal/
├── data/
│   ├── raw/                  # 71,382 synthetic booking transactions (2024-2025)
│   ├── processed/            # 30,702 leakage-safe daily as-of pickup feature rows
│   └── app.db                # SQLite user authentication store (gitignored)
├── src/
│   ├── config.py             # Room capacities, ADR, asymmetric costs, reproducibility seed
│   ├── auth.py               # SQLite user model, bcrypt, rate limiting, JWT, RBAC dependencies
│   ├── data/
│   │   └── generate_data.py  # Synthetic PMS generator & automated quality checks
│   ├── features/
│   │   └── build_features.py # Leakage-safe pickup curves, lags (T0, T-7), festival features
│   ├── models/
│   │   ├── baselines.py      # Naive (same weekday), Seasonal Naive, Historical Pickup
│   │   ├── candidate_models.py # Ridge Regression & Gradient Boosted Trees Pipelines
│   │   ├── conformal.py      # Locally adaptive 80% Split-Conformal calibrator
│   │   ├── backtest.py       # Rolling backtest & horizon evaluation
│   │   ├── metrics.py        # MAE, RMSE, WAPE (%) calculation and Markdown formatting
│   │   └── train_pipeline.py # End-to-end training & artifact persistence orchestrator
│   ├── decision/
│   │   └── optimizer.py      # Asymmetric Newsvendor decision layer (hold/overbook)
│   └── explainability/
│       └── shap_explain.py   # TreeSHAP summary, single prediction & error post-mortem
├── notebooks/
│   ├── 01_exploratory_data_analysis.ipynb # Interactive EDA walkthrough notebook
│   └── eda_analysis.py       # Automated script generating 5 publication-ready EDA plots
├── tests/
│   ├── test_auth.py          # 11 tests: bcrypt hashing, registration validation, rate limits, RBAC, approvals
│   ├── test_data_leakage.py  # Strict verification of zero look-ahead leakage
│   ├── test_preprocessing.py # Schema integrity, physical capacity bounds, null checks
│   └── test_inference.py     # FastAPI endpoints, dynamic inference, interval bounds
├── app/
│   ├── api.py                # FastAPI REST service with JWT Auth, Registration & RBAC
│   └── streamlit_app.py      # Streamlit dashboard with Sign In / Register tabs & Access Requests page
├── scripts/
│   └── seed_users.py         # Seeds demo manager, front desk, and pending accounts
├── artifacts/
│   ├── best_forecasting_pipeline.joblib # Winning trained Gradient Boosting model
│   ├── conformal_calibrator.joblib      # Calibrated conformal prediction bounds
│   ├── pickup_lookup.joblib             # Historical pace reference lookup
│   └── eda_plots/                       # 10 generated visualization plots
├── docs/
│   ├── problem_definition.md # Mathematical formulation & timeline specs
│   ├── technical_report.md   # Comprehensive technical report & governance outline
│   ├── explainability_report.md # SHAP and failure mode post-mortem
│   └── demo_video_script.md  # 3-5 minute presentation script & live defense guide
├── requirements.txt          # Python dependencies
├── .env.example              # Environment variables template
├── run.sh                    # End-to-end execution script
└── README.md                 # Project documentation & walkthrough guide
```

---

## ⚡ Quickstart & Reproduction

### 1. Environment Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

### 2. Execution Commands via `run.sh`
```bash
# 1. Generate synthetic data & run quality checks (71,382 records)
bash run.sh generate

# 2. Seed SQLite demo users (manager, front desk, pending trainee)
bash run.sh seed

# 3. Train candidate models, backtest, calibrate conformal intervals & persist artifacts
bash run.sh train

# 4. Run automated pytest test suite (27 passed tests: leakage, preprocessing, inference, auth)
bash run.sh test

# 5. Launch FastAPI REST service with JWT Auth & RBAC (port 8000)
bash run.sh serve-api

# 6. Launch Streamlit interactive dashboard with Sign In / Register & RBAC (port 8501)
bash run.sh serve-ui

# Or execute full pipeline end-to-end:
bash run.sh all
```

---

## 🚀 Deployment Architecture & Cloud Hosting

To ensure scalability and reliable authentication persistence, the system separates the frontend dashboard from the API backend:

> [!IMPORTANT]
> **Separation of Concerns & SQLite Persistence**:
> - **FastAPI Backend (Entry: `main.py` -> `uvicorn main:app`)**: Deployed on a platform with persistent disk storage (e.g., **Render** using [`render.yaml`](file:///Users/saurabhsingh/Downloads/hotal/render.yaml), or **Railway** / **Docker**). SQLite (`data/app.db`) stores user accounts, bcrypt password hashes, and manager approval states; serverless platforms (like Vercel or AWS Lambda) reset ephemeral filesystem state on each cold start, which would erase registered users.
> - **Streamlit Dashboard (Entry: `app/streamlit_app.py`)**: Deployed independently on **Streamlit Community Cloud** (using [`requirements-app.txt`](file:///Users/saurabhsingh/Downloads/hotal/requirements-app.txt)). The dashboard securely connects to the deployed FastAPI backend via REST API.
> - **Entrypoint Structure**: Root-level `main.py` cleanly exposes the FastAPI instance (`from app.api import app`), preventing `[FASTAPI_ENTRYPOINT_NOT_FOUND]` errors while keeping application logic organized in `app/api.py`. Tests are excluded from deployments via `.vercelignore`, `.dockerignore`, and `.slugignore`.

---

## 📊 Benchmark Backtest Results (Untouched Test Partition: Oct–Dec 2025)

### Overall Model Performance
| Model | MAE | RMSE | WAPE (%) [Lower is better] |
| :--- | :---: | :---: | :---: |
| **1. Naive (Same Weekday)** | 7.475 | 10.315 | 24.02% |
| **2. Seasonal Naive (4-Wk)** | 6.308 | 9.205 | 20.27% |
| **3. Historical Pickup Baseline** | 3.111 | 5.179 | 10.00% |
| **4. Ridge Regression** | 3.069 | 4.748 | 9.86% |
| **5. Gradient Boosted Trees (Winning)** | **2.792** | **4.275** | **8.97%** |

### Horizon Progression (WAPE %)
- **Horizon 1 Day**: Historical Pickup: 3.81% | Gradient Boosting: **5.08%**
- **Horizon 7 Days (Key Milestone)**: Naive: 24.06% | Historical Pickup: 9.20% | Gradient Boosting: **8.88%**
- **Horizon 14 Days (Key Milestone)**: Naive: 23.99% | Historical Pickup: 16.19% | Gradient Boosting: **12.04%**

---

## 🛡️ Uncertainty & Asymmetric Decision Layer

### 1. 80% Conformal Prediction Intervals
- **Method**: Locally-adaptive, variance-stabilized Split Conformal Prediction ($s_i = \frac{|y_i - \hat{y}_i|}{\sqrt{\hat{y}_i}}$).
- **PICP (Coverage)**: **81.31%** (Target $\ge 80\%$).
- **MPIW (Mean Width)**: **9.02 rooms** ($\pm 1.8$ rooms at $h=1$, expanding smoothly to $\pm 6.8$ rooms at $h=14$).

### 2. Newsvendor Economic Loss Comparison
Walking a confirmed guest costs $1.75\times - 2.5\times$ ADR in taxi, refunds, and competitor re-accommodation.
- **Fixed Capacity Heuristic**: ₹34,014,400 loss (excess empty room spoilage).
- **Naive Point Forecast Heuristic**: ₹74,983,350 loss (**catastrophic walk penalties**).
- **Asymmetric Conformal Newsvendor (Ours)**: **₹39,085,725 loss** (**47.9% cost reduction** vs naive point forecast).

---

## 🔒 Security and Privacy

### 1. Authentication & Account Lifecycle
- **Storage**: SQLite database at `data/app.db` (strictly ignored by `.gitignore`). Schema: `id, name, email (UNIQUE), password_hash (bcrypt), role, status, created_at`.
- **Public Registration (`POST /auth/register`)**: Open to new staff. Validates email format, minimum password length ($\ge 8$ chars), and password confirmation match.
- **Zero Client Role Injection**: The server strictly enforces `role='front_desk'` and `status='pending'` for all registrations. Clients can never elevate themselves to `manager` or self-approve.
- **Manager Approval Flow**:
  - `GET /admin/requests`: Manager reviews pending staff accounts.
  - `POST /admin/requests/{id}/approve`: Manager approves account, immediately granting login access.
  - `POST /admin/requests/{id}/reject`: Manager rejects unauthorized account requests.
- **Login Status Enforcement (`POST /auth/login`)**: Only accounts with `status='approved'` receive JWT tokens. Pending users receive HTTP 403 Forbidden with `"Account registration is pending approval by a hotel manager"`.
- **Sliding Window Rate Limiting**: Built-in protection against brute-force attacks (5 failed attempts per minute per email/IP address triggers **HTTP 429 Too Many Requests**).
- **JWT Authorization**: 60-minute signed JSON Web Tokens (`HS256`). Required for `/predict`, `/recommend`, `/auth/me`, and `/admin/*`.

### 2. Role-Based Access Control (RBAC) Matrix
Role permissions are strictly enforced at the FastAPI dependency level (`require_role`), ensuring direct HTTP/curl calls cannot bypass the UI:

| Role | Sign In / Register | Demand Forecast (`/predict`) | Asymmetric Decisions (`/recommend`) | Access Requests (`/admin/*`) |
| :--- | :---: | :---: | :---: | :---: |
| **Public / Unauthenticated** | ✅ | ❌ (401) | ❌ (401) | ❌ (401) |
| **Pending User** | ❌ (403 Pending) | ❌ (401) | ❌ (401) | ❌ (401) |
| **Front Desk (`front_desk`)** | ✅ | ✅ (200) | ❌ (403 Forbidden) | ❌ (403 Forbidden) |
| **Revenue Manager (`manager`)** | ✅ | ✅ (200) | ✅ (200) | ✅ (200) |

### 3. Demo User Credentials (from `scripts/seed_users.py`)
| Role | Email | Password | Status | Intended Use |
| :--- | :--- | :--- | :--- | :--- |
| **Manager** | `manager@hotel.com` | `ManagerPass@123` | `approved` | Full operational access & approving pending staff |
| **Front Desk** | `frontdesk@hotel.com` | `FrontDeskPass@123` | `approved` | Daily check-in demand view (decision layer blocked) |
| **Pending Trainee** | `pending_trainee@hotel.com` | `TraineePass@123` | `pending` | Video walkthrough showing live manager approval flow |

### 4. Data Privacy & Zero Real PII
- **Synthetic Reservation Ledger**: All guest identifiers (`BK-000001` through `BK-071382`) and bookings are algorithmically synthesized.
- **No Real Personal Data**: Zero real customer personal data, phone numbers, credit cards, or actual addresses are stored or handled.
- **Zero Secrets in Git**: Secret keys, database paths, and demo passwords use environment variables with sensible defaults defined in `.env.example`.

### 5. Operational Governance & Guardrails
- **Anomaly Velocity Alerts**: 3-day booking pace velocity alerts (`pickup_last_3d == 0`) notify managers of demand anomalies or channel outages.
- **Cold-Start Fallback**: Properties or categories with $< 60$ days of pace history fallback to baseline occupancy curves.
- **Manager-in-the-Loop Safeguards**: Authorizing overbooking $> 5$ rooms or holding $> 25\%$ capacity requires manager sign-off.
