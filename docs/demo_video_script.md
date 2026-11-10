# 🎬 3 to 5 Minute Demo Video Script & Walkthrough Defense Guide

**Project**: Hotel Occupancy & Room-Demand Forecasting (7/14-Day Horizon)  
**Presenter**: Machine Learning Engineer  
**Target Duration**: 3 minutes 45 seconds  

---

## ⏱️ Video Script Breakdown

### [0:00 - 0:45] Act 1: The Real Problem (Asymmetric Risk, Not MSE)
- **Visual**: Title slide / Terminal showing `git status` and project tree.
- **Presenter (Narration)**:
  > *"Hello! Most ML projects treat hotel forecasting as a generic time series, predicting daily stays using ARIMA or Prophet. But in actual hotel revenue management, that approach fails for two reasons:*
  > 
  > *First, it ignores the reservation ledger: days before arrival, we already have confirmed bookings on the books.*
  > 
  > *Second, errors are asymmetric. If we underestimate demand and turn away a guest with a confirmed reservation, re-accommodating them at a competitor hotel costs 1.75 to 2.5 times the room rate in taxi, refunds, and lost loyalty. But an empty room only loses our profit margin.*
  > 
  > *Today, I’m presenting an end-to-end forecasting and decision system built around one standout idea: **booking-pace pickup modeling, 80% conformal prediction intervals, and a Newsvendor decision layer**."*

---

### [0:45 - 1:30] Act 2: The Standout Idea (Booking-Pace Pickup)
- **Visual**: Show `artifacts/eda_plots/01_booking_pace_curves.png` and `artifacts/eda_plots/06_model_comparison_by_horizon.png`.
- **Presenter (Narration)**:
  > *"Here is the core hospitality data signal: our booking pace curves. As lead time counts down from 60 days to arrival, different room categories pick up demand at vastly different velocities. Standard rooms book last-minute; luxury suites book weeks in advance.*
  > 
  > *Instead of predicting demand from scratch, our model forecasts **net pickup**—future reservations minus cancellations—anchored to current On-The-Books reservations.*
  > 
  > *Notice the result: at a 7-day horizon, naive weekly baselines suffer a 24% WAPE. By incorporating booking pace, our Gradient Boosted Trees model slashes error down to **8.88%**, beating the industry pickup benchmark across all 14 horizons."*

---

### [1:30 - 2:30] Act 3: Live Engineering & Leakage-Safe Architecture
- **Visual**: Terminal running `pytest tests/ -v`. Show all 27 tests passing.
- **Presenter (Narration)**:
  > *"Let’s look at the engineering. Multi-horizon forecasting is notorious for look-ahead data leakage. In `tests/test_data_leakage.py`, we have automated tests verifying two non-negotiable invariant rules:*
  > 
  > *1. Every feature is computed strictly as of prediction date T, 23:59. If a guest cancelled on day T+2, that cancellation is invisible at day T and treated as active on the books.*
  > *2. All autoregressive demand lags are relative to as-of date T, NOT the stay date in the future. Accessing demand at (stay_date - 7) when forecasting 14 days out would peek 7 days into the future.*
  > 
  > *In addition, `tests/test_auth.py` tests our bcrypt password hashing, public self-registration, sliding-window rate limiting (5 failed attempts per min), and RBAC boundaries.*
  > 
  > *As you can see, all 27 tests pass in under 5 seconds."*

---

### [2:30 - 3:15] Act 4: Conformal Uncertainty & Asymmetric Decision Layer
- **Visual**: Show `artifacts/eda_plots/08_shap_summary_beeswarm.png` and Table 4 (Newsvendor Loss).
- **Presenter (Narration)**:
  > *"Point forecasts are not enough to make decisions. We implemented variance-stabilized Split Conformal Prediction, achieving **81.31% empirical coverage** on the untouched test partition.*
  > 
  > *Then comes our decision layer. Using the Newsvendor critical ratio, we weigh the cost of empty room spoilage against the cost of walking a guest.*
  > 
  > *If you use a naive symmetric point forecast to set booking limits, the hotel suffers ₹75 million in walk penalties on the test period! Our asymmetric Newsvendor policy cuts that loss by **47.9%**, authorizing targeted overbooking only when cancellation probability reliably absorbs the excess."*

---

### [3:15 - 4:15] Act 5: Live Dashboard, Authentication & Manager Approvals Demo
- **Visual**: Open Streamlit dashboard (`localhost:8501`).
- **Presenter (Narration)**:
  > *"Here is the production dashboard. Notice our secure entry:*
  > - *1. **Sign In & Register Tabs**: New staff register under the 'Register' tab. Notice that self-registration automatically forces `role='front_desk'` and `status='pending'`. The client can never elevate their own privileges.*
  > - *2. **Pending User Lockout**: Clicking 'Trainee (Pending)' demonstrates that unapproved accounts are rejected with a clear message: 'Account registration is pending manager approval'.*
  > - *3. **Manager Access & Approvals**: Logging in as the Revenue Manager unlocks the manager-only **Access Requests** tab. The manager can see pending registration requests and approve or reject them with a single click.*
  > - *4. **RBAC in the Dashboard**: Front Desk users can view daily demand forecasts, but the Newsvendor Overbooking controls and SHAP analytics tabs are strictly locked.*
  > 
  > *Every forecast queries our trained `.joblib` model artifact—100% dynamic, zero hardcoding."*

---

### [4:15 - 4:45] Act 6: Responsible AI & Wrap-Up
- **Presenter (Narration)**:
  > *"Finally, we designed this system for responsible governance: zero personal data is collected, moving Indian festival calendars prevent drift, and high-impact actions over 5 rooms require human-in-the-loop sign-off.*
  > 
  > *The entire repository is reproducible via a single command: `bash run.sh all`. Thank you!"*

---

## 🎯 Live Walkthrough Defense Guide: Tricky Questions & Expert Answers

| Likely Interviewer Question | Trap / Why They Ask | Your Exact Winning Answer |
| :--- | :--- | :--- |
| **"Why did you use WAPE instead of MAPE or MSE?"** | Testing if you understand hospitality zero-demand edge cases. | *"In luxury suites or small boutique categories, realized daily demand can be zero or single digits. Standard MAPE divides by zero or explodes. WAPE weights absolute errors by total volume ($\sum \|y - \hat{y}\| / \sum y$), making it scale-stable across room tiers."* |
| **"If your model predicts 63 rooms for a 60-room category, why not just sell 63 rooms?"** | Testing if you understand asymmetric risk vs point prediction. | *"Because 63 is the conditional mean, with a 50% chance actual demand is even higher. Walking 3 guests costs $2\times$ ADR each in compensation and competitor lodging. Our Newsvendor critical ratio shades capacity down to 60 or 61 unless cancellation certainty is $>74\%$."* |
| **"How do you prevent data leakage when using lag features in a 14-day horizon forecast?"** | Testing multi-step time-series leakage knowledge. | *"All lag features are indexed by $T$ (as-of date), NOT $S$ (stay date). For horizon $h=14$, $S - 7 = T + 7$, which is 7 days in the future! By indexing strictly to $T$ (e.g. $T0, T-1, T-7$), all lags are physically observed before the forecast is generated."* |
| **"Why use Conformal Prediction over quantile regression loss?"** | Testing uncertainty quantification rigor. | *"Quantile regression optimizes a pinball loss during training, but does not provide distribution-free finite-sample coverage guarantees. Conformal prediction guarantees valid empirical coverage on unseen data regardless of model complexity."* |
