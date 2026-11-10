# Hotel Occupancy & Room-Demand Forecasting: Problem Definition & Architecture

## 1. Executive Summary & Business Decision

Hotel revenue management is inherently an **asymmetric decision problem under uncertainty**:
- **If demand is underestimated** (underage), rooms are either held empty unnecessarily, sold out too early at lower rates, or turn away profitable high-paying travelers.
- **If demand is overestimated** (overage), the hotel either holds unallocated rooms that spoil at midnight (empty room opportunity cost = lost net margin), or overbooks excessively resulting in **"walking" a confirmed guest** to a competitor hotel (re-accommodation costs, taxi, brand damage, refunds).

### The Core Decision to Automate
Every evening at **23:59**, the revenue manager must determine inventory controls for the next **1 to 14 days**:
1. **Rooms to Hold / Protect**: Minimum rooms reserved for high-yield last-minute bookings.
2. **Authorized Overbooking Level**: Number of rooms permitted to be sold beyond physical capacity to offset anticipated cancellations and no-shows.

---

## 2. Formal Problem Formulation

| Concept | Specification | Detail |
| :--- | :--- | :--- |
| **Prediction Unit** | `(stay_date, room_category)` | Daily room count required for each distinct inventory pool. |
| **Room Categories** | `Standard` (60 rooms)<br>`Deluxe` (35 rooms)<br>`Suite` (15 rooms) | Total physical capacity = 110 rooms. |
| **Prediction Time Point ($T_{\text{as\_of}}$)** | 23:59:59 on date $T$ | Strict cut-off: Only data recorded $\le T$ is visible. |
| **Forecast Horizons ($h$)** | $h \in \{1, 2, \dots, 14\}$ days | Target arrival date = $T + h$. Key milestones: $h=7$ and $h=14$. |
| **Target Variable ($Y_{t, c}$)** | Realized Net Occupied Rooms | Final rooms occupied on `stay_date` $t$ for category $c$ after all cancellations, modifications, and no-shows. |
| **Information Set ($\mathcal{F}_T$)** | All records $\le T$ | Cumulative On-The-Books (OTB) bookings, cancellations, pace curves, calendar & festival calendars. |

```
Timeline Illustration:
========================================================================================
[Past History ...] ===> [ Prediction Date T (23:59) ] ------------ Horizon h ------------> [ Stay Date T+h ]
                       ^                                                                   ^
                       |--- Information Cutoff (OTB_h, historical pickup, lags)            |--- Target: Realized Demand Y
```

---

## 3. The Standout Idea: Booking-Pace (Pickup) Modeling

Standard time-series forecasting (ARIMA, Prophet) treats hotel demand as a 1D sequence of past realized occupancies. This ignores the single most valuable data asset in hospitality: **the reservation ledger**.

On prediction date $T$, for a stay date $T+h$, we already know:
1. **$OTB(T, T+h)$ (On-The-Books)**: How many guests have already booked and not yet cancelled as of $T$.
2. **Booking Pace ($\Delta_{pace}$)**: How many new reservations were booked in the last 3, 7, and 14 days for that stay date.
3. **Cancellation Velocity ($\Delta_{canc}$)**: Proportion of OTB that has historically cancelled between $h$ days out and arrival.
4. **Historical Pickup Curve**: On average, for a given room category and weekday/festival, what fraction of total demand books between $h$ days out and day 0?

Our forecasting model combines:
$$\widehat{Y}(T+h) = f\Big(OTB(T, T+h), \text{Pace features}, \text{Calendar/Holiday features}, \text{Historical demand lags}\Big)$$

---

## 4. Evaluation & Success Metrics

### 4.1 Forecast Accuracy Metrics
Evaluated strictly on the untouched out-of-time test partition (and validation partition during tuning) across horizons $h=1..14$ and room categories:

1. **MAE (Mean Absolute Error)**:
   $$\text{MAE} = \frac{1}{N} \sum_{i=1}^N |y_i - \hat{y}_i|$$
2. **RMSE (Root Mean Squared Error)**:
   $$\text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2}$$
3. **WAPE (Weighted Absolute Percentage Error)**:
   Hospitality industry standard (handles zero or low-demand days without division-by-zero bias of MAPE):
   $$\text{WAPE} = \frac{\sum_{i=1}^N |y_i - \hat{y}_i|}{\sum_{i=1}^N y_i}$$

### 4.2 Uncertainty & Interval Metrics
Targeting an **80% Prediction Interval** $[L_{0.10}, U_{0.90}]$:
1. **PICP (Prediction Interval Coverage Probability)**:
   $$\text{PICP} = \frac{1}{N} \sum_{i=1}^N \mathbb{I}(L_i \le y_i \le U_i) \quad (\text{Target} \ge 80\%)$$
2. **MPIW (Mean Prediction Interval Width)**:
   $$\text{MPIW} = \frac{1}{N} \sum_{i=1}^N (U_i - L_i) \quad (\text{Sharpness metric: narrower is better at equal coverage})$$

### 4.3 Decision Layer Metric: Newsvendor Loss
For room category with marginal profit margin (ADR - marginal cost) $C_u$ and walk penalty $C_o$:
$$\text{Loss}(Q, y) = C_u \cdot \max(0, y - Q) + C_o \cdot \max(0, Q - y)$$
Where $Q$ is the recommended capacity / booking limit. We evaluate whether our decision layer minimizes realized operational loss compared to deterministic mean capacity allocations.

---

## 5. Baselines for Fair Comparison
1. **Naive (Last Week Same Weekday)**:
   $$\hat{Y}_{T+h} = Y_{T+h - 7}$$
2. **Seasonal Naive (4-Week Rolling Same Weekday Average)**:
   $$\hat{Y}_{T+h} = \frac{1}{4} \sum_{w=1}^4 Y_{T+h - 7w}$$
3. **OTB + Historical Average Pickup Baseline**:
   $$\hat{Y}_{T+h} = OTB(T, T+h) + \bar{\Delta}_{pickup}(h, \text{weekday})$$
4. **Machine Learning Candidates**:
   - Model A: Regularized Linear Model (`Ridge` / `ElasticNet`)
   - Model B: Gradient Boosted Trees (`LightGBM`)
