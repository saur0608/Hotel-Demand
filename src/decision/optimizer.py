"""
Asymmetric Decision Layer: Newsvendor Inventory & Overbooking Optimization.

Translates ML point forecasts and Conformal Prediction intervals into
operational revenue management decisions:
1. Authorized Overbooking Limits (rooms permitted to sell above physical capacity)
2. Inventory Protection / Rooms to Hold (protected for high-yield late bookings)
3. Newsvendor financial loss evaluation comparing asymmetric policy against naive heuristics.
"""

from typing import Dict, List, Tuple
import logging
import numpy as np
import pandas as pd
from scipy.stats import norm

from src.config import ROOM_CATEGORIES

logger = logging.getLogger(__name__)


class NewsvendorDecisionLayer:
    """
    Hospitality Decision Engine under Asymmetric Costs.
    
    Economic parameters per category:
    - Underage Cost (Cu): Lost profit margin = ADR - marginal_cost
    - Overage Cost (Co): Re-accommodation walk cost = walk_penalty_ratio * ADR
    - Critical Fractile: CR = Cu / (Cu + Co)
    """

    def __init__(self, room_specs: Dict = None):
        self.room_specs = room_specs or ROOM_CATEGORIES

    def get_cost_parameters(self, category: str) -> Tuple[float, float, float]:
        """
        Compute Cu (underage), Co (overage), and CR (critical ratio) for a room category.
        """
        spec = self.room_specs[category]
        adr = spec["base_adr"]
        marginal_cost = spec["marginal_cost"]
        walk_multiplier = spec["walk_penalty_ratio"]

        # Cu: Opportunity cost of an unsold/turned-away room
        cu = adr - marginal_cost

        # Co: Penalty of walking a confirmed guest (taxi + competitor room + voucher)
        co = walk_multiplier * adr

        # Critical Fractile
        critical_ratio = cu / (cu + co)
        return cu, co, critical_ratio

    def optimize_decision(
        self,
        room_category: str,
        point_forecast: float,
        conformal_lower: float,
        conformal_upper: float,
        otb_rooms: int,
        horizon: int,
    ) -> Dict:
        """
        Generate operational recommendations for a specific query:
        - Overbooking allowance (rooms to sell beyond physical capacity)
        - Rooms to hold (inventory protected from discount channels)
        - Actionable plain-language summary for the Revenue Manager
        """
        spec = self.room_specs[room_category]
        capacity = spec["capacity"]
        cu, co, cr = self.get_cost_parameters(room_category)

        # 1. Asymmetric Demand Target (Q*) via Conformal Quantile Interpolation
        # Conformal interval represents [10th percentile, 90th percentile]
        # We linearly map critical ratio CR into the calibrated uncertainty band:
        cr_clamped = np.clip(cr, 0.05, 0.95)
        # Interpolate between conformal_lower (p=0.10) and conformal_upper (p=0.90)
        q_star = conformal_lower + ((cr_clamped - 0.10) / 0.80) * (conformal_upper - conformal_lower)
        q_star = np.clip(q_star, 0, capacity * 1.25)

        # 2. Authorized Overbooking Allowance
        # How many bookings above physical capacity should the PMS accept?
        # Only overbook if expected demand exceeds capacity and downside risk is acceptable
        if point_forecast > capacity * 0.92:
            # Overbooking buffer authorized above capacity
            raw_ob = q_star - capacity
            overbooking_rooms = int(max(0, np.floor(raw_ob)))
            # Cap maximum overbooking at 10% of capacity to protect hotel reputation
            overbooking_rooms = min(overbooking_rooms, int(np.ceil(capacity * 0.10)))
        else:
            overbooking_rooms = 0

        authorized_booking_limit = capacity + overbooking_rooms

        # 3. Rooms to Hold / Protect from Discount Channels for High-Yield Late Bookings
        # Applicable for horizons >= 3 days out
        remaining_capacity = max(0, authorized_booking_limit - otb_rooms)
        expected_late_pickup = max(0.0, point_forecast - otb_rooms)

        if horizon >= 3 and remaining_capacity > 0:
            # High-yield late bookings (walk-ins / corporate) pay full rack rate
            # Protect a conservative fraction of remaining expected pickup
            hold_target = int(np.clip(np.round(expected_late_pickup * (1.0 - cr)), 0, remaining_capacity))
        else:
            hold_target = 0

        # Available to sell to early/discount channels right now
        available_discount_rooms = max(0, remaining_capacity - hold_target)

        # 4. Human-Readable Action Summary
        if overbooking_rooms > 0:
            action_text = (
                f"High-demand period detected. Authorize {overbooking_rooms} overbooking room(s) "
                f"(Total booking cap: {authorized_booking_limit}/{capacity}). "
                f"Asymmetric risk protects against ₹{co:,.0f} walk cost while capturing ₹{cu:,.0f} net margin."
            )
        elif hold_target > 0:
            action_text = (
                f"Protect {hold_target} room(s) from discount channels for anticipated late-booking pickup. "
                f"Leave {available_discount_rooms} room(s) open for general sales."
            )
        else:
            action_text = (
                f"Normal operational pace. Maintain physical capacity ({capacity} rooms). "
                f"{remaining_capacity} room(s) remaining on open sale."
            )

        return {
            "room_category": room_category,
            "capacity": capacity,
            "point_forecast": round(point_forecast, 1),
            "conformal_interval": (round(conformal_lower, 1), round(conformal_upper, 1)),
            "critical_ratio": round(cr, 3),
            "underage_cost_inr": cu,
            "overage_cost_inr": co,
            "authorized_booking_limit": authorized_booking_limit,
            "authorized_overbooking_rooms": overbooking_rooms,
            "rooms_to_hold": hold_target,
            "available_discount_rooms": available_discount_rooms,
            "action_text": action_text,
        }

    def evaluate_financial_loss(self, test_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compare financial performance over the untouched test set under 3 policies:
        - Policy 1: Fixed Capacity Heuristic (never overbook, sell up to C)
        - Policy 2: Symmetric Point Forecast Policy (sell up to round(y_hat))
        - Policy 3: Asymmetric Conformal Newsvendor Policy (Our method)
        """
        logger.info("Evaluating financial performance under asymmetric costs...")
        policies = {"Fixed Capacity (No Overbook)": 0.0, "Point Forecast Heuristic": 0.0, "Asymmetric Newsvendor (Ours)": 0.0}

        for _, row in test_df.iterrows():
            cat = row["room_category"]
            cap = row["capacity"]
            actual_y = row["target_net_demand"]
            y_hat = row["pred_lightgbm"]
            c_low = row["conformal_lower"]
            c_high = row["conformal_upper"]
            otb = row["otb_rooms"]
            hz = row["horizon"]

            cu, co, cr = self.get_cost_parameters(cat)

            # Policy 1: Fixed Capacity (Limit = C)
            q1 = cap
            # Loss = lost margin on unsatisfied demand above capacity
            loss1 = cu * max(0, actual_y - q1)
            policies["Fixed Capacity (No Overbook)"] += loss1

            # Policy 2: Symmetric Point Forecast (Limit = round(y_hat))
            q2 = int(np.round(y_hat))
            if q2 > cap:
                # Walked guests if actual stays also exceed physical cap
                walked = max(0, min(actual_y, q2) - cap)
                loss2 = (co * walked) + (cu * max(0, actual_y - q2))
            else:
                loss2 = cu * max(0, actual_y - q2)
            policies["Point Forecast Heuristic"] += loss2

            # Policy 3: Asymmetric Newsvendor
            dec = self.optimize_decision(cat, y_hat, c_low, c_high, otb, hz)
            q3 = dec["authorized_booking_limit"]
            if q3 > cap:
                walked = max(0, min(actual_y, q3) - cap)
                loss3 = (co * walked) + (cu * max(0, actual_y - q3))
            else:
                loss3 = cu * max(0, actual_y - q3)
            policies["Asymmetric Newsvendor (Ours)"] += loss3

        results = []
        baseline_loss = policies["Fixed Capacity (No Overbook)"]
        for pol, loss in policies.items():
            savings = baseline_loss - loss
            results.append({
                "Policy": pol,
                "Total Loss (INR)": f"₹{loss:,.0f}",
                "Financial Savings vs Heuristic": f"₹{savings:,.0f}" if savings >= 0 else f"-₹{abs(savings):,.0f}",
                "Cost Reduction (%)": f"{(savings / baseline_loss) * 100:+.1f}%" if baseline_loss > 0 else "0.0%",
            })

        return pd.DataFrame(results)
