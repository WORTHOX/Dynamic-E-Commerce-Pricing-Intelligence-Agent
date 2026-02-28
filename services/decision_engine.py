"""
services/decision_engine.py
----------------------------
Rule-Based Decision Engine Layer for the E-Commerce Pricing Intelligence Agent.

Applies structured pricing heuristics to recommend:
  - INCREASE  : raise price to capture margin
  - DECREASE  : lower price to defend volume / clear inventory
  - HOLD      : maintain current price

All thresholds are named constants — no magic numbers in rule bodies.

Decision output:
  {
    "recommended_price":  float,
    "decision_type":      "increase" | "decrease" | "hold",
    "reasoning_factors":  list[str],
  }
"""

import logging
import math
from typing import Any

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Rule thresholds (named constants — tune without touching rule logic)
# --------------------------------------------------------------------------- #
COMPETITOR_UNDERCUT_THRESHOLD = 0.10   # >10 % cheaper triggers defensive action
COMPETITOR_PREMIUM_THRESHOLD  = 0.08   # >8 % more expensive allows upward move
DEMAND_HIGH_THRESHOLD         = 10.0   # units/day rolling avg considered "strong"
DEMAND_LOW_THRESHOLD          = 2.0    # units/day rolling avg considered "weak"
INVENTORY_HIGH_THRESHOLD      = 500.0  # cumulative qty considered "high stock"
INVENTORY_LOW_THRESHOLD       = 50.0   # cumulative qty considered "low stock"
PRICE_INCREASE_FACTOR         = 0.05   # recommend +5 % of current_price
PRICE_DECREASE_FACTOR         = 0.07   # recommend -7 % of current_price
VOLATILITY_HIGH_THRESHOLD     = 0.50   # std dev as proxy for noisy pricing


def make_pricing_decision(
    current_price: float,
    competitor_price: float,
    features: dict[str, Any],
) -> dict[str, Any]:
    """
    Apply the rule-based pricing decision engine.

    Parameters
    ----------
    current_price : float
        The merchant's current selling price.
    competitor_price : float
        The reference competitor price (from API input).
    features : dict[str, Any]
        Output of ``services.feature_engineering.generate_product_features()``.

    Returns
    -------
    dict[str, Any]
        Keys:
        - ``recommended_price``  : float
        - ``decision_type``      : str   ('increase' | 'decrease' | 'hold')
        - ``reasoning_factors``  : list[str]
    """
    if current_price <= 0:
        raise ValueError(f"current_price must be positive; got {current_price}")
    if competitor_price <= 0:
        raise ValueError(f"competitor_price must be positive; got {competitor_price}")

    demand_velocity: float  = features.get("demand_velocity", 0.0)
    sales_trend: str        = features.get("sales_trend", "stable")
    inventory_proxy: float  = features.get("inventory_proxy", 0.0)
    price_volatility: float = features.get("price_volatility", 0.0)

    # Pre-compute comparison ratios
    price_gap_ratio = (competitor_price - current_price) / current_price  # +ve = cheaper competitor
    competitor_cheaper = price_gap_ratio < -COMPETITOR_UNDERCUT_THRESHOLD
    competitor_pricier = price_gap_ratio > COMPETITOR_PREMIUM_THRESHOLD

    demand_strong = demand_velocity >= DEMAND_HIGH_THRESHOLD
    demand_weak   = demand_velocity <= DEMAND_LOW_THRESHOLD
    inventory_high = inventory_proxy >= INVENTORY_HIGH_THRESHOLD
    inventory_low  = inventory_proxy <= INVENTORY_LOW_THRESHOLD

    decision_type      = "hold"
    reasoning_factors: list[str] = []
    recommended_price  = current_price

    # ── Rule 1: Defensive decrease — competitor undercuts AND conditions weak ──
    if competitor_cheaper and (sales_trend == "decreasing" or demand_weak) and inventory_high:
        decision_type     = "decrease"
        recommended_price = _round_price(current_price * (1 - PRICE_DECREASE_FACTOR))
        reasoning_factors.extend([
            f"Competitor price is {abs(price_gap_ratio)*100:.1f}% below ours — elasticity risk.",
            f"Sales trend is '{sales_trend}' with low demand velocity ({demand_velocity:.1f} units/day).",
            "High inventory warrants volume-preserving markdown.",
        ])

    # ── Rule 2: Opportunistic increase — strong demand AND competitor is pricier ─
    elif competitor_pricier and demand_strong and sales_trend == "increasing":
        decision_type     = "increase"
        recommended_price = _round_price(current_price * (1 + PRICE_INCREASE_FACTOR))
        reasoning_factors.extend([
            f"Competitor price is {price_gap_ratio*100:.1f}% above ours — margin headroom exists.",
            f"Demand velocity is strong at {demand_velocity:.1f} units/day.",
            "Upward sales trend supports a price lift without volume risk.",
        ])

    # ── Rule 3: Soft decrease — competitor cheaper AND low inventory ───────────
    elif competitor_cheaper and inventory_low and sales_trend != "increasing":
        decision_type     = "decrease"
        recommended_price = _round_price(current_price * (1 - PRICE_DECREASE_FACTOR * 0.5))
        reasoning_factors.extend([
            f"Competitor price undercuts ours by {abs(price_gap_ratio)*100:.1f}%.",
            "Low inventory limits room for aggressive cuts; applying conservative markdown.",
            "Trend is not improving; price stimulus may revive velocity.",
        ])

    # ── Rule 4: Moderate increase — demand high, competitor not cheaper ─────────
    elif demand_strong and not competitor_cheaper and price_volatility < VOLATILITY_HIGH_THRESHOLD:
        decision_type     = "increase"
        recommended_price = _round_price(current_price * (1 + PRICE_INCREASE_FACTOR * 0.6))
        reasoning_factors.extend([
            f"Demand velocity ({demand_velocity:.1f} units/day) indicates strong market pull.",
            "Competitor is not undercutting; price stability in market supports increase.",
            "Low price volatility signals a stable pricing environment.",
        ])

    # ── Rule 5: Hold — mixed or insufficient signals ────────────────────────────
    else:
        reasoning_factors.extend([
            "No dominant pricing signal detected.",
            f"Current price is within acceptable bounds of competitor (gap: {price_gap_ratio*100:.1f}%).",
            f"Sales trend is '{sales_trend}'; demand velocity at {demand_velocity:.1f} units/day.",
            "Maintaining current price to avoid unnecessary market disruption.",
        ])

    logger.info(
        "Decision='%s', recommended=%.4f (current=%.4f, competitor=%.4f)",
        decision_type,
        recommended_price,
        current_price,
        competitor_price,
    )

    return {
        "recommended_price": recommended_price,
        "decision_type": decision_type,
        "reasoning_factors": reasoning_factors,
    }


def _round_price(price: float) -> float:
    """Round price to 2 decimal places; guard against floating-point drift."""
    return round(math.floor(price * 100) / 100, 2)
