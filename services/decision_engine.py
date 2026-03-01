"""
services/decision_engine.py
----------------------------
Band-Anchored Parabolic Pricing Engine — E-Commerce Pricing Intelligence Agent.

PHILOSOPHY
----------
Two ideas are combined into a single coherent formula:

  1. TARGET BAND (demand-scaled premium above competitor)
     ─────────────────────────────────────────────────────
     We never price AT competitor — we always aim for a profitable premium:

       target_mult  = PREMIUM_HIGH − demand_scale × PREMIUM_RANGE
       target_price = competitor_price × target_mult

     • Low  demand (→ demand_scale ≈ 0) → target_mult ≈ 1.20 (20% above competitor)
     • High demand (→ demand_scale ≈ 1) → target_mult ≈ 1.05 ( 5% above competitor)

     This reflects price elasticity: niche buyers tolerate bigger premiums;
     high-volume buyers are more price-sensitive.

  2. PARABOLIC CONVERGENCE (toward the target, not toward raw competitor)
     ─────────────────────────────────────────────────────────────────────
     How fast we move to the target follows a parabolic curve:

       gap        = (target_price − current_price) / reference_price   [signed]
       convergence = |gap|^EXPONENT × demand_scale × GAIN
       recommended = current_price + gap × convergence

     • Small gap from target → tiny nudge  (parabola is flat near zero)
     • Large gap from target → large move   (parabola accelerates)
     • Demand also governs speed: high demand → faster convergence

  3. HOLD BAND (±HOLD_TOLERANCE around the target)
     ────────────────────────────────────────────────
     If current_price is within ±HOLD_TOLERANCE of target_price → HOLD.
     Outside that → move toward target parabolically.

  4. PROFIT FLOOR (time-aware, never below recent_price_median × floor_mult)
     ─────────────────────────────────────────────────────────────────────────
     Hard lower bound to protect margin. Uses recent 30-day median, not
     all-time avg_price, adjusted for price_trend_direction.

Decision table (summary):
  ┌───────────────────────────────────────────────────────────────┐
  │  current < target × (1 − tol)  →  INCREASE  (underpricing)  │
  │  current > target × (1 + tol)  →  DECREASE  (overpriced)    │
  │  within band                   →  HOLD       (optimal zone)  │
  └───────────────────────────────────────────────────────────────┘
"""

import logging
import math
from typing import Any

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Constants — all named, none buried in logic
# --------------------------------------------------------------------------- #

# ── Target-band (demand-scaled premium above competitor) ──────────────────────
PREMIUM_HIGH    = 1.20   # target mult for near-zero demand  (20% above competitor)
PREMIUM_LOW     = 1.05   # target mult for max demand        ( 5% above competitor)
PREMIUM_RANGE   = PREMIUM_HIGH - PREMIUM_LOW   # 0.15 — range covered by demand

# ── Hold tolerance: ±% around target where price is "good enough" ────────────
HOLD_TOLERANCE  = 0.025  # ±2.5% around target_price → hold

# ── Parabolic convergence parameters ─────────────────────────────────────────
PARABOLIC_EXPONENT  = 1.4    # shape: >1 → parabolic (big gaps move fast)
CONVERGENCE_GAIN    = 0.85   # cap on fraction of gap closed per call
MAX_CONVERGENCE     = 0.92   # absolute max convergence (don't overshoot)

# ── Demand scaling (maps demand_velocity → [0, 1]) ───────────────────────────
DEMAND_NORMALIZER   = 30.0   # units/day considered "full" demand (scale = 1.0)
DEMAND_FLOOR_SCALE  = 0.30   # minimum convergence speed even at zero demand

# ── Inventory modifier (high stock → lean toward lower premium) ───────────────
INVENTORY_HIGH      = 500.0
INVENTORY_PREMIUM_DISCOUNT = 0.03   # reduce target_mult by 3% when stock is high

# ── Profit floor (time-aware, using recent_price_median) ─────────────────────
PROFIT_MARGIN_MULTIPLIER = 1.10   # standard: floor = recent_median × 1.10
PROFIT_FLOOR_MINIMUM     = 1.05   # hard floor: never below recent_median × 1.05
RISING_PRICE_BONUS       = 0.05   # floor +5% when product price is trending up
FALLING_PRICE_DISCOUNT   = 0.03   # floor -3% when product price is trending down

# ── Ceiling: never recommend above PREMIUM_HIGH × competitor ─────────────────
MAX_ABOVE_COMPETITOR = PREMIUM_HIGH + 0.02   # 1.22× competitor hard ceiling


def make_pricing_decision(
    current_price: float,
    competitor_price: float,
    features: dict[str, Any],
) -> dict[str, Any]:
    """
    Band-anchored parabolic pricing decision.

    Returns
    -------
    dict with keys:
        recommended_price   float
        decision_type       'increase' | 'decrease' | 'hold'
        reasoning_factors   list[str]
        convergence_score   float   (0–1)
        gap_fraction        float   (signed, vs target_price)
        target_price        float   (what we're converging toward)
        target_mult         float   (target = competitor × this)
    """
    if current_price <= 0:
        raise ValueError(f"current_price must be positive; got {current_price}")
    if competitor_price <= 0:
        raise ValueError(f"competitor_price must be positive; got {competitor_price}")

    # ── Extract features ──────────────────────────────────────────────────────
    demand_velocity: float  = features.get("demand_velocity", 0.0)
    sales_trend: str        = features.get("sales_trend", "stable")
    inventory_proxy: float  = features.get("inventory_proxy", 0.0)
    price_volatility: float = features.get("price_volatility", 0.0)
    avg_price: float        = features.get("avg_price", current_price)
    recent_median: float    = features.get("recent_price_median", avg_price)
    price_trend_dir: str    = features.get("price_trend_direction", "stable")

    # ── Step 1: Demand scale  [DEMAND_FLOOR_SCALE, 1.0] ──────────────────────
    # log1p gives smooth diminishing returns; clip at 1.0 for full-demand products
    raw_demand   = math.log1p(demand_velocity / DEMAND_NORMALIZER)
    demand_scale = DEMAND_FLOOR_SCALE + (1.0 - DEMAND_FLOOR_SCALE) * min(raw_demand, 1.0)
    # demand_scale ∈ [0.30, 1.0]

    # ── Step 2: Dynamic target multiplier ────────────────────────────────────
    # Slide from PREMIUM_HIGH (low demand) to PREMIUM_LOW (high demand)
    target_mult = PREMIUM_HIGH - (demand_scale - DEMAND_FLOOR_SCALE) / (1.0 - DEMAND_FLOOR_SCALE) * PREMIUM_RANGE
    # target_mult ∈ [PREMIUM_LOW, PREMIUM_HIGH] = [1.05, 1.20]

    # Inventory modifier: high stock → accept lower premium (move it)
    if inventory_proxy >= INVENTORY_HIGH:
        target_mult = max(target_mult - INVENTORY_PREMIUM_DISCOUNT, PREMIUM_LOW)

    # Price trend modifier: if OWN price is rising, hold premium higher
    if price_trend_dir == "rising":
        target_mult = min(target_mult + 0.02, PREMIUM_HIGH)
    elif price_trend_dir == "falling":
        target_mult = max(target_mult - 0.02, PREMIUM_LOW)

    target_price = round(competitor_price * target_mult, 4)

    # ── Step 3: Hold band ─────────────────────────────────────────────────────
    lower_band = target_price * (1 - HOLD_TOLERANCE)
    upper_band = target_price * (1 + HOLD_TOLERANCE)

    # ── Step 4: Decision type (direction based on band position) ─────────────
    if lower_band <= current_price <= upper_band:
        decision_type = "hold"
    elif current_price < lower_band:
        decision_type = "increase"
    else:
        decision_type = "decrease"

    # ── Step 4b: Competitive position guard ───────────────────────────────────
    # If we are ALREADY priced above the competitor, INCREASE makes no market
    # sense — raising price further only widens the gap against a cheaper rival.
    # Maximum allowed decision when current > competitor is HOLD.
    competitive_gap = (current_price - competitor_price) / competitor_price  # +ve = we're pricier
    if competitive_gap > 0.0 and decision_type == "increase":
        decision_type = "hold"   # cap at hold — never increase when already above competitor

    # Classify our competitive position for reasoning transparency
    if competitive_gap > 0.05:
        competitive_position = f"premium (+{competitive_gap*100:.1f}% above competitor)"
    elif competitive_gap < -0.05:
        competitive_position = f"discount ({abs(competitive_gap)*100:.1f}% below competitor)"
    else:
        competitive_position = f"parity ({competitive_gap*100:+.1f}% vs competitor)"

    # ── Step 5: Parabolic convergence toward target ───────────────────────────
    reference    = max(target_price, current_price)
    gap_fraction = (target_price - current_price) / reference   # signed
    abs_gap      = abs(gap_fraction)

    convergence  = (abs_gap ** PARABOLIC_EXPONENT) * demand_scale * CONVERGENCE_GAIN
    convergence  = min(convergence, MAX_CONVERGENCE)

    price_distance   = target_price - current_price
    raw_recommended  = current_price + price_distance * convergence

    # ── Step 6: Time-aware profit floor ──────────────────────────────────────
    if price_trend_dir == "rising":
        floor_mult = PROFIT_MARGIN_MULTIPLIER + RISING_PRICE_BONUS    # 1.15
    elif price_trend_dir == "falling":
        floor_mult = PROFIT_MARGIN_MULTIPLIER - FALLING_PRICE_DISCOUNT # 1.07
    else:
        floor_mult = PROFIT_MARGIN_MULTIPLIER                           # 1.10

    profit_floor = _round_price(recent_median * floor_mult)
    hard_floor   = _round_price(recent_median * PROFIT_FLOOR_MINIMUM)
    ceiling      = _round_price(competitor_price * MAX_ABOVE_COMPETITOR)

    # ── Step 7: Apply constraints ─────────────────────────────────────────────
    # HOLD → no price movement; use current_price directly (don't apply convergence)
    if decision_type == "hold":
        recommended_price = current_price
    else:
        recommended_price = _round_price(raw_recommended)
        recommended_price = max(recommended_price, hard_floor)   # profit floor
        recommended_price = min(recommended_price, ceiling)       # never above ceiling

    # Final check: if movement is negligible, call it a hold
    if abs(recommended_price - current_price) < 0.01:
        decision_type     = "hold"
        recommended_price = current_price

    # ── Step 8: Reasoning factors ─────────────────────────────────────────────
    reasoning_factors = [
        f"Competitive position: {competitive_position} — "
        f"{'we are already above competitor, increase blocked.' if competitive_gap > 0 and decision_type == 'hold' and current_price > competitor_price else 'pricing room assessed vs target band.'}",

        f"Target band: competitor £{competitor_price:.2f} × {target_mult:.2f} = £{target_price:.2f} "
        f"(demand_scale={demand_scale:.2f} → {(target_mult-1)*100:.0f}% premium above competitor).",

        f"Current £{current_price:.2f} vs band [£{lower_band:.2f} – £{upper_band:.2f}] (±{HOLD_TOLERANCE*100:.0f}%) "
        f"→ {decision_type.upper()}.",

        f"Parabolic convergence: {convergence*100:.1f}% of gap closed "
        f"(gap={gap_fraction*100:+.1f}%, exponent={PARABOLIC_EXPONENT}, demand_scale={demand_scale:.2f}).",

        f"Demand velocity: {demand_velocity:.1f} units/day "
        f"({'high' if demand_scale >= 0.7 else 'medium' if demand_scale >= 0.5 else 'low'} — "
        f"{'closer to competitor' if demand_scale >= 0.7 else 'larger premium applied'}).",

        f"Time-aware profit floor: £{profit_floor:.2f} "
        f"(recent 30d median £{recent_median:.2f} × {floor_mult:.2f}, "
        f"price trend: '{price_trend_dir}').",

        f"Recommended: £{recommended_price:.2f} "
        f"[moved £{abs(recommended_price - current_price):.2f} toward target £{target_price:.2f}].",
    ]

    if inventory_proxy >= INVENTORY_HIGH:
        reasoning_factors.append(
            f"Inventory pressure ({inventory_proxy:,.0f} units) → "
            f"target premium reduced by {INVENTORY_PREMIUM_DISCOUNT*100:.0f}% to keep stock moving."
        )
    if price_trend_dir in ("rising", "falling"):
        reasoning_factors.append(
            f"Own price trend is '{price_trend_dir}' → target premium "
            f"{'raised' if price_trend_dir == 'rising' else 'reduced'} by 2%."
        )

    logger.info(
        "Decision='%s' | target_mult=%.2f | target=£%.2f | current=£%.2f | "
        "recommended=£%.2f | convergence=%.1f%% | demand_scale=%.2f",
        decision_type, target_mult, target_price, current_price,
        recommended_price, convergence * 100, demand_scale,
    )

    return {
        "recommended_price": recommended_price,
        "decision_type":     decision_type,
        "reasoning_factors": reasoning_factors,
        "convergence_score": round(convergence, 4),
        "gap_fraction":      round(gap_fraction, 4),
        "target_price":      round(target_price, 2),
        "target_mult":       round(target_mult, 4),
    }


def _round_price(price: float) -> float:
    """Round to 2 decimal places."""
    return round(price, 2)
