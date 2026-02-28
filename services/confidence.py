"""
services/confidence.py
-----------------------
Confidence Scoring Layer for the E-Commerce Pricing Intelligence Agent.

Produces a single float score in [0, 1] reflecting how trustworthy the
recommended price is, based on:
  - Data completeness (record count and date coverage)
  - Demand stability (inverse of rolling demand variance)
  - Trend consistency (clear directional trend vs. noise)
  - Price stability (inverse of price volatility)

Each dimension is independently scored in [0, 1] and then combined via
a weighted average. Weights are named constants.
"""

import logging
import math
from typing import Any

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Scoring weights (must sum to 1.0)
# --------------------------------------------------------------------------- #
W_DATA_COMPLETENESS = 0.30
W_DEMAND_STABILITY  = 0.25
W_TREND_CONSISTENCY = 0.25
W_PRICE_STABILITY   = 0.20

assert math.isclose(
    W_DATA_COMPLETENESS + W_DEMAND_STABILITY + W_TREND_CONSISTENCY + W_PRICE_STABILITY,
    1.0,
    rel_tol=1e-6,
), "Confidence weights must sum to 1.0"

# --------------------------------------------------------------------------- #
# Saturation thresholds
# --------------------------------------------------------------------------- #
MIN_RECORDS_FOR_FULL_CONFIDENCE  = 100   # >= this → full data completeness score
MAX_VOLATILITY_FOR_ZERO_SCORE    = 5.0   # std dev >= this → price stability = 0
MAX_DEMAND_VARIANCE_FOR_ZERO     = 200.0 # variance in demand → score clamps to 0


def compute_confidence_score(features: dict[str, Any]) -> float:
    """
    Compute an overall confidence score for a pricing recommendation.

    Parameters
    ----------
    features : dict[str, Any]
        Output of ``services.feature_engineering.generate_product_features()``.

    Returns
    -------
    float
        A value in [0.0, 1.0]. Higher is more confident.
        - ≥ 0.75 : High confidence
        - 0.50–0.74 : Moderate confidence
        - < 0.50 : Low confidence — treat with caution
    """
    record_count        = int(features.get("record_count", 0))
    data_coverage       = float(features.get("data_coverage_ratio", 0.0))
    demand_velocity     = float(features.get("demand_velocity", 0.0))
    price_volatility    = float(features.get("price_volatility", 0.0))
    sales_trend         = str(features.get("sales_trend", "stable"))

    # ── Dimension 1: Data Completeness ─────────────────────────────────────
    # Combines raw record count with coverage ratio
    count_score    = min(record_count / MIN_RECORDS_FOR_FULL_CONFIDENCE, 1.0)
    coverage_score = data_coverage  # already 0–1
    d_completeness = (count_score + coverage_score) / 2.0

    # ── Dimension 2: Demand Stability ─────────────────────────────────────
    # Low demand velocity → lower confidence; high variance → lower confidence
    # We use a simple heuristic: demand_velocity capped at 50 gives a base,
    # divided by variance proxy (volatility in demand isn't directly available,
    # so we use the coverage gap as a demand irregularity proxy).
    gap_penalty    = 1.0 - data_coverage           # sparse data = unstable demand
    velocity_score = min(demand_velocity / 50.0, 1.0)
    d_demand       = velocity_score * (1.0 - gap_penalty * 0.5)
    d_demand       = _clamp(d_demand)

    # ── Dimension 3: Trend Consistency ────────────────────────────────────
    # A clear directional trend (not "stable") is more reliable for decisions.
    # "stable" with high coverage is also acceptable.
    if sales_trend in ("increasing", "decreasing"):
        d_trend = 0.85 + (0.15 * data_coverage)
    else:
        # stable can still be confident if coverage is high
        d_trend = 0.50 + (0.30 * data_coverage)
    d_trend = _clamp(d_trend)

    # ── Dimension 4: Price Stability ──────────────────────────────────────
    # Low price std dev → pricing is consistent → higher reliability
    d_price = 1.0 - _clamp(price_volatility / MAX_VOLATILITY_FOR_ZERO_SCORE)

    # ── Weighted Combination ───────────────────────────────────────────────
    score = (
        W_DATA_COMPLETENESS * d_completeness
        + W_DEMAND_STABILITY  * d_demand
        + W_TREND_CONSISTENCY * d_trend
        + W_PRICE_STABILITY   * d_price
    )
    score = _clamp(round(score, 4))

    logger.info(
        "Confidence score=%.4f (completeness=%.3f, demand=%.3f, trend=%.3f, price=%.3f)",
        score, d_completeness, d_demand, d_trend, d_price,
    )
    return score


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Clamp a float to [lo, hi]."""
    return max(lo, min(hi, value))
