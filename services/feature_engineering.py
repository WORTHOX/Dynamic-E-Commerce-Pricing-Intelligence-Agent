"""
services/feature_engineering.py
---------------------------------
Feature Engineering Layer for the E-Commerce Pricing Intelligence Agent.

For a given StockCode, this module derives product-level signals:
  - avg_price              : all-time mean unit price (kept for reference)
  - recent_price_median    : median unit price over last RECENT_PRICE_DAYS days  ← TIME-AWARE
  - recent_price_avg       : mean unit price over last RECENT_PRICE_DAYS days    ← TIME-AWARE
  - price_trend_direction  : 'rising' | 'falling' | 'stable' (OLS on daily median price)
  - demand_velocity        : 7-day rolling average quantity sold
  - price_volatility       : standard deviation of unit price (instability signal)
  - total_sales_last_30d   : total revenue in the last 30 days of data
  - sales_trend            : 'increasing' | 'decreasing' | 'stable' (OLS slope on qty)
  - inventory_proxy        : cumulative net quantity (proxy for stock on hand)
  - data_coverage_ratio    : fraction of days-in-range with at least one sale

Why recent_price_* instead of avg_price for cost proxy:
  all-time avg_price is biased by historical low prices. If a product was sold at
  £1 for 2 years then £5 for the last month, avg = ~£1.30 — wrong floor.
  Recent median/avg uses only the last RECENT_PRICE_DAYS of actual transactions.
"""

import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Tunables — no magic numbers in code
# --------------------------------------------------------------------------- #
ROLLING_WINDOW_DAYS   = 7
TRAILING_SALES_DAYS   = 30
RECENT_PRICE_DAYS     = 30    # Window for time-aware price reference
TREND_SLOPE_THRESHOLD = 0.05  # Units / day; below this is "stable"
PRICE_SLOPE_THRESHOLD = 0.01  # £ / day; below this is "stable" price trend
MIN_RECORDS_REQUIRED  = 3     # Cannot calculate meaningful stats below this


def generate_product_features(
    df_clean: pd.DataFrame,
    stock_code: str,
) -> dict[str, Any]:
    """
    Derive pricing intelligence features for a single product.

    Parameters
    ----------
    df_clean : pd.DataFrame
        The cleaned dataset from ``data.loader.load_and_clean_data()``.
    stock_code : str
        The product identifier (StockCode column value).

    Returns
    -------
    dict[str, Any]
        Keys:
        - ``avg_price``             : float  (all-time mean, kept for reference)
        - ``recent_price_median``   : float  (median price, last 30d — TIME-AWARE)
        - ``recent_price_avg``      : float  (mean price, last 30d — TIME-AWARE)
        - ``price_trend_direction`` : str    ('rising' | 'falling' | 'stable')
        - ``demand_velocity``       : float  (latest 7-day rolling avg)
        - ``price_volatility``      : float  (std dev of unit prices)
        - ``total_sales_last_30d``  : float  (revenue in last 30 days)
        - ``sales_trend``           : str    ('increasing' | 'decreasing' | 'stable')
        - ``inventory_proxy``       : float  (cumulative quantity)
        - ``data_coverage_ratio``   : float  (0–1 data density signal)
        - ``record_count``          : int    (raw rows for this product)

    Raises
    ------
    ValueError
        If the stock code is not found in the dataset.
    """
    stock_code = str(stock_code).strip().upper()
    product_df = df_clean[df_clean["StockCode"].str.upper() == stock_code].copy()

    if product_df.empty:
        raise ValueError(
            f"StockCode '{stock_code}' not found in the dataset. "
            "Verify the code or re-check the data source."
        )

    record_count = len(product_df)
    logger.info("Engineering features for StockCode='%s' (%d records)", stock_code, record_count)

    # ── Temporal index ─────────────────────────────────────────────────────
    product_df = product_df.sort_values("InvoiceDate")
    product_df["Date"] = product_df["InvoiceDate"].dt.normalize()

    # ── 1. avg_price (all-time) ────────────────────────────────────────────
    avg_price: float = float(product_df["UnitPrice"].mean())

    # ── 1b. TIME-AWARE price references (last RECENT_PRICE_DAYS) ───────────
    latest_date       = product_df["InvoiceDate"].max()
    recent_cutoff     = latest_date - pd.Timedelta(days=RECENT_PRICE_DAYS)
    recent_df         = product_df[product_df["InvoiceDate"] >= recent_cutoff]

    if len(recent_df) >= 2:
        recent_price_median: float = float(recent_df["UnitPrice"].median())
        recent_price_avg: float    = float(recent_df["UnitPrice"].mean())
    else:
        # Fallback: not enough recent data — use all-time median (more robust than mean)
        recent_price_median = float(product_df["UnitPrice"].median())
        recent_price_avg    = avg_price
        logger.debug("StockCode='%s': insufficient recent price data, falling back to all-time.", stock_code)

    # ── 1c. Price trend direction (OLS on daily median price over time) ─────
    daily_price = (
        product_df.groupby("Date")["UnitPrice"]
        .median()
        .reset_index()
        .rename(columns={"UnitPrice": "median_price"})
    )
    price_trend_direction: str = _compute_price_trend(daily_price["median_price"].values)

    # ── 2. price_volatility ────────────────────────────────────────────────
    price_volatility: float = (
        float(product_df["UnitPrice"].std()) if record_count > 1 else 0.0
    )

    # ── 3. Daily aggregation for demand features ───────────────────────────
    daily = (
        product_df.groupby("Date")
        .agg(qty=("Quantity", "sum"), revenue=("LineRevenue", "sum"))
        .reset_index()
    )

    # Fill date gaps so rolling window is accurate
    full_range = pd.date_range(daily["Date"].min(), daily["Date"].max(), freq="D")
    daily = (
        daily.set_index("Date")
        .reindex(full_range, fill_value=0)
        .rename_axis("Date")
        .reset_index()
    )

    # ── 4. demand_velocity (rolling 7-day avg, most recent value) ──────────
    daily["rolling_qty"] = (
        daily["qty"].rolling(window=ROLLING_WINDOW_DAYS, min_periods=1).mean()
    )
    demand_velocity: float = float(daily["rolling_qty"].iloc[-1])

    # ── 5. total_sales_last_30d ────────────────────────────────────────────
    cutoff_date = daily["Date"].max() - pd.Timedelta(days=TRAILING_SALES_DAYS)
    total_sales_last_30d: float = float(
        daily[daily["Date"] >= cutoff_date]["revenue"].sum()
    )

    # ── 6. sales_trend via OLS slope ──────────────────────────────────────
    sales_trend = _compute_trend(daily["qty"].values)

    # ── 7. inventory_proxy ────────────────────────────────────────────────
    inventory_proxy: float = float(product_df["Quantity"].sum())

    # ── 8. data_coverage_ratio ────────────────────────────────────────────
    total_days = max(len(full_range), 1)
    active_days = int((daily["qty"] > 0).sum())
    data_coverage_ratio: float = round(active_days / total_days, 4)

    features = {
        # ── All-time reference (kept for backwards-compat) ──────────────────
        "avg_price":            round(avg_price, 4),
        # ── Time-aware price references (preferred for floor calculation) ───
        "recent_price_median":  round(recent_price_median, 4),
        "recent_price_avg":     round(recent_price_avg, 4),
        "price_trend_direction": price_trend_direction,
        # ── Demand & supply ─────────────────────────────────────────────────
        "demand_velocity":      round(demand_velocity, 4),
        "price_volatility":     round(price_volatility, 4),
        "total_sales_last_30d": round(total_sales_last_30d, 4),
        "sales_trend":          sales_trend,
        "inventory_proxy":      round(inventory_proxy, 2),
        "data_coverage_ratio":  data_coverage_ratio,
        "record_count":         record_count,
    }

    logger.debug("Features for %s: %s", stock_code, features)
    return features


def _compute_trend(qty_series: np.ndarray) -> str:
    """
    Fit a simple OLS regression through the quantity-over-time series
    and classify the slope as increasing, decreasing, or stable.
    """
    n = len(qty_series)
    if n < MIN_RECORDS_REQUIRED:
        return "stable"
    x = np.arange(n, dtype=float)
    slope, _intercept, _r, _p, _se = stats.linregress(x, qty_series.astype(float))
    if slope > TREND_SLOPE_THRESHOLD:
        return "increasing"
    elif slope < -TREND_SLOPE_THRESHOLD:
        return "decreasing"
    return "stable"


def _compute_price_trend(price_series: np.ndarray) -> str:
    """
    Fit OLS through daily median *price* values to detect if the product's
    own price has been rising, falling, or stable over time.

    This is distinct from sales_trend (which tracks quantity).
    Uses PRICE_SLOPE_THRESHOLD (£/day) — a much smaller threshold than qty slope.

    Returns
    -------
    str
        One of: 'rising', 'falling', 'stable'.
    """
    n = len(price_series)
    if n < MIN_RECORDS_REQUIRED:
        return "stable"
    x = np.arange(n, dtype=float)
    slope, _intercept, _r, _p, _se = stats.linregress(x, price_series.astype(float))
    if slope > PRICE_SLOPE_THRESHOLD:
        return "rising"
    elif slope < -PRICE_SLOPE_THRESHOLD:
        return "falling"
    return "stable"
