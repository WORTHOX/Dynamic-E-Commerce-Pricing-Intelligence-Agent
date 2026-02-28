"""
services/feature_engineering.py
---------------------------------
Feature Engineering Layer for the E-Commerce Pricing Intelligence Agent.

For a given StockCode, this module derives product-level signals:
  - avg_price              : mean historical unit price
  - demand_velocity        : 7-day rolling average quantity sold
  - price_volatility       : standard deviation of unit price (instability signal)
  - total_sales_last_30d   : total revenue in the last 30 days of data
  - sales_trend            : 'increasing' | 'decreasing' | 'stable' (OLS slope)
  - inventory_proxy        : cumulative net quantity (proxy for stock on hand)
  - data_coverage_ratio    : fraction of days-in-range with at least one sale
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
ROLLING_WINDOW_DAYS = 7
TRAILING_SALES_DAYS = 30
TREND_SLOPE_THRESHOLD = 0.05   # Units / day; below this is "stable"
MIN_RECORDS_REQUIRED = 3       # Cannot calculate meaningful stats below this


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
        - ``avg_price``             : float
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

    # ── 1. avg_price ───────────────────────────────────────────────────────
    avg_price: float = float(product_df["UnitPrice"].mean())

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
        "avg_price": round(avg_price, 4),
        "demand_velocity": round(demand_velocity, 4),
        "price_volatility": round(price_volatility, 4),
        "total_sales_last_30d": round(total_sales_last_30d, 4),
        "sales_trend": sales_trend,
        "inventory_proxy": round(inventory_proxy, 2),
        "data_coverage_ratio": data_coverage_ratio,
        "record_count": record_count,
    }

    logger.debug("Features for %s: %s", stock_code, features)
    return features


def _compute_trend(qty_series: np.ndarray) -> str:
    """
    Fit a simple OLS regression through the quantity-over-time series
    and classify the slope as increasing, decreasing, or stable.

    Parameters
    ----------
    qty_series : np.ndarray
        Ordered daily quantity values.

    Returns
    -------
    str
        One of: 'increasing', 'decreasing', 'stable'.
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
