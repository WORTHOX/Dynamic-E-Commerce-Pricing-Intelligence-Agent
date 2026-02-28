"""
data/loader.py
--------------
Data ingestion and cleaning layer for the E-Commerce Pricing Intelligence Agent.

Responsibilities:
- Load raw CSV data
- Apply quality rules (missing values, cancellations, negatives)
- Return a clean, typed DataFrame ready for feature engineering
"""

import logging
import os
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
DATA_FILE = os.getenv("DATA_FILE", str(Path(__file__).parent.parent / "data.csv"))
CANCELLATION_PREFIX = "C"


def load_and_clean_data() -> pd.DataFrame:
    """
    Load and clean the Online Retail dataset.

    Cleaning steps applied:
    1. Drop rows with missing CustomerID (anonymous transactions).
    2. Remove cancelled invoices (InvoiceNo starting with 'C').
    3. Drop rows with non-positive Quantity (returns / errors).
    4. Drop rows with non-positive UnitPrice (erroneous entries).
    5. Drop rows with null Description.
    6. Parse InvoiceDate to datetime.
    7. Strip whitespace from string columns.

    Returns
    -------
    pd.DataFrame
        A clean DataFrame with guaranteed column types and no nulls in
        critical columns.

    Raises
    ------
    FileNotFoundError
        If the dataset file does not exist at the configured path.
    ValueError
        If the dataset is empty after cleaning.
    """
    if not Path(DATA_FILE).exists():
        raise FileNotFoundError(
            f"Dataset not found at '{DATA_FILE}'. "
            "Set the DATA_FILE environment variable to the correct path."
        )

    logger.info("Loading dataset from: %s", DATA_FILE)
    df = pd.read_csv(
        DATA_FILE,
        encoding="ISO-8859-1",
        dtype={
            "InvoiceNo": str,
            "StockCode": str,
            "CustomerID": str,
        },
    )

    raw_count = len(df)
    logger.info("Raw rows loaded: %d", raw_count)

    # ── Step 1: Drop missing CustomerID ────────────────────────────────────
    df = df.dropna(subset=["CustomerID"])
    logger.debug("After dropping missing CustomerID: %d rows", len(df))

    # ── Step 2: Remove cancelled invoices ──────────────────────────────────
    df = df[~df["InvoiceNo"].str.startswith(CANCELLATION_PREFIX, na=False)]
    logger.debug("After removing cancellations: %d rows", len(df))

    # ── Step 3: Drop non-positive Quantity ─────────────────────────────────
    df = df[df["Quantity"] > 0]
    logger.debug("After filtering positive Quantity: %d rows", len(df))

    # ── Step 4: Drop non-positive UnitPrice ────────────────────────────────
    df = df[df["UnitPrice"] > 0]
    logger.debug("After filtering positive UnitPrice: %d rows", len(df))

    # ── Step 5: Drop null Description ──────────────────────────────────────
    df = df.dropna(subset=["Description"])
    logger.debug("After dropping null Description: %d rows", len(df))

    # ── Step 6: Parse InvoiceDate ─────────────────────────────────────────
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

    # ── Step 7: Strip whitespace ──────────────────────────────────────────
    for col in ["StockCode", "Description", "InvoiceNo", "Country"]:
        df[col] = df[col].str.strip()

    # ── Derived column: revenue per line ──────────────────────────────────
    df["LineRevenue"] = df["Quantity"] * df["UnitPrice"]

    cleaned_count = len(df)
    dropped = raw_count - cleaned_count
    logger.info(
        "Cleaning complete. Retained %d / %d rows (dropped %d, %.1f%%).",
        cleaned_count,
        raw_count,
        dropped,
        100.0 * dropped / max(raw_count, 1),
    )

    if cleaned_count == 0:
        raise ValueError("No data remains after cleaning. Check your input file.")

    return df.reset_index(drop=True)
