"""
main.py
--------
FastAPI Application — E-Commerce Pricing Intelligence Agent

Entry point for the pricing analysis API. Orchestrates the full pipeline:

  Raw Input → Data Layer → Feature Engineering →
  Decision Engine → Confidence Scoring → LLM Explanation → JSON Response

Run with:
  uvicorn main:app --reload --host 0.0.0.0 --port 8000
"""

import logging
import os
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from data.loader import load_and_clean_data
from services.confidence import compute_confidence_score
from services.decision_engine import make_pricing_decision
from services.feature_engineering import generate_product_features
from services.llm_explainer import generate_explanation

# --------------------------------------------------------------------------- #
# Bootstrap
# --------------------------------------------------------------------------- #
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("pricing_agent")

BASE_DIR = Path(__file__).parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


# --------------------------------------------------------------------------- #
# Application State — load data once at startup
# --------------------------------------------------------------------------- #
class AppState:
    df_clean: pd.DataFrame | None = None


app_state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load and cache the cleaned dataset on startup."""
    logger.info("Loading and cleaning dataset on startup …")
    try:
        app_state.df_clean = load_and_clean_data()
        logger.info("Dataset ready. Shape: %s", app_state.df_clean.shape)
    except Exception as exc:
        logger.critical("Failed to load dataset: %s", exc)
        raise
    yield
    logger.info("Application shutting down.")


# --------------------------------------------------------------------------- #
# FastAPI App
# --------------------------------------------------------------------------- #
app = FastAPI(
    title="Dynamic E-Commerce Pricing Intelligence Agent",
    description=(
        "Production-grade API that combines rule-based pricing decisions "
        "with Gemini LLM explanations to recommend optimal product prices."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------- #
# Request / Response Schemas
# --------------------------------------------------------------------------- #
class AnalyzeRequest(BaseModel):
    stock_code: str = Field(..., min_length=1, max_length=20, examples=["85123A"])
    current_price: float = Field(..., gt=0, examples=[2.55])
    competitor_price: float = Field(..., gt=0, examples=[2.30])

    @field_validator("stock_code")
    @classmethod
    def sanitise_stock_code(cls, v: str) -> str:
        return v.strip().upper()


class AnalyzeResponse(BaseModel):
    stock_code: str
    recommended_price: float
    decision_type: str
    confidence_score: float
    explanation: str
    executive_summary: str
    risk_level: str
    risk_rationale: str
    reasoning_factors: list[str]
    features: dict[str, Any]
    convergence_score: float | None = None   # parabolic convergence fraction used
    gap_fraction: float | None = None        # signed price gap vs target_price
    target_price: float | None = None        # competitor × demand-scaled multiplier
    target_mult: float | None = None         # the multiplier used (1.05–1.20)


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #
@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def dashboard(request: Request) -> HTMLResponse:
    """Serve the interactive pricing analysis dashboard."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/health", tags=["ops"])
async def health_check() -> dict[str, str]:
    """Lightweight liveness probe for load balancers / k8s."""
    if app_state.df_clean is None:
        raise HTTPException(status_code=503, detail="Dataset not loaded.")
    return {"status": "ok", "rows": str(len(app_state.df_clean))}


@app.get("/products", tags=["catalog"])
async def list_products() -> list[dict]:
    """
    Return all unique products with:
      - description   : product name
      - last_price    : UnitPrice from the MOST RECENT transaction (by InvoiceDate)
      - avg_price     : all-time mean (for reference only)

    last_price is the timestamp-accurate current market price, used to
    auto-fill the 'Current Price' field in the dashboard.
    avg_price is shown in the dropdown label only as context.
    """
    if app_state.df_clean is None:
        raise HTTPException(status_code=503, detail="Dataset not initialised.")

    df = app_state.df_clean

    # Most recent transaction price per product — sort by date, take last row
    latest_price = (
        df.sort_values("InvoiceDate")
        .groupby("StockCode", as_index=False)
        .agg(
            last_price=("UnitPrice",  "last"),       # price at most recent timestamp
            last_date= ("InvoiceDate","last"),       # when was that price recorded
            description=("Description", "last"),    # desc from same period
        )
    )

    # All-time avg for reference in dropdown label
    avg_by_code = (
        df.groupby("StockCode")["UnitPrice"].mean().round(2).rename("avg_price")
    )

    catalog = latest_price.join(avg_by_code, on="StockCode").sort_values("StockCode")

    return [
        {
            "stock_code":  row.StockCode,
            "description": str(row.description).strip().title(),
            "last_price":  round(row.last_price, 2),           # ← use this for current price
            "avg_price":   round(row.avg_price, 2),            # ← shown in dropdown label
            "last_date":   str(row.last_date.date()),          # ← when this price was recorded
        }
        for row in catalog.itertuples()
    ]


@app.post("/analyze", response_model=AnalyzeResponse, tags=["pricing"])
async def analyze_product(body: AnalyzeRequest) -> AnalyzeResponse:
    """
    Full pricing analysis pipeline for a single product.

    - Validates input
    - Derives product features from historical data
    - Applies rule-based decision engine
    - Computes confidence score
    - Generates LLM explanation via Gemini
    """
    if app_state.df_clean is None:
        raise HTTPException(status_code=503, detail="Dataset not initialised.")

    logger.info(
        "Analysis request: stock_code=%s current_price=%.2f competitor_price=%.2f",
        body.stock_code, body.current_price, body.competitor_price,
    )

    # ── Step 1: Feature Engineering ─────────────────────────────────────────
    try:
        features = generate_product_features(app_state.df_clean, body.stock_code)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        logger.exception("Feature engineering error for %s", body.stock_code)
        raise HTTPException(status_code=500, detail=f"Feature engineering failed: {exc}")

    # ── Step 2: Decision Engine ─────────────────────────────────────────────
    try:
        decision = make_pricing_decision(
            current_price    = body.current_price,
            competitor_price = body.competitor_price,
            features         = features,
        )
    except Exception as exc:
        logger.exception("Decision engine error for %s", body.stock_code)
        raise HTTPException(status_code=500, detail=f"Decision engine failed: {exc}")

    # ── Step 3: Confidence Score ────────────────────────────────────────────
    try:
        confidence_score = compute_confidence_score(features)
    except Exception as exc:
        logger.exception("Confidence scoring error")
        raise HTTPException(status_code=500, detail=f"Confidence scoring failed: {exc}")

    # ── Step 4: LLM Explanation ─────────────────────────────────────────────
    try:
        llm_output = generate_explanation(
            current_price    = body.current_price,
            competitor_price = body.competitor_price,
            features         = features,
            decision         = decision,
            confidence_score = confidence_score,
        )
    except EnvironmentError as exc:
        logger.warning("LLM skipped: %s", exc)
        llm_output = {
            "explanation":       "LLM explanation unavailable — GEMINI_API_KEY not configured.",
            "executive_summary": "Configure GEMINI_API_KEY to enable AI-generated summaries.",
            "risk_level":        "Unknown",
            "risk_rationale":    "Cannot assess risk without LLM.",
        }
    except RuntimeError as exc:
        logger.error("LLM call failed: %s", exc)
        llm_output = {
            "explanation":       f"LLM call failed: {exc}",
            "executive_summary": "AI summary temporarily unavailable.",
            "risk_level":        "Unknown",
            "risk_rationale":    "LLM error — please retry.",
        }

    return AnalyzeResponse(
        stock_code        = body.stock_code,
        recommended_price = decision["recommended_price"],
        decision_type     = decision["decision_type"],
        confidence_score  = confidence_score,
        explanation       = llm_output["explanation"],
        executive_summary = llm_output["executive_summary"],
        risk_level        = llm_output["risk_level"],
        risk_rationale    = llm_output["risk_rationale"],
        reasoning_factors = decision["reasoning_factors"],
        features          = features,
        convergence_score = decision.get("convergence_score"),
        gap_fraction      = decision.get("gap_fraction"),
        target_price      = decision.get("target_price"),
        target_mult       = decision.get("target_mult"),
    )
