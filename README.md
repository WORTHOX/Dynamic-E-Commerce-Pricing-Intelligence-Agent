<div align="center">

```
╔══════════════════════════════════════════════════════════════════════════╗
║                                                                          ║
║    ██████╗ ██╗   ██╗███╗  ██╗    ██████╗ ██████╗ ██╗ ██████╗███████╗     ║
║    ██╔══██╗╚██╗ ██╔╝████╗ ██║    ██╔══██╗██╔══██╗██║██╔════╝██╔════╝     ║
║    ██║  ██║ ╚████╔╝ ██╔██╗██║    ██████╔╝██████╔╝██║██║     █████╗       ║
║    ██║  ██║  ╚██╔╝  ██║╚████║    ██╔═══╝ ██╔══██╗██║██║     ██╔══╝       ║
║    ██████╔╝   ██║   ██║ ╚███║    ██║     ██║  ██║██║╚██████╗███████╗     ║
║    ╚═════╝    ╚═╝   ╚═╝  ╚══╝    ╚═╝     ╚═╝  ╚═╝╚═╝ ╚═════╝╚══════╝     ║
║                                                                          ║
║         ── Dynamic E-Commerce Pricing Intelligence Agent ──              ║
║                                                                          ║
╚══════════════════════════════════════════════════════════════════════════╝
```

![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111%2B-009688?logo=fastapi&logoColor=white)
![Gemini](https://img.shields.io/badge/LLM-Gemini%20Flash-4285F4?logo=google&logoColor=white)
![Pandas](https://img.shields.io/badge/Data-Pandas%202.0%2B-150458?logo=pandas&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-22C55E)

</div>

> **AI-powered dynamic pricing engine** — rule-based decisions, parabolic convergence model, Gemini LLM reasoning, confidence scoring, FastAPI backend, and a fully interactive dashboard. Built on 541K real e-commerce transactions.

---

## What This Actually Is

Most "AI pricing tools" are either a lookup table or a black-box ML model. This is neither.

This is a **layered intelligence pipeline** where every recommendation is:
1. **Derived from real product transaction signals** — not guesses
2. **Computed by a continuous parabolic convergence formula** — not a set of if/else rules
3. **Constrained by market reality** — profit floor, competitive position guard, demand-scaled premium
4. **Explained in business English** — Gemini LLM writes the executive summary a CFO can act on

---

## System Architecture

```
Raw CSV (541k rows)
       │
       ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Layer 1 · Data Loader  (data/loader.py)                            │
│  • Remove cancelled invoices (InvoiceNo starts with 'C')            │
│  • Drop negative Quantity (returns)                                 │
│  • Drop null CustomerID, null Description                           │
│  • Clip zero/negative UnitPrice                                     │
│  • Parse InvoiceDate → datetime                                     │
│  • Compute LineRevenue = Quantity × UnitPrice                       │
│  Result: 397,884 clean rows                                         │
└──────────────────────────┬──────────────────────────────────────────┘
                           │  per-product slice
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Layer 2 · Feature Engineering  (services/feature_engineering.py)   │
│                                                                     │
│  TIME-AWARE PRICE SIGNALS (why recent > avg explained below):       │
│  • recent_price_median  — median UnitPrice, last 30 days            │
│  • recent_price_avg     — mean UnitPrice, last 30 days              │
│  • price_trend_direction— OLS on daily median price ('rising' etc.) │
│                                                                     │
│  DEMAND SIGNALS:                                                    │
│  • demand_velocity      — 7-day rolling avg qty/day (most recent)   │
│  • sales_trend          — OLS on daily qty ('increasing' etc.)      │
│  • total_sales_last_30d — revenue last 30 days                      │
│                                                                     │
│  SUPPLY SIGNALS:                                                    │
│  • inventory_proxy      — cumulative net quantity sold              │
│  • price_volatility     — std dev of UnitPrice (pricing instability)│
│  • data_coverage_ratio  — % of days in range with ≥1 sale           │
└──────────────────────────┬──────────────────────────────────────────┘
                           │  feature dict
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Layer 3 · Decision Engine  (services/decision_engine.py)           │
│                                                                     │
│  BAND-ANCHORED PARABOLIC CONVERGENCE MODEL:                         │
│                                                                     │
│  1. demand_scale  = log(velocity / 30) → [0.30, 1.0]                │
│  2. target_mult   = 1.20 − demand_scale × 0.15  → [1.05, 1.20]      │
│  3. target_price  = competitor × target_mult                        │
│  4. gap_fraction  = (target − current) / max(target, current)       │
│  5. convergence   = |gap|^1.4 × demand_scale × 0.85  ← PARABOLIC    │
│  6. recommended   = current + (target − current) × convergence      │
│  7. Apply: profit_floor, competitive_position_guard, hold_band      │
│                                                                     │
│  COMPETITIVE POSITION GUARD:                                        │
│  If current > competitor → max decision = HOLD (never INCREASE)     │
│                                                                     │
│  PROFIT FLOOR (time-aware):                                         │
│  floor = recent_price_median × dynamic_multiplier                   │
│         (1.15 if price rising, 1.07 if falling, 1.10 stable)        │
└──────────────────────────┬──────────────────────────────────────────┘
                           │  decision + recommended_price
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Layer 4 · Confidence Scorer  (services/confidence.py)              │
│  4-dimension weighted score [0, 1]:                                 │
│  • Data completeness   30% (record count + day coverage ratio)      │
│  • Demand stability    25% (velocity consistency)                   │
│  • Trend consistency   25% (OLS slope clarity)                      │
│  • Price stability     20% (inverse of price_volatility)            │
└──────────────────────────┬──────────────────────────────────────────┘
                           │  confidence_score
                           ▼
┌───────────────────────────────────────────────────────────────────────┐
│  Layer 5 · LLM Explainer  (services/llm_explainer.py)                 │
│  • Google Gemini Flash via google-genai SDK                           │
│  • Structured JSON output (no prompt injection risk)                  │
│  • Returns: explanation, executive_summary, risk_level, risk_rationale│
└───────────────────────────────────────────────────────────────────────┘
```

---

## The Core Pricing Model — Thought Process

### Why Not Simple if/elif Rules?

The first version used 5 fixed percentage rules (`current × 1.05`, `current × 0.93`). The problem:

```
Rule: INCREASE → recommended = current_price × 1.05
If current = £2.55 and competitor = £3.50 → recommended = £2.68
That's only £0.13 above current, completely ignoring the £0.95 margin headroom
```

Rules anchor to current price. **The market doesn't care about your current price — it cares about the competitor's price.**

### Why Parabolic Convergence?

The key insight: **the size of the recommended move should be proportional to how far we are from the optimal target** — and that relationship should be non-linear (parabolic).

```
gap_fraction = 5%  → convergence = 5%^1.4 = 1.8%  → tiny nudge
gap_fraction = 20% → convergence = 20%^1.4 = 10%   → moderate move
gap_fraction = 50% → convergence = 50%^1.4 = 35%   → large move
```

Small deviations from target get tiny corrections. Large deviations get aggressive corrections. This is how real markets self-correct — not in fixed jumps.

### Why a Target Band (Not Raw Competitor Price)?

Always pricing at exactly competitor's price is a commodity race to the bottom. The insight:

- **High-demand products** have price-sensitive buyers → stay close (5% above competitor)
- **Low-demand products** are niche/specialty items → buyers are less price-elastic → can command 20% premium

```python
target_mult   = 1.20 − demand_scale × 0.15   # slides between 5% and 20%
target_price  = competitor_price × target_mult
```

The parabolic convergence then moves current price toward this target — not toward the raw competitor price.

### Why `recent_price_median` Instead of `avg_price` for the Profit Floor?

If a product sold for £1 for 2 years and £5 for the last month:
- `avg_price` = £1.30 → profit floor = £1.43 → way too low
- `recent_price_median` (last 30 days) = £5.00 → floor = £5.50 → accurate

All-time averages are biased by historical pricing. The profit floor must reflect **what the market currently values the product at**, not what it was valued at historically.

### Why the Competitive Position Guard?

When `current_price > competitor_price`, increasing prices further is irrational:
- You are already more expensive than the market
- Raising price widens the competitive gap against a cheaper rival
- Even if the target band math suggests INCREASE (because target > current), the market reality says HOLD

```python
if current_price > competitor_price and decision_type == "increase":
    decision_type = "hold"   # cap at hold — never increase when already above competitor
```

This one rule eliminates a whole class of incorrect recommendations.

---

## Feature Engineering — Why Each Signal Matters

| Feature | How Computed | Why It Matters |
|---|---|---|
| `demand_velocity` | 7-day rolling avg units/day | Direct pricing power — high demand = more room to price up |
| `sales_trend` | OLS regression on daily qty, slope vs threshold | Direction of demand — rising trend supports increases |
| `recent_price_median` | Median UnitPrice, last 30 days of transactions | Time-accurate cost proxy for profit floor |
| `price_trend_direction` | OLS regression on daily median price | Are we systematically raising/lowering price? Adjusts floor |
| `inventory_proxy` | Cumulative net quantity sold | High stock + low demand = clearance pressure |
| `price_volatility` | Std dev of UnitPrice over all records | High volatility = unreliable pricing → more conservative decisions |
| `data_coverage_ratio` | Active selling days / total days in range | Data density signal — thin data = lower confidence |
| `total_sales_last_30d` | Sum of LineRevenue in last 30 days | Revenue health — context for the LLM explanation |

---

## Confidence Scoring — Why 4 Dimensions?

A single-number confidence score is only meaningful if it captures multiple independent risk axes:

| Dimension | What Fails With Just One Metric |
|---|---|
| **Data Completeness** | A product with 3 transactions looks the same as one with 3,000 |
| **Demand Stability** | High velocity is meaningless if it's erratic |
| **Trend Consistency** | A clear trending signal is more trustworthy than a noisy one |
| **Price Stability** | Erratic historical pricing = unreliable feature inputs |

Threshold interpretation:
- `≥ 0.75` → **High confidence** — act on recommendation
- `0.50–0.74` → **Moderate** — use with judgment
- `< 0.50` → **Low** — treat as directional signal only

---

## Frontend — Design Decisions

Built with **pure Vanilla HTML + CSS + JS** — no React, no Vue, no framework overhead.

**Searchable Dropdown for Stock Code:**
- Fetches all 3,665 products from `/products` on page load
- Filters by code OR description in real-time with keyboard navigation
- Dropdown shows: code | description | last recorded price | date

**Auto-fill Chain on Product Selection:**
1. `current_price` → filled with `last_price` (most recent transaction by timestamp, NOT avg)
2. `description` → filled from catalog
3. `last sale date` → shown below price field as context

**Why `last_price` not `avg_price` for auto-fill:**
The avg would show £2.89 for a product that last sold at £2.95 — the analysis would start from a factually wrong current price.

---

## Endpoints

| Endpoint | Method | Purpose |
|---|---|---|
| `/` | GET | Serves the dashboard HTML |
| `/health` | GET | Liveness probe (`{"status":"ok","rows":"397884"}`) |
| `/products` | GET | Full product catalog with last_price, avg_price, last_date |
| `/analyze` | POST | Full 5-layer pipeline for one product |

**`POST /analyze` Request:**
```json
{
  "stock_code": "85123A",
  "current_price": 2.95,
  "competitor_price": 3.50
}
```

**`POST /analyze` Response includes:**
```json
{
  "recommended_price": 3.10,
  "decision_type": "increase",
  "confidence_score": 0.93,
  "target_price": 3.67,
  "target_mult": 1.05,
  "convergence_score": 0.088,
  "gap_fraction": 0.197,
  "reasoning_factors": [...],
  "executive_summary": "...",
  "risk_level": "Low",
  "features": {...}
}
```

---

## Quick Start

```bash
git clone https://github.com/WORTHOX/Dynamic-E-Commerce-Pricing-Intelligence-Agent.git
cd Dynamic-E-Commerce-Pricing-Intelligence-Agent

pip install -r requirements.txt
cp .env.example .env
# Add your GEMINI_API_KEY to .env

uvicorn main:app --reload --host 0.0.0.0 --port 8000
# Open http://localhost:8000
```

---

## Project Structure

```
Dynamic-E-Commerce-Pricing-Intelligence-Agent/
│
├── ecommerce.csv                     ← Online Retail dataset (541k rows)
├── main.py                           ← FastAPI orchestrator (4 endpoints)
│
├── data/
│   ├── __init__.py
│   └── loader.py                     ← Layer 1: 7 cleaning rules → 397,884 rows
│
├── services/
│   ├── __init__.py
│   ├── feature_engineering.py        ← Layer 2: 9 per-product signals, OLS trend
│   ├── decision_engine.py            ← Layer 3: Band-anchored parabolic model
│   ├── confidence.py                 ← Layer 4: 4-dimension weighted confidence
│   └── llm_explainer.py              ← Layer 5: Gemini Flash structured output
│
├── templates/
│   └── index.html                    ← Dark-mode dashboard, searchable dropdown
│
├── requirements.txt
├── .env.example
└── .gitignore
```

---

## Tech Stack

| Layer | Technology | Why |
|---|---|---|
| API Framework | FastAPI | Async, Pydantic validation, auto OpenAPI docs |
| ASGI Server | Uvicorn | Production-grade ASGI, hot reload in dev |
| Data Processing | Pandas 2.0, NumPy | Vectorised operations on 400k rows without OOM |
| Statistical Trend | SciPy (`linregress`) | OLS slope for both qty trend and price trend |
| LLM | Google Gemini Flash | Structured JSON mode, cost-effective, fast |
| Validation | Pydantic v2 | Type-safe request/response models |
| Frontend | Vanilla HTML/CSS/JS | Zero JS framework overhead, fully auditable |
| Config | python-dotenv | 12-factor app configuration |

---

## Known Limitations & What Production Would Add

| Limitation | Root Cause | Production Fix |
|---|---|---|
| Competitor price is manually entered | No competitor data in dataset | Daily price scraper / competitor API feed |
| No cost-of-goods data | Dataset has only revenue, not margin | Plug in COGS column → use actual margin floor |
| Seasonality not normalised | Garden parasol has 0.3 units/day in winter — not dead | STL decomposition to separate trend / seasonality / noise |
| Zero-demand products | demand_velocity ~0 doesn't mean HOLD forever | Add clearance flag: low velocity + high inventory = CLEARANCE |
| Static historical data (2010–2011) | Cannot validate current market | Replace with live transaction feed; engine logic unchanged |
| Single-market data (UK) | Calibration tuned for UK retail | Re-calibrate PREMIUM_HIGH/LOW constants per market |

> Architecture is production-grade. The gaps above are **known with identified solutions** — which is the correct engineering position: build the right structure first, feed it better data second.

---

## Anticipated Questions & Answers

**Q: Why not use a machine learning model instead of this rule-based approach?**

> ML requires labelled training data — i.e., "given these features, the correct price was £X." We have no such labels. What we have is transaction history. The parabolic convergence model is theoretically grounded (price elasticity + market positioning), fully explainable, and doesn't need training. A future ML layer could learn the optimal `PREMIUM_HIGH/LOW` constants from outcomes data if we close the feedback loop.

**Q: How do you know the parabolic exponent of 1.4 is correct?**

> It's a calibrated starting point. Exponent = 1.0 gives linear convergence (fixed %). Exponent > 1 gives accelerating convergence — larger gaps get proportionally larger corrections. 1.4 was chosen because it gives a ~10x ratio between small-gap (5%) and large-gap (50%) corrections, which matches intuitive business behaviour. In production it would be a tunable constant backed by A/B test outcomes.

**Q: What happens when demand_velocity is zero? Is the product assumed dead?**

> No — zero velocity means the 7-day rolling average is zero, but the product may be seasonal. The model falls back to `demand_scale = 0.30` (the floor), applies `target_mult = 1.20` (maximum premium), and recommends hold unless there's a strong competitive signal. A production system would add seasonality decomposition to distinguish "seasonal pause" from "dead product."

**Q: Why use `recent_price_median` (30-day) instead of `recent_price_avg` as the floor?**

> Median is robust to outliers. If a product had one bulk-order transaction at an unusually low unit price in the last 30 days, the mean would be pulled down — making the floor artificially low and risking a loss-making recommendation. Median ignores such outliers.

**Q: How does the competitive position guard interact with the parabolic model?**

> The parabolic model computes a direction and magnitude. The guard is a post-processing step that overrides direction: if `current > competitor`, the maximum direction is HOLD regardless of what the target band math says. This separates the "what should our target be" question from the "is it appropriate to move toward it right now" question. It's the same separation used in PID controllers.

**Q: Is this system production-ready?**

> The **architecture** is production-ready: clean separation of layers, typed Pydantic models, async FastAPI, stateless request handling, health endpoint, structured LLM output. The **data inputs** are not — competitor prices need a live feed and costs need a COGS column. The correct framing is: *"The pipeline is built. Plug in better data and the quality of recommendations scales automatically."*

**Q: Why not a simpler approach — just match competitor price and add a fixed margin?**

> That ignores demand elasticity entirely. A product selling 92 units/day has far more pricing power than one selling 0.3 units/day — charging the same fixed margin above competitor for both is leaving money on the table for the first and pushing away customers for the second. The demand-scaled premium (5%–20%) is the minimum necessary sophistication to avoid those errors.

---

## Dataset

[Online Retail Dataset](https://archive.ics.uci.edu/dataset/352/online+retail) — 541,909 transactions, UK-based e-commerce, 2010–2011.

After 7-rule cleaning pipeline: **397,884 rows**, 3,665+ unique products.

---

## License

MIT — see [LICENSE](LICENSE) for details.
