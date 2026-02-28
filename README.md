<div align="center">

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  │
│  ░  ▄▄▄  ▄  ▄  ▄  ▄     ▄▄▄  ▄▄▄  ▄  ▄▄▄  ▄  ▄  ▄▄▄     ▄▄▄  ▄     ░░░░░░  │
│  ░  █ █  ▀▄▀  █▄█     █ █  █ ██▄   █  █ █ ██─██  █ █    █▀█  █     ░░░░░░  │
│  ░  ▀▀▀   ▀   █ █  ▀  ▀▀▀  ▀  ▀▀▀  ▀  ▀ ▀  ▀ ▀   ▀▀▀  ▀ ▀▀▀  ▀▀▀   ░░░░░░  │
│  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  │
│                                                                                 │
│         ⬡ DYNAMIC   ⬡ PRICING   ⬡ INTELLIGENCE   ⬡ AGENT                     │
│                                                                                 │
│   ──◈── data layer   ──◈── decision engine   ──◈── gemini llm ──◈──           │
└─────────────────────────────────────────────────────────────────────────────────┘
```

**AI-powered dynamic pricing engine for e-commerce intelligence**  
*Rule-based decisions · Gemini LLM reasoning · Confidence scoring · FastAPI backend*

![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111%2B-009688?logo=fastapi&logoColor=white)
![Gemini](https://img.shields.io/badge/LLM-Gemini%20Flash-4285F4?logo=google&logoColor=white)
![Pandas](https://img.shields.io/badge/Data-Pandas%202.0%2B-150458?logo=pandas&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-22C55E)

</div>


---

## What is This?

A production-structured AI pricing intelligence system that analyzes historical e-commerce sales data and recommends optimal product prices — backed by data signals, business rules, and Gemini LLM-generated explanations a CFO can actually read.

It does three things no simple pricing script does:

1. **Derives product signals** — demand velocity, sales trend (OLS regression), price volatility, inventory proxy — from 400k+ historical transactions.
2. **Makes rule-based pricing decisions** — 5 named business rules produce `increase`, `decrease`, or `hold` with explicit reasoning factors.
3. **Explains decisions in business English** — Gemini Flash writes a 2–3 line executive summary and risk assessment for every recommendation.

---

## Architecture

```
                    ┌──────────────────────────────────┐
                    │          Browser Dashboard       │
                    │   (Vanilla HTML + CSS + JS)      │
                    └────────────────┬─────────────────┘
                                     │  POST /analyze
                    ┌────────────────▼─────────────────┐
                    │        FastAPI  (main.py)        │
                    │  GET  /          → Dashboard      │
                    │  GET  /health    → Liveness probe │
                    │  POST /analyze   → Full pipeline  │
                    └───┬────────┬────────┬────────┬───┘
                        │        │        │        │
              ┌─────────▼──┐ ┌───▼────┐ ┌─▼─────┐ ┌▼──────────────┐
              │  Layer 1   │ │Layer 2 │ │Layer 3│ │   Layer 4      │
              │  Data      │ │Feature │ │Decision│ │  Confidence    │
              │  Loader    │ │Engin.  │ │Engine │ │  Scorer        │
              │            │ │        │ │       │ │                │
              │ • CSV clean│ │• 8 per-│ │• 5    │ │• 4-dim score   │
              │ • 7 rules  │ │  product│ │ rules │ │• 0–1 float     │
              │ • 397k rows│ │  signals│ │• OLS  │ │• weighted avg  │
              └────────────┘ └────────┘ └───────┘ └────────────────┘
                                                           │
                                              ┌────────────▼──────────┐
                                              │      Layer 5          │
                                              │   LLM Explainer       │
                                              │                       │
                                              │ • Gemini Flash        │
                                              │ • JSON-mode output    │
                                              │ • Explanation         │
                                              │ • Executive Summary   │
                                              │ • Risk: Low/Med/High  │
                                              └───────────────────────┘
```

---

## Quick Start

### Prerequisites
- Python 3.11+
- Google Gemini API key → [Get one free](https://aistudio.google.com/app/apikey)

### 1 — Clone & Install

```bash
git clone https://github.com/WORTHOX/Dynamic-E-Commerce-Pricing-Intelligence-Agent.git
cd Dynamic-E-Commerce-Pricing-Intelligence-Agent

pip install -r requirements.txt
```

### 2 — Configure

```bash
cp .env.example .env
```

Edit `.env` and set your key:
```
GEMINI_API_KEY=your_key_here
GEMINI_MODEL=gemini-flash-latest
```

### 3 — Run

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Open **http://localhost:8000** — the dashboard loads instantly.

---

## API Reference

### `POST /analyze`

Runs the full 5-layer pipeline for a single product.

**Request**
```json
{
  "stock_code": "85123A",
  "current_price": 2.55,
  "competitor_price": 2.30
}
```

**Response**
```json
{
  "stock_code": "85123A",
  "recommended_price": 2.62,
  "decision_type": "increase",
  "confidence_score": 0.9325,
  "explanation": "The recommendation to increase the price is supported by strong demand velocity of 92.3 units/day...",
  "executive_summary": "We recommend a price increase to £2.62 to capture margin while demand is strong and the market is stable.",
  "risk_level": "Low",
  "risk_rationale": "High confidence score and stable market indicators suggest minimal risk.",
  "reasoning_factors": [
    "Demand velocity (92.3 units/day) indicates strong market pull.",
    "Competitor is not undercutting; price stability supports increase.",
    "Low price volatility signals a stable pricing environment."
  ],
  "features": {
    "avg_price": 2.8931,
    "demand_velocity": 92.2857,
    "price_volatility": 0.2471,
    "total_sales_last_30d": 10666.47,
    "sales_trend": "decreasing",
    "inventory_proxy": 36782.0,
    "data_coverage_ratio": 0.8155,
    "record_count": 2035
  }
}
```

### `GET /health`

Liveness probe for load balancers / Kubernetes.

```json
{ "status": "ok", "rows": "397884" }
```

---

## Decision Engine — The 5 Rules

| Rule | Condition | Action |
|---|---|---|
| **Defensive Decrease** | Competitor >10% cheaper AND demand weak AND high inventory | Decrease 7% |
| **Opportunistic Increase** | Competitor >8% pricier AND demand strong AND trend increasing | Increase 5% |
| **Soft Decrease** | Competitor >10% cheaper AND low inventory | Decrease 3.5% |
| **Moderate Increase** | Demand strong AND competitor not undercutting AND price stable | Increase 3% |
| **Hold** | No dominant signal | Maintain price |

All thresholds are named constants — no magic numbers buried in logic.

---

## Confidence Scoring

A 4-dimension weighted score (0–1) tells you how much to trust the recommendation:

| Dimension | Weight | Measures |
|---|---|---|
| Data Completeness | 30% | Record count + day coverage ratio |
| Demand Stability | 25% | Velocity consistency |
| Trend Consistency | 25% | OLS clarity of direction |
| Price Stability | 20% | Inverse of UnitPrice std dev |

- `≥ 0.75` → **High confidence** — act on it
- `0.50–0.74` → **Moderate** — use with context
- `< 0.50` → **Low** — treat as a signal only

---

## Project Structure

```
Dynamic-E-Commerce-Pricing-Intelligence-Agent/
│
├── data.csv                         ← Online Retail dataset (~542k rows)
├── main.py                          ← FastAPI app (orchestrator + 3 endpoints)
│
├── data/
│   ├── __init__.py
│   └── loader.py                    ← Layer 1: 7-rule data cleaning
│
├── services/
│   ├── __init__.py
│   ├── feature_engineering.py       ← Layer 2: 8 per-product signals (OLS trend)
│   ├── decision_engine.py           ← Layer 3: 5 named pricing rules
│   ├── confidence.py                ← Layer 4: 4-weighted-dimension scoring
│   └── llm_explainer.py             ← Layer 5: Gemini Flash JSON explanation
│
├── templates/
│   └── index.html                   ← Dark-mode dashboard (no frameworks)
│
├── requirements.txt
├── .env.example                     ← Copy to .env and add your API key
└── .gitignore
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| API Framework | FastAPI |
| ASGI Server | Uvicorn |
| Data Processing | Pandas, NumPy |
| Trend Analysis | SciPy (OLS regression) |
| LLM | Google Gemini Flash (`google-genai` SDK) |
| Input Validation | Pydantic v2 |
| Templating | Jinja2 |
| Config | python-dotenv |
| Frontend | Pure HTML + CSS + Vanilla JS |
| Runtime | Python 3.11+ |

---

## Dataset

Uses the [Online Retail Dataset](https://archive.ics.uci.edu/dataset/352/online+retail) — 541,909 transactions from a UK-based e-commerce retailer (2010–2011).

After cleaning: **397,884 rows** across hundreds of unique products (`StockCode`).

---

## License

MIT — see [LICENSE](LICENSE) for details.
