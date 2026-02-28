"""
services/llm_explainer.py
--------------------------
LLM Explanation Layer for the E-Commerce Pricing Intelligence Agent.

Uses the Google Gemini API (google-genai SDK) to generate:
  - A business-language explanation of the pricing recommendation.
  - A risk assessment (Low / Medium / High).
  - A 2 – 3 line executive summary.

Environment Variables
---------------------
GEMINI_API_KEY : str
    Required. Google Generative AI API key.
GEMINI_MODEL : str
    Optional. Defaults to 'gemini-1.5-flash'.
LLM_TEMPERATURE : float
    Optional. Generation temperature. Defaults to 0.3.
LLM_MAX_TOKENS : int
    Optional. Max output tokens. Defaults to 512.
"""

import json
import logging
import os
from typing import Any

from google import genai
from google.genai import types as genai_types

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
GEMINI_MODEL    = os.getenv("GEMINI_MODEL", "gemini-flash-latest")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.3"))
LLM_MAX_TOKENS  = int(os.getenv("LLM_MAX_TOKENS", "1024"))

_PROMPT_TEMPLATE = """
You are a pricing intelligence assistant for an e-commerce business.

The following data has been derived from sales analysis:

- Current Price     : {current_price}
- Competitor Price  : {competitor_price}
- Demand Velocity   : {demand_velocity} units/day (7-day rolling avg)
- Sales Trend       : {sales_trend}
- Inventory Level   : {inventory_proxy} units (cumulative proxy)
- Recommended Price : {recommended_price}
- Decision Type     : {decision_type}
- Key Factors       : {reasoning_factors}
- Confidence Score  : {confidence_score} (0 = no confidence, 1 = full confidence)

Your task is to respond ONLY with a valid JSON object (no markdown, no code fences) using this exact schema:

{{
  "explanation": "<2–4 sentence business explanation of why this price was recommended>",
  "executive_summary": "<2–3 line executive summary suitable for a VP or CFO>",
  "risk_level": "<one of: Low | Medium | High>",
  "risk_rationale": "<1 sentence explaining the risk level>"
}}

Guidelines:
- Use plain business English. Avoid jargon.
- If confidence is below 0.5, flag it explicitly in risk_rationale.
- Risk level should reflect combined uncertainty: data quality, market dynamics, and magnitude of recommended change.
"""


def generate_explanation(
    current_price: float,
    competitor_price: float,
    features: dict[str, Any],
    decision: dict[str, Any],
    confidence_score: float,
) -> dict[str, str]:
    """
    Call the Gemini LLM to generate a business explanation for the pricing decision.

    Parameters
    ----------
    current_price : float
        Merchant's current price.
    competitor_price : float
        Reference competitor price.
    features : dict[str, Any]
        Output from feature engineering layer.
    decision : dict[str, Any]
        Output from decision engine layer.
    confidence_score : float
        Output from confidence scoring layer.

    Returns
    -------
    dict[str, str]
        Keys:
        - ``explanation``        : str
        - ``executive_summary``  : str
        - ``risk_level``         : str  ('Low' | 'Medium' | 'High')
        - ``risk_rationale``     : str

    Raises
    ------
    EnvironmentError
        If GEMINI_API_KEY is not set.
    RuntimeError
        If the Gemini API call fails or returns non-parseable output.
    """
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise EnvironmentError(
            "GEMINI_API_KEY is not set. "
            "Add it to your .env file or export it as an environment variable."
        )

    client = genai.Client(api_key=api_key)

    prompt = _PROMPT_TEMPLATE.format(
        current_price     = round(current_price, 2),
        competitor_price  = round(competitor_price, 2),
        demand_velocity   = features.get("demand_velocity", "N/A"),
        sales_trend       = features.get("sales_trend", "N/A"),
        inventory_proxy   = features.get("inventory_proxy", "N/A"),
        recommended_price = decision.get("recommended_price", "N/A"),
        decision_type     = decision.get("decision_type", "N/A"),
        reasoning_factors = "; ".join(decision.get("reasoning_factors", [])),
        confidence_score  = confidence_score,
    )

    logger.info("Calling Gemini model='%s' for explanation.", GEMINI_MODEL)
    try:
        response = client.models.generate_content(
            model    = GEMINI_MODEL,
            contents = prompt,
            config   = genai_types.GenerateContentConfig(
                temperature        = LLM_TEMPERATURE,
                max_output_tokens  = LLM_MAX_TOKENS,
                response_mime_type = "application/json",
            ),
        )
        raw_text = response.text.strip()
    except Exception as exc:
        logger.error("Gemini API call failed: %s", exc)
        raise RuntimeError(f"LLM call failed: {exc}") from exc

    # ── Parse structured JSON response ────────────────────────────────────
    # response_mime_type forces clean JSON output; fallback handles edge cases
    try:
        parsed: dict[str, str] = json.loads(raw_text)
    except json.JSONDecodeError:
        import re
        match = re.search(r'\{[\s\S]+\}', raw_text)
        try:
            parsed = json.loads(match.group()) if match else {}
            if not parsed:
                raise ValueError("empty")
        except Exception as exc:
            logger.warning("Could not parse LLM JSON. Error: %s | Raw: %.300s", exc, raw_text)
            parsed = {
                "explanation":       raw_text,
                "executive_summary": "See explanation field.",
                "risk_level":        "Medium",
                "risk_rationale":    "Structured output parsing failed; review raw explanation.",
            }

    # Ensure all expected keys are present
    for key in ("explanation", "executive_summary", "risk_level", "risk_rationale"):
        parsed.setdefault(key, "")

    logger.info("LLM explanation generated. Risk level: %s", parsed.get("risk_level"))
    return parsed
