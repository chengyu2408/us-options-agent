"""
OptionsAgent — the LLM-driven decision core.

Orchestrates market analysis → strategy selection → trade execution
via an LLM "analyst" that reasons about options chains, Greeks,
and macro conditions before acting.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class AnalysisResult:
    symbol: str
    timestamp: datetime
    sentiment: str  # bullish | bearish | neutral
    confidence: float  # 0.0 - 1.0
    strategy: str | None
    reasoning: str
    signals: dict[str, Any] = field(default_factory=dict)


class OptionsAgent:
    """Multi-step LLM agent for US options trading decisions."""

    def __init__(self, llm_client: Any | None = None, model: str = "gpt-4o"):
        self.model = model
        self.llm = llm_client
        self.history: list[AnalysisResult] = []

    async def analyze(self, symbol: str, market_data: dict) -> AnalysisResult:
        """Run the full analysis pipeline for a symbol."""
        # 1. Gather signals
        signals = self._compute_signals(market_data)

        # 2. LLM synthesis (when a client is attached)
        if self.llm:
            reasoning = await self._llm_analyze(symbol, signals)
        else:
            reasoning = "No LLM client configured — using signal-only mode."

        # 3. Decide strategy
        strategy = self._pick_strategy(signals, reasoning)

        result = AnalysisResult(
            symbol=symbol,
            timestamp=datetime.utcnow(),
            sentiment=signals.get("sentiment", "neutral"),
            confidence=signals.get("confidence", 0.5),
            strategy=strategy,
            reasoning=reasoning,
            signals=signals,
        )
        self.history.append(result)
        return result

    def _compute_signals(self, data: dict) -> dict:
        """Compute technical + fundamental signals."""
        signals: dict[str, Any] = {"indicators": {}}
        prices = data.get("prices", [])

        if len(prices) > 20:
            sma_short = sum(prices[-20:]) / 20
            sma_long = sum(prices[-50:]) / 50 if len(prices) >= 50 else sma_short
            signals["indicators"]["sma_20"] = sma_short
            signals["indicators"]["sma_50"] = sma_long
            signals["indicators"]["trend"] = (
                "up" if sma_short > sma_long else "down"
            )

        iv = data.get("implied_volatility")
        if iv:
            signals["indicators"]["iv_rank"] = iv
            # High IV → premium selling opportunity
            signals["indicators"]["iv_signal"] = "sell_premium" if iv > 0.5 else "neutral"

        # Sentiment heuristic
        trend = signals["indicators"].get("trend", "neutral")
        sentiment_map = {"up": "bullish", "down": "bearish", "neutral": "neutral"}
        signals["sentiment"] = sentiment_map.get(trend, "neutral")
        signals["confidence"] = 0.6 if trend != "neutral" else 0.3

        return signals

    async def _llm_analyze(self, symbol: str, signals: dict) -> str:
        """Ask the LLM for a reasoned market read with rich context."""
        # Extract extra data
        options_chain = signals.pop("options_chain", {})
        realtime_quote = signals.pop("realtime_quote", {})

        prompt_parts = [f"You are an expert US options trader. Analyze {symbol}."]

        if realtime_quote:
            prompt_parts.append(f"\nCurrent Quote:\n  Price: ${realtime_quote.get('last_price', 'N/A')}")
            prompt_parts.append(f"  Day Range: ${realtime_quote.get('low_price', 'N/A')} - ${realtime_quote.get('high_price', 'N/A')}")
            prompt_parts.append(f"  Volume: {realtime_quote.get('volume', 'N/A')}")

        prompt_parts.append(f"\nTechnical Signals:\n{json.dumps(signals, indent=2)}")

        if options_chain and options_chain.get("calls"):
            calls = options_chain["calls"][:3]
            puts = options_chain["puts"][:3]
            prompt_parts.append("\nNear-term Options Chain:")
            for c in calls:
                strike = c.get("strike", "?")
                prompt_parts.append(f"  CALL ${strike}")
            for p in puts:
                strike = p.get("strike", "?")
                prompt_parts.append(f"  PUT  ${strike}")

        prompt_parts.append("""

Provide a concise analysis covering:
1. Market regime (bullish/bearish/neutral) and why
2. An appropriate options strategy (covered call, cash-secured put, iron condor, vertical spread, long call/put)
3. Key risks to watch
4. Suggested trade structure (strike, expiry bias)

Be specific and actionable.""")

        prompt = "\n".join(prompt_parts)
        try:
            resp = await self.llm.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
            )
            return resp.choices[0].message.content or "No response."
        except Exception as e:
            logger.warning("LLM analysis failed: %s", e)
            return f"LLM unavailable: {e}"

    def _pick_strategy(self, signals: dict, reasoning: str) -> str | None:
        """Rule-based strategy fallback."""
        sentiment = signals.get("sentiment", "neutral")
        iv_signal = signals.get("indicators", {}).get("iv_signal", "neutral")

        if sentiment == "bullish" and iv_signal == "sell_premium":
            return "cash_secured_put"
        elif sentiment == "bearish" and iv_signal == "sell_premium":
            return "covered_call"
        elif sentiment == "bullish":
            return "long_call"
        elif sentiment == "bearish":
            return "long_put"
        return None
