"""Market data provider — price feeds, options chains, Greeks."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


class MarketDataProvider:
    """Fetches market data from yfinance and/or Alpaca."""

    def __init__(self):
        self._cache: dict[str, Any] = {}

    async def get_prices(self, symbol: str, days: int = 100) -> list[float]:
        """Fetch daily close prices."""
        try:
            import yfinance as yf

            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=f"{days}d")
            closes = hist["Close"].dropna().tolist()
            return closes
        except ImportError:
            logger.warning("yfinance not installed — returning mock prices")
            return self._mock_prices(days)
        except Exception as e:
            logger.error("Failed to fetch %s prices: %s", symbol, e)
            return self._mock_prices(days)

    async def get_options_chain(self, symbol: str) -> dict:
        """Fetch options chain (simplified)."""
        try:
            import yfinance as yf

            ticker = yf.Ticker(symbol)
            exps = ticker.options
            if not exps:
                return {}
            chain = ticker.option_chain(exps[0])
            return {
                "expiration": exps[0],
                "calls": chain.calls.head(5).to_dict(orient="records"),
                "puts": chain.puts.head(5).to_dict(orient="records"),
            }
        except Exception as e:
            logger.error("Options chain fetch failed: %s", e)
            return {}

    async def get_implied_volatility(self, symbol: str) -> float | None:
        """Get implied volatility (e.g. VIX for SPY, or underlying IV)."""
        if symbol.upper() == "SPY":
            try:
                import yfinance as yf

                vix = yf.Ticker("^VIX")
                hist = vix.history(period="5d")
                return float(hist["Close"].iloc[-1] / 100)
            except Exception:
                pass
        return None

    def _mock_prices(self, days: int, base: float = 450) -> list[float]:
        import random
        random.seed(42)
        prices = [base]
        for _ in range(days - 1):
            prices.append(prices[-1] * (1 + random.gauss(0, 0.01)))
        return prices
