"""Futu market data provider — real-time quotes, options chains, K-line."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any

logger = logging.getLogger(__name__)


class FutuDataProvider:
    """Market data via Futu OpenAPI.

    Requires the Futu OpenAPI server running locally.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 11111):
        self._host = host
        self._port = port
        self._ctx: Any = None

    async def connect(self) -> bool:
        try:
            import futu as ft

            self._ctx = ft.OpenQuoteContext(host=self._host, port=self._port)
            logger.info("Futu data provider connected")
            return True
        except Exception as e:
            logger.error("Futu data connect failed: %s", e)
            return False

    async def get_prices(self, code: str, days: int = 100) -> list[float]:
        """Fetch daily K-line close prices."""
        if not self._ctx:
            return []
        try:
            import futu as ft

            end = datetime.now()
            start = end - timedelta(days=days)
            ret, data, page_key = self._ctx.request_history_kline(
                code=code,
                start=start.strftime("%Y-%m-%d"),
                end=end.strftime("%Y-%m-%d"),
                ktype=ft.KLType.K_DAY,
            )
            if ret == ft.RET_OK and data is not None and not data.empty:
                return data["close"].dropna().tolist()
            return []
        except Exception as e:
            logger.warning("Futu get_prices failed for %s: %s", code, e)
            return []

    async def get_realtime_quote(self, code: str) -> dict:
        """Get real-time quote for a stock."""
        if not self._ctx:
            return {}
        try:
            import futu as ft

            # Try market snapshot first (more data)
            ret, snap = self._ctx.get_market_snapshot([code])
            if ret == ft.RET_OK and snap is not None and not snap.empty:
                s = snap.iloc[0]
                return {
                    "code": code,
                    "name": s.get("code", ""),
                    "last_price": float(s.get("lastPrice", 0)),
                    "open_price": float(s.get("openPrice", 0)),
                    "high_price": float(s.get("highPrice", 0)),
                    "low_price": float(s.get("lowPrice", 0)),
                    "volume": int(s.get("volume", 0)),
                    "turnover": float(s.get("turnover", 0)),
                    "change_pct": float(s.get("changePct", 0)),
                }

            # Fallback to stock quote
            ret, q = self._ctx.get_stock_quote([code])
            if ret == ft.RET_OK and q is not None and not q.empty:
                s = q.iloc[0]
                return {
                    "code": s.get("code", code),
                    "last_price": float(s.get("lastPrice", 0)),
                    "open_price": float(s.get("openPrice", 0)),
                    "high_price": float(s.get("highPrice", 0)),
                    "low_price": float(s.get("lowPrice", 0)),
                    "volume": int(s.get("volume", 0)),
                    "turnover": float(s.get("turnover", 0)),
                    "change_pct": float(s.get("changePct", 0)),
                }
            return {}
        except Exception as e:
            logger.debug("Futu quote (fallback) failed: %s", e)
            return {}

    async def get_options_chain(self, us_code: str) -> dict:
        """Fetch US stock options chain."""
        if not self._ctx:
            return {}
        try:
            import futu as ft

            ret, data = self._ctx.get_option_chain(
                code=us_code,
                start=datetime.now().strftime("%Y-%m-%d"),
                end=(datetime.now() + timedelta(days=60)).strftime("%Y-%m-%d"),
            )
            if ret == ft.RET_OK and data is not None and not data.empty:
                return {
                    "source": "futu",
                    "calls": data[data["optionType"] == ft.OptionType.CALL].head(5).to_dict(orient="records"),
                    "puts": data[data["optionType"] == ft.OptionType.PUT].head(5).to_dict(orient="records"),
                }
            return {}
        except Exception as e:
            logger.warning("Futu options chain failed: %s", e)
            return {}

    async def get_option_quote(self, option_code: str) -> dict:
        """Get real-time quote + Greeks for an option contract."""
        if not self._ctx:
            return {}
        try:
            import futu as ft

            ret, snap = self._ctx.get_market_snapshot([option_code])
            if ret == ft.RET_OK and snap is not None and not snap.empty:
                s = snap.iloc[0]
                return {
                    "code": option_code,
                    "bid": float(s.get("bidPrice", 0)),
                    "ask": float(s.get("askPrice", 0)),
                    "last": float(s.get("lastPrice", 0)),
                    "implied_vol": float(s.get("impliedVolatility", 0)),
                    "delta": float(s.get("delta", 0)),
                    "gamma": float(s.get("gamma", 0)),
                    "theta": float(s.get("theta", 0)),
                    "vega": float(s.get("vega", 0)),
                    "volume": int(s.get("volume", 0)),
                    "open_interest": int(s.get("openInterest", 0)),
                }
            return {}
        except Exception as e:
            logger.warning("Futu option quote failed: %s", e)
            return {}

    def close(self):
        if self._ctx:
            try:
                self._ctx.close()
            except Exception:
                pass
