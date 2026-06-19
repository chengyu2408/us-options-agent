"""Alpaca broker integration for paper & live trading."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class AlpacaBroker:
    """Thin wrapper around Alpaca's trading API."""

    def __init__(self, api_key: str = "", secret_key: str = "", paper: bool = True):
        self._api_key = api_key
        self._secret_key = secret_key
        self._paper = paper
        self._client: Any = None

    async def connect(self):
        """Initialize the Alpaca client."""
        if not self._api_key or not self._secret_key:
            logger.warning("Alpaca credentials not set — running in dry mode")
            return False
        try:
            from alpaca.trading.client import TradingClient

            self._client = TradingClient(
                api_key=self._api_key,
                secret_key=self._secret_key,
                paper=self._paper,
            )
            account = self._client.get_account()
            logger.info(
                "Alpaca connected — equity=%.2f, buying_power=%.2f",
                float(account.equity),
                float(account.buying_power),
            )
            return True
        except Exception as e:
            logger.error("Alpaca connect failed: %s", e)
            return False

    async def get_account_summary(self) -> dict:
        if not self._client:
            return {"status": "not connected"}
        try:
            acct = self._client.get_account()
            return {
                "equity": float(acct.equity),
                "buying_power": float(acct.buying_power),
                "cash": float(acct.cash),
                "day_trade_count": int(acct.daytrade_count),
                "status": acct.status,
            }
        except Exception as e:
            return {"error": str(e)}

    async def submit_order(self, order: dict) -> dict:
        """Place an options or equity order."""
        if not self._client:
            return {"status": "dry_run", "order": order}
        try:
            from alpaca.trading.requests import MarketOrderRequest
            from alpaca.trading.enums import OrderSide, TimeInForce

            req = MarketOrderRequest(
                symbol=order["symbol"],
                qty=order["qty"],
                side=OrderSide.BUY if order.get("side") == "buy" else OrderSide.SELL,
                time_in_force=TimeInForce.DAY,
            )
            resp = self._client.submit_order(req)
            return {"status": "submitted", "id": str(resp.id), "filled_qty": str(resp.filled_qty)}
        except Exception as e:
            return {"error": str(e)}
