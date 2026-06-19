"""Futu (富途牛牛) broker integration — market data, trading, account info."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class FutuBroker:
    """Futu OpenAPI trading interface.

    Requires:
      1. Futu account + OpenAPI application approval
      2. Futu OpenAPI server running locally (default port 11111)
      https://openapi.futunn.com/
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 11111, pwd_unlock: str = ""):
        self._host = host
        self._port = port
        self._pwd = pwd_unlock
        self._quote_ctx: Any = None
        self._trade_ctx: Any = None

    async def connect_quote(self) -> bool:
        """Connect to Futu's quote context."""
        try:
            import futu as ft

            self._quote_ctx = ft.OpenQuoteContext(host=self._host, port=self._port)
            logger.info("Futu quote connected → %s:%s", self._host, self._port)
            return True
        except Exception as e:
            logger.error("Futu quote connect failed: %s", e)
            return False

    async def connect_trade(self, acc_id: int = 0, trd_env: int = 1) -> bool:
        """Connect to Futu's trade context (1=simulate, 0=real)."""
        try:
            import futu as ft

            self._trade_ctx = ft.OpenSecTradeContext(
                host=self._host,
                port=self._port,
                security_firm=ft.SecurityFirm.FUTUSECURITIES,
            )
            if self._pwd:
                ret, _ = self._trade_ctx.unlock_trade(password=self._pwd)
                if ret != ft.RET_OK:
                    logger.warning("Futu trade unlock failed")
                    return False
            logger.info("Futu trade connected (env=%s)", "simulate" if trd_env else "real")
            return True
        except Exception as e:
            logger.error("Futu trade connect failed: %s", e)
            return False

    async def get_account_summary(self) -> dict:
        """Fetch account summary via get_acc_list."""
        if not self._trade_ctx:
            return {"status": "not connected"}
        try:
            import futu as ft

            ret, data = self._trade_ctx.get_acc_list()
            if ret != ft.RET_OK:
                return {"error": "get_acc_list failed"}
            accounts = []
            for _, row in data.iterrows():
                accounts.append({
                    "acc_id": int(row.get("accID", 0)),
                    "name": row.get("accName", ""),
                    "type": str(row.get("accType", "")),
                    "status": str(row.get("accStatus", "")),
                })
            return {"accounts": accounts}
        except Exception as e:
            return {"error": str(e)}

    async def get_positions(self, trd_env: int = 1) -> list[dict]:
        """Get current positions."""
        if not self._trade_ctx:
            return []
        try:
            import futu as ft

            ret, data = self._trade_ctx.position_list_query(trd_env=trd_env)
            if ret != ft.RET_OK:
                return []
            positions = []
            for _, row in data.iterrows():
                positions.append({
                    "code": row.get("code", ""),
                    "stock_name": row.get("stockName", ""),
                    "qty": int(row.get("qty", 0)),
                    "cost_price": float(row.get("costPrice", 0)),
                    "market_val": float(row.get("marketVal", 0)),
                    "pl_ratio": float(row.get("plRatio", 0)),
                })
            return positions
        except Exception as e:
            logger.warning("get_positions failed: %s", e)
            return []

    async def place_order(
        self,
        code: str,
        qty: int,
        side: str,
        price: float = 0.0,
        order_type: str = "market",
        trd_env: int = 1,
    ) -> dict:
        """Place an order."""
        if not self._trade_ctx:
            return {"status": "not connected"}
        try:
            import futu as ft

            trd_side = ft.TrdSide.BUY if side == "buy" else ft.TrdSide.SELL
            price_type = ft.PriceType.MARKET if order_type == "market" else ft.PriceType.NORMAL

            ret, data = self._trade_ctx.place_order(
                price=price,
                qty=qty,
                code=code,
                trd_side=trd_side,
                order_type=price_type,
                trd_env=trd_env,
            )
            if ret == ft.RET_OK:
                return {
                    "status": "submitted",
                    "order_id": str(data.iloc[0].get("orderID", "")),
                }
            return {"status": "failed", "error": str(data)}
        except Exception as e:
            return {"error": str(e)}

    def close(self):
        if self._quote_ctx:
            try:
                self._quote_ctx.close()
            except Exception:
                pass
        if self._trade_ctx:
            try:
                self._trade_ctx.close()
            except Exception:
                pass
