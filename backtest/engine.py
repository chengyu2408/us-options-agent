"""Backtest engine — replay agent decisions against historical data."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import pandas as pd


@dataclass
class BacktestTrade:
    symbol: str
    entry_date: datetime
    exit_date: datetime | None
    strategy: str
    pnl: float
    entry_price: float
    exit_price: float | None


@dataclass
class BacktestResult:
    total_pnl: float
    total_trades: int
    win_rate: float
    max_drawdown: float
    sharpe: float
    trades: list[BacktestTrade] = field(default_factory=list)


class BacktestEngine:
    """Walk-forward backtester for options strategies."""

    def __init__(self, initial_capital: float = 100_000):
        self.capital = initial_capital
        self.trades: list[BacktestTrade] = []

    async def run(
        self,
        symbol: str,
        price_history: pd.DataFrame,
        agent: Any,
    ) -> BacktestResult:
        """Run a backtest by feeding the agent historical slices."""
        for i in range(50, len(price_history)):  # start after warmup
            window = price_history.iloc[:i]
            current = price_history.iloc[i]

            # Feed the agent the window
            data = {
                "prices": window["Close"].tolist(),
                "volume": window["Volume"].tolist() if "Volume" in window else [],
            }

            result = await agent.analyze(symbol, data)

            if result.strategy:
                self.trades.append(
                    BacktestTrade(
                        symbol=symbol,
                        entry_date=current.name if hasattr(current, "name") else datetime.now(),
                        exit_date=None,
                        strategy=result.strategy,
                        pnl=0.0,
                        entry_price=float(current["Close"]),
                        exit_price=None,
                    )
                )

        return self._compute_results()

    def _compute_results(self) -> BacktestResult:
        if not self.trades:
            return BacktestResult(0, 0, 0, 0, 0)

        pnls = [t.pnl for t in self.trades]
        wins = [p for p in pnls if p > 0]
        total_pnl = sum(pnls)
        win_rate = len(wins) / len(pnls) if pnls else 0
        max_dd = self._max_drawdown(pnls)
        sharpe = self._sharpe(pnls)

        return BacktestResult(
            total_pnl=total_pnl,
            total_trades=len(self.trades),
            win_rate=win_rate,
            max_drawdown=max_dd,
            sharpe=sharpe,
            trades=self.trades,
        )

    @staticmethod
    def _max_drawdown(pnls: list[float]) -> float:
        cumulative = pd.Series(pnls).cumsum()
        peak = cumulative.expanding().max()
        dd = (cumulative - peak).min()
        return float(dd) if pd.notna(dd) else 0.0

    @staticmethod
    def _sharpe(pnls: list[float], rf: float = 0.05) -> float:
        s = pd.Series(pnls)
        if s.std() == 0:
            return 0.0
        return float((s.mean() * 252 - rf) / (s.std() * (252**0.5)))
