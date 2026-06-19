"""Options strategy definitions and P&L calculations."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal


class StrategyType(str, Enum):
    LONG_CALL = "long_call"
    LONG_PUT = "long_put"
    COVERED_CALL = "covered_call"
    CASH_SECURED_PUT = "cash_secured_put"
    IRON_CONDOR = "iron_condor"
    VERTICAL_SPREAD = "vertical_spread"


@dataclass
class OptionsStrategy:
    """A concrete options trade configuration."""

    type: StrategyType
    symbol: str
    strike: float
    expiration: str  # YYYY-MM-DD
    direction: Literal["buy", "sell"]
    premium: float  # per share
    contracts: int = 1

    @property
    def notional_value(self) -> float:
        return self.strike * 100 * self.contracts

    @property
    def max_risk(self) -> float:
        if self.type in (StrategyType.LONG_CALL, StrategyType.LONG_PUT):
            return self.premium * 100 * self.contracts
        return self.notional_value

    @property
    def max_profit(self) -> float | None:
        if self.type == StrategyType.LONG_CALL:
            return None  # uncapped
        if self.type == StrategyType.LONG_PUT:
            return self.max_risk  # spot goes to 0
        return self.premium * 100 * self.contracts


def payoff_at_expiry(strategy: OptionsStrategy, spot: float) -> float:
    """Calculate P&L at expiry for a single-leg option."""
    multiplier = 100 * strategy.contracts
    if strategy.type == StrategyType.LONG_CALL:
        return max(0, spot - strategy.strike) * multiplier - strategy.premium * multiplier
    if strategy.type == StrategyType.LONG_PUT:
        return max(0, strategy.strike - spot) * multiplier - strategy.premium * multiplier
    raise NotImplementedError(f"Payoff for {strategy.type} not implemented")
