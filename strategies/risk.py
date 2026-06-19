"""Risk management — position sizing, Greeks thresholds, portfolio constraints."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RiskLimits:
    max_position_size_pct: float = 0.05  # 5% of portfolio per trade
    max_portfolio_theta: float = -0.02  # daily theta not below -2%
    max_portfolio_gamma: float = 0.10
    max_concentrated_single_name: float = 0.20  # max 20% in one ticker
    min_days_to_expiry: int = 7
    max_days_to_expiry: int = 90


DEFAULT_RISK_LIMITS = RiskLimits()


def validate_trade(
    strategy_pct: float,
    ticker_concentration: float,
    dte: int,
    limits: RiskLimits = DEFAULT_RISK_LIMITS,
) -> tuple[bool, str]:
    """Check a proposed trade against risk limits. Returns (pass, reason)."""
    if strategy_pct > limits.max_position_size_pct:
        return False, f"Position {strategy_pct:.1%} exceeds {limits.max_position_size_pct:.1%} limit"
    if ticker_concentration > limits.max_concentrated_single_name:
        return False, f"Ticker concentration {ticker_concentration:.1%} exceeds limit"
    if dte < limits.min_days_to_expiry:
        return False, f"DTE {dte} too low (min {limits.min_days_to_expiry})"
    if dte > limits.max_days_to_expiry:
        return False, f"DTE {dte} too high (max {limits.max_days_to_expiry})"
    return True, "Trade passes risk checks"
