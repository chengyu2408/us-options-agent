"""
US Options Agent — configuration.
"""

from pydantic_settings import BaseSettings
from typing import Literal


class Settings(BaseSettings):
    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    # Alpaca
    alpaca_api_key: str = ""
    alpaca_secret_key: str = ""
    alpaca_paper: bool = True

    # LLM
    llm_model: str = "gpt-4o"
    llm_api_key: str = ""
    llm_api_base: str = ""

    # Agent behavior
    max_concurrent_analyses: int = 3
    default_risk_per_trade: float = 0.02  # 2% account risk
    backtest_start: str = "2024-01-01"
    backtest_end: str = "2025-01-01"

    # Watchlist (comma-separated symbols)
    watchlist: str = "SPY,QQQ,IWM,AAPL,MSFT,TSLA,AMZN,GOOGL,META,NVDA"


settings = Settings()
