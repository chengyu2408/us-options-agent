# US Options Agent 🤖📈

AI-powered US stock options trading agent with LLM decision-making,
strategy analytics, and backtesting.

## Quick Start

```bash
pip install -e ".[dev]"

# Set up your API keys
cp .env.example .env
# Edit .env with your Alpaca + LLM keys

# Run the example
python examples/basic_usage.py
```

## Architecture

```
us-options-agent/
├── agent/          ← LLM orchestration, reasoning, prompts
│   ├── core.py         Agent loop & signal computation
│   └── prompts.py      LLM system prompts
├── strategies/     ← Options strategy definitions & risk
│   ├── options.py      Strategy types, payoff calculations
│   └── risk.py         Position sizing, Greeks limits
├── market/         ← Data feeds & broker integration
│   ├── data.py         yfinance / Alpaca market data
│   └── broker.py       Alpaca paper & live trading
├── backtest/       ← Walk-forward backtesting engine
│   └── engine.py       P&L, win rate, Sharpe, drawdown
└── config/         ← Pydantic settings
```

## Agent Pipeline

1. **Signal computation** — technical indicators, IV rank, sentiment
2. **LLM synthesis** — optional LLM reads market and recommends strategy
3. **Risk checking** — validate against position limits & Greeks thresholds
4. **Execution** — via Alpaca (paper by default)

## Requirements

- Python 3.11+
- Alpaca account (free paper trading)
- LLM API key (OpenAI / DeepSeek / Anthropic)

## Disclaimer

For educational purposes only. Not financial advice. Options trading
involves substantial risk.
