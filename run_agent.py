"""
US Options Agent — entry point.

Usage:
  python run_agent.py --llm                      # yfinance + DeepSeek LLM
  python run_agent.py --futu --llm               # Futu + DeepSeek LLM
  python run_agent.py --symbol SPY,QQQ            # custom watchlist

Environment:
  LLM_API_KEY   DeepSeek / OpenAI API key (or find it from your OpenClaw config)
  LLM_MODEL     Model name, default: deepseek-chat
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("agent")


def parse_args():
    p = argparse.ArgumentParser(description="US Options Agent")
    p.add_argument("--futu", action="store_true", help="Use Futu for data + trading")
    p.add_argument(
        "--symbol",
        default="SPY",
        help="Stock symbol(s), comma-separated (default: SPY)",
    )
    p.add_argument("--paper", action="store_true", default=True, help="Paper trade (default)")
    p.add_argument("--llm", action="store_true", help="Enable LLM market analysis")
    return p.parse_args()


def build_llm_client():
    """Build an OpenAI-compatible LLM client (DeepSeek by default)."""
    from openai import AsyncOpenAI

    api_key = os.getenv("LLM_API_KEY") or os.getenv("DEEPSEEK_API_KEY")
    model = os.getenv("LLM_MODEL", "deepseek-chat")
    base_url = os.getenv("LLM_API_BASE", "https://api.deepseek.com")

    if not api_key:
        # Try to read from OpenClaw's auth store
        try:
            import sqlite3

            db = os.path.expanduser("~/.openclaw/agents/main/agent/openclaw-agent.sqlite")
            if os.path.exists(db):
                conn = sqlite3.connect(db)
                cur = conn.execute("SELECT store_json FROM auth_profile_store WHERE store_key='primary'")
                row = cur.fetchone()
                conn.close()
                if row:
                    store = json.loads(row[0])
                    dp = store.get("profiles", {}).get("deepseek:default", {})
                    api_key = dp.get("key", "")
                    if api_key:
                        logger.info("Read DeepSeek API key from OpenClaw auth store")
        except Exception as e:
            logger.debug("Could not read auth store: %s", e)

    if not api_key:
        print("\n⚠  No API key found. Set LLM_API_KEY or DEEPSEEK_API_KEY env var.")
        print("   Or get it from your OpenClaw config and set it:")
        print("   export LLM_API_KEY=sk-xxx")
        return None, model

    client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    return client, model


async def run_with_yfinance(symbols: list[str], llm_config: tuple | None):
    """Run agent using yfinance data."""
    from agent.core import OptionsAgent
    from market.data import MarketDataProvider

    provider = MarketDataProvider()
    llm_client, model = llm_config if llm_config else (None, "gpt-4o")
    agent = OptionsAgent(llm_client=llm_client, model=model)

    print("\n" + "=" * 65)
    print("  🤖 US OPTIONS AGENT — yfinance mode" + (" (with LLM 🧠)" if llm_client else ""))
    print("=" * 65)

    for symbol in symbols:
        print(f"\n📊 Analyzing {symbol}...")

        prices = await provider.get_prices(symbol)
        iv = await provider.get_implied_volatility(symbol)
        chain = await provider.get_options_chain(symbol)

        market_data = {
            "prices": prices,
            "implied_volatility": iv,
            "options_chain": chain,
        }

        if not prices:
            print(f"  ⚠ No price data for {symbol}")
            continue

        result = await agent.analyze(symbol, market_data)

        # Print summary
        print(f"  {'Price':20s} ${prices[-1]:<8.2f} ({len(prices)} days)")
        print(f"  {'Sentiment':20s} {result.sentiment.upper():>8}  ({result.confidence:.0%} confidence)")
        print(f"  {'Recommendation':20s} {result.strategy or 'HOLD / no signal'}")
        print(f"  {'IV':20s} {f'{iv:.1%}' if iv else 'N/A'}")

        ind = result.signals.get("indicators", {})
        if ind.get("trend"):
            print(f"  {'Trend':20s} {ind['trend']} (SMA20={ind.get('sma_20', 0):.1f})")

        if chain and chain.get("calls"):
            print(f"  {'Options Chain':20s} {len(chain['calls'])} calls, {len(chain['puts'])} puts")

        # LLM reasoning
        if result.reasoning and not result.reasoning.startswith("No LLM"):
            print(f"\n  🧠 LLM Analysis:")
            for line in result.reasoning.strip().split("\n"):
                print(f"     {line}")

    print("\n" + "=" * 65)
    print(f"  ✅ Done — {len(agent.history)} analyses completed")
    print("=" * 65 + "\n")


async def run_with_futu(symbols: list[str], paper: bool, llm_config: tuple | None):
    """Run agent using Futu OpenAPI data + trading."""
    from agent.core import OptionsAgent
    from market.futu_data import FutuDataProvider
    from market.futu_broker import FutuBroker

    provider = FutuDataProvider()
    broker = FutuBroker(pwd_unlock=os.getenv("FUTU_TRADE_PWD", ""))
    llm_client, model = llm_config if llm_config else (None, "gpt-4o")
    agent = OptionsAgent(llm_client=llm_client, model=model)

    print("\n" + "=" * 65)
    print("  🤖 US OPTIONS AGENT — Futu OpenAPI" + (" (with LLM 🧠)" if llm_client else ""))
    print("=" * 65)

    print("\n🔌 Connecting to Futu OpenAPI...")
    if not await provider.connect():
        print("  ❌ Cannot connect to Futu OpenAPI server.")
        print("  Make sure Futu OpenAPI is running (port 11111).")
        return
    if paper:
        await broker.connect_trade(trd_env=1)

    acct = await broker.get_account_summary()
    print(f"  Account: {acct.get('accounts', [])}")

    for symbol in symbols:
        us_code = f"US.{symbol}" if not symbol.startswith("US.") else symbol
        print(f"\n📊 Analyzing {us_code}...")

        # 1. Real-time quote
        quote = await provider.get_realtime_quote(us_code)
        if quote.get("last_price"):
            print(f"  {'Last Price':20s} ${quote['last_price']:<8.2f}  ({quote.get('change_pct', 0):+.2f}%)")

        # 2. K-line prices for signal computation
        prices = await provider.get_prices(us_code)
        print(f"  {'K-line Data':20s} {len(prices)} days of data")

        # 3. Options chain
        chain = await provider.get_options_chain(us_code)
        if chain and chain.get("calls"):
            print(f"  {'Options Chain':20s} {len(chain['calls'])} near-term calls available")

        # 4. Run agent analysis
        market_data = {
            "prices": prices or [],
            "implied_volatility": None,
            "options_chain": chain,
            "realtime_quote": quote,
        }

        result = await agent.analyze(symbol, market_data)

        print(f"  {'Sentiment':20s} {result.sentiment.upper():>8}  ({result.confidence:.0%})")
        print(f"  {'Recommendation':20s} {result.strategy or 'HOLD'}")

        if result.reasoning and not result.reasoning.startswith("No LLM"):
            print(f"\n  🧠 LLM Analysis:")
            for line in result.reasoning.strip().split("\n"):
                print(f"     {line}")

    provider.close()
    broker.close()
    print("\n✅ Done — connections closed.")


async def main():
    args = parse_args()

    llm_config = None
    if args.llm:
        client, model = build_llm_client()
        if client:
            llm_config = (client, model)

    symbols = [s.strip() for s in args.symbol.split(",")]

    if args.futu:
        await run_with_futu(symbols, args.paper, llm_config)
    else:
        await run_with_yfinance(symbols, llm_config)


if __name__ == "__main__":
    asyncio.run(main())
