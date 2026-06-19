"""
US Options Agent — entry point.

Usage:
  python run_agent.py --llm                           # yfinance + default LLM
  python run_agent.py --futu --llm                    # Futu + default LLM
  python run_agent.py --futu --llm --provider claude  # Futu + Claude Sonnet
  python run_agent.py --symbol SPY,QQQ                 # custom watchlist

Environment:
  LLM_PROVIDER  openai (default) | claude (Anthropic)
  LLM_API_KEY   Your API key
  LLM_MODEL     Model name (default: deepseek-chat or claude-sonnet-4-20250514)
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
    p.add_argument("--provider", default="openai", choices=["openai", "claude", "gemini"],
                    help="LLM provider: openai, claude, or gemini")
    return p.parse_args()


def _read_key_from_store(profile_key: str) -> str | None:
    """Read API key from OpenClaw's auth SQLite store."""
    try:
        import sqlite3
        db = os.path.expanduser("~/.openclaw/agents/main/agent/openclaw-agent.sqlite")
        if not os.path.exists(db):
            return None
        conn = sqlite3.connect(db)
        cur = conn.execute("SELECT store_json FROM auth_profile_store WHERE store_key='primary'")
        row = cur.fetchone()
        conn.close()
        if not row:
            return None
        store = json.loads(row[0])
        profile = store.get("profiles", {}).get(profile_key, {})
        return profile.get("key") or None
    except Exception as e:
        logger.debug("Auth store read failed: %s", e)
        return None


def build_llm_client(provider: str = "openai"):
    """Build LLM client for the requested provider."""
    if provider == "claude":
        return _build_claude()
    if provider == "gemini":
        return _build_gemini()
    return _build_openai()


def _build_openai():
    """Build OpenAI-compatible client (DeepSeek, GPT, etc.)."""
    from openai import AsyncOpenAI

    api_key = os.getenv("LLM_API_KEY") or os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
    model = os.getenv("LLM_MODEL", "deepseek-chat")
    base_url = os.getenv("LLM_API_BASE", "https://api.deepseek.com")

    if not api_key:
        api_key = _read_key_from_store("deepseek:default")
        if api_key:
            logger.info("Using DeepSeek key from OpenClaw auth store")
            base_url = "https://api.deepseek.com"

    if not api_key:
        print("\n⚠  No OpenAI-compatible API key found.")
        print("   Set LLM_API_KEY, DEEPSEEK_API_KEY, or OPENAI_API_KEY")
        return None, model

    client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    return client, model


def _build_claude():
    """Build Anthropic Claude client."""
    from anthropic import AsyncAnthropic

    api_key = os.getenv("ANTHROPIC_API_KEY")
    model = os.getenv("LLM_MODEL", "claude-sonnet-4-20250514")

    if not api_key:
        api_key = _read_key_from_store("anthropic:default")
        if api_key:
            logger.info("Using Anthropic key from OpenClaw auth store")

    if not api_key:
        print("\n⚠  No Anthropic API key found. Set ANTHROPIC_API_KEY env var.")
        return None, model

    client = AsyncAnthropic(api_key=api_key)
    return client, model


def _build_gemini():
    """Build Gemini client using OpenAI-compatible endpoint (free tier).
    Get free API key from https://aistudio.google.com/apikey
    """
    from openai import AsyncOpenAI

    api_key = os.getenv("GEMINI_API_KEY")
    model = os.getenv("LLM_MODEL", "gemini-2.0-flash")
    base_url = os.getenv("LLM_API_BASE", "https://generativelanguage.googleapis.com/v1beta/openai/")

    if not api_key:
        api_key = _read_key_from_store("gemini:default")
        if api_key:
            logger.info("Using Gemini key from OpenClaw auth store")

    if not api_key:
        print("\n⚠  No Gemini API key found.")
        print("   Get a free key: https://aistudio.google.com/apikey")
        print("   Then: export GEMINI_API_KEY=***")
        return None, model

    client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    return client, model


_LLM_TAGS = {"claude": " (with Claude 🧠)", "gemini": " (with Gemini 🧠)", "openai": " (with LLM 🧠)"}


async def run_with_yfinance(symbols: list[str], llm_client, model: str, provider: str = ""):
    from agent.core import OptionsAgent
    from market.data import MarketDataProvider

    f_data = MarketDataProvider()
    agent = OptionsAgent(llm_client=llm_client, model=model, llm_provider=provider)

    tag = _LLM_TAGS.get(provider, "") if llm_client else ""
    print("\n" + "=" * 65)
    print(f"  🤖 US OPTIONS AGENT — yfinance{tag}")
    print("=" * 65)

    for symbol in symbols:
        print(f"\n📊 {symbol}...")
        prices = await f_data.get_prices(symbol)
        iv = await f_data.get_implied_volatility(symbol)
        chain = await f_data.get_options_chain(symbol)
        market_data = {"prices": prices or [], "implied_volatility": iv, "options_chain": chain}
        if not prices:
            print(f"  ⚠ No data")
            continue
        result = await agent.analyze(symbol, market_data)
        print(f"  {'Price':15s} ${prices[-1]:<8.2f} ({len(prices)}d)")
        print(f"  {'Sentiment':15s} {result.sentiment.upper():>7}  ({result.confidence:.0%})")
        print(f"  {'Strategy':15s} {result.strategy or 'HOLD'}")
        if result.reasoning and not result.reasoning.startswith("No LLM"):
            print(f"\n  🧠 Analysis:")
            for line in result.reasoning.strip().split("\n"):
                print(f"     {line}")

    print(f"\n✅ Done — {len(agent.history)} analyses\n")


async def run_with_futu(symbols: list[str], paper: bool, llm_client, model: str, provider: str = ""):
    from agent.core import OptionsAgent
    from market.futu_data import FutuDataProvider
    from market.futu_broker import FutuBroker

    f_data = FutuDataProvider()
    broker = FutuBroker(pwd_unlock=os.getenv("FUTU_TRADE_PWD", ""))
    agent = OptionsAgent(llm_client=llm_client, model=model, llm_provider=provider)

    tag = _LLM_TAGS.get(provider, "") if llm_client else ""
    print("\n" + "=" * 65)
    print(f"  🤖 US OPTIONS AGENT — Futu{tag}")
    print("=" * 65)

    print("\n🔌 Connecting...")
    if not await f_data.connect():
        print("  ❌ Cannot connect to Futu OpenAPI (port 11111)")
        return
    if paper:
        await broker.connect_trade(trd_env=1)

    for symbol in symbols:
        us_code = f"US.{symbol}" if not symbol.startswith("US.") else symbol
        print(f"\n📊 {us_code}...")
        quote = await f_data.get_realtime_quote(us_code)
        if quote.get("last_price"):
            print(f"  {'Price':15s} ${quote['last_price']:<8.2f} ({quote.get('change_pct', 0):+.2f}%)")
        prices = await f_data.get_prices(us_code)
        chain = await f_data.get_options_chain(us_code)
        print(f"  {'Data':15s} {len(prices)}d K-line")

        market_data = {"prices": prices or [], "implied_volatility": None, "options_chain": chain, "realtime_quote": quote}
        result = await agent.analyze(symbol, market_data)
        print(f"  {'Sentiment':15s} {result.sentiment.upper():>7} ({result.confidence:.0%})")
        print(f"  {'Strategy':15s} {result.strategy or 'HOLD'}")
        if result.reasoning and not result.reasoning.startswith("No LLM"):
            print(f"\n  🧠 Analysis:")
            for line in result.reasoning.strip().split("\n"):
                print(f"     {line}")

    f_data.close()
    broker.close()
    print("\n✅ Done\n")


async def main():
    args = parse_args()
    symbols = [s.strip() for s in args.symbol.split(",")]
    llm_client = None
    model = ""

    if args.llm:
        llm_client, model = build_llm_client(args.provider)

    prov = args.provider if args.llm else ""
    if args.futu:
        await run_with_futu(symbols, args.paper, llm_client, model, prov)
    else:
        await run_with_yfinance(symbols, llm_client, model, prov)


if __name__ == "__main__":
    asyncio.run(main())
