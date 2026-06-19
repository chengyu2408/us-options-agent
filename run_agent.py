"""
US Options Agent — entry point.

Usage:
  python run_agent.py                      # yfinance (no broker connection needed)
  python run_agent.py --futu               # Futu data + paper trading
  python run_agent.py --symbol SPY,QQQ     # custom watchlist

Environment:
  Export your API keys in .env or env vars when needed.
"""

from __future__ import annotations

import argparse
import asyncio
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
    p.add_argument("--no-llm", action="store_true", help="Skip LLM, signal-only")
    return p.parse_args()


async def run_with_yfinance(symbols: list[str], use_llm: bool):
    """Run agent using yfinance data (no broker connection needed)."""
    from agent.core import OptionsAgent
    from market.data import MarketDataProvider

    provider = MarketDataProvider()
    agent = OptionsAgent() if not use_llm else OptionsAgent(
        model=os.getenv("LLM_MODEL", "gpt-4o"),
    )

    print("\n" + "=" * 60)
    print("  🤖 US OPTIONS AGENT — yfinance mode")
    print("=" * 60)

    for symbol in symbols:
        print(f"\n📊 Analyzing {symbol}...")

        # Fetch data
        prices = await provider.get_prices(symbol)
        iv = await provider.get_implied_volatility(symbol)

        if not prices:
            print(f"  ⚠ No price data for {symbol}")
            continue

        # Run agent
        result = await agent.analyze(symbol, {
            "prices": prices,
            "implied_volatility": iv,
        })

        # Print analysis
        print(f"  {'Price samples':20s} {prices[-1]:>8.2f} (last) / {len(prices)} days")
        print(f"  {'Sentiment':20s} {result.sentiment.upper():>8}  ({result.confidence:.0%} confidence)")
        print(f"  {'Recommended':20s} {result.strategy or 'HOLD / no signal'}")
        print(f"  {'Current IV':20s} {f'{iv:.1%}' if iv else 'N/A'}")

        # Print key indicators
        ind = result.signals.get("indicators", {})
        if ind.get("trend"):
            print(f"  {'Trend':20s} {ind['trend']} (SMA20={ind.get('sma_20', 0):.1f}, SMA50={ind.get('sma_50', 0):.1f})")
        if ind.get("iv_signal"):
            print(f"  {'IV Signal':20s} {ind['iv_signal']}")

        # Brief reasoning
        if result.reasoning and not result.reasoning.startswith("No LLM"):
            print(f"  {'Reasoning':20s} {result.reasoning[:150]}...")

        # Try options chain
        chain = await provider.get_options_chain(symbol)
        if chain and chain.get("calls"):
            print(f"\n  📋 Near-term options chain ({chain.get('expiration', '?')}):")
            for c in chain["calls"][:3]:
                strike = c.get("strike", 0)
                price = c.get("lastPrice", c.get("bid", "?"))
                print(f"    CALL  ${strike:<8} ${price:<8}")
            for p in chain["puts"][:3]:
                strike = p.get("strike", 0)
                price = p.get("lastPrice", p.get("bid", "?"))
                print(f"    PUT   ${strike:<8} ${price:<8}")

    print("\n" + "=" * 60)
    print(f"  ✅ Done — {len(agent.history)} analyses completed")
    print("=" * 60 + "\n")


async def run_with_futu(symbols: list[str], paper: bool):
    """Run agent using Futu OpenAPI data + trading."""
    from agent.core import OptionsAgent
    from market.futu_data import FutuDataProvider
    from market.futu_broker import FutuBroker

    provider = FutuDataProvider()
    broker = FutuBroker(
        pwd_unlock=os.getenv("FUTU_TRADE_PWD", ""),
    )

    print("\n" + "=" * 60)
    print("  🤖 US OPTIONS AGENT — Futu OpenAPI mode")
    print("=" * 60)

    # Connect
    print("\n🔌 Connecting to Futu OpenAPI...")
    if not await provider.connect():
        print("  ❌ Cannot connect to Futu OpenAPI server.")
        print("  Make sure Futu OpenAPI is running on your machine.")
        print("  https://openapi.futunn.com/")
        return

    if paper:
        await broker.connect_trade(trd_env=1)
    else:
        await broker.connect_trade(trd_env=0)

    # Check account
    acct = await broker.get_account_summary()
    print(f"  Account: {acct}")

    # Build agent (no LLM client by default)
    agent = OptionsAgent()

    for symbol in symbols:
        us_code = f"US.{symbol}" if not symbol.startswith("US.") else symbol
        print(f"\n📊 Analyzing {us_code}...")

        # Futu real-time data
        quote = await provider.get_realtime_quote(us_code)
        if quote.get("last_price"):
            print(f"  {'Last Price':20s} ${quote['last_price']:<8.2f}  ({quote.get('change_pct', 0):.2f}%)")
            print(f"  {'Day Range':20s} ${quote.get('low_price', 0):.2f} - ${quote.get('high_price', 0):.2f}")
            print(f"  {'Volume':20s} {quote.get('volume', 0):,}")

        # K-line prices
        prices = await provider.get_prices(us_code)
        if not prices:
            print(f"  ⚠ No K-line data for {symbol}")
            continue

        # Futures options chain
        chain = await provider.get_options_chain(us_code)
        if chain:
            print(f"\n  📋 Options chain (Futu):")
            for c in chain.get("calls", [])[:3]:
                strike = c.get("strike", "?")
                print(f"    CALL  ${strike}")
            for p in chain.get("puts", [])[:3]:
                strike = p.get("strike", "?")
                print(f"    PUT   ${strike}")

        # Run analysis
        result = await agent.analyze(symbol, {"prices": prices})
        print(f"  {'Sentiment':20s} {result.sentiment.upper():>8}")
        print(f"  {'Recommendation':20s} {result.strategy or 'HOLD'}")

        # Option Greeks for a specific contract (example)
        if chain and chain.get("calls"):
            opt_code = chain["calls"][0].get("code", "")
            if opt_code:
                greeks = await provider.get_option_quote(opt_code)
                if greeks:
                    print(f"\n  📐 Greeks (near-term ATM call):")
                    print(f"    {'Delta':10s} {greeks.get('delta', 'N/A')}")
                    print(f"    {'Gamma':10s} {greeks.get('gamma', 'N/A')}")
                    print(f"    {'Theta':10s} {greeks.get('theta', 'N/A')}")
                    print(f"    {'Vega':10s} {greeks.get('vega', 'N/A')}")
                    print(f"    {'IV':10s} {greeks.get('implied_vol', 'N/A')}")

    # Cleanup
    provider.close()
    broker.close()
    print("\n✅ Done — connections closed.")


async def main():
    args = parse_args()
    symbols = [s.strip() for s in args.symbol.split(",")]
    use_llm = not args.no_llm

    if args.futu:
        await run_with_futu(symbols, args.paper)
    else:
        await run_with_yfinance(symbols, use_llm)


if __name__ == "__main__":
    asyncio.run(main())
