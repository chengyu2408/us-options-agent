"""
Basic usage example — analyze a stock and get an options recommendation.
"""

import asyncio
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


async def main():
    from agent.core import OptionsAgent
    from market.data import MarketDataProvider

    agent = OptionsAgent()  # no LLM — signal-only mode
    data_provider = MarketDataProvider()

    symbol = "SPY"
    print(f"\n🔍 Analyzing {symbol}...\n")

    # Get market data
    prices = await data_provider.get_prices(symbol)
    iv = await data_provider.get_implied_volatility(symbol)

    # Run agent analysis
    result = await agent.analyze(symbol, {
        "prices": prices,
        "implied_volatility": iv,
    })

    print(f"  Sentiment:   {result.sentiment} ({result.confidence:.0%} confidence)")
    print(f"  Strategy:    {result.strategy or 'none — hold'}")
    print(f"  Indicators:  {result.signals.get('indicators', {})}")
    print(f"  Reasoning:   {result.reasoning[:200]}...\n")

    # Show trade history
    print(f"  Total analyses so far: {len(agent.history)}")

    # Try options chain
    print(f"\n📋 Options chain (near-term):")
    chain = await data_provider.get_options_chain(symbol)
    if chain.get("calls"):
        for c in chain["calls"][:3]:
            print(f"  CALL  {c['strike']:>8}  {c['lastBid'] if 'lastBid' in c else c.get('lastPrice', '?'):>8}  expiry={chain.get('expiration', '?')}")
    if chain.get("puts"):
        for p in chain["puts"][:3]:
            print(f"  PUT   {p['strike']:>8}  {p['lastBid'] if 'lastBid' in p else p.get('lastPrice', '?'):>8}  expiry={chain.get('expiration', '?')}")


if __name__ == "__main__":
    asyncio.run(main())
