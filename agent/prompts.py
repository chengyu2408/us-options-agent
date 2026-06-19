"""System prompts and templates for the agent's LLM interactions."""

ANALYST_SYSTEM_PROMPT = """You are a professional US options trader and analyst.
You reason step-by-step about market conditions, implied volatility, Greeks,
and risk before recommending any trade. You are conservative and always
acknowledge uncertainty."""

STRATEGY_DESCRIPTIONS = {
    "long_call": "Buy a call option — bullish, defined risk (premium), unlimited upside.",
    "long_put": "Buy a put option — bearish, defined risk (premium), unlimited downside protection.",
    "covered_call": "Own 100 shares, sell 1 call — neutral/bullish, generates income, caps upside.",
    "cash_secured_put": "Sell a put with cash collateral — neutral/bullish, collects premium, may acquire shares.",
    "iron_condor": "Sell OTM put spread + OTM call spread — neutral, defined risk, theta positive.",
    "vertical_spread": "Buy and sell same-expiry calls/puts — directional, defined risk/reward.",
}
