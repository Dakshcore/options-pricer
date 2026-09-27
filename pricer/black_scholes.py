"""Black-Scholes-Merton prices and Greeks for European options.

Inputs use the notation of Hull, Options, Futures and Other Derivatives:
    S      spot price of the underlying
    K      strike price
    T      time to expiry, in years
    r      risk-free rate, continuously compounded
    sigma  annualised volatility
    q      continuous dividend yield (0 for a stock that pays no dividends)
    kind   "call" or "put"
"""
from math import exp, log, sqrt

from scipy.stats import norm


def _d1_d2(S, K, T, r, sigma, q):
    d1 = (log(S / K) + (r - q + sigma**2 / 2) * T) / (sigma * sqrt(T))
    return d1, d1 - sigma * sqrt(T)


def _validate(S, K, T, sigma, kind):
    if kind not in ("call", "put"):
        raise ValueError(f"kind must be 'call' or 'put', not {kind!r}")
    if min(S, K, T, sigma) <= 0:
        raise ValueError("S, K, T and sigma must all be positive")


def price(S, K, T, r, sigma, q=0.0, kind="call"):
    """Option price."""
    _validate(S, K, T, sigma, kind)
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    if kind == "call":
        return S * exp(-q * T) * norm.cdf(d1) - K * exp(-r * T) * norm.cdf(d2)
    return K * exp(-r * T) * norm.cdf(-d2) - S * exp(-q * T) * norm.cdf(-d1)


def delta(S, K, T, r, sigma, q=0.0, kind="call"):
    """Change in price per 1 unit change in S."""
    _validate(S, K, T, sigma, kind)
    d1, _ = _d1_d2(S, K, T, r, sigma, q)
    if kind == "call":
        return exp(-q * T) * norm.cdf(d1)
    return exp(-q * T) * (norm.cdf(d1) - 1)


def gamma(S, K, T, r, sigma, q=0.0, kind="call"):
    """Change in delta per 1 unit change in S (the same for calls and puts)."""
    _validate(S, K, T, sigma, kind)
    d1, _ = _d1_d2(S, K, T, r, sigma, q)
    return exp(-q * T) * norm.pdf(d1) / (S * sigma * sqrt(T))


def vega(S, K, T, r, sigma, q=0.0, kind="call"):
    """Change in price per 1.00 change in sigma (the same for calls and puts); divide by 100 for per vol point."""
    _validate(S, K, T, sigma, kind)
    d1, _ = _d1_d2(S, K, T, r, sigma, q)
    return S * exp(-q * T) * norm.pdf(d1) * sqrt(T)


def theta(S, K, T, r, sigma, q=0.0, kind="call"):
    """Change in price per year of time passing; divide by 365 for per calendar day."""
    _validate(S, K, T, sigma, kind)
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    decay = -S * exp(-q * T) * norm.pdf(d1) * sigma / (2 * sqrt(T))
    if kind == "call":
        return decay - r * K * exp(-r * T) * norm.cdf(d2) + q * S * exp(-q * T) * norm.cdf(d1)
    return decay + r * K * exp(-r * T) * norm.cdf(-d2) - q * S * exp(-q * T) * norm.cdf(-d1)


def rho(S, K, T, r, sigma, q=0.0, kind="call"):
    """Change in price per 1.00 change in r; divide by 100 for per percentage point."""
    _validate(S, K, T, sigma, kind)
    _, d2 = _d1_d2(S, K, T, r, sigma, q)
    if kind == "call":
        return K * T * exp(-r * T) * norm.cdf(d2)
    return -K * T * exp(-r * T) * norm.cdf(-d2)
