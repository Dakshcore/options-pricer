"""Implied volatility, and the forward price implied by put-call parity.

For options on an index, pass the forward price F as S and set q equal to r. Black-Scholes then
becomes Black's formula for options on a forward, so no dividend yield or funding assumption is needed:
the market's own forward carries both.
"""
from math import exp

import numpy as np
from scipy.optimize import brentq

from pricer import black_scholes as bs

LOW_VOL, HIGH_VOL = 1e-4, 5.0   # search range: 0.01% to 500% a year


def implied_vol(price, S, K, T, r, q=0.0, kind="call"):
    """The volatility at which Black-Scholes gives `price`, found with Brent's method."""
    low = bs.price(S, K, T, r, LOW_VOL, q, kind)
    high = bs.price(S, K, T, r, HIGH_VOL, q, kind)
    if not low < price < high:
        raise ValueError(f"no volatility gives a price of {price}: the model ranges from {low:.4f} to {high:.4f}")
    return brentq(lambda sigma: bs.price(S, K, T, r, sigma, q, kind) - price, LOW_VOL, HIGH_VOL, xtol=1e-10)


def implied_forward(strikes, calls, puts, T, r):
    """Forward price from put-call parity, C - P = e^(-rT) (F - K), at the strike where the call and put are closest."""
    strikes, calls, puts = (np.asarray(a, dtype=float) for a in (strikes, calls, puts))
    i = int(np.argmin(np.abs(calls - puts)))
    return float(strikes[i] + exp(r * T) * (calls[i] - puts[i]))
