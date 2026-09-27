"""Cox-Ross-Rubinstein binomial tree for European and American options.

Uses the same inputs as black_scholes.py, plus:
    steps     number of time steps in the tree
    american  True to allow exercise at any node, not just at expiry
"""
from math import exp, sqrt

import numpy as np


def price(S, K, T, r, sigma, q=0.0, kind="call", steps=500, american=False):
    """Option price from a recombining binomial tree."""
    if kind not in ("call", "put"):
        raise ValueError(f"kind must be 'call' or 'put', not {kind!r}")
    if min(S, K, T, sigma) <= 0 or steps < 1:
        raise ValueError("S, K, T and sigma must all be positive, and steps at least 1")

    dt = T / steps
    u = exp(sigma * sqrt(dt))                  # size of an up move
    d = 1 / u                                  # a down move undoes an up move, so the tree recombines
    p = (exp((r - q) * dt) - d) / (u - d)      # risk-neutral probability of an up move
    if not 0 < p < 1:
        raise ValueError("too few steps for these inputs: the up-move probability falls outside 0-1")
    discount = exp(-r * dt)
    sign = 1 if kind == "call" else -1         # payoff is sign * (stock - K), floored at 0

    # At expiry, node j has had j up moves and (steps - j) down moves: stock = S * u**(j - (steps - j))
    j = np.arange(steps + 1)
    value = np.maximum(sign * (S * u ** (2 * j - steps) - K), 0.0)

    # Work backwards one step at a time: each node is the discounted risk-neutral average of its two children
    for n in range(steps - 1, -1, -1):
        value = discount * (p * value[1:] + (1 - p) * value[:-1])
        if american:
            stock = S * u ** (2 * j[:n + 1] - n)
            value = np.maximum(value, sign * (stock - K))   # exercise now if that is worth more than holding
    return float(value[0])
