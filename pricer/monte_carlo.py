"""Monte Carlo prices under risk-neutral geometric Brownian motion.

European options have an exact price (black_scholes.py), so here they mainly check the method.
Monte Carlo earns its place with options whose payoff depends on the whole price path, like the
arithmetic-average Asian option below, which has no closed-form price.

Uses the same inputs as black_scholes.py, plus:
    paths       number of simulated price paths
    antithetic  pair every random draw Z with -Z to reduce the error (variance reduction)
    seed        fix the random numbers so a result can be reproduced
"""
from math import exp, sqrt
from typing import NamedTuple

import numpy as np


class Estimate(NamedTuple):
    price: float
    std_error: float

    def interval(self, z=1.96):
        """Confidence interval around the price; z=1.96 gives 95%."""
        return self.price - z * self.std_error, self.price + z * self.std_error


def simulate_paths(S, T, r, sigma, q=0.0, steps=1, paths=100_000, antithetic=True, seed=None):
    """Stock prices at the end of each of `steps` equal time steps, one row per path.

    With antithetic=True, the second half of the rows reuse the first half's random draws with the sign flipped.
    """
    if min(S, T, sigma) <= 0 or steps < 1 or paths < 2:
        raise ValueError("S, T and sigma must be positive, steps at least 1 and paths at least 2")
    if antithetic and paths % 2:
        raise ValueError("paths must be even when antithetic=True")
    rng = np.random.default_rng(seed)
    dt = T / steps
    z = rng.standard_normal((paths // 2 if antithetic else paths, steps))
    if antithetic:
        z = np.concatenate([z, -z])
    # Exact for geometric Brownian motion: each step, the log price moves by (r - q - sigma^2/2) dt + sigma sqrt(dt) Z
    log_moves = (r - q - sigma**2 / 2) * dt + sigma * sqrt(dt) * z
    return S * np.exp(np.cumsum(log_moves, axis=1))


def european(S, K, T, r, sigma, q=0.0, kind="call", paths=100_000, antithetic=True, seed=None):
    """European option: the payoff depends only on the price at expiry."""
    _validate(K, kind)
    final = simulate_paths(S, T, r, sigma, q, 1, paths, antithetic, seed)[:, -1]
    return _estimate(_payoff(final, K, kind), exp(-r * T), antithetic)


def asian(S, K, T, r, sigma, q=0.0, kind="call", steps=12, paths=100_000, antithetic=True, seed=None):
    """Asian option on the arithmetic average of the price on `steps` equally spaced dates, the last at expiry."""
    _validate(K, kind)
    average = simulate_paths(S, T, r, sigma, q, steps, paths, antithetic, seed).mean(axis=1)
    return _estimate(_payoff(average, K, kind), exp(-r * T), antithetic)


def _validate(K, kind):
    if kind not in ("call", "put"):
        raise ValueError(f"kind must be 'call' or 'put', not {kind!r}")
    if K <= 0:
        raise ValueError("K must be positive")


def _payoff(price, K, kind):
    return np.maximum(price - K, 0.0) if kind == "call" else np.maximum(K - price, 0.0)


def _estimate(payoffs, discount, antithetic):
    """Discounted average payoff and its standard error."""
    if antithetic:
        # Z and -Z are not independent draws, so each pair counts as one sample
        half = len(payoffs) // 2
        payoffs = (payoffs[:half] + payoffs[half:]) / 2
    return Estimate(float(discount * payoffs.mean()), float(discount * payoffs.std(ddof=1) / sqrt(len(payoffs))))
