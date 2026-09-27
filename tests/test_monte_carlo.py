import unittest
from math import exp, log, sqrt

import numpy as np
from scipy.stats import norm

from pricer import black_scholes as bs
from pricer import monte_carlo as mc

CASES = [  # (S, K, T, r, sigma, q)
    (42, 40, 0.5, 0.10, 0.20, 0.0),
    (100, 110, 1.0, 0.05, 0.30, 0.02),
    (24000, 24500, 30 / 365, 0.065, 0.14, 0.012),
]
HULL = dict(S=42, K=40, T=0.5, r=0.10, sigma=0.20)


def geometric_asian_call(S, K, T, r, sigma, q, n):
    """Exact price of a call on the geometric average of n equally spaced prices: that average is lognormal."""
    dt = T / n
    mean = log(S) + (r - q - sigma**2 / 2) * dt * (n + 1) / 2
    var = sigma**2 * dt * (n + 1) * (2 * n + 1) / (6 * n)
    d1 = (mean - log(K) + var) / sqrt(var)
    return exp(-r * T) * (exp(mean + var / 2) * norm.cdf(d1) - K * norm.cdf(d1 - sqrt(var)))


class EuropeanOptions(unittest.TestCase):
    def test_matches_black_scholes_within_four_standard_errors(self):
        for S, K, T, r, sigma, q in CASES:
            for kind in ("call", "put"):
                exact = bs.price(S, K, T, r, sigma, q, kind)
                low, high = mc.european(S, K, T, r, sigma, q, kind, paths=200_000, seed=1).interval(z=4)
                self.assertTrue(low <= exact <= high, f"{kind}, S={S}: BS {exact} outside ({low}, {high})")

    def test_standard_error_halves_with_four_times_the_paths(self):
        small = mc.european(**HULL, paths=100_000, seed=2).std_error
        large = mc.european(**HULL, paths=400_000, seed=3).std_error
        self.assertAlmostEqual(large / small, 0.5, delta=0.05)

    def test_antithetic_variates_reduce_the_error(self):
        plain = mc.european(**HULL, paths=100_000, antithetic=False, seed=4).std_error
        paired = mc.european(**HULL, paths=100_000, antithetic=True, seed=4).std_error
        self.assertLess(paired, plain)

    def test_same_seed_gives_the_same_result(self):
        self.assertEqual(mc.european(**HULL, seed=5), mc.european(**HULL, seed=5))


class AsianOptions(unittest.TestCase):
    def test_simulated_paths_match_the_exact_geometric_average_price(self):
        S, K, T, r, sigma, q, n = 100, 100, 1.0, 0.05, 0.30, 0.02, 12
        prices = mc.simulate_paths(S, T, r, sigma, q, steps=n, paths=200_000, antithetic=False, seed=6)
        payoffs = exp(-r * T) * np.maximum(np.exp(np.log(prices).mean(axis=1)) - K, 0.0)
        error = payoffs.std(ddof=1) / sqrt(len(payoffs))
        self.assertLess(abs(payoffs.mean() - geometric_asian_call(S, K, T, r, sigma, q, n)), 4 * error)

    def test_arithmetic_average_call_is_worth_more_than_geometric(self):
        # On every path the arithmetic average is at least the geometric average
        S, K, T, r, sigma, q, n = 100, 100, 1.0, 0.05, 0.30, 0.02, 12
        arithmetic = mc.asian(S, K, T, r, sigma, q, steps=n, paths=200_000, seed=7)
        self.assertGreater(arithmetic.interval(z=4)[0], geometric_asian_call(S, K, T, r, sigma, q, n))

    def test_one_averaging_date_is_a_european_option(self):
        self.assertEqual(mc.asian(**HULL, steps=1, paths=50_000, seed=8), mc.european(**HULL, paths=50_000, seed=8))

    def test_averaging_makes_the_option_cheaper(self):
        self.assertLess(mc.asian(**HULL, steps=6, seed=9).price, mc.european(**HULL, seed=9).price)


class InputChecks(unittest.TestCase):
    def test_rejects_bad_inputs(self):
        with self.assertRaises(ValueError):
            mc.european(100, 100, 1, 0.05, 0.2, kind="straddle")
        with self.assertRaises(ValueError):
            mc.european(100, 100, 1, 0.05, 0.2, paths=99_999)   # antithetic pairs need an even number of paths
        with self.assertRaises(ValueError):
            mc.asian(100, 100, 1, 0.05, -0.2)


if __name__ == "__main__":
    unittest.main()
