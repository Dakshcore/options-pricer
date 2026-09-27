import unittest
from math import exp, isclose

from pricer import black_scholes as bs

CASES = [  # (S, K, T, r, sigma, q)
    (42, 40, 0.5, 0.10, 0.20, 0.0),                  # Hull's worked example
    (100, 110, 1.0, 0.05, 0.30, 0.02),               # out-of-the-money call, with dividends
    (24000, 24500, 30 / 365, 0.065, 0.14, 0.012),    # NIFTY-like index option, one month out
]


def bumped(args, name, h):
    """Central finite difference of the price with respect to one input."""
    up, down = dict(args), dict(args)
    up[name] += h
    down[name] -= h
    return (bs.price(**up) - bs.price(**down)) / (2 * h)


def second_difference(args, h):
    """Second finite difference of the price with respect to S."""
    up, down = dict(args), dict(args)
    up["S"] += h
    down["S"] -= h
    return (bs.price(**up) - 2 * bs.price(**args) + bs.price(**down)) / h**2


class TextbookValues(unittest.TestCase):
    def test_hull_example(self):
        # Hull: S=42, K=40, T=0.5, r=10%, sigma=20% gives a call of 4.76 and a put of 0.81
        self.assertAlmostEqual(bs.price(42, 40, 0.5, 0.10, 0.20, kind="call"), 4.76, places=2)
        self.assertAlmostEqual(bs.price(42, 40, 0.5, 0.10, 0.20, kind="put"), 0.81, places=2)


class NoArbitrage(unittest.TestCase):
    def test_put_call_parity(self):
        for S, K, T, r, sigma, q in CASES:
            call = bs.price(S, K, T, r, sigma, q, "call")
            put = bs.price(S, K, T, r, sigma, q, "put")
            self.assertAlmostEqual(call - put, S * exp(-q * T) - K * exp(-r * T), places=8)

    def test_call_bounds(self):
        for S, K, T, r, sigma, q in CASES:
            call = bs.price(S, K, T, r, sigma, q, "call")
            self.assertGreaterEqual(call, max(S * exp(-q * T) - K * exp(-r * T), 0))
            self.assertLessEqual(call, S * exp(-q * T))


class GreeksMatchFiniteDifferences(unittest.TestCase):
    """Each Greek should match bumping one input and repricing."""

    def assert_matches(self, greek, numerical):
        for S, K, T, r, sigma, q in CASES:
            for kind in ("call", "put"):
                args = dict(S=S, K=K, T=T, r=r, sigma=sigma, q=q, kind=kind)
                expected, actual = numerical(args), greek(**args)
                self.assertTrue(isclose(actual, expected, rel_tol=1e-5, abs_tol=1e-12),
                                f"{greek.__name__} ({kind}): {actual} vs finite difference {expected}")

    def test_delta(self):
        self.assert_matches(bs.delta, lambda a: bumped(a, "S", 1e-4 * a["S"]))

    def test_gamma(self):
        self.assert_matches(bs.gamma, lambda a: second_difference(a, 1e-4 * a["S"]))

    def test_vega(self):
        self.assert_matches(bs.vega, lambda a: bumped(a, "sigma", 1e-5))

    def test_theta(self):
        # theta is the change as time passes, so it is minus the derivative with respect to T
        self.assert_matches(bs.theta, lambda a: -bumped(a, "T", 1e-6))

    def test_rho(self):
        self.assert_matches(bs.rho, lambda a: bumped(a, "r", 1e-5))


class InputChecks(unittest.TestCase):
    def test_rejects_bad_inputs(self):
        with self.assertRaises(ValueError):
            bs.price(100, 100, 0, 0.05, 0.2)                    # already expired
        with self.assertRaises(ValueError):
            bs.price(100, 100, 1, 0.05, -0.2)                   # negative volatility
        with self.assertRaises(ValueError):
            bs.price(100, 100, 1, 0.05, 0.2, kind="straddle")   # unknown option type


if __name__ == "__main__":
    unittest.main()
