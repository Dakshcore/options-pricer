import unittest
from math import exp, isclose

from pricer import binomial as tree
from pricer import black_scholes as bs

CASES = [  # (S, K, T, r, sigma, q)
    (42, 40, 0.5, 0.10, 0.20, 0.0),
    (100, 110, 1.0, 0.05, 0.30, 0.02),
    (24000, 24500, 30 / 365, 0.065, 0.14, 0.012),
]
HULL_PUT = dict(S=50, K=50, T=5 / 12, r=0.10, sigma=0.40, kind="put")   # Hull's American put example


class TextbookValues(unittest.TestCase):
    def test_hull_american_put(self):
        # Hull: 4.49 with 5 steps, converging to about 4.283 with 500
        self.assertAlmostEqual(tree.price(**HULL_PUT, steps=5, american=True), 4.49, places=2)
        self.assertAlmostEqual(tree.price(**HULL_PUT, steps=500, american=True), 4.283, places=3)


class EuropeanOptions(unittest.TestCase):
    def test_converges_to_black_scholes(self):
        for S, K, T, r, sigma, q in CASES:
            for kind in ("call", "put"):
                exact = bs.price(S, K, T, r, sigma, q, kind)
                approx = tree.price(S, K, T, r, sigma, q, kind, steps=1000)
                self.assertTrue(isclose(approx, exact, rel_tol=1e-3), f"{kind}, S={S}: tree {approx} vs BS {exact}")

    def test_put_call_parity_holds_exactly(self):
        for S, K, T, r, sigma, q in CASES:
            call = tree.price(S, K, T, r, sigma, q, "call", steps=200)
            put = tree.price(S, K, T, r, sigma, q, "put", steps=200)
            self.assertTrue(isclose(call - put, S * exp(-q * T) - K * exp(-r * T), rel_tol=1e-9))


class AmericanOptions(unittest.TestCase):
    def test_call_without_dividends_is_never_exercised_early(self):
        for S, K, T, r, sigma in [(42, 40, 0.5, 0.10, 0.20), (100, 110, 1.0, 0.05, 0.30)]:
            european = tree.price(S, K, T, r, sigma, kind="call", steps=300)
            american = tree.price(S, K, T, r, sigma, kind="call", steps=300, american=True)
            self.assertAlmostEqual(american, european, places=10)

    def test_dividends_can_make_early_exercise_of_a_call_worthwhile(self):
        args = dict(S=100, K=90, T=1.0, r=0.05, sigma=0.20, q=0.10, kind="call", steps=300)
        self.assertGreater(tree.price(**args, american=True), tree.price(**args) + 0.01)

    def test_put_carries_an_early_exercise_premium(self):
        american = tree.price(**HULL_PUT, steps=500, american=True)
        european = bs.price(**HULL_PUT)
        self.assertGreater(american, european + 0.1)   # about 4.283 vs 4.076

    def test_deep_in_the_money_put_is_exercised_immediately(self):
        self.assertAlmostEqual(tree.price(1, 100, 1.0, 0.05, 0.20, kind="put", steps=100, american=True), 99.0,
                               places=10)


class InputChecks(unittest.TestCase):
    def test_rejects_bad_inputs(self):
        with self.assertRaises(ValueError):
            tree.price(100, 100, 1, 0.05, 0.2, kind="straddle")
        with self.assertRaises(ValueError):
            tree.price(100, 100, 1, 0.05, 0.2, steps=0)
        with self.assertRaises(ValueError):
            tree.price(100, 100, 1, 0.10, 0.01, steps=1)   # the up-move probability would exceed 1


if __name__ == "__main__":
    unittest.main()
