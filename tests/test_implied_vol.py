import unittest

import numpy as np

from pricer import black_scholes as bs
from pricer.implied_vol import implied_forward, implied_vol

CASES = [  # (S, K, T, r, q)
    (42, 40, 0.5, 0.10, 0.0),
    (100, 110, 1.0, 0.05, 0.02),
    (24000, 24500, 30 / 365, 0.065, 0.012),
]


class ImpliedVol(unittest.TestCase):
    def test_recovers_the_volatility_used_to_price_the_option(self):
        for S, K, T, r, q in CASES:
            for sigma in (0.1, 0.25, 0.6):
                for kind in ("call", "put"):
                    price = bs.price(S, K, T, r, sigma, q, kind)
                    self.assertAlmostEqual(implied_vol(price, S, K, T, r, q, kind), sigma, places=6)

    def test_works_on_a_forward_with_blacks_formula(self):
        F, T, r = 23258.7, 32 / 365, 0.053   # an index option priced off its forward: S = F, q = r
        for K in (21000, 23300, 25500):
            for kind in ("call", "put"):
                price = bs.price(F, K, T, r, 0.14, q=r, kind=kind)
                self.assertAlmostEqual(implied_vol(price, F, K, T, r, q=r, kind=kind), 0.14, places=6)

    def test_rejects_prices_no_volatility_can_produce(self):
        with self.assertRaises(ValueError):
            implied_vol(3.0, 42, 40, 0.5, 0.10, kind="call")    # below 42 - 40e^(-0.05), the no-arbitrage floor
        with self.assertRaises(ValueError):
            implied_vol(43.0, 42, 40, 0.5, 0.10, kind="call")   # more than the stock itself


class ImpliedForward(unittest.TestCase):
    def test_recovers_the_forward_from_put_call_parity(self):
        F, T, r, sigma = 23258.7, 32 / 365, 0.053, 0.12
        strikes = np.arange(22000, 24600, 100)
        calls = [bs.price(F, k, T, r, sigma, q=r, kind="call") for k in strikes]
        puts = [bs.price(F, k, T, r, sigma, q=r, kind="put") for k in strikes]
        self.assertAlmostEqual(implied_forward(strikes, calls, puts, T, r), F, places=6)


if __name__ == "__main__":
    unittest.main()
