# Options Pricer

European option pricing in Python, built step by step and tested against theory.

## Progress

- [x] Black-Scholes-Merton prices and Greeks (delta, gamma, vega, theta, rho), with a dividend yield
- [x] Binomial tree (Cox-Ross-Rubinstein), European and American
- [x] Monte Carlo: European and Asian options, with standard errors and antithetic variates
- [x] Implied volatility, and a NIFTY volatility smile from NSE's end-of-day prices

## NIFTY volatility smile

![NIFTY implied volatility smile](nifty_smile.png)

`python nifty_smile.py` downloads NSE's end-of-day F&O file (the bhavcopy) for 25 Sep 2026, backs out the forward price from put-call parity, and solves for the implied volatility of every out-of-the-money NIFTY option expiring 27 Oct 2026 that traded that day.

- **Forward:** 23,258.7 from put-call parity, within 5 points of the October future (23,263.6).
- **At-the-money volatility:** 11.2%. India VIX closed at 12.13 that day; it sits higher because it averages in the more expensive out-of-the-money options.
- **Skew:** 14.3% for strikes 5% below the forward and 19.2% for 10% below, against 10.6% for 5% above. Buyers pay up for protection against a fall.

## Example

```python
from pricer import black_scholes as bs

bs.price(S=42, K=40, T=0.5, r=0.10, sigma=0.20, kind="call")   # 4.7594
bs.delta(S=42, K=40, T=0.5, r=0.10, sigma=0.20, kind="call")   # 0.7791

from pricer import binomial as tree

# American put: worth more than the European put (4.076) because early exercise can pay
tree.price(S=50, K=50, T=5/12, r=0.10, sigma=0.40, kind="put", steps=500, american=True)   # 4.283

from pricer import monte_carlo as mc

mc.european(S=42, K=40, T=0.5, r=0.10, sigma=0.20, seed=0)         # 4.761, standard error 0.008
mc.asian(S=42, K=40, T=0.5, r=0.10, sigma=0.20, steps=6, seed=0)   # call on the average of 6 monthly prices: about 3.49
```

## Tests

```bash
python -m unittest -v
```

The tests check:

- **Black-Scholes:** the price against a worked example in Hull's *Options, Futures and Other Derivatives*, put-call parity and no-arbitrage bounds, and every Greek against a bump-and-reprice finite difference.
- **Binomial tree:** Hull's American put example (4.49 with 5 steps, 4.283 with 500), convergence to Black-Scholes for European options, exact put-call parity, and the early-exercise rules: never early for a call without dividends, sometimes early for a call with dividends, and immediately for a deep in-the-money put.
- **Monte Carlo:** European prices within 4 standard errors of Black-Scholes, the error halving with 4x the paths, antithetic variates reducing it, simulated paths reproducing the exact price of a geometric-average Asian option, and an arithmetic Asian call worth more than the geometric one but less than the European.
- **Implied volatility:** solving recovers the volatility used to price an option, including on a forward with Black's formula; prices outside the no-arbitrage range are rejected; and put-call parity recovers the forward exactly.
- **NSE data:** the loader keeps only traded contracts for the requested index and expiry, finds the future and the index level, reads the zipped file, and reuses a file already downloaded.
