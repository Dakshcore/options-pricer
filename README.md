# Options Pricer

European option pricing in Python, built step by step and tested against theory.

## Progress

- [x] Black-Scholes-Merton prices and Greeks (delta, gamma, vega, theta, rho), with a dividend yield
- [x] Binomial tree (Cox-Ross-Rubinstein), European and American
- [ ] Monte Carlo pricer
- [ ] Implied volatility and a volatility smile from real option prices

## Example

```python
from pricer import black_scholes as bs

bs.price(S=42, K=40, T=0.5, r=0.10, sigma=0.20, kind="call")   # 4.7594
bs.delta(S=42, K=40, T=0.5, r=0.10, sigma=0.20, kind="call")   # 0.7791

from pricer import binomial as tree

# American put: worth more than the European put (4.076) because early exercise can pay
tree.price(S=50, K=50, T=5/12, r=0.10, sigma=0.40, kind="put", steps=500, american=True)   # 4.283
```

## Tests

```bash
python -m unittest -v
```

The tests check:

- **Black-Scholes:** the price against a worked example in Hull's *Options, Futures and Other Derivatives*, put-call parity and no-arbitrage bounds, and every Greek against a bump-and-reprice finite difference.
- **Binomial tree:** Hull's American put example (4.49 with 5 steps, 4.283 with 500), convergence to Black-Scholes for European options, exact put-call parity, and the early-exercise rules: never early for a call without dividends, sometimes early for a call with dividends, and immediately for a deep in-the-money put.
