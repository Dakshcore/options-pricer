# Options Pricer

European option pricing in Python, built step by step and tested against theory.

## Progress

- [x] Black-Scholes-Merton prices and Greeks (delta, gamma, vega, theta, rho), with a dividend yield
- [ ] Binomial tree, European and American
- [ ] Monte Carlo pricer
- [ ] Implied volatility and a volatility smile from real option prices

## Example

```python
from pricer import black_scholes as bs

bs.price(S=42, K=40, T=0.5, r=0.10, sigma=0.20, kind="call")   # 4.7594
bs.delta(S=42, K=40, T=0.5, r=0.10, sigma=0.20, kind="call")   # 0.7791
```

## Tests

```bash
python -m unittest -v
```

The tests check the price against a worked example in Hull's *Options, Futures and Other Derivatives*, check put-call parity and no-arbitrage bounds, and check that every Greek matches a bump-and-reprice finite difference.
