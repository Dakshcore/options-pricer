"""NIFTY implied volatility smile from NSE's end-of-day prices.

Downloads the day's F&O bhavcopy (cached in data/), backs out the forward price from put-call parity,
solves for the implied volatility of every out-of-the-money option that traded, and saves a chart.

    python nifty_smile.py
"""
from datetime import date

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

from pricer import nse
from pricer.implied_vol import implied_forward, implied_vol

SYMBOL = "NIFTY"
TRADE_DATE = date(2026, 9, 25)
EXPIRY = date(2026, 10, 27)   # October monthly expiry: 32 days out, with 184 options traded
RATE = 0.053                  # about the 91-day T-bill yield at September 2026 auctions (5.26-5.39%)
MIN_PRICE = 1.0               # below Rs 1, the Rs 0.05 tick makes the implied volatility unreliable
MONEYNESS = (0.85, 1.15)      # strikes from 15% below to 15% above the forward
CHART = "nifty_smile.png"

INK, SECONDARY, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
COLORS = {"put": "#2a78d6", "call": "#eb6834"}


def build_smile(bhavcopy):
    """Forward price and the implied volatility of each out-of-the-money option that traded."""
    options = nse.index_options(bhavcopy, SYMBOL, EXPIRY)
    T = (EXPIRY - TRADE_DATE).days / 365
    both = options.pivot(index="strike", columns="kind", values="price").dropna()
    forward = implied_forward(both.index, both["call"], both["put"], T, RATE)

    # Puts below the forward and calls above it: the out-of-the-money, more actively traded side of each strike
    otm = options[((options["kind"] == "put") & (options["strike"] < forward))
                  | ((options["kind"] == "call") & (options["strike"] >= forward))]
    otm = otm[(otm["price"] >= MIN_PRICE)
              & otm["strike"].between(MONEYNESS[0] * forward, MONEYNESS[1] * forward)].copy()

    def solve(row):
        try:   # an index option on the forward: S = forward and q = r (Black's formula)
            return implied_vol(row.price, forward, row.strike, T, RATE, q=RATE, kind=row.kind)
        except ValueError:
            return np.nan

    otm["iv"] = [solve(row) for row in otm.itertuples()]
    return forward, T, otm.dropna(subset=["iv"]), int(otm["iv"].isna().sum())


def plot(smile, forward, atm_vol):
    plt.rcParams["font.family"] = ["Segoe UI", "DejaVu Sans"]
    fig, ax = plt.subplots(figsize=(9, 5.4), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    for kind, label in (("put", "Puts (strikes below the forward)"), ("call", "Calls (strikes above the forward)")):
        part = smile[smile["kind"] == kind]
        ax.plot(part["strike"], part["iv"] * 100, color=COLORS[kind], lw=1.5, marker="o", ms=6.5,
                mec=SURFACE, mew=1.5, solid_joinstyle="round", solid_capstyle="round", label=label, zorder=3)

    ax.axvline(forward, color=AXIS, lw=0.8, zorder=1)
    ax.annotate(f"Forward {forward:,.0f}\nat-the-money vol {atm_vol:.1%}", xy=(forward, atm_vol * 100),
                xytext=(12, 28), textcoords="offset points", color=SECONDARY, fontsize=9,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    low = smile.iloc[0]
    ax.annotate(f"{low['iv']:.1%} at {low['strike']:,.0f}", xy=(low["strike"], low["iv"] * 100),
                xytext=(10, 0), textcoords="offset points", va="center", color=SECONDARY, fontsize=9)

    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelcolor=SECONDARY, labelsize=9, length=0)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.0f}%"))
    ax.set_xlabel("Strike", color=SECONDARY, fontsize=9.5)
    ax.set_ylabel("Implied volatility (annualised)", color=SECONDARY, fontsize=9.5)
    legend = ax.legend(loc="upper right", frameon=False, fontsize=9)
    for text in legend.get_texts():
        text.set_color(SECONDARY)

    fig.text(0.07, 0.955, "NIFTY implied volatility smile", color=INK, fontsize=14, fontweight="bold")
    fig.text(0.07, 0.915, f"Options expiring {EXPIRY:%d %b %Y}, priced at the close on {TRADE_DATE:%d %b %Y} "
                          f"({(EXPIRY - TRADE_DATE).days} days to expiry)", color=SECONDARY, fontsize=9.5)
    fig.text(0.07, 0.025, f"Source: NSE F&O bhavcopy. Forward from put-call parity; risk-free rate {RATE:.1%}. "
                          f"Out-of-the-money options that traded, priced at Rs {MIN_PRICE:.0f} or more.",
             color=MUTED, fontsize=8)
    fig.subplots_adjust(left=0.07, right=0.97, top=0.86, bottom=0.14)
    fig.savefig(CHART, facecolor=SURFACE)


def main():
    bhavcopy = nse.load_bhavcopy(nse.download_bhavcopy(TRADE_DATE))
    forward, T, smile, unsolved = build_smile(bhavcopy)
    smile = smile.sort_values("strike")
    at = lambda strike: float(np.interp(strike, smile["strike"], smile["iv"]))
    future = nse.future_price(bhavcopy, SYMBOL, EXPIRY)

    print(f"{SYMBOL} on {TRADE_DATE:%d-%b-%Y}: index {nse.index_level(bhavcopy, SYMBOL):,.1f}, "
          f"future {future:,.1f}, forward from put-call parity {forward:,.1f}")
    print(f"Expiry {EXPIRY:%d-%b-%Y} ({T * 365:.0f} days): {len(smile)} options used, {unsolved} with no solution")
    print(f"Implied vol: at the forward {at(forward):.2%} | 5% below {at(0.95 * forward):.2%} | "
          f"10% below {at(0.90 * forward):.2%} | 5% above {at(1.05 * forward):.2%}")
    plot(smile, forward, at(forward))
    print(f"Chart saved to {CHART}")


if __name__ == "__main__":
    main()
