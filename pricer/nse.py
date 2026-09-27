"""NSE's daily F&O bhavcopy: the official end-of-day prices for every futures and options contract."""
import urllib.request
from pathlib import Path

import pandas as pd

URL = "https://nsearchives.nseindia.com/content/fo/BhavCopy_NSE_FO_0_0_0_{:%Y%m%d}_F_0000.csv.zip"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)", "Referer": "https://www.nseindia.com/"}


def download_bhavcopy(day, folder="data"):
    """Download the bhavcopy for trading day `day` into `folder`, unless it is already there, and return its path."""
    path = Path(folder) / f"fo_bhavcopy_{day:%Y%m%d}.csv.zip"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(urllib.request.Request(URL.format(day), headers=HEADERS), timeout=60) as response:
            path.write_bytes(response.read())
    return path


def load_bhavcopy(path):
    """Every contract in a bhavcopy file (.csv or .csv.zip)."""
    return pd.read_csv(path, parse_dates=["TradDt", "XpryDt"])


def index_options(bhavcopy, symbol, expiry):
    """Options on an index for one expiry that traded that day, sorted by strike.

    Columns: strike, kind ('call' or 'put'), price (NSE's closing price), volume (contracts), open_interest.
    Contracts that did not trade are dropped: their closing price is just the previous day's, carried forward.
    """
    rows = bhavcopy[(bhavcopy["FinInstrmTp"] == "IDO") & (bhavcopy["TckrSymb"] == symbol)
                    & (bhavcopy["XpryDt"] == pd.Timestamp(expiry)) & (bhavcopy["TtlTradgVol"] > 0)]
    options = pd.DataFrame({"strike": rows["StrkPric"], "kind": rows["OptnTp"].map({"CE": "call", "PE": "put"}),
                            "price": rows["ClsPric"], "volume": rows["TtlTradgVol"],
                            "open_interest": rows["OpnIntrst"]})
    return options.sort_values(["strike", "kind"]).reset_index(drop=True)


def future_price(bhavcopy, symbol, expiry):
    """Closing price of the index future for `expiry`, or None if that expiry has no future."""
    rows = bhavcopy[(bhavcopy["FinInstrmTp"] == "IDF") & (bhavcopy["TckrSymb"] == symbol)
                    & (bhavcopy["XpryDt"] == pd.Timestamp(expiry))]
    return float(rows["ClsPric"].iloc[0]) if len(rows) else None


def index_level(bhavcopy, symbol):
    """Closing level of the index itself, as reported alongside its options."""
    rows = bhavcopy[(bhavcopy["FinInstrmTp"] == "IDO") & (bhavcopy["TckrSymb"] == symbol)]
    return float(rows["UndrlygPric"].iloc[0])
