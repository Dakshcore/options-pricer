import tempfile
import unittest
from datetime import date
from pathlib import Path

import pandas as pd

from pricer import nse

COLUMNS = ["TradDt", "FinInstrmTp", "TckrSymb", "XpryDt", "StrkPric", "OptnTp", "ClsPric", "TtlTradgVol",
           "OpnIntrst", "UndrlygPric"]
ROWS = [  # a few rows in NSE's bhavcopy layout
    ("2026-09-25", "IDO", "NIFTY", "2026-10-27", 23000.0, "PE", 208.00, 65459, 4766580, 23140.5),
    ("2026-09-25", "IDO", "NIFTY", "2026-10-27", 23000.0, "CE", 465.15, 38781, 1888575, 23140.5),
    ("2026-09-25", "IDO", "NIFTY", "2026-10-27", 22500.0, "PE", 99.00, 0, 0, 23140.5),        # did not trade
    ("2026-09-25", "IDO", "NIFTY", "2026-10-06", 23000.0, "CE", 300.00, 5000, 10000, 23140.5),  # another expiry
    ("2026-09-25", "IDO", "BANKNIFTY", "2026-10-27", 52000.0, "CE", 900.00, 700, 5000, 51800.0),  # another index
    ("2026-09-25", "IDF", "NIFTY", "2026-10-27", 0.0, None, 23263.6, 46784, 0, 23140.5),       # the future
]


def sample_bhavcopy():
    frame = pd.DataFrame(ROWS, columns=COLUMNS)
    return frame.assign(TradDt=pd.to_datetime(frame["TradDt"]), XpryDt=pd.to_datetime(frame["XpryDt"]))


class Bhavcopy(unittest.TestCase):
    def test_index_options_keeps_traded_contracts_for_one_index_and_expiry(self):
        options = nse.index_options(sample_bhavcopy(), "NIFTY", date(2026, 10, 27))
        self.assertEqual(list(options["kind"]), ["call", "put"])
        self.assertEqual(list(options["strike"]), [23000.0, 23000.0])
        self.assertEqual(list(options["price"]), [465.15, 208.00])

    def test_future_price(self):
        self.assertEqual(nse.future_price(sample_bhavcopy(), "NIFTY", date(2026, 10, 27)), 23263.6)
        self.assertIsNone(nse.future_price(sample_bhavcopy(), "NIFTY", date(2026, 10, 6)))

    def test_index_level(self):
        self.assertEqual(nse.index_level(sample_bhavcopy(), "NIFTY"), 23140.5)

    def test_load_bhavcopy_reads_a_zipped_file_and_parses_dates(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "sample.csv.zip"
            pd.DataFrame(ROWS, columns=COLUMNS).to_csv(path, index=False)
            loaded = nse.load_bhavcopy(path)
        self.assertEqual(len(loaded), len(ROWS))
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(loaded["XpryDt"]))

    def test_download_reuses_a_file_already_on_disk(self):
        with tempfile.TemporaryDirectory() as folder:
            cached = Path(folder) / "fo_bhavcopy_20260925.csv.zip"
            cached.write_bytes(b"already here")
            self.assertEqual(nse.download_bhavcopy(date(2026, 9, 25), folder), cached)
            self.assertEqual(cached.read_bytes(), b"already here")

    def test_url_names_the_trading_day(self):
        self.assertIn("20260925", nse.URL.format(date(2026, 9, 25)))


if __name__ == "__main__":
    unittest.main()
