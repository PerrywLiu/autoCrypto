import tempfile
import unittest
from pathlib import Path

from web.equity import history, initialize, save, valuation


class EquityHistoryTests(unittest.TestCase):
    def test_shared_account_includes_realized_and_open_strategy_pnl(self):
        state = {
            "dry_run": True, "paper_cash": 200, "paper_core_btc": 0.01,
            "total_pnl": 15,
            "position": {"side": "long", "entry": 10000, "notional": 100},
        }
        sample = valuation(state, 11000, 123)
        self.assertAlmostEqual(sample["spot_value"], 110)
        self.assertAlmostEqual(sample["strategy_unrealized_pnl"], 10)
        self.assertAlmostEqual(sample["equity"], 335)
        state["position"] = {"side": "short", "entry": 10000, "notional": 100}
        self.assertAlmostEqual(valuation(state, 11000, 124)["equity"], 200 + 110 + 15 + 100 * (10000 / 11000 - 1))
        state["dry_run"] = False
        self.assertIsNone(valuation(state, 11000, 125))

    def test_history_filters_by_start_and_preserves_endpoints(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "equity.sqlite3"
            initialize(path)
            for timestamp in range(10):
                save(path, valuation({"dry_run": True, "paper_cash": timestamp}, 100, timestamp))
            result = history(path, 2, 8, max_points=3)
            self.assertEqual(result["first_sample_time"], 0)
            self.assertEqual(result["sample_count"], 7)
            self.assertEqual([point["time"] for point in result["points"]], [2, 5, 8])


if __name__ == "__main__":
    unittest.main()
