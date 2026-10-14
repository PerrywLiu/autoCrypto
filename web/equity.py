"""Paper account valuation and persisted equity samples."""

import math
from pathlib import Path
import sqlite3


def valuation(state, price, timestamp):
    """Value the shared paper account without treating swap notional as an asset."""
    if state.get("dry_run") is not True or not isinstance(price, (int, float)) or not math.isfinite(price) or price <= 0:
        return None
    cash = float(state.get("paper_cash", 0))
    spot_value = float(state.get("paper_core_btc", 0)) * price
    realized = float(state.get("total_pnl", 0))
    unrealized = 0.0
    position = state.get("position")
    if position:
        entry = float(position["entry"])
        notional = float(position["notional"])
        if entry <= 0:
            return None
        if position["side"] == "long":
            unrealized = notional * (price / entry - 1)
        elif position["side"] == "short":
            unrealized = notional * (entry / price - 1)
        else:
            return None
    amounts = (cash, spot_value, realized, unrealized)
    if not all(math.isfinite(value) for value in amounts):
        return None
    return {
        "time": int(timestamp), "price": price, "cash": cash,
        "spot_value": spot_value, "strategy_realized_pnl": realized,
        "strategy_unrealized_pnl": unrealized,
        "equity": sum(amounts),
    }


def initialize(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as db:
        db.execute("""CREATE TABLE IF NOT EXISTS equity_samples (
            time INTEGER PRIMARY KEY, price REAL NOT NULL, cash REAL NOT NULL,
            spot_value REAL NOT NULL, strategy_realized_pnl REAL NOT NULL,
            strategy_unrealized_pnl REAL NOT NULL, equity REAL NOT NULL
        )""")


def save(path, sample):
    columns = ("time", "price", "cash", "spot_value", "strategy_realized_pnl", "strategy_unrealized_pnl", "equity")
    with sqlite3.connect(path) as db:
        db.execute(
            "INSERT OR REPLACE INTO equity_samples VALUES (?, ?, ?, ?, ?, ?, ?)",
            tuple(sample[key] for key in columns),
        )


def history(path, start, end, max_points=500):
    with sqlite3.connect(path) as db:
        db.row_factory = sqlite3.Row
        first = db.execute("SELECT MIN(time) FROM equity_samples").fetchone()[0]
        rows = db.execute(
            "SELECT * FROM equity_samples WHERE time BETWEEN ? AND ? ORDER BY time",
            (start, end),
        ).fetchall()
    count = len(rows)
    if count > max_points:
        rows = [rows[round(i * (count - 1) / (max_points - 1))] for i in range(max_points)]
    return {"first_sample_time": first, "sample_count": count, "points": [dict(row) for row in rows]}
