"""Read-only paper trading dashboard on port 8765."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from datetime import datetime, timezone
import json
import sqlite3
from zoneinfo import ZoneInfo
from pathlib import Path
import threading
import time
import sys
from urllib.parse import parse_qs, urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.live.trade_history import open_event, recover_open_time

import ccxt
import yaml
try:
    from .equity import history, initialize, save, valuation
except ImportError:
    from equity import history, initialize, save, valuation


OUTPUTS = Path("outputs")
CONFIG = Path("config/production.yaml")
EQUITY_DB = Path("/app/dashboard-data/equity.sqlite3")
PAGE = Path(__file__).with_name("index.html").read_bytes()
_exchange = ccxt.okx({"enableRateLimit": True, "options": {"defaultType": "swap"}})
_ticker = {"price": None, "change_24h_pct": None, "updated_at": None}
_ticker_time = 0.0
_history = []
_history_time = 0.0
_ticker_lock = threading.Lock()


def read_json(name, default):
    try:
        with (OUTPUTS / name).open(encoding="utf-8") as file:
            return json.load(file)
    except (FileNotFoundError, OSError, ValueError):
        return default


def activity_timestamp(row):
    try:
        stamp = datetime.fromisoformat(row.get("time"))
        return stamp.timestamp() if stamp.tzinfo is not None else float("-inf")
    except (TypeError, ValueError, OverflowError):
        return float("-inf")


def market_ticker():
    global _ticker, _ticker_time, _history, _history_time
    with _ticker_lock:
        now = time.monotonic()
        if now - _ticker_time >= 15:
            _ticker_time = now
            try:
                item = _exchange.fetch_ticker("BTC/USDT:USDT")
                _ticker = {
                    "price": item.get("last"),
                    "change_24h_pct": item.get("percentage"),
                    "updated_at": item.get("timestamp"),
                }
            except (ccxt.BaseError, OSError, ValueError):
                pass
        if now - _history_time >= 60:
            _history_time = now
            try:
                candles = _exchange.fetch_ohlcv("BTC/USDT:USDT", "1h", limit=25)
                _history = [{"time": row[0], "close": row[4]} for row in candles]
            except (ccxt.BaseError, OSError, ValueError):
                pass
        return {**_ticker, "history": _history}


def performance_summary(current, initial, core_trades):
    result = {"total_pnl": None, "total_return_pct": None, "today_pnl": None,
              "today_return_pct": None, "today_start": None, "today_start_equity": None,
              "recorded_spot_fees": sum(float(t.get("fee") or 0) for t in core_trades if isinstance(t, dict)),
              "strategy_realized_pnl": current.get("strategy_realized_pnl") if current else None}
    if not current:
        return result
    if initial and initial > 0:
        result.update(total_pnl=current["equity"] - initial,
                      total_return_pct=(current["equity"] / initial - 1) * 100)
    now = datetime.fromtimestamp(current["time"] / 1000, ZoneInfo("Asia/Shanghai"))
    start = int(now.replace(hour=0, minute=0, second=0, microsecond=0).timestamp() * 1000)
    try:
        # Read only; a missing DB must never create a second history file.
        with sqlite3.connect(f"file:{EQUITY_DB}?mode=ro", uri=True) as db:
            first = db.execute("SELECT time, equity FROM equity_samples WHERE time BETWEEN ? AND ? ORDER BY time LIMIT 1",
                               (start, current["time"])).fetchone()
        if first and first[0] < current["time"]:
            result.update(today_start=first[0], today_start_equity=first[1],
                          today_pnl=current["equity"] - first[1],
                          today_return_pct=(current["equity"] / first[1] - 1) * 100 if first[1] > 0 else None)
    except sqlite3.Error:
        pass
    return result


def dashboard_data():
    snapshot = read_json("production_account_snapshot.json", {})
    state = read_json("production_state.json", {})
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    if snapshot.get("mode") not in (None, "paper") or state.get("dry_run") is False:
        return None
    trades = read_json("production_trades.json", state.get("trade_log", []))
    if not isinstance(trades, list):
        trades = []
    fields = ("time", "side", "entry", "exit", "notional", "pnl", "pnl_pct", "reason")
    position = state.get("position") if "position" in state else snapshot.get("position")
    if not isinstance(position, dict):
        position = None
    position = recover_open_time(position, OUTPUTS / "production.log")
    saved_opens = state.get("strategy_open_log", [])
    opens = [dict(row) for row in saved_opens if isinstance(row, dict)] if isinstance(saved_opens, list) else []
    # Saved close records may retain an entry timestamp even without an open ledger.
    for item in [*trades, position]:
        if isinstance(item, dict) and item.get("opened_at"):
            event = open_event(item)
            if not any(row.get("time") == event['time'] and row.get('side') == event['side'] for row in opens):
                opens.append(event)
    strategy = config.get("strategy", {})
    risk = config.get("risk", {})
    market = market_ticker()
    market_time = market.get("updated_at")
    fresh_market = market_time and abs(int(time.time() * 1000) - market_time) < 120_000
    price = market.get("price") if fresh_market else None
    current = valuation(state, price, market_time) if price is not None else None
    account = {key: snapshot.get(key) for key in (
        "mode", "price", "equity", "cash", "core_btc",
        "net_exposure", "gross_exposure", "updated_at"
    )}
    if current:
        account.update(price=price, equity=current["equity"], cash=current["cash"],
                       core_btc=float(state.get("paper_core_btc", 0)))
    activity = []
    core_btc = float(account.get("core_btc") or 0)
    snapshot_price = float(account.get("price") or 0)
    if core_btc > 0:
        activity.append({
            "kind": "spot_position", "status": "open", "side": "buy",
            "price": snapshot_price, "size": core_btc,
            "notional": core_btc * snapshot_price,
            "time": snapshot.get("updated_at"),
        })
    if position:
        activity.append({
            "kind": "strategy_position", "status": "open",
            "side": position.get("side"), "price": position.get("entry"),
            "notional": position.get("notional"), "time": position.get("opened_at"),
            "opened_at_source": position.get("opened_at_source"),
        })
    activity.extend({**row, "kind": "strategy_open", "status": "filled",
                     "price": row.get("entry")} for row in opens)
    activity.extend({
        "kind": "strategy_close", "status": "closed",
        "side": trade.get("side"), "price": trade.get("entry"),
        "exit": trade.get("exit"), "notional": trade.get("notional"),
        "pnl": trade.get("pnl"), "time": trade.get("time"),
        **{key: trade.get(key) for key in ("reason", "entry_analysis", "account_after", "total_fee", "opened_at", "opened_at_source")},
    } for trade in reversed(trades[-50:]) if isinstance(trade, dict))
    core_trades = state.get("core_trade_log", [])
    if isinstance(core_trades, list):
        activity.extend({
            "kind": "spot_trade", "status": "filled",
            **{key: trade.get(key) for key in ("time", "side", "price", "size", "notional", "fee", "analysis", "account_after")},
        } for trade in core_trades[-50:] if isinstance(trade, dict))
    # Current holdings remain separate from the dated simulated fills.
    current_activity = [row for row in activity if row["status"] == "open"]
    completed = sorted((row for row in activity if row["status"] != "open"),
                       key=activity_timestamp, reverse=True)
    activity = current_activity + completed[:50]
    health = read_json("production_health.json", {})
    if not isinstance(health, dict):
        health = {}
    try:
        stamp = datetime.fromisoformat(health["updated_at"])
        if stamp.tzinfo is None:
            raise ValueError("heartbeat has no timezone")
        age = (datetime.now(timezone.utc) - stamp).total_seconds()
        stale = age < -5 or age > 180
    except (KeyError, TypeError, ValueError):
        stale = True
    bot_health = {key: health.get(key) for key in (
        "updated_at", "running", "paused_reason", "api_error_count", "market_error_count",
        "last_market_success_at", "last_closed_candle_ts", "cooldown", "signal",
    )}
    bot_health["stale"] = stale
    check = (health.get("signal") or {}).get("holding_check")
    # A fresh heartbeat alone does not prove an exit check for this position/bar.
    if (not position or stale or not health.get("running") or health.get("paused_reason")
            or health.get("market_error_count", 0) or not isinstance(check, dict)
            or check.get("candle_ts") != health.get("last_closed_candle_ts")
            or check.get("side") != position.get("side")
            or check.get("entry") != position.get("entry")
            or check.get("opened_at") != position.get("opened_at")):
        check = None
    return {
        "market": market,
        "account": account,
        "valuation": current,
        "initial_capital": risk.get("initial_capital"),
        "performance": performance_summary(current, risk.get("initial_capital"), core_trades if isinstance(core_trades, list) else []),
        "strategy": {
            "name": strategy.get("name"),
            "symbol": config.get("symbol"),
            "timeframe": config.get("timeframe"),
            "configured_leverage": risk.get("leverage"),
            "position": {key: position.get(key) for key in (
                "side", "entry", "size", "notional", "tp", "sl", "opened_at", "opened_at_source"
            )} if position else None,
            "holding_check": check,
        },
        "trades": [{key: trade.get(key) for key in fields} for trade in trades[-50:] if isinstance(trade, dict)],
        "activity": activity,
        "bot_updated_at": health.get("updated_at"),
        "bot_health": bot_health,
    }


def sample_loop():
    while True:
        try:
            state = read_json("production_state.json", {})
            if state.get("dry_run") is True:
                market = market_ticker()
                timestamp = market.get("updated_at")
                if timestamp and abs(int(time.time() * 1000) - timestamp) < 120_000:
                    sample = valuation(state, market.get("price"), timestamp)
                    if sample:
                        save(EQUITY_DB, sample)
        except (OSError, ValueError, KeyError, TypeError) as error:
            print(f"Equity sample failed: {error}", flush=True)
        time.sleep(60)


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        url = urlsplit(self.path)
        if url.path == "/":
            self.respond(200, "text/html; charset=utf-8", PAGE)
        elif url.path == "/api/status":
            data = dashboard_data()
            if data is None:
                self.respond(403, "application/json", b'{"error":"Only paper mode is available"}')
            else:
                self.respond(200, "application/json", json.dumps(data, ensure_ascii=False).encode())
        elif url.path == "/api/equity-history":
            state = read_json("production_state.json", {})
            if state.get("dry_run") is not True:
                self.respond(403, "application/json", b'{"error":"Only paper mode is available"}')
                return
            query = parse_qs(url.query)
            now = int(time.time() * 1000)
            ranges = {"1d": 86_400_000, "7d": 7 * 86_400_000, "30d": 30 * 86_400_000}
            try:
                selected = query.get("range", ["1d"])[0]
                if selected == "custom":
                    start = int(query["start"][0])
                    end = int(query.get("end", [now])[0])
                    if start < 0 or end > now or start >= end:
                        raise ValueError("invalid time window")
                else:
                    start = now - ranges[selected]
                    end = now
            except (KeyError, ValueError, IndexError):
                self.respond(400, "application/json", b'{"error":"Invalid time range"}')
                return
            data = history(EQUITY_DB, start, end)
            data.update(range=selected, requested_start=start, end=end)
            self.respond(200, "application/json", json.dumps(data).encode())
        else:
            self.respond(404, "text/plain; charset=utf-8", b"Not found")

    def respond(self, code, content_type, body):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


if __name__ == "__main__":
    initialize(EQUITY_DB)
    threading.Thread(target=sample_loop, daemon=True).start()
    ThreadingHTTPServer(("0.0.0.0", 8765), Handler).serve_forever()
