#!/usr/bin/env python
"""Run OKX live-trading preflight checks."""

import argparse
import json
import os
import sys
from pathlib import Path

import ccxt
import yaml
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.live.alerts import LocalAlertManager
from src.live.preflight import (
    PreflightCheck,
    PreflightReport,
    build_snapshot_from_exchange,
    load_state_file,
    needs_okx_account,
    validate_account_snapshot,
    validate_live_state_position,
    validate_preflight_config,
)


def run_preflight(
    config_path: str = "config/production.yaml",
    price: float = 0.0,
    allow_existing_state: bool = False,
    emit_output: bool = True,
) -> int:
    load_dotenv()
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    alert_path = cfg.get("output", {}).get("alert_log", "outputs/alerts.jsonl")
    alerts = LocalAlertManager(alert_path)
    checks = validate_preflight_config(cfg, os.environ)
    report = PreflightReport(checks=checks)

    if report.ok and needs_okx_account(cfg):
        try:
            exchange = _build_exchange(cfg)
            btc_price = price or _fetch_price(exchange, cfg["symbol"])
            snapshot = build_snapshot_from_exchange(exchange, cfg["symbol"], btc_price)
            state = load_state_file(
                cfg.get("output", {}).get("state_file", "outputs/production_state.json")
            )
            if allow_existing_state:
                checks.extend(validate_live_state_position(state, allow_empty=False))
            checks.extend(validate_account_snapshot(
                snapshot,
                allow_existing_state,
                state=state,
            ))
            report = PreflightReport(checks=checks, snapshot=snapshot)
        except Exception as e:
            checks.append(PreflightCheck(
                "okx_account_snapshot",
                False,
                "error",
                f"读取 OKX 账户快照失败: {e}",
            ))
            report = PreflightReport(checks=checks)

    if emit_output:
        print(json.dumps(report.as_dict(), ensure_ascii=False, indent=2, default=str))
    if not report.ok:
        alerts.send(
            "critical",
            "okx_preflight_failed",
            "OKX 实盘前检查失败",
            report.as_dict(),
        )
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="OKX live preflight checks")
    parser.add_argument("--config", default="config/production.yaml", help="配置文件路径")
    parser.add_argument("--price", type=float, default=0.0, help="手动指定 BTC/USDT 价格")
    parser.add_argument(
        "--allow-existing-state",
        action="store_true",
        help="允许交易所有合约仓位，前提是本地状态可恢复",
    )
    args = parser.parse_args()
    return run_preflight(
        config_path=args.config,
        price=args.price,
        allow_existing_state=args.allow_existing_state,
    )


def _build_exchange(cfg: dict):
    exchange = ccxt.okx({
        "apiKey": os.getenv("OKX_API_KEY", ""),
        "secret": os.getenv("OKX_SECRET", ""),
        "password": os.getenv("OKX_PASSPHRASE", ""),
        "enableRateLimit": True,
        "options": {"defaultType": cfg.get("market_type", "swap")},
    })
    if cfg.get("mode", {}).get("testnet", True):
        exchange.set_sandbox_mode(True)
    return exchange


def _fetch_price(exchange, symbol: str) -> float:
    ticker = exchange.fetch_ticker(symbol)
    price = float(ticker.get("last") or ticker.get("close") or 0.0)
    if price <= 0:
        raise RuntimeError("无法获取有效 BTC/USDT 价格")
    return price


if __name__ == "__main__":
    raise SystemExit(main())
