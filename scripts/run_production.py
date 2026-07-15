#!/usr/bin/env python
"""
TripleEma 4h 2x 实盘交易机器人 — 启动入口

用法:
    # 纸上交易 (推荐起步)
    uv run python scripts/run_production.py --dry-run

    # 模拟盘 (需 OKX Demo API Key)
    uv run python scripts/run_production.py

    # 实盘 (⚠️ 真金白银)
    uv run python scripts/run_production.py --live

    # 自定义配置
    uv run python scripts/run_production.py --config my_config.yaml
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
from src.live.production_bot import run_bot


def main():
    parser = argparse.ArgumentParser(
        description="TripleEma 4h 2x 实盘交易机器人")
    parser.add_argument("--config", type=str, default="config/production.yaml",
                        help="配置文件路径")
    parser.add_argument("--dry-run", action="store_true",
                        help="纸上交易模式 (只看不下单)")
    parser.add_argument("--live", action="store_true",
                        help="⚠️ 实盘模式 (真金白银!)")
    args = parser.parse_args()

    config_path = args.config

    # 快速覆盖模式
    if args.dry_run:
        import yaml
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        cfg["mode"]["dry_run"] = True
        cfg["mode"]["testnet"] = True
        # 写入临时配置
        import tempfile
        import os
        fd, tmp = tempfile.mkstemp(suffix=".yaml")
        with os.fdopen(fd, "w") as f:
            yaml.dump(cfg, f)
        config_path = tmp
        print("🧪 纸上交易模式: 使用实时行情, 模拟信号, 不下单")
    elif args.live:
        import yaml
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        cfg["mode"]["dry_run"] = False
        cfg["mode"]["testnet"] = False
        import tempfile, os
        fd, tmp = tempfile.mkstemp(suffix=".yaml")
        with os.fdopen(fd, "w") as f:
            yaml.dump(cfg, f)
        config_path = tmp
        print("⚠️ 实盘模式: 使用真实资金交易!")
        print("   确认? (Ctrl+C 取消) ")
        try:
            input()
        except (KeyboardInterrupt, EOFError):
            print("\n已取消")
            return

    run_bot(config_path)


if __name__ == "__main__":
    main()
