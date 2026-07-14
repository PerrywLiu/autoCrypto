#!/usr/bin/env python3
"""
实盘交易启动脚本

用法:
    # 模拟盘 (默认)
    uv run python scripts/live.py

    # 自定义参数
    uv run python scripts/live.py --symbol BTC/USDT --capital 500 --leverage 3

配置 API 密钥:
    export OKX_API_KEY="your_key"
    export OKX_SECRET="your_secret"
    export OKX_PASSPHRASE="your_passphrase"
"""

import os
import sys
import argparse
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()  # 加载 .env 文件中的环境变量

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.live.bot import LiveTradingBot
from loguru import logger


def main():
    parser = argparse.ArgumentParser(description="autoCrypto 实盘交易机器人")
    parser.add_argument("--symbol", default="BTC/USDT", help="交易币种")
    parser.add_argument("--capital", type=float, default=500, help="初始资金 USDT")
    parser.add_argument("--leverage", type=int, default=3, help="杠杆倍数")
    parser.add_argument("--risk", type=float, default=0.02, help="单笔风险比例")
    parser.add_argument("--live", action="store_true", help="实盘模式 (默认模拟盘)")
    parser.add_argument("--dry-run", action="store_true", help="纸上交易模式 (只看不下单, 模拟盈亏)")
    args = parser.parse_args()

    # API 密钥
    api_key = os.getenv("OKX_API_KEY")
    secret = os.getenv("OKX_SECRET")
    passphrase = os.getenv("OKX_PASSPHRASE")

    if not all([api_key, secret, passphrase]):
        logger.error("请设置环境变量: OKX_API_KEY, OKX_SECRET, OKX_PASSPHRASE")
        logger.info("获取方式: OKX 官网 → API → 创建 API Key (需交易权限)")
        sys.exit(1)

    testnet = not args.live

    if args.dry_run:
        print("╔══════════════════════════════════════╗")
        print("║   🧪 纸上交易 — 只看信号不下单       ║")
        print("║   用实时数据, 模拟成交和盈亏          ║")
        print("╚══════════════════════════════════════╝")
    elif not args.live:
        print("╔══════════════════════════════════════╗")
        print("║   🧪 模拟盘模式 — 不会使用真实资金   ║")
        print("╚══════════════════════════════════════╝")
    else:
        print("╔══════════════════════════════════════╗")
        print("║   ⚠️  实盘模式 — 真实资金交易!!     ║")
        print("╚══════════════════════════════════════╝")
        confirm = input("确认进入实盘? 输入 'yes' 继续: ")
        if confirm != "yes":
            print("已取消")
            sys.exit(0)

    bot = LiveTradingBot(
        api_key=api_key,
        secret=secret,
        passphrase=passphrase,
        symbol=args.symbol,
        testnet=testnet,
        initial_capital=args.capital,
        risk_per_trade=args.risk,
        leverage=args.leverage,
    )
    bot.dry_run = args.dry_run

    try:
        bot.start()
    except KeyboardInterrupt:
        bot.stop()


if __name__ == "__main__":
    main()
