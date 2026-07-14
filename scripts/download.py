#!/usr/bin/env python3
"""
OKX 历史 K 线数据下载脚本

用法:
    python scripts/download.py --symbol BTC/USDT --timeframe 1h --start 2024-01-01 --end 2025-06-01

依赖:
    uv add ccxt pyarrow    # 数据下载所需
"""

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys

# 将项目根目录加入 Python 路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.downloader import OKXDataDownloader
from loguru import logger


def main():
    parser = argparse.ArgumentParser(
        description="下载 OKX 历史 K 线数据并保存为 Parquet 格式"
    )
    parser.add_argument(
        "--symbol", default="BTC/USDT",
        help="交易对（默认: BTC/USDT）"
    )
    parser.add_argument(
        "--timeframe", default="1h",
        choices=["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w"],
        help="K线周期（默认: 1h）"
    )
    parser.add_argument(
        "--start", default="2024-01-01",
        help="起始日期 YYYY-MM-DD（默认: 2024-01-01）"
    )
    parser.add_argument(
        "--end", default="2025-06-01",
        help="结束日期 YYYY-MM-DD（默认: 2025-06-01）"
    )
    parser.add_argument(
        "--data-dir", default="data/raw",
        help="数据存储目录（默认: data/raw）"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="强制重新下载（忽略已有数据）"
    )
    parser.add_argument(
        "--tfs", nargs="+", default=None,
        help="批量下载多个时间框架，如: --tfs 1h 4h 1d"
    )
    parser.add_argument(
        "--symbols", nargs="+", default=None,
        help="批量下载多个交易对，如: --symbols BTC/USDT ETH/USDT SOL/USDT"
    )

    args = parser.parse_args()

    downloader = OKXDataDownloader(data_dir=args.data_dir)

    symbols = args.symbols or [args.symbol]
    timeframes = args.tfs or [args.timeframe]

    start_dt = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end_dt = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    for symbol in symbols:
        for tf in timeframes:
            logger.info(f">> 下载 {symbol} {tf} ({args.start} ~ {args.end})")
            try:
                df = downloader.download_and_save(
                    symbol=symbol, timeframe=tf,
                    since=start_dt, until=end_dt,
                    force=args.force,
                )
                if not df.empty:
                    logger.info(f"✅ {symbol} {tf}: {len(df)} 根K线")
                else:
                    logger.warning(f"⚠️ {symbol} {tf}: 无数据")
            except Exception as e:
                logger.error(f"❌ {symbol} {tf}: {e}")

    logger.info("全部下载完成！")


if __name__ == "__main__":
    main()
