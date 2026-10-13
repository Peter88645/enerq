"""Command line interface.

enerq price  examples/portfolio.json --market examples/market.json --workers 4
enerq risk   examples/portfolio.json --market examples/market.json
enerq battery-attribution examples/portfolio.json --market examples/market.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from enerq.engine import EngineConfig, price_many, to_frame
from enerq.instruments import BatteryStorage, Instrument, instrument_from_dict
from enerq.market import MarketData
from enerq.models.battery import constraint_attribution
from enerq.risk import risk_report


def _load_json(path: str) -> Any:
    return json.loads(Path(path).read_text())


def load_portfolio(path: str) -> list[Instrument]:
    return [instrument_from_dict(d) for d in _load_json(path)["trades"]]


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="enerq", description="Energy derivatives pricing & risk")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)
    for name, help_ in [
        ("price", "price a portfolio"),
        ("risk", "delta/gamma/vega and bucketed delta ladder"),
        ("battery-attribution", "value impact of each battery constraint"),
    ]:
        sp = sub.add_parser(name, help=help_)
        sp.add_argument("portfolio", help="JSON file with a 'trades' list")
        sp.add_argument("--market", required=True, help="market data JSON")
        sp.add_argument("--workers", type=int, default=1, help="processes (0 = all CPUs)")
        sp.add_argument(
            "--asian", choices=["moment_matching", "monte_carlo"], default="moment_matching"
        )
        sp.add_argument("--csv", help="also write the result table to this CSV file")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)
    trades = load_portfolio(args.portfolio)
    mkt = MarketData.from_dict(_load_json(args.market))
    cfg = EngineConfig(asian_method=args.asian)
    workers = None if args.workers == 0 else args.workers

    table: pd.DataFrame
    if args.command == "price":
        table = to_frame(price_many(trades, mkt, cfg, workers=workers))
    elif args.command == "risk":
        table = risk_report(trades, mkt, cfg, workers=workers)
    else:
        frames = [
            constraint_attribution(t, mkt).assign(trade_id=t.trade_id)
            for t in trades
            if isinstance(t, BatteryStorage)
        ]
        if not frames:
            print("no battery trades in portfolio", file=sys.stderr)
            return 1
        table = pd.concat(frames).set_index(["trade_id", "constraint"])

    with pd.option_context("display.float_format", "{:,.4f}".format, "display.width", 160):
        print(table.to_string())
    if args.csv:
        table.to_csv(args.csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
