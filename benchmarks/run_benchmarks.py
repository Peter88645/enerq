"""Reproducible performance / accuracy benchmarks. Prints a Markdown table.

python benchmarks/run_benchmarks.py
"""

from __future__ import annotations

import json
import platform
import time
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from enerq.cli import load_portfolio
from enerq.instruments import AsianOption, BatteryStorage, GasStorage
from enerq.market import MarketData
from enerq.models import asian, battery, black76, storage

EX = Path(__file__).resolve().parents[1] / "examples"


def timeit(fn: Callable[[], Any], repeat: int = 3) -> tuple[float, Any]:
    best, out = float("inf"), None
    for _ in range(repeat):
        t0 = time.perf_counter()
        out = fn()
        best = min(best, time.perf_counter() - t0)
    return best, out


def main() -> None:
    mkt = MarketData.from_dict(json.loads((EX / "market.json").read_text()))
    trades = {t.trade_id: t for t in load_portfolio(str(EX / "portfolio.json"))}
    rows: list[tuple[str, str, str]] = []

    rng = np.random.default_rng(0)
    n = 1_000_000
    F = rng.uniform(20, 80, n)
    K = F * rng.uniform(0.8, 1.25, n)
    T = rng.uniform(0.1, 3, n)
    sig = rng.uniform(0.15, 1.2, n)
    t, prices = timeit(lambda: black76.price(F, K, T, sig, 0.97))
    rows.append(("Black-76 price, 1,000,000 options", f"{t * 1e3:,.0f} ms", "vectorised NumPy"))
    t, _ = timeit(lambda: black76.greeks(F, K, T, sig, 0.03))
    rows.append(("Black-76 price + 4 Greeks, 1,000,000 options", f"{t * 1e3:,.0f} ms", "analytic"))
    t, iv = timeit(lambda: black76.implied_vol(prices, F, K, T, 0.97), repeat=1)
    err = np.nanmax(np.abs(iv - sig))
    rows.append(
        (
            "Implied vol, 1,000,000 options",
            f"{t * 1e3:,.0f} ms",
            f"max abs error vs true vol {err:.1e}",
        )
    )

    a = trades["ASIAN-DEC"]
    assert isinstance(a, AsianOption)
    t, mm = timeit(lambda: asian.moment_matching(a, mkt), repeat=20)
    rows.append(("Asian, moment matching (31 fixings)", f"{t * 1e6:,.0f} µs", f"{mm:.5f}"))
    for method, cv in [("pseudo", False), ("pseudo", True), ("sobol", True)]:
        t, r = timeit(
            lambda m=method, c=cv: asian.monte_carlo(  # type: ignore[misc]
                a, mkt, n_paths=100_000, method=m, control_variate=c
            )
        )
        label = f"Asian MC 100k paths, {method}{' + geometric CV' if cv else ''}"
        rows.append((label, f"{t * 1e3:,.0f} ms", f"{r.price:.5f} ± {r.stderr:.1e}"))

    s = trades["STORAGE-1Y"]
    assert isinstance(s, GasStorage)
    t, r = timeit(lambda: storage.lsmc_value(s, mkt, n_paths=20_000), repeat=1)
    rows.append(
        (
            "Gas storage LSMC, 52 steps x 21 levels x 20k paths",
            f"{t:,.2f} s",
            f"intrinsic {r.intrinsic:.1f}, total {r.value:.1f} ± {r.stderr:.1f}",
        )
    )

    b = trades["BESS-10MW-2H"]
    assert isinstance(b, BatteryStorage)
    t, r = timeit(lambda: battery.optimise_dispatch(b, mkt))
    rows.append(
        ("BESS MILP, 168 h (672 vars, 168 binaries)", f"{t * 1e3:,.0f} ms", f"{r.value:,.0f}")
    )
    month = np.tile(mkt.hourly_prices, 5)[:744]
    mkt_m = MarketData(mkt.curve, mkt.vol, hourly_prices=month)
    b_m = replace(b, max_cycles=31.0)
    t, r = timeit(lambda: battery.optimise_dispatch(b_m, mkt_m), repeat=1)
    rows.append(("BESS MILP, 744 h (one month)", f"{t:,.2f} s", f"{r.value:,.0f}"))

    print(f"Python {platform.python_version()}, NumPy {np.__version__}, {platform.machine()}\n")
    print("| Benchmark | Time | Result |\n|---|---:|---|")
    for name, tm, res in rows:
        print(f"| {name} | {tm} | {res} |")


if __name__ == "__main__":
    main()
