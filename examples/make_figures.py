"""Regenerate the figures used in the README.

python examples/make_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from enerq.cli import load_portfolio
from enerq.instruments import AsianOption, BatteryStorage, GasStorage
from enerq.market import MarketData
from enerq.models import asian, battery, storage

EX = Path(__file__).resolve().parent
OUT = EX.parent / "docs"
OUT.mkdir(exist_ok=True)
plt.rcParams.update({"figure.dpi": 140, "axes.grid": True, "grid.alpha": 0.3, "font.size": 9})


def storage_figure(mkt: MarketData, inst: GasStorage) -> None:
    r = storage.lsmc_value(inst, mkt, n_paths=20_000)
    t = np.linspace(inst.start, inst.end, inst.n_steps + 1)
    fig, ax1 = plt.subplots(figsize=(7, 3.2))
    ax1.plot(t, mkt.curve(t), color="#444", label="forward curve")
    ax1.set_ylabel("price")
    ax1.set_xlabel("years")
    ax2 = ax1.twinx()
    ax2.fill_between(t, r.expected_inventory, step="post", alpha=0.3, color="#2b7bba")
    ax2.set_ylabel("expected inventory", color="#2b7bba")
    ax2.grid(False)
    ax1.set_title(
        f"Gas storage LSMC: intrinsic {r.intrinsic:,.0f}, total {r.value:,.0f} "
        f"(extrinsic {r.extrinsic / r.value:.0%})"
    )
    fig.tight_layout()
    fig.savefig(OUT / "storage.png")


def battery_figure(mkt: MarketData, inst: BatteryStorage) -> None:
    r = battery.optimise_dispatch(inst, mkt)
    s = r.schedule.iloc[:72]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 4), sharex=True)
    ax1.step(s.index, s.price, where="post", color="#444")
    ax1.set_ylabel("price")
    net = s.discharge_mw - s.charge_mw
    ax2.bar(s.index, net, color=np.where(net >= 0, "#d95f02", "#1b9e77"), width=0.9)
    ax2.set_ylabel("discharge(+) / charge(-) MW")
    ax2.set_xlabel("hour")
    ax1.set_title(f"BESS optimal dispatch (MILP), first 72h of 168h, value {r.value:,.0f}")
    fig.tight_layout()
    fig.savefig(OUT / "battery.png")


def asian_figure(mkt: MarketData) -> None:
    """Moment-matching error vs MC across strike and averaging-window length."""
    strikes = np.linspace(30, 50, 9)
    windows = [1 / 12, 3 / 12, 6 / 12, 1.0]
    err = np.zeros((len(windows), len(strikes)))
    for i, w in enumerate(windows):
        fix = tuple(np.linspace(0.25, 0.25 + w, max(int(w * 250), 2)))
        for j, k in enumerate(strikes):
            opt = AsianOption("a", strike=float(k), fixings=fix)
            mc = asian.monte_carlo(opt, mkt, n_paths=100_000).price
            err[i, j] = 1e4 * (asian.moment_matching(opt, mkt) - mc) / max(mc, 1e-8)
    fig, ax = plt.subplots(figsize=(7, 3))
    for i, w in enumerate(windows):
        ax.plot(strikes, err[i], marker="o", ms=3, label=f"{w * 12:.0f}m window")
    ax.axhline(0, color="#444", lw=0.8)
    ax.set_xlabel("strike")
    ax.set_ylabel("MM - MC (bp of price)")
    ax.set_title("Asian call: moment-matching approximation error vs Sobol+CV Monte Carlo")
    ax.legend(ncol=4, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "asian_error.png")


def main() -> None:
    mkt = MarketData.from_dict(json.loads((EX / "market.json").read_text()))
    trades = {t.trade_id: t for t in load_portfolio(str(EX / "portfolio.json"))}
    s, b = trades["STORAGE-1Y"], trades["BESS-10MW-2H"]
    assert isinstance(s, GasStorage) and isinstance(b, BatteryStorage)
    storage_figure(mkt, s)
    battery_figure(mkt, b)
    asian_figure(mkt)
    print(f"figures written to {OUT}")


if __name__ == "__main__":
    main()
