"""Bump-and-revalue risk and scenario analysis.

All sensitivities are central finite differences on immutable market objects. Monte Carlo
pricers reuse the engine's fixed seed, so base and bumped valuations share random numbers
(common random numbers) and the Greeks are not swamped by simulation noise.

Every revaluation needed (trade x market) is submitted as one batch, so a full ladder is a
single parallel job rather than a loop of small ones.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

import numpy as np
import pandas as pd

from enerq.engine import EngineConfig, price_batch
from enerq.instruments import Instrument
from enerq.market import MarketData


def _price_shift(mkt: MarketData, h: float, pillar: int | None = None) -> MarketData:
    out = mkt.with_curve(mkt.curve.shifted(h, pillar))
    if pillar is None and mkt.hourly_prices is not None:
        out = replace(out, hourly_prices=mkt.hourly_prices + h)
    return out


def _revalue(
    trades: Sequence[Instrument],
    markets: Sequence[MarketData],
    cfg: EngineConfig,
    workers: int | None,
) -> np.ndarray:
    """Value matrix of shape (n_markets, n_trades), computed as one parallel batch."""
    results = price_batch([(t, m, cfg) for m in markets for t in trades], workers=workers)
    failed = [r for r in results if r.error]
    if failed:
        raise RuntimeError(f"{len(failed)} revaluations failed, first: {failed[0].error}")
    return np.array([r.value for r in results]).reshape(len(markets), len(trades))


def risk_report(
    trades: Sequence[Instrument],
    mkt: MarketData,
    cfg: EngineConfig | None = None,
    *,
    price_bump: float = 0.5,
    vol_bump: float = 0.01,
    workers: int | None = 1,
) -> pd.DataFrame:
    """Per-trade PV, parallel delta/gamma, vega (per 1 vol point) and bucketed delta ladder."""
    cfg = cfg or EngineConfig()
    h, v = price_bump, vol_bump
    n_pillars = mkt.curve.tenors.size
    markets = [
        mkt,
        _price_shift(mkt, +h),
        _price_shift(mkt, -h),
        mkt.with_vol(mkt.vol.shifted(+v)),
        mkt.with_vol(mkt.vol.shifted(-v)),
    ]
    for i in range(n_pillars):
        markets += [_price_shift(mkt, +h, i), _price_shift(mkt, -h, i)]

    V = _revalue(trades, markets, cfg, workers)
    base, up, dn, vup, vdn = V[:5]
    report = pd.DataFrame(
        {
            "pv": base,
            "delta": (up - dn) / (2 * h),
            "gamma": (up - 2 * base + dn) / h**2,
            "vega_1pt": (vup - vdn) / (2 * v) * 0.01,
        },
        index=pd.Index([t.trade_id for t in trades], name="trade_id"),
    )
    buckets = V[5:].reshape(n_pillars, 2, len(trades))
    ladder = (buckets[:, 0] - buckets[:, 1]) / (2 * h)
    for i, tenor in enumerate(mkt.curve.tenors):
        report[f"delta_{tenor:g}y"] = ladder[i]
    report.loc["TOTAL"] = report.sum()
    return report


def scenario_grid(
    trades: Sequence[Instrument],
    mkt: MarketData,
    price_shifts: Sequence[float],
    vol_shifts: Sequence[float],
    cfg: EngineConfig | None = None,
    *,
    workers: int | None = 1,
) -> pd.DataFrame:
    """Portfolio P&L versus base under joint parallel price / vol shocks."""
    cfg = cfg or EngineConfig()
    markets = [
        _price_shift(mkt, dp).with_vol(mkt.vol.shifted(dv))
        for dv in vol_shifts
        for dp in price_shifts
    ]
    V = _revalue(trades, [mkt, *markets], cfg, workers).sum(axis=1)
    pnl = (V[1:] - V[0]).reshape(len(vol_shifts), len(price_shifts))
    return pd.DataFrame(
        pnl,
        index=pd.Index(vol_shifts, name="vol_shift"),
        columns=pd.Index(price_shifts, name="price_shift"),
    )
