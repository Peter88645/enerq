"""Pricing engine: maps instrument types to models and prices portfolios in parallel.

Pricers are registered with a decorator, so adding a product is one function and no change to
the engine (open/closed principle)::

    @register(MyProduct)
    def _price_my_product(inst: MyProduct, mkt: MarketData, cfg: EngineConfig) -> PricingResult:
        ...

Portfolio pricing fans out over a ``ProcessPoolExecutor``: the heavy models (LSMC, MILP) are
CPU-bound NumPy/HiGHS code, and frozen instruments/market data pickle cheaply.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Literal, TypeVar

import pandas as pd

from enerq.instruments import (
    AsianOption,
    BatteryStorage,
    EuropeanOption,
    GasStorage,
    Instrument,
    OptionType,
)
from enerq.market import MarketData
from enerq.models import asian, battery, black76, storage

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class EngineConfig:
    """Numerical settings. Fixed seeds give common random numbers across bumped revaluations."""

    asian_method: Literal["moment_matching", "monte_carlo"] = "moment_matching"
    mc_paths: int = 100_000
    lsmc_paths: int = 20_000
    seed: int = 42


@dataclass(frozen=True, slots=True)
class PricingResult:
    trade_id: str
    model: str
    value: float
    stderr: float = 0.0
    elapsed_ms: float = 0.0
    details: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


InstT = TypeVar("InstT", bound=Instrument)
Pricer = Callable[[Any, MarketData, EngineConfig], PricingResult]
_REGISTRY: dict[type[Instrument], Pricer] = {}


def register(
    inst_type: type[InstT],
) -> Callable[[Callable[[InstT, MarketData, EngineConfig], PricingResult]], Pricer]:
    def deco(fn: Callable[[InstT, MarketData, EngineConfig], PricingResult]) -> Pricer:
        if inst_type in _REGISTRY:
            raise ValueError(f"pricer already registered for {inst_type.__name__}")
        _REGISTRY[inst_type] = fn
        return fn

    return deco


def price(inst: Instrument, mkt: MarketData, cfg: EngineConfig | None = None) -> PricingResult:
    """Price one instrument. Errors are captured in the result rather than raised, so one bad
    trade never kills a portfolio run."""
    cfg = cfg or EngineConfig()
    pricer = _REGISTRY.get(type(inst))
    if pricer is None:
        return PricingResult(
            inst.trade_id, "none", float("nan"), error=f"no pricer for {type(inst).__name__}"
        )
    t0 = time.perf_counter()
    try:
        res = pricer(inst, mkt, cfg)
    except Exception as exc:
        log.exception("pricing failed for %s", inst.trade_id)
        return PricingResult(inst.trade_id, pricer.__name__, float("nan"), error=repr(exc))
    elapsed = (time.perf_counter() - t0) * 1e3
    return PricingResult(
        res.trade_id, res.model, res.value, res.stderr, round(elapsed, 3), res.details, res.error
    )


def _price_job(args: tuple[Instrument, MarketData, EngineConfig]) -> PricingResult:  # picklable
    return price(*args)


Job = tuple[Instrument, MarketData, EngineConfig]


def price_batch(jobs: Sequence[Job], *, workers: int | None = 1) -> list[PricingResult]:
    """Price arbitrary (instrument, market, config) jobs; order is preserved.

    ``workers=1`` runs in-process; ``None`` uses one process per CPU.
    """
    if workers == 1 or len(jobs) <= 1:
        return [_price_job(j) for j in jobs]
    chunk = max(1, len(jobs) // (4 * (workers or 8)))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(_price_job, jobs, chunksize=chunk))


def price_many(
    trades: Sequence[Instrument],
    mkt: MarketData,
    cfg: EngineConfig | None = None,
    *,
    workers: int | None = 1,
) -> list[PricingResult]:
    """Price a portfolio against one market snapshot."""
    cfg = cfg or EngineConfig()
    return price_batch([(t, mkt, cfg) for t in trades], workers=workers)


def to_frame(results: Iterable[PricingResult]) -> pd.DataFrame:
    rows = [
        {
            "trade_id": r.trade_id,
            "model": r.model,
            "value": r.value,
            "stderr": r.stderr,
            "ms": r.elapsed_ms,
            "error": r.error,
        }
        for r in results
    ]
    return pd.DataFrame(rows).set_index("trade_id")


# --------------------------------------------------------------------------------------------
# Registered pricers
# --------------------------------------------------------------------------------------------


@register(EuropeanOption)
def _european(inst: EuropeanOption, mkt: MarketData, cfg: EngineConfig) -> PricingResult:
    delivery = inst.delivery if inst.delivery is not None else inst.expiry
    F = float(mkt.curve(delivery))
    sigma = float(mkt.vol.vol(inst.expiry))
    g = black76.greeks(
        F, inst.strike, inst.expiry, sigma, mkt.rate, inst.option_type is OptionType.CALL
    )
    n = inst.notional
    return PricingResult(
        inst.trade_id,
        "black76",
        n * float(g.price),
        details={
            "forward": F,
            "vol": sigma,
            "delta": n * float(g.delta),
            "vega": n * float(g.vega),
        },
    )


@register(AsianOption)
def _asian(inst: AsianOption, mkt: MarketData, cfg: EngineConfig) -> PricingResult:
    n = inst.notional
    if cfg.asian_method == "monte_carlo":
        r = asian.monte_carlo(inst, mkt, n_paths=cfg.mc_paths, seed=cfg.seed)
        return PricingResult(
            inst.trade_id,
            "asian_mc_cv",
            n * r.price,
            n * r.stderr,
            details={"cv_beta": r.control_variate_beta},
        )
    return PricingResult(
        inst.trade_id, "asian_moment_matching", n * asian.moment_matching(inst, mkt)
    )


@register(GasStorage)
def _gas_storage(inst: GasStorage, mkt: MarketData, cfg: EngineConfig) -> PricingResult:
    r = storage.lsmc_value(inst, mkt, n_paths=cfg.lsmc_paths, seed=cfg.seed)
    n = inst.notional
    return PricingResult(
        inst.trade_id,
        "storage_lsmc",
        n * r.value,
        n * r.stderr,
        details={"intrinsic": n * r.intrinsic, "extrinsic": n * r.extrinsic},
    )


@register(BatteryStorage)
def _battery(inst: BatteryStorage, mkt: MarketData, cfg: EngineConfig) -> PricingResult:
    r = battery.optimise_dispatch(inst, mkt)
    return PricingResult(
        inst.trade_id,
        "bess_milp",
        r.value,
        details={"full_cycles": r.full_cycles, "mip_gap": r.mip_gap},
    )
