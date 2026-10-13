from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from enerq.instruments import GasStorage
from enerq.market import ForwardCurve, MarketData, VolTermStructure
from enerq.models import storage


@pytest.fixture
def seasonal_mkt() -> MarketData:
    t = np.linspace(0.0, 1.0, 13)
    return MarketData(
        curve=ForwardCurve(t, 35 + 8 * np.cos(2 * np.pi * t)),  # expensive at start/end, cheap mid
        vol=VolTermStructure.flat(0.4),
        rate=0.02,
        mean_reversion=8.0,
        spot_vol=0.6,
    )


@pytest.fixture
def contract() -> GasStorage:
    return GasStorage(
        "s",
        start=0.0,
        end=1.0,
        n_steps=24,
        capacity=60,
        initial=0,
        final=0,
        max_inject=10,
        max_withdraw=20,
        volume_step=5,
    )


def test_intrinsic_matches_hand_calculation() -> None:
    # Two steps, flat->up curve: inject full at t0 price 10, withdraw at t1 price 20.
    mkt = MarketData(ForwardCurve([0.0, 0.5], [10.0, 20.0]), VolTermStructure.flat(0.3))
    c = GasStorage(
        "s",
        start=0,
        end=1,
        n_steps=2,
        capacity=10,
        initial=0,
        final=0,
        max_inject=10,
        max_withdraw=10,
        volume_step=5,
    )
    assert storage.intrinsic_value(c, mkt) == pytest.approx(10 * (20 - 10))


def test_lsmc_at_least_intrinsic(seasonal_mkt: MarketData, contract: GasStorage) -> None:
    r = storage.lsmc_value(contract, seasonal_mkt, n_paths=8_000)
    assert r.value >= r.intrinsic - 3 * r.stderr
    assert r.extrinsic > 0
    assert r.expected_inventory[0] == contract.initial
    assert r.expected_inventory[-1] == pytest.approx(contract.final)


def test_zero_vol_collapses_to_intrinsic(seasonal_mkt: MarketData, contract: GasStorage) -> None:
    mkt = replace(seasonal_mkt, spot_vol=0.0)
    r = storage.lsmc_value(contract, mkt, n_paths=500)
    assert r.value == pytest.approx(r.intrinsic, rel=1e-9)
    assert r.stderr == pytest.approx(0.0, abs=1e-9)


def test_value_increases_with_volatility(seasonal_mkt: MarketData, contract: GasStorage) -> None:
    lo = storage.lsmc_value(contract, replace(seasonal_mkt, spot_vol=0.3), n_paths=8_000)
    hi = storage.lsmc_value(contract, replace(seasonal_mkt, spot_vol=0.9), n_paths=8_000)
    assert hi.value > lo.value


def test_in_and_out_of_sample_agree(seasonal_mkt: MarketData, contract: GasStorage) -> None:
    r = storage.lsmc_value(contract, seasonal_mkt, n_paths=20_000)
    assert abs(r.in_sample - r.value) < 4 * r.stderr * np.sqrt(2)


def test_costs_reduce_value(seasonal_mkt: MarketData, contract: GasStorage) -> None:
    costly = replace(contract, inject_cost=1.0, withdraw_cost=1.0)
    assert storage.intrinsic_value(costly, seasonal_mkt) < storage.intrinsic_value(
        contract, seasonal_mkt
    )


def test_validation() -> None:
    with pytest.raises(ValueError, match="multiple"):
        GasStorage("s", capacity=12, volume_step=5)
    with pytest.raises(ValueError, match="unreachable"):
        GasStorage("s", n_steps=2, capacity=100, initial=0, final=100, max_inject=10, volume_step=5)
