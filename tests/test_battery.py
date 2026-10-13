from __future__ import annotations

from dataclasses import replace
from itertools import pairwise

import numpy as np
import pytest

from enerq.instruments import BatteryStorage
from enerq.market import ForwardCurve, MarketData, VolTermStructure
from enerq.models.battery import InfeasibleDispatchError, constraint_attribution, optimise_dispatch

TOL = 1e-6


def _mkt(prices: list[float] | np.ndarray) -> MarketData:
    return MarketData(ForwardCurve([1.0], [50.0]), VolTermStructure.flat(0.5), hourly_prices=prices)


def test_simple_arbitrage_value() -> None:
    # Charge 1h at 0, discharge 1h at 100, lossless, 1MW/1MWh, start and end empty.
    b = BatteryStorage(
        "b", power_mw=1, energy_mwh=1, charge_eff=1, discharge_eff=1, initial_soc=0, terminal_soc=0
    )
    r = optimise_dispatch(b, _mkt([0.0, 100.0]))
    assert r.value == pytest.approx(100.0)
    assert r.full_cycles == pytest.approx(1.0)


def test_constraints_respected(mkt: MarketData) -> None:
    b = BatteryStorage(
        "b",
        power_mw=10,
        energy_mwh=20,
        charge_eff=0.9,
        discharge_eff=0.9,
        initial_soc=0.5,
        terminal_soc=0.5,
        max_cycles=4,
        ramp_mw_per_h=6,
        min_soc=0.1,
        max_soc=0.9,
    )
    r = optimise_dispatch(b, mkt)
    s = r.schedule
    E = b.energy_mwh
    assert s.soc_mwh.min() >= 0.1 * E - TOL and s.soc_mwh.max() <= 0.9 * E + TOL
    assert s.soc_mwh.iloc[-1] == pytest.approx(0.5 * E, abs=TOL)
    assert ((s.charge_mw > TOL) & (s.discharge_mw > TOL)).sum() == 0
    net = s.discharge_mw - s.charge_mw
    assert net.diff().abs().max() <= 6 + TOL
    assert r.full_cycles <= 4 + TOL
    # Energy balance holds hour by hour.
    soc_prev = np.r_[0.5 * E, s.soc_mwh.to_numpy()[:-1]]
    balance = soc_prev + 0.9 * s.charge_mw - s.discharge_mw / 0.9
    np.testing.assert_allclose(balance, s.soc_mwh, atol=1e-6)


def test_relaxing_constraints_never_reduces_value(mkt: MarketData, portfolio: list) -> None:
    b = next(t for t in portfolio if isinstance(t, BatteryStorage))
    attr = constraint_attribution(b, mkt)
    assert (attr.uplift >= -1e-6).all()
    assert set(attr.constraint) == {
        "cycling_limit",
        "ramp_limit",
        "terminal_soc",
        "no_simultaneous",
    }


def test_cycling_limit_monotone(mkt: MarketData) -> None:
    base = BatteryStorage("b", power_mw=10, energy_mwh=20)
    values = [optimise_dispatch(replace(base, max_cycles=n), mkt).value for n in (1, 3, 7, None)]
    assert all(a <= b + 1e-6 for a, b in pairwise(values))


def test_negative_prices_make_mode_constraint_bind() -> None:
    # With negative prices a lossy battery can earn by charging and discharging at once.
    prices = [-50.0] * 4
    b = BatteryStorage(
        "b",
        power_mw=1,
        energy_mwh=1,
        charge_eff=0.8,
        discharge_eff=0.8,
        initial_soc=0.5,
        terminal_soc=None,
    )
    strict = optimise_dispatch(b, _mkt(prices)).value
    relaxed = optimise_dispatch(replace(b, no_simultaneous=False), _mkt(prices)).value
    assert relaxed > strict + 1.0


def test_infeasible_raises() -> None:
    b = BatteryStorage("b", power_mw=1, energy_mwh=10, initial_soc=0.0, terminal_soc=1.0)
    with pytest.raises(InfeasibleDispatchError):
        optimise_dispatch(b, _mkt([10.0, 20.0]))


def test_missing_prices() -> None:
    b = BatteryStorage("b")
    with pytest.raises(ValueError, match="hourly_prices"):
        optimise_dispatch(b, MarketData(ForwardCurve([1.0], [1.0]), VolTermStructure.flat(0.1)))
