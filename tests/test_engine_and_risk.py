from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest

from enerq.engine import EngineConfig, price, price_many, to_frame
from enerq.instruments import EuropeanOption, Instrument, instrument_from_dict
from enerq.market import ForwardCurve, MarketData, VolTermStructure
from enerq.models import black76
from enerq.risk import risk_report, scenario_grid

FAST = EngineConfig(mc_paths=20_000, lsmc_paths=3_000)


def test_roundtrip_serialisation(portfolio: list[Instrument]) -> None:
    for t in portfolio:
        clone = instrument_from_dict(json.loads(json.dumps(t.to_dict())))
        assert clone == t


def test_unknown_type() -> None:
    with pytest.raises(ValueError, match="unknown"):
        instrument_from_dict({"type": "swaption", "trade_id": "x"})


def test_market_is_immutable(mkt: MarketData) -> None:
    with pytest.raises(ValueError):
        mkt.curve.prices[0] = 1.0
    bumped = mkt.curve.shifted(1.0)
    assert bumped is not mkt.curve and mkt.curve.prices[0] + 1 == pytest.approx(bumped.prices[0])


def test_vol_term_structure_rejects_calendar_arbitrage() -> None:
    with pytest.raises(ValueError, match="calendar"):
        VolTermStructure([0.5, 1.0], [0.8, 0.3])


def test_portfolio_prices_and_errors_are_isolated(
    mkt: MarketData, portfolio: list[Instrument]
) -> None:
    bad = replace(portfolio[0], trade_id="BAD")
    bad_mkt_trade = [*portfolio, bad]
    results = price_many(bad_mkt_trade, mkt, FAST)
    df = to_frame(results)
    assert df.loc[[t.trade_id for t in portfolio], "error"].isna().all()
    assert np.isfinite(df.loc[[t.trade_id for t in portfolio], "value"]).all()


def test_failing_trade_does_not_raise(flat_mkt: MarketData) -> None:
    # Battery without hourly prices fails inside the model; the engine reports, not raises.
    from enerq.instruments import BatteryStorage

    r = price(BatteryStorage("b"), flat_mkt)
    assert r.error is not None and np.isnan(r.value)


def test_parallel_equals_serial(mkt: MarketData, portfolio: list[Instrument]) -> None:
    serial = to_frame(price_many(portfolio, mkt, FAST, workers=1))
    parallel = to_frame(price_many(portfolio, mkt, FAST, workers=2))
    np.testing.assert_allclose(serial.value, parallel.value, rtol=1e-12)


def test_engine_european_matches_black(flat_mkt: MarketData) -> None:
    opt = EuropeanOption("e", notional=100, strike=42, expiry=0.5, option_type="put")
    expected = 100 * float(black76.price(40, 42, 0.5, 0.5, flat_mkt.df(0.5), False))
    assert price(opt, flat_mkt).value == pytest.approx(expected)


def test_risk_report_against_analytic(flat_mkt: MarketData) -> None:
    opt = EuropeanOption("e", strike=40, expiry=1.0)
    rep = risk_report([opt], flat_mkt, price_bump=0.01, vol_bump=1e-4)
    g = black76.greeks(40, 40, 1.0, 0.5, flat_mkt.rate)
    assert rep.loc["e", "delta"] == pytest.approx(float(g.delta), rel=1e-6)
    assert rep.loc["e", "gamma"] == pytest.approx(float(g.gamma), rel=1e-3)
    assert rep.loc["e", "vega_1pt"] == pytest.approx(0.01 * float(g.vega), rel=1e-5)
    ladder = rep.filter(like="delta_").loc["e"]
    assert ladder.sum() == pytest.approx(rep.loc["e", "delta"], rel=1e-6)  # buckets add up


def test_bucketed_delta_lands_on_right_pillars() -> None:
    mkt = MarketData(ForwardCurve([0.25, 0.5, 1.0], [40, 40, 40]), VolTermStructure.flat(0.4))
    opt = EuropeanOption("e", strike=40, expiry=0.5)  # forward exactly on the 0.5y pillar
    rep = risk_report([opt], mkt)
    assert rep.loc["e", "delta_0.25y"] == pytest.approx(0.0, abs=1e-12)
    assert rep.loc["e", "delta_1y"] == pytest.approx(0.0, abs=1e-12)
    assert rep.loc["e", "delta_0.5y"] == pytest.approx(rep.loc["e", "delta"])


def test_mc_greeks_stable_with_common_random_numbers(flat_mkt: MarketData) -> None:
    from enerq.instruments import AsianOption

    opt = AsianOption("a", strike=40, fixings=tuple(np.linspace(0.4, 0.5, 10)))
    cfg = EngineConfig(asian_method="monte_carlo", mc_paths=20_000)
    mc = risk_report([opt], flat_mkt, cfg).loc["a"]
    mm = risk_report([opt], flat_mkt).loc["a"]
    assert mc.delta == pytest.approx(mm.delta, rel=5e-3)


def test_scenario_grid_shape_and_zero(flat_mkt: MarketData) -> None:
    opt = EuropeanOption("e", strike=40, expiry=1.0)
    grid = scenario_grid([opt], flat_mkt, [-5, 0, 5], [-0.1, 0, 0.1])
    assert grid.shape == (3, 3)
    assert grid.loc[0, 0] == pytest.approx(0.0, abs=1e-12)
    assert grid.loc[0, 5] > 0 > grid.loc[0, -5]  # long call gains when prices rise
