from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from enerq.models import black76

prices = st.floats(1.0, 500.0)
vols = st.floats(0.05, 2.0)
times = st.floats(0.01, 5.0)
rates = st.floats(-0.01, 0.10)


@given(F=prices, K=prices, T=times, sigma=vols, r=rates)
def test_put_call_parity(F: float, K: float, T: float, sigma: float, r: float) -> None:
    df = np.exp(-r * T)
    call = black76.price(F, K, T, sigma, df, True)
    put = black76.price(F, K, T, sigma, df, False)
    assert call - put == pytest.approx(df * (F - K), abs=1e-9 * max(F, K))


@given(F=prices, K=prices, T=times, sigma=vols)
def test_no_arbitrage_bounds(F: float, K: float, T: float, sigma: float) -> None:
    c = float(black76.price(F, K, T, sigma))
    assert max(F - K, 0.0) - 1e-9 <= c <= F + 1e-9


@settings(max_examples=200)
@given(F=prices, K=prices, T=times, sigma=vols, r=rates, call=st.booleans())
def test_greeks_match_finite_differences(
    F: float, K: float, T: float, sigma: float, r: float, call: bool
) -> None:
    g = black76.greeks(F, K, T, sigma, r, call)
    df = np.exp(-r * T)
    hF, hs, hT = 1e-4 * F, 1e-5, 1e-6

    def p(F_: float = F, s_: float = sigma, T_: float = T) -> float:
        return float(black76.price(F_, K, T_, s_, np.exp(-r * T_), call))

    assert g.delta == pytest.approx((p(F + hF) - p(F - hF)) / (2 * hF), rel=1e-5, abs=1e-6)
    assert g.vega == pytest.approx(
        (p(s_=sigma + hs) - p(s_=sigma - hs)) / (2 * hs), rel=1e-5, abs=1e-6
    )
    assert g.theta == pytest.approx(-(p(T_=T + hT) - p(T_=T - hT)) / (2 * hT), rel=1e-4, abs=1e-5)
    hg = 1e-2 * F * min(1.0, sigma * np.sqrt(T))  # step well inside the price distribution
    gamma_fd = (p(F + hg) - 2 * p() + p(F - hg)) / hg**2
    # Second differences lose ~eps * price / h^2 to rounding; deep ITM prices are large.
    rounding = 1e-14 * max(F, K) / hg**2
    assert g.gamma == pytest.approx(gamma_fd, rel=1e-3, abs=rounding + 1e-9 * df)


def test_degenerate_inputs_give_intrinsic() -> None:
    assert black76.price(50.0, 40.0, 0.0, 0.3) == pytest.approx(10.0)
    assert black76.price(50.0, 40.0, 1.0, 0.0, 0.9, False) == pytest.approx(0.0)
    g = black76.greeks(50.0, 40.0, 0.0, 0.3)
    assert np.isfinite([g.delta, g.gamma, g.vega, g.theta]).all()


def test_vectorised_broadcasting() -> None:
    F = np.linspace(20, 60, 50)[:, None]
    K = np.linspace(20, 60, 30)[None, :]
    out = black76.price(F, K, 1.0, 0.4, 0.97, True)
    assert out.shape == (50, 30)
    assert np.all(np.diff(out, axis=1) <= 1e-12)  # call price decreasing in strike


@settings(max_examples=300)
@given(F=prices, m=st.floats(0.5, 2.0), T=times, sigma=vols, call=st.booleans())
def test_implied_vol_roundtrip(F: float, m: float, T: float, sigma: float, call: bool) -> None:
    K = F * m
    target = black76.price(F, K, T, sigma, 0.98, call)
    # Deep OTM options with price ~0 carry no vol information; skip them.
    if target < 1e-10 * F:
        return
    iv = black76.implied_vol(target, F, K, T, 0.98, call)
    assert float(black76.price(F, K, T, iv, 0.98, call)) == pytest.approx(
        float(target), rel=1e-8, abs=1e-10
    )


def test_implied_vol_vectorised_and_invalid() -> None:
    rng = np.random.default_rng(0)
    n = 10_000
    F = rng.uniform(20, 80, n)
    K = F * rng.uniform(0.7, 1.4, n)
    T = rng.uniform(0.05, 3, n)
    sigma = rng.uniform(0.1, 1.5, n)
    target = black76.price(F, K, T, sigma)
    iv = black76.implied_vol(target, F, K, T)
    time_value = target - np.maximum(F - K, 0.0)
    informative = time_value > 1e-8 * F  # time value ~1e-50 carries no vol information
    assert informative.mean() > 0.95
    assert np.max(np.abs(iv - sigma)[informative]) < 1e-8
    # Above the upper bound / below intrinsic -> NaN, not an exception.
    bad = black76.implied_vol([60.0, 1.0], [50.0, 50.0], [40.0, 40.0], [1.0, 1.0])
    assert np.isnan(bad).all()
