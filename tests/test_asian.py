from __future__ import annotations

import numpy as np
import pytest

from enerq.instruments import AsianOption
from enerq.market import MarketData
from enerq.models import asian, black76


def _daily(start: float, end: float, n: int = 22) -> tuple[float, ...]:
    return tuple(np.linspace(start, end, n))


@pytest.mark.parametrize("call", [True, False])
@pytest.mark.parametrize("strike", [32.0, 40.0, 48.0])
def test_moment_matching_close_to_monte_carlo(
    flat_mkt: MarketData, call: bool, strike: float
) -> None:
    opt = AsianOption(
        "a", strike=strike, fixings=_daily(0.5, 0.58), option_type="call" if call else "put"
    )
    mm = asian.moment_matching(opt, flat_mkt)
    mc = asian.monte_carlo(opt, flat_mkt, n_paths=200_000)
    # Lognormal moment matching is an approximation; for a one-month window it is tight.
    assert mm == pytest.approx(mc.price, abs=max(5 * mc.stderr, 2e-3 * mc.price + 1e-3))


def test_single_fixing_reduces_to_black(flat_mkt: MarketData) -> None:
    opt = AsianOption("a", strike=42.0, fixings=(0.75,))
    bl = float(black76.price(40.0, 42.0, 0.75, 0.5, flat_mkt.df(0.75)))
    assert asian.moment_matching(opt, flat_mkt) == pytest.approx(bl, rel=1e-12)
    assert asian.monte_carlo(opt, flat_mkt, n_paths=100_000).price == pytest.approx(bl, rel=2e-3)


def test_averaging_reduces_value(flat_mkt: MarketData) -> None:
    euro = AsianOption("e", strike=40.0, fixings=(1.0,))
    avg = AsianOption("a", strike=40.0, fixings=_daily(0.0 + 1e-3, 1.0, 250))
    assert asian.moment_matching(avg, flat_mkt) < asian.moment_matching(euro, flat_mkt)


def test_control_variate_reduces_error(flat_mkt: MarketData) -> None:
    opt = AsianOption("a", strike=40.0, fixings=_daily(0.2, 0.3))
    plain = asian.monte_carlo(opt, flat_mkt, n_paths=40_000, method="pseudo", control_variate=False)
    cv = asian.monte_carlo(opt, flat_mkt, n_paths=40_000, method="pseudo", control_variate=True)
    assert cv.stderr < plain.stderr / 10
    assert cv.control_variate_beta == pytest.approx(1.0, abs=0.1)


def test_partially_fixed(flat_mkt: MarketData) -> None:
    opt = AsianOption(
        "a",
        strike=40.0,
        fixings=(-0.02, -0.01, 0.1, 0.2),
        past_fixings=(44.0, 46.0),
        option_type="put",
    )
    mm = asian.moment_matching(opt, flat_mkt)
    sobol = asian.monte_carlo(opt, flat_mkt, n_paths=200_000, method="sobol")
    pseudo = asian.monte_carlo(opt, flat_mkt, n_paths=200_000, method="pseudo", seed=1)
    # The two MC estimators agree to within their errors...
    assert sobol.price == pytest.approx(pseudo.price, abs=5 * (sobol.stderr + pseudo.stderr))
    # ...while moment matching carries a visible approximation error (~1%) with only two widely
    # spaced remaining fixings and a low effective strike. This is the model-risk number a
    # validator would report against the closed form.
    assert mm == pytest.approx(sobol.price, rel=0.015)
    assert abs(mm - sobol.price) > 10 * sobol.stderr


def test_deep_in_the_money_known_part(flat_mkt: MarketData) -> None:
    # Known fixings alone exceed the strike: the call is a forward on the remaining average.
    opt = AsianOption("a", strike=10.0, fixings=(-0.1, 0.5), past_fixings=(30.0,))
    expected = float(flat_mkt.df(0.5)) * (0.5 * 30.0 + 0.5 * 40.0 - 10.0)
    assert asian.moment_matching(opt, flat_mkt) == pytest.approx(expected)
    assert (
        asian.moment_matching(
            AsianOption(
                "p", strike=10.0, fixings=(-0.1, 0.5), past_fixings=(30.0,), option_type="put"
            ),
            flat_mkt,
        )
        == 0.0
    )


def test_fully_fixed(flat_mkt: MarketData) -> None:
    opt = AsianOption(
        "a", strike=40.0, fixings=(-0.2, -0.1), past_fixings=(42.0, 44.0), payment=0.1
    )
    assert asian.moment_matching(opt, flat_mkt) == pytest.approx(3.0 * float(flat_mkt.df(0.1)))


def test_validation() -> None:
    with pytest.raises(ValueError, match="past_fixings"):
        AsianOption("a", strike=1.0, fixings=(-0.1, 0.2))
    with pytest.raises(ValueError, match="increasing"):
        AsianOption("a", strike=1.0, fixings=(0.3, 0.2))
