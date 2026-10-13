from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from enerq.cli import load_portfolio
from enerq.instruments import Instrument
from enerq.market import ForwardCurve, MarketData, VolTermStructure

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


@pytest.fixture(scope="session")
def mkt() -> MarketData:
    return MarketData.from_dict(json.loads((EXAMPLES / "market.json").read_text()))


@pytest.fixture(scope="session")
def portfolio() -> list[Instrument]:
    return load_portfolio(str(EXAMPLES / "portfolio.json"))


@pytest.fixture
def flat_mkt() -> MarketData:
    tenors = np.linspace(0.05, 2.0, 40)
    return MarketData(
        curve=ForwardCurve(tenors, np.full_like(tenors, 40.0)),
        vol=VolTermStructure.flat(0.5),
        rate=0.03,
    )
