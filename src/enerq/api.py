"""Minimal pricing service (optional extra: ``pip install enerq[api]``).

    uvicorn enerq.api:app --workers 4

Request bodies are validated by Pydantic at the boundary and converted to the library's own
frozen dataclasses, so the core stays free of web-framework dependencies.
"""

from __future__ import annotations

from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from enerq import __version__
from enerq.engine import EngineConfig, price_many
from enerq.instruments import instrument_from_dict
from enerq.market import MarketData
from enerq.risk import risk_report


class CurveIn(BaseModel):
    tenors: list[float] = Field(min_length=1)
    prices: list[float] = Field(min_length=1)


class VolIn(BaseModel):
    tenors: list[float] = Field(min_length=1)
    vols: list[float] = Field(min_length=1)


class MarketIn(BaseModel):
    curve: CurveIn
    vol: VolIn
    rate: float = 0.0
    mean_reversion: float = 5.0
    spot_vol: float = 0.6
    hourly_prices: list[float] | None = None


class PriceRequest(BaseModel):
    market: MarketIn
    trades: list[dict[str, Any]] = Field(min_length=1, max_length=5_000)
    asian_method: Literal["moment_matching", "monte_carlo"] = "moment_matching"


class TradeResult(BaseModel):
    trade_id: str
    model: str
    value: float | None
    stderr: float
    error: str | None = None


app = FastAPI(title="enerq pricing service", version=__version__)


def _parse(req: PriceRequest) -> tuple[list[Any], MarketData, EngineConfig]:
    try:
        trades = [instrument_from_dict(t) for t in req.trades]
        mkt = MarketData.from_dict(req.market.model_dump())
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return trades, mkt, EngineConfig(asian_method=req.asian_method)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": __version__}


@app.post("/price", response_model=list[TradeResult])
def price_endpoint(req: PriceRequest) -> list[TradeResult]:
    trades, mkt, cfg = _parse(req)
    return [
        TradeResult(
            trade_id=r.trade_id,
            model=r.model,
            value=None if r.error else r.value,
            stderr=r.stderr,
            error=r.error,
        )
        for r in price_many(trades, mkt, cfg)
    ]


@app.post("/risk")
def risk_endpoint(req: PriceRequest) -> dict[str, dict[str, float]]:
    trades, mkt, cfg = _parse(req)
    try:
        report = risk_report(trades, mkt, cfg)
    except RuntimeError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {str(k): {c: float(v) for c, v in row.items()} for k, row in report.iterrows()}
