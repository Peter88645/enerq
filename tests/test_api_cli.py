from __future__ import annotations

import json
from pathlib import Path

import pytest

from enerq.cli import main

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from enerq.api import app  # noqa: E402

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
client = TestClient(app)


def _payload() -> dict:
    market = json.loads((EXAMPLES / "market.json").read_text())
    trades = [
        t
        for t in json.loads((EXAMPLES / "portfolio.json").read_text())["trades"]
        if t["type"] in {"european", "asian"}
    ]
    return {"market": market, "trades": trades}


def test_health() -> None:
    assert client.get("/health").json()["status"] == "ok"


def test_price_endpoint() -> None:
    r = client.post("/price", json=_payload())
    assert r.status_code == 200
    body = r.json()
    assert len(body) == 4 and all(row["error"] is None for row in body)


def test_risk_endpoint() -> None:
    r = client.post("/risk", json=_payload())
    assert r.status_code == 200
    assert "TOTAL" in r.json()


def test_bad_trade_is_422() -> None:
    payload = _payload()
    payload["trades"] = [{"type": "european", "trade_id": "x", "expiry": -1}]
    assert client.post("/price", json=payload).status_code == 422


def test_cli_price(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    out = tmp_path / "pv.csv"
    rc = main(
        [
            "price",
            str(EXAMPLES / "portfolio.json"),
            "--market",
            str(EXAMPLES / "market.json"),
            "--csv",
            str(out),
        ]
    )
    assert rc == 0
    assert "STORAGE-1Y" in capsys.readouterr().out
    assert out.read_text().count("\n") == 7
