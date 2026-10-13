"""Immutable market data containers.

All times are year fractions measured from the valuation date. Objects are frozen so they can be
shared safely across threads/processes and used as the base for bump-and-revalue risk: every
"bump" returns a *new* object and never mutates the original.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from enerq._types import FloatArray, FloatLike


def _frozen_array(values: FloatLike, name: str) -> FloatArray:
    arr = np.array(values, dtype=np.float64, copy=True).reshape(-1)
    if arr.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite")
    arr.setflags(write=False)
    return arr


@dataclass(frozen=True, slots=True, eq=False)
class ForwardCurve:
    """Piecewise-linear forward curve F(t) with flat extrapolation."""

    tenors: FloatArray
    prices: FloatArray

    def __post_init__(self) -> None:
        tenors = _frozen_array(self.tenors, "tenors")
        prices = _frozen_array(self.prices, "prices")
        if tenors.shape != prices.shape:
            raise ValueError("tenors and prices must have the same length")
        if np.any(np.diff(tenors) <= 0):
            raise ValueError("tenors must be strictly increasing")
        object.__setattr__(self, "tenors", tenors)
        object.__setattr__(self, "prices", prices)

    def __call__(self, t: FloatLike) -> FloatArray:
        return np.interp(np.asarray(t, dtype=np.float64), self.tenors, self.prices)

    def shifted(self, shift: float, pillar: int | None = None) -> ForwardCurve:
        """Additive shift of the whole curve, or of a single pillar (bucketed delta)."""
        prices = self.prices.copy()
        if pillar is None:
            prices += shift
        else:
            prices[pillar] += shift
        return ForwardCurve(self.tenors, prices)

    def to_dict(self) -> dict[str, list[float]]:
        return {"tenors": self.tenors.tolist(), "prices": self.prices.tolist()}


@dataclass(frozen=True, slots=True, eq=False)
class VolTermStructure:
    """ATM implied-vol term structure, interpolated linearly in total variance w(t) = sigma^2 t."""

    tenors: FloatArray
    vols: FloatArray

    def __post_init__(self) -> None:
        tenors = _frozen_array(self.tenors, "tenors")
        vols = _frozen_array(self.vols, "vols")
        if tenors.shape != vols.shape:
            raise ValueError("tenors and vols must have the same length")
        if np.any(tenors <= 0) or np.any(np.diff(tenors) <= 0):
            raise ValueError("tenors must be positive and strictly increasing")
        if np.any(vols < 0):
            raise ValueError("vols must be non-negative")
        if np.any(np.diff(vols**2 * tenors) < -1e-14):
            raise ValueError("total variance must be non-decreasing (calendar arbitrage)")
        object.__setattr__(self, "tenors", tenors)
        object.__setattr__(self, "vols", vols)

    @classmethod
    def flat(cls, vol: float) -> VolTermStructure:
        return cls(np.array([1.0]), np.array([vol]))

    def total_variance(self, t: FloatLike) -> FloatArray:
        t = np.maximum(np.asarray(t, dtype=np.float64), 0.0)
        w_pillars = self.vols**2 * self.tenors
        # Linear in w between pillars, w = sigma_0^2 t before the first pillar, flat vol after.
        w = np.interp(t, self.tenors, w_pillars)
        before = t < self.tenors[0]
        after = t > self.tenors[-1]
        w = np.where(before, self.vols[0] ** 2 * t, w)
        return np.where(after, self.vols[-1] ** 2 * t, w)

    def vol(self, t: FloatLike) -> FloatArray:
        t = np.asarray(t, dtype=np.float64)
        w = self.total_variance(t)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(t > 0, np.sqrt(w / np.where(t > 0, t, 1.0)), self.vols[0])

    def shifted(self, shift: float) -> VolTermStructure:
        return VolTermStructure(self.tenors, np.maximum(self.vols + shift, 0.0))

    def to_dict(self) -> dict[str, list[float]]:
        return {"tenors": self.tenors.tolist(), "vols": self.vols.tolist()}


@dataclass(frozen=True, slots=True, eq=False)
class MarketData:
    """Snapshot of everything a pricer may need.

    ``mean_reversion`` and ``spot_vol`` parameterise the one-factor spot model used by the
    storage pricer; they are ignored by the forward-based models.
    """

    curve: ForwardCurve
    vol: VolTermStructure
    rate: float = 0.0
    mean_reversion: float = 5.0
    spot_vol: float = 0.6
    hourly_prices: FloatArray | None = field(default=None)

    def __post_init__(self) -> None:
        if self.mean_reversion <= 0:
            raise ValueError("mean_reversion must be positive")
        if self.spot_vol < 0:
            raise ValueError("spot_vol must be non-negative")
        if self.hourly_prices is not None:
            object.__setattr__(
                self, "hourly_prices", _frozen_array(self.hourly_prices, "hourly_prices")
            )

    def df(self, t: FloatLike) -> FloatArray:
        return np.exp(-self.rate * np.asarray(t, dtype=np.float64))

    def with_curve(self, curve: ForwardCurve) -> MarketData:
        return replace(self, curve=curve)

    def with_vol(self, vol: VolTermStructure) -> MarketData:
        return replace(self, vol=vol)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> MarketData:
        return cls(
            curve=ForwardCurve(**d["curve"]),
            vol=VolTermStructure(**d["vol"]),
            rate=float(d.get("rate", 0.0)),
            mean_reversion=float(d.get("mean_reversion", 5.0)),
            spot_vol=float(d.get("spot_vol", 0.6)),
            hourly_prices=d.get("hourly_prices"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "curve": self.curve.to_dict(),
            "vol": self.vol.to_dict(),
            "rate": self.rate,
            "mean_reversion": self.mean_reversion,
            "spot_vol": self.spot_vol,
            "hourly_prices": None if self.hourly_prices is None else self.hourly_prices.tolist(),
        }
