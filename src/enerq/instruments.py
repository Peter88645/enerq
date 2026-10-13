"""Trade representations.

Instruments are plain frozen dataclasses: they describe *what* is traded and validate their own
terms, but know nothing about *how* they are priced (see :mod:`enerq.engine`). A ``type`` tag in
the serialised form lets portfolios round-trip through JSON.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any, ClassVar

import numpy as np


class OptionType(StrEnum):
    CALL = "call"
    PUT = "put"


@dataclass(frozen=True, slots=True)
class Instrument:
    trade_id: str
    notional: float = 1.0

    TYPE: ClassVar[str] = "instrument"

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["type"] = self.TYPE
        return d


@dataclass(frozen=True, slots=True)
class EuropeanOption(Instrument):
    """Option on a futures/forward contract delivering at ``delivery`` (Black-76)."""

    strike: float = 0.0
    expiry: float = 1.0
    delivery: float | None = None
    option_type: OptionType = OptionType.CALL

    TYPE: ClassVar[str] = "european"

    def __post_init__(self) -> None:
        object.__setattr__(self, "option_type", OptionType(self.option_type))
        if self.expiry <= 0:
            raise ValueError(f"{self.trade_id}: expiry must be positive")
        if self.delivery is not None and self.delivery < self.expiry:
            raise ValueError(f"{self.trade_id}: delivery must be on/after expiry")


@dataclass(frozen=True, slots=True)
class AsianOption(Instrument):
    """Arithmetic-average option, e.g. a monthly-average gas or power option.

    ``fixings`` are fixing times (years). Fixings at or before 0 are treated as already set and
    must have a matching entry in ``past_fixings``. ``payment`` defaults to the last fixing.
    """

    strike: float = 0.0
    fixings: tuple[float, ...] = ()
    option_type: OptionType = OptionType.CALL
    past_fixings: tuple[float, ...] = ()
    payment: float | None = None

    TYPE: ClassVar[str] = "asian"

    def __post_init__(self) -> None:
        object.__setattr__(self, "option_type", OptionType(self.option_type))
        object.__setattr__(self, "fixings", tuple(float(t) for t in self.fixings))
        object.__setattr__(self, "past_fixings", tuple(float(x) for x in self.past_fixings))
        if not self.fixings:
            raise ValueError(f"{self.trade_id}: at least one fixing required")
        if any(b <= a for a, b in zip(self.fixings, self.fixings[1:], strict=False)):
            raise ValueError(f"{self.trade_id}: fixings must be strictly increasing")
        n_past = sum(t <= 0 for t in self.fixings)
        if n_past != len(self.past_fixings):
            raise ValueError(
                f"{self.trade_id}: {n_past} fixings are in the past but "
                f"{len(self.past_fixings)} past_fixings were supplied"
            )

    @property
    def payment_time(self) -> float:
        return self.payment if self.payment is not None else max(self.fixings[-1], 0.0)


@dataclass(frozen=True, slots=True)
class GasStorage(Instrument):
    """Virtual gas storage contract valued by least-squares Monte Carlo.

    Volumes are in a common unit (e.g. MWh); rates are *per decision step*. All volumes must be
    integer multiples of ``volume_step``, which defines the inventory grid.
    """

    start: float = 0.0
    end: float = 1.0
    n_steps: int = 52
    capacity: float = 100.0
    initial: float = 0.0
    final: float = 0.0
    max_inject: float = 10.0
    max_withdraw: float = 10.0
    volume_step: float = 5.0
    inject_cost: float = 0.0
    withdraw_cost: float = 0.0

    TYPE: ClassVar[str] = "gas_storage"

    def __post_init__(self) -> None:
        if not 0 <= self.start < self.end:
            raise ValueError(f"{self.trade_id}: need 0 <= start < end")
        if self.n_steps < 1:
            raise ValueError(f"{self.trade_id}: n_steps must be >= 1")
        for name in ("capacity", "initial", "final", "max_inject", "max_withdraw"):
            value = getattr(self, name)
            ratio = value / self.volume_step
            if value < 0 or not np.isclose(ratio, round(ratio)):
                raise ValueError(
                    f"{self.trade_id}: {name} must be a non-negative multiple of "
                    f"volume_step={self.volume_step}"
                )
        if not (self.initial <= self.capacity and self.final <= self.capacity):
            raise ValueError(f"{self.trade_id}: initial/final inventory exceed capacity")
        steps_needed = abs(self.final - self.initial) / max(
            self.max_inject if self.final > self.initial else self.max_withdraw, 1e-12
        )
        if steps_needed > self.n_steps + 1e-9:
            raise ValueError(f"{self.trade_id}: final inventory unreachable")


@dataclass(frozen=True, slots=True)
class BatteryStorage(Instrument):
    """Battery energy storage system (BESS) tolling-style valuation over an hourly horizon.

    The optimal dispatch against ``MarketData.hourly_prices`` is solved as a mixed-integer
    programme. Optional constraints can be switched off (set to ``None``) to measure their value.
    """

    power_mw: float = 10.0
    energy_mwh: float = 20.0
    charge_eff: float = 0.95
    discharge_eff: float = 0.95
    initial_soc: float = 0.5  # fraction of energy_mwh
    terminal_soc: float | None = 0.5  # fraction; None = free
    max_cycles: float | None = None  # full-equivalent cycles over the horizon
    ramp_mw_per_h: float | None = None
    degradation_cost: float = 0.0  # currency per MWh discharged
    min_soc: float = 0.0
    max_soc: float = 1.0
    no_simultaneous: bool = field(default=True)

    TYPE: ClassVar[str] = "battery"

    def __post_init__(self) -> None:
        if self.power_mw <= 0 or self.energy_mwh <= 0:
            raise ValueError(f"{self.trade_id}: power and energy must be positive")
        for name in ("charge_eff", "discharge_eff"):
            if not 0 < getattr(self, name) <= 1:
                raise ValueError(f"{self.trade_id}: {name} must be in (0, 1]")
        if not 0 <= self.min_soc <= self.initial_soc <= self.max_soc <= 1:
            raise ValueError(f"{self.trade_id}: need 0 <= min_soc <= initial_soc <= max_soc <= 1")


INSTRUMENT_TYPES: dict[str, type[Instrument]] = {
    cls.TYPE: cls for cls in (EuropeanOption, AsianOption, GasStorage, BatteryStorage)
}


def instrument_from_dict(d: dict[str, Any]) -> Instrument:
    """Build an instrument from its serialised form (``{"type": ..., **fields}``)."""
    d = dict(d)
    try:
        cls = INSTRUMENT_TYPES[d.pop("type")]
    except KeyError as exc:
        raise ValueError(f"unknown or missing instrument type: {exc}") from None
    for key in ("fixings", "past_fixings"):
        if key in d:
            d[key] = tuple(d[key])
    return cls(**d)
