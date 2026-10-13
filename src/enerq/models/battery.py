"""Battery energy storage (BESS) dispatch valuation as a mixed-integer programme.

Decision variables per hour t (all 1h periods, so MW == MWh per period):

    c_t  charge power        0 <= c_t <= P
    d_t  discharge power     0 <= d_t <= P
    s_t  state of charge     E_min <= s_t <= E_max   (end of hour)
    u_t  charging mode       binary, prevents simultaneous charge/discharge

Constraints:

    s_t = s_{t-1} + eta_c c_t - d_t / eta_d          energy balance
    c_t <= P u_t,  d_t <= P (1 - u_t)                 mode exclusivity (optional)
    |(d_t - c_t) - (d_{t-1} - c_{t-1})| <= R          ramp limit (optional)
    sum_t d_t / eta_d <= N * E                        cycling / throughput limit (optional)
    s_{H-1} = s_terminal                              terminal state of charge (optional)

Objective: maximise sum_t p_t (d_t - c_t) - degradation * d_t.

The model is assembled with sparse matrices and solved with HiGHS via ``scipy.optimize.milp``.
:func:`constraint_attribution` re-solves with each optional constraint relaxed to quantify its
value impact - the standard way to explain a battery valuation to a desk.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import Bounds, LinearConstraint, milp

from enerq._types import FloatArray
from enerq.instruments import BatteryStorage
from enerq.market import MarketData


class InfeasibleDispatchError(RuntimeError):
    """Raised when the dispatch problem has no feasible solution."""


@dataclass(frozen=True, slots=True)
class BatteryResult:
    value: float
    schedule: pd.DataFrame  # price, charge, discharge, soc per hour
    full_cycles: float
    mip_gap: float

    @property
    def n_hours(self) -> int:
        return len(self.schedule)


def _prices(mkt: MarketData) -> FloatArray:
    if mkt.hourly_prices is None:
        raise ValueError("BatteryStorage valuation needs MarketData.hourly_prices")
    return mkt.hourly_prices


def optimise_dispatch(
    inst: BatteryStorage,
    mkt: MarketData,
    *,
    time_limit: float = 30.0,
    mip_rel_gap: float = 1e-6,
) -> BatteryResult:
    """Solve the optimal dispatch and return value and hourly schedule."""
    p = _prices(mkt)
    H = p.size
    P, E = inst.power_mw, inst.energy_mwh
    eta_c, eta_d = inst.charge_eff, inst.discharge_eff
    use_u = inst.no_simultaneous
    n = (4 if use_u else 3) * H
    ic, id_, is_, iu = 0, H, 2 * H, 3 * H
    eye = sparse.identity(H, format="csr")
    shift = sparse.eye(H, k=-1, format="csr")  # (shift @ x)_t = x_{t-1}

    def block(**parts: sparse.spmatrix) -> sparse.csr_matrix:
        """Row block [A_c | A_d | A_s | A_u], filling unspecified variable groups with zeros."""
        m = next(iter(parts.values())).shape[0]
        cols = [parts.get(k, sparse.csr_matrix((m, H))) for k in ("c", "d", "s", "u")[: n // H]]
        return sparse.hstack(cols, format="csr")

    rows: list[sparse.spmatrix] = []
    lbs: list[FloatArray] = []
    ubs: list[FloatArray] = []

    def add(a: sparse.spmatrix, lb: FloatArray | float, ub: FloatArray | float) -> None:
        rows.append(a)
        lbs.append(np.broadcast_to(np.asarray(lb, dtype=float), (a.shape[0],)))
        ubs.append(np.broadcast_to(np.asarray(ub, dtype=float), (a.shape[0],)))

    # Energy balance: s_t - s_{t-1} - eta_c c_t + d_t / eta_d = 0  (s_{-1} = initial)
    rhs = np.zeros(H)
    rhs[0] = inst.initial_soc * E
    add(block(c=-eta_c * eye, d=eye / eta_d, s=eye - shift), rhs, rhs)

    if use_u:
        add(block(c=eye, u=-P * eye), -np.inf, 0.0)  # c_t <= P u_t
        add(block(d=eye, u=P * eye), -np.inf, P)  # d_t <= P (1 - u_t)

    if inst.ramp_mw_per_h is not None and H > 1:
        diff = (eye - shift)[1:]  # net_t - net_{t-1} for t >= 1
        R = inst.ramp_mw_per_h
        add(block(c=-diff, d=diff), -R, R)

    if inst.max_cycles is not None:
        throughput = sparse.csr_matrix(
            np.r_[np.zeros(H), np.full(H, 1 / eta_d), np.zeros(n - 2 * H)]
        )
        add(throughput, -np.inf, inst.max_cycles * E)

    lb = np.zeros(n)
    ub = np.full(n, P)
    lb[is_ : is_ + H], ub[is_ : is_ + H] = inst.min_soc * E, inst.max_soc * E
    if inst.terminal_soc is not None:
        lb[is_ + H - 1] = ub[is_ + H - 1] = inst.terminal_soc * E
    integrality = np.zeros(n)
    if use_u:
        ub[iu:] = 1.0
        integrality[iu:] = 1

    cost = np.zeros(n)
    cost[ic : ic + H] = p
    cost[id_ : id_ + H] = -p + inst.degradation_cost

    res = milp(
        cost,
        constraints=LinearConstraint(
            sparse.vstack(rows, format="csr"), np.concatenate(lbs), np.concatenate(ubs)
        ),
        bounds=Bounds(lb, ub),
        integrality=integrality,
        options={"time_limit": time_limit, "mip_rel_gap": mip_rel_gap, "disp": False},
    )
    if res.x is None:
        raise InfeasibleDispatchError(f"{inst.trade_id}: {res.message}")

    x = res.x
    charge, discharge, soc = x[ic : ic + H], x[id_ : id_ + H], x[is_ : is_ + H]
    schedule = pd.DataFrame(
        {"price": p, "charge_mw": charge, "discharge_mw": discharge, "soc_mwh": soc},
        index=pd.RangeIndex(H, name="hour"),
    )
    gap = float(getattr(res, "mip_gap", 0.0) or 0.0)
    return BatteryResult(
        value=-float(res.fun) * inst.notional,
        schedule=schedule.round(10),
        full_cycles=float(discharge.sum() / eta_d / E),
        mip_gap=gap,
    )


_RELAXATIONS: dict[str, dict[str, object]] = {
    "cycling_limit": {"max_cycles": None},
    "ramp_limit": {"ramp_mw_per_h": None},
    "terminal_soc": {"terminal_soc": None},
    "no_simultaneous": {"no_simultaneous": False},
}


def constraint_attribution(inst: BatteryStorage, mkt: MarketData) -> pd.DataFrame:
    """Value uplift from relaxing each active optional constraint, one at a time.

    A large uplift flags a binding constraint whose specification deserves scrutiny; a zero
    uplift shows a constraint that does not bite under the current price shape.
    """
    base = optimise_dispatch(inst, mkt).value
    records = []
    for name, change in _RELAXATIONS.items():
        current = {k: getattr(inst, k) for k in change}
        if current == change:
            continue  # already relaxed
        relaxed = optimise_dispatch(replace(inst, **change), mkt).value  # type: ignore[arg-type]
        records.append({"constraint": name, "relaxed_value": relaxed, "uplift": relaxed - base})
    df = pd.DataFrame(records, columns=["constraint", "relaxed_value", "uplift"])
    df.attrs["base_value"] = base
    return df.sort_values("uplift", ascending=False, ignore_index=True)
