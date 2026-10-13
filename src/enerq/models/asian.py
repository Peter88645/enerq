"""Arithmetic-average (Asian) options on energy forwards.

Model: one-factor lognormal. The fixing at time t_i is the forward F_i = F(t_i) observed at
t_i, all fixings are driven by a single Brownian motion, and the log-covariance between two
fixings is the total variance up to the earlier one:

    Cov(ln S_i, ln S_j) = w(min(t_i, t_j)),     w(t) = sigma(t)^2 t

Two pricers are provided:

* ``moment_matching`` - closed form. Matches the first two moments of the average to a
  lognormal (Levy / Turnbull-Wakeman) and prices with Black-76.
* ``monte_carlo`` - exact simulation of the fixings with a geometric-average control variate
  (whose price is known in closed form), optionally with randomised Sobol points.

The MC is the benchmark for the moment-matching approximation; their difference is the
model-approximation error and is checked in the test suite.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from scipy.special import ndtri
from scipy.stats import qmc

from enerq._types import FloatArray
from enerq.instruments import AsianOption, OptionType
from enerq.market import MarketData
from enerq.models import black76


@dataclass(frozen=True, slots=True)
class _Setup:
    """Pre-computed quantities shared by both pricers."""

    t: FloatArray  # future fixing times
    F: FloatArray  # forwards for future fixings
    w: FloatArray  # total variance to each future fixing
    weight: float  # 1 / total number of fixings
    known: float  # contribution of past fixings to the average
    strike_eff: float  # strike net of the known part
    df: float
    is_call: bool
    payment: float = field(default=0.0)


def _setup(inst: AsianOption, mkt: MarketData) -> _Setup:
    fixings = np.asarray(inst.fixings)
    future = fixings > 0
    t = fixings[future]
    weight = 1.0 / fixings.size
    known = weight * float(np.sum(inst.past_fixings))
    return _Setup(
        t=t,
        F=mkt.curve(t),
        w=mkt.vol.total_variance(t),
        weight=weight,
        known=known,
        strike_eff=inst.strike - known,
        df=float(mkt.df(inst.payment_time)),
        is_call=inst.option_type is OptionType.CALL,
        payment=inst.payment_time,
    )


def _degenerate(s: _Setup) -> float | None:
    """Handle cases with no remaining optionality."""
    m1 = s.weight * float(s.F.sum())
    if s.t.size == 0:
        # Fully fixed: the average equals the known part, so avg - K = -strike_eff.
        sign = 1.0 if s.is_call else -1.0
        return s.df * max(-sign * s.strike_eff, 0.0)
    if s.strike_eff <= 0:
        # The known part alone already exceeds the strike: call is a forward, put is worthless.
        return s.df * (m1 - s.strike_eff) if s.is_call else 0.0
    return None


def moment_matching(inst: AsianOption, mkt: MarketData) -> float:
    """Closed-form lognormal moment-matching price."""
    s = _setup(inst, mkt)
    if (shortcut := _degenerate(s)) is not None:
        return shortcut
    m1 = s.weight * s.F.sum()
    w_min = np.minimum.outer(s.w, s.w)  # w(min(t_i, t_j)) since w is non-decreasing
    m2 = s.weight**2 * float(s.F @ np.exp(w_min) @ s.F)
    total_var = max(np.log(m2 / m1**2), 0.0)
    return float(black76.price(m1, s.strike_eff, 1.0, np.sqrt(total_var), s.df, s.is_call))


@dataclass(frozen=True, slots=True)
class MCResult:
    price: float
    stderr: float
    n_paths: int
    control_variate_beta: float | None


def _normals(n_paths: int, dim: int, rng: np.random.Generator, method: str) -> FloatArray:
    if method == "sobol":
        m = int(np.ceil(np.log2(max(n_paths, 2))))
        u = qmc.Sobol(d=dim, scramble=True, seed=rng).random_base2(m)[:n_paths]
        return np.asarray(ndtri(np.clip(u, 1e-16, 1 - 1e-16)), dtype=np.float64)
    return rng.standard_normal((n_paths, dim))


def monte_carlo(
    inst: AsianOption,
    mkt: MarketData,
    *,
    n_paths: int = 100_000,
    seed: int | None = 42,
    method: Literal["pseudo", "sobol"] = "sobol",
    control_variate: bool = True,
    antithetic: bool = True,
    n_batches: int = 8,
) -> MCResult:
    """Monte Carlo price with optional geometric control variate.

    The standard error is estimated from ``n_batches`` independent batches (independent
    scrambles for Sobol), which is valid for both pseudo-random and randomised QMC.
    """
    s = _setup(inst, mkt)
    if (shortcut := _degenerate(s)) is not None:
        return MCResult(shortcut, 0.0, 0, None)

    rng = np.random.default_rng(seed)
    sign = 1.0 if s.is_call else -1.0
    dim = s.t.size
    dw = np.diff(s.w, prepend=0.0)  # variance increments between fixings
    vol_inc = np.sqrt(dw)
    drift = np.log(s.F) - 0.5 * s.w

    # Control variate: G = k * geometric mean of the future fixings, where k is the share of the
    # average still to fix. ln(G / k) is Gaussian, so E[(G - K')^+] is a Black-76 price.
    k = s.weight * dim
    g_mu = float(drift.mean())
    g_var = float(np.mean(np.minimum.outer(s.w, s.w)))
    g_fwd = k * np.exp(g_mu + 0.5 * g_var)
    g_price = float(black76.price(g_fwd, s.strike_eff, 1.0, np.sqrt(g_var), 1.0, s.is_call))

    per_batch = max(n_paths // n_batches, 2)
    estimates, betas = [], []
    for _ in range(n_batches):
        half = per_batch // 2 if antithetic else per_batch
        z = _normals(half, dim, rng, method)
        if antithetic:
            z = np.vstack([z, -z])
        log_s = drift + np.cumsum(z * vol_inc, axis=1)  # exact: shared Brownian motion
        arith = s.weight * np.exp(log_s).sum(axis=1)
        payoff = np.maximum(sign * (arith - s.strike_eff), 0.0)
        if control_variate:
            geo = k * np.exp(log_s.mean(axis=1))
            g_pay = np.maximum(sign * (geo - s.strike_eff), 0.0)
            cov = np.cov(payoff, g_pay)
            beta = cov[0, 1] / cov[1, 1] if cov[1, 1] > 0 else 0.0
            payoff = payoff - beta * (g_pay - g_price)
            betas.append(beta)
        estimates.append(payoff.mean())

    est = np.asarray(estimates)
    price = s.df * float(est.mean())
    stderr = s.df * float(est.std(ddof=1) / np.sqrt(n_batches))
    beta_avg = float(np.mean(betas)) if betas else None
    return MCResult(price, stderr, per_batch * n_batches, beta_avg)
