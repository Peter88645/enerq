"""Vectorised Black-76 pricing, analytic Greeks and implied volatility.

Every function broadcasts over its inputs with NumPy, so pricing a 10^6-option grid is a single
call with no Python loop. Degenerate inputs (zero vol or zero time) fall back to discounted
intrinsic value instead of producing NaNs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import ndtr

from enerq._types import FloatArray, FloatLike

_SQRT_2PI = np.sqrt(2.0 * np.pi)


def _pdf(x: FloatArray) -> FloatArray:
    return np.asarray(np.exp(-0.5 * x * x) / _SQRT_2PI, dtype=np.float64)


def _sign(is_call: FloatLike) -> FloatArray:
    return np.where(np.asarray(is_call, dtype=bool), 1.0, -1.0)


def _d1_d2(
    F: FloatArray, K: FloatArray, T: FloatArray, sigma: FloatArray
) -> tuple[FloatArray, FloatArray, FloatArray]:
    std = sigma * np.sqrt(np.maximum(T, 0.0))
    safe = np.where(std > 0, std, 1.0)
    with np.errstate(divide="ignore"):
        d1 = (np.log(F / K) + 0.5 * std * std) / safe
    return d1, d1 - std, std


def price(
    F: FloatLike,
    K: FloatLike,
    T: FloatLike,
    sigma: FloatLike,
    df: FloatLike = 1.0,
    is_call: FloatLike = True,
) -> FloatArray:
    """Black-76 price of a European option on a forward."""
    F, K, T, sigma, df = (np.asarray(x, dtype=np.float64) for x in (F, K, T, sigma, df))
    w = _sign(is_call)
    d1, d2, std = _d1_d2(F, K, T, sigma)
    pv = df * w * (F * ndtr(w * d1) - K * ndtr(w * d2))
    intrinsic = df * np.maximum(w * (F - K), 0.0)
    return np.asarray(np.where(std > 0, pv, intrinsic), dtype=np.float64)


@dataclass(frozen=True, slots=True)
class Greeks:
    price: FloatArray
    delta: FloatArray  # dV/dF
    gamma: FloatArray  # d2V/dF2
    vega: FloatArray  # dV/dsigma (per 1.00 vol, i.e. 100 vol points)
    theta: FloatArray  # dV/dt  (calendar time, discounting held at fixed rate)


def greeks(
    F: FloatLike,
    K: FloatLike,
    T: FloatLike,
    sigma: FloatLike,
    rate: FloatLike = 0.0,
    is_call: FloatLike = True,
) -> Greeks:
    """Analytic Black-76 Greeks with continuous discounting ``df = exp(-rate * T)``."""
    F, K, T, sigma, r = (np.asarray(x, dtype=np.float64) for x in (F, K, T, sigma, rate))
    w = _sign(is_call)
    df = np.exp(-r * T)
    d1, _, std = _d1_d2(F, K, T, sigma)
    live = std > 0
    sqrtT = np.sqrt(np.maximum(T, 0.0))
    safe_std = np.where(live, std, 1.0)

    pv = price(F, K, T, sigma, df, is_call)
    delta = np.where(live, df * w * ndtr(w * d1), df * w * (w * (F - K) > 0))
    gamma = np.where(live, df * _pdf(d1) / (F * safe_std), 0.0)
    vega = np.where(live, df * F * _pdf(d1) * sqrtT, 0.0)
    # Theta = -dV/dT: time decay of optionality plus the carry of the discount factor.
    safe_sqrtT = np.where(sqrtT > 0, sqrtT, 1.0)
    theta = np.where(live, -df * F * _pdf(d1) * sigma / (2.0 * safe_sqrtT) + r * pv, r * pv)
    return Greeks(pv, delta, gamma, vega, theta)


def implied_vol(
    target: FloatLike,
    F: FloatLike,
    K: FloatLike,
    T: FloatLike,
    df: FloatLike = 1.0,
    is_call: FloatLike = True,
    *,
    tol: float = 1e-12,
    max_iter: int = 100,
    vol_upper: float = 10.0,
) -> FloatArray:
    """Vectorised implied vol: safeguarded Newton with a bisection bracket.

    Black price is monotone in sigma, so each element keeps a bracket ``[lo, hi]`` that always
    contains the root. A Newton step is accepted only if it lands inside the bracket; otherwise
    we bisect. This converges quadratically in the usual case and never diverges in the wings.
    Prices outside the no-arbitrage bounds return NaN.
    """
    arrays = np.broadcast_arrays(
        *(np.asarray(x, dtype=np.float64) for x in (target, F, K, T, df)),
        np.asarray(is_call, dtype=bool),
    )
    shape = arrays[0].shape
    target, F, K, T, df, is_call = (np.array(a, ndmin=1).reshape(-1) for a in arrays)
    w = _sign(is_call)
    lower = df * np.maximum(w * (F - K), 0.0)
    upper = np.where(is_call, df * F, df * K)
    valid = (target >= lower - tol) & (target < upper) & (T > 0)
    # Invert the out-of-the-money option (via put-call parity): the ITM price is dominated by
    # intrinsic value, so its time value - the only part that depends on sigma - is better
    # conditioned when read off the OTM side.
    itm = w * (F - K) > 0
    target = np.where(itm, target - df * w * (F - K), target)
    is_call = np.where(itm, ~is_call, is_call)

    lo = np.zeros_like(target)
    hi = np.full_like(target, vol_upper)
    # Brenner-Subrahmanyam style initial guess, clipped into the bracket.
    sigma = np.clip(
        np.sqrt(2.0 * np.pi / np.maximum(T, 1e-12)) * target / np.maximum(df * F, 1e-12),
        1e-4,
        vol_upper / 2,
    )
    active = valid.copy()
    for _ in range(max_iter):
        if not active.any():
            break
        s, lo_a, hi_a = sigma[active], lo[active], hi[active]
        Fa, Ka, Ta, dfa = F[active], K[active], T[active], df[active]
        wa = np.where(is_call[active], 1.0, -1.0)
        d1, d2, std = _d1_d2(Fa, Ka, Ta, s)
        model = dfa * wa * (Fa * ndtr(wa * d1) - Ka * ndtr(wa * d2))
        model = np.where(std > 0, model, dfa * np.maximum(wa * (Fa - Ka), 0.0))
        diff = model - target[active]
        vega = dfa * Fa * _pdf(d1) * np.sqrt(Ta)
        lo_a = np.where(diff < 0, s, lo_a)
        hi_a = np.where(diff > 0, s, hi_a)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            newton = s - diff / vega
        ok = np.isfinite(newton) & (newton >= lo_a) & (newton <= hi_a)
        # Converged when the (OTM) price matches to relative precision ``tol``; keep the current
        # sigma in that case - a further bisection step would throw it away.
        hit = np.abs(diff) <= tol * np.maximum(target[active], 1e-300)
        s_new = np.where(hit, s, np.where(ok, newton, 0.5 * (lo_a + hi_a)))
        sigma[active], lo[active], hi[active] = s_new, lo_a, hi_a
        done = hit | (np.abs(s_new - s) < tol) | (hi_a - lo_a < tol)
        idx = np.flatnonzero(active)
        active[idx[done]] = False
    return np.where(valid, sigma, np.nan).reshape(shape)
