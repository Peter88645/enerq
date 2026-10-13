"""Gas storage valuation: deterministic intrinsic (DP) and least-squares Monte Carlo (LSMC).

Spot model (one-factor, mean-reverting, fitted to the forward curve)::

    S_t = F(t) * exp(X_t - Var(X_t) / 2),    dX = -kappa X dt + sigma dW,    X_0 = 0

so that E[S_t] = F(t) exactly. X is simulated with its exact Gaussian transition.

Inventory lives on a grid of ``volume_step`` increments. At each decision date the holder picks
an inventory change within the injection/withdrawal limits. Reachability of the required final
inventory is deterministic, so infeasible states are pruned once up-front rather than handled
with penalties inside the regression.

LSMC follows Longstaff-Schwartz: continuation values are *estimated* by regression (one
``lstsq`` call solves all inventory levels at once) but the value carried backwards is the
*realised* path value of the chosen action. The fitted policy is then replayed on an
independent set of paths, giving a low-biased estimate; the gap to the in-sample estimate is
a direct measure of regression (policy) error.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from enerq._types import FloatArray
from enerq.instruments import GasStorage
from enerq.market import MarketData


@dataclass(frozen=True, slots=True)
class StorageResult:
    value: float
    intrinsic: float
    extrinsic: float
    stderr: float
    in_sample: float  # high-biased estimate from the regression paths
    expected_inventory: FloatArray  # mean inventory path under the fitted policy


@dataclass(frozen=True, slots=True)
class _Grid:
    times: FloatArray  # decision times, length n_steps
    levels: FloatArray  # inventory grid
    moves: NDArray[np.int_]  # allowed inventory moves in grid units, e.g. [-2, -1, 0, 1]
    feasible: NDArray[np.bool_]  # (n_steps + 1, n_levels)
    start_idx: int
    step_df: float


def _grid(inst: GasStorage, mkt: MarketData) -> _Grid:
    dv = inst.volume_step
    n_levels = round(inst.capacity / dv) + 1
    levels = np.arange(n_levels, dtype=np.float64) * dv
    n_inj, n_wd = round(inst.max_inject / dv), round(inst.max_withdraw / dv)
    moves = np.arange(-n_wd, n_inj + 1)
    times = np.linspace(inst.start, inst.end, inst.n_steps, endpoint=False)
    dt = (inst.end - inst.start) / inst.n_steps

    # feasible[i, k]: from level k at step i the final inventory is still reachable.
    final_idx = round(inst.final / dv)
    remaining = (inst.n_steps - np.arange(inst.n_steps + 1))[:, None]
    k = np.arange(n_levels)[None, :]
    feasible = (final_idx - k <= remaining * n_inj) & (k - final_idx <= remaining * n_wd)
    return _Grid(
        times, levels, moves, feasible, round(inst.initial / dv), float(np.exp(-mkt.rate * dt))
    )


def _cashflow(move_volume: float, price: FloatArray, inst: GasStorage) -> FloatArray:
    """Cash from changing inventory by ``move_volume`` (inject < 0 cash, withdraw > 0)."""
    if move_volume > 0:
        return -move_volume * (price + inst.inject_cost)
    if move_volume < 0:
        return -move_volume * (price - inst.withdraw_cost)
    return np.zeros_like(price)


def _basis(s: FloatArray, scale: float, degree: int) -> FloatArray:
    return np.vander(s / scale, degree + 1, increasing=True)


def _backward(
    inst: GasStorage, g: _Grid, prices: FloatArray, regress: bool, degree: int = 3
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Backward induction over ``prices`` of shape (n_steps, n_paths).

    Returns the realised path values at t=0 for the start level, plus the regression
    coefficients (n_steps, degree + 1, n_levels) and price scalings (n_steps,) that define the
    exercise policy, so it can be replayed on independent paths.
    """
    n_steps, n_paths = prices.shape
    n_levels = g.levels.size
    dv = inst.volume_step
    value = np.where(g.feasible[-1][:, None], 0.0, np.nan) * np.ones((1, n_paths))
    coefs = np.full((n_steps, degree + 1, n_levels), np.nan)
    scales = np.ones(n_steps)

    for i in range(n_steps - 1, -1, -1):
        s = prices[i]
        future = g.step_df * value  # realised discounted next-step values (n_levels, n_paths)
        ok = g.feasible[i + 1]
        if regress:
            # One least-squares solve for all inventory levels (multiple right-hand sides).
            scales[i] = s.mean()
            if s.std() > 0:
                coef, *_ = np.linalg.lstsq(_basis(s, scales[i], degree), future[ok].T, rcond=None)
            else:  # degenerate cross-section (t = 0): the regression is just the mean
                coef = np.zeros((degree + 1, int(ok.sum())))
                coef[0] = future[ok].mean(axis=1)
            coefs[i][:, ok] = coef
            cont = np.full_like(future, np.nan)
            cont[ok] = (_basis(s, scales[i], degree) @ coef).T
        else:
            cont = future

        best = np.full((n_levels, n_paths), -np.inf)
        realised = np.full((n_levels, n_paths), np.nan)
        for m in g.moves:
            src = np.arange(max(0, -m), min(n_levels, n_levels - m))
            dst = src + m
            keep = g.feasible[i, src] & g.feasible[i + 1, dst]
            src, dst = src[keep], dst[keep]
            if src.size == 0:
                continue
            cash = _cashflow(m * dv, s, inst)
            score = cash + cont[dst]
            better = score > best[src]
            best[src] = np.where(better, score, best[src])
            # Longstaff-Schwartz: decide on the estimate, carry back the realised value.
            realised[src] = np.where(better, cash + future[dst], realised[src])
        realised[~g.feasible[i]] = np.nan
        value = realised
    return value[g.start_idx], coefs, scales


def _forward(
    inst: GasStorage,
    g: _Grid,
    prices: FloatArray,
    coefs: FloatArray,
    scales: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    """Apply a fitted policy to independent paths. Returns path PVs and mean inventory path.

    Because the policy is fixed before seeing these paths, the estimate is low-biased: a
    sub-optimal policy can only lose value. Together with the in-sample estimate it brackets
    the true value.
    """
    n_steps, n_paths = prices.shape
    degree = coefs.shape[1] - 1
    n_levels = g.levels.size
    level = np.full(n_paths, g.start_idx)
    pv = np.zeros(n_paths)
    path = [float(g.levels[g.start_idx])]
    for i in range(n_steps):
        s = prices[i]
        basis = _basis(s, scales[i], degree)
        best = np.full(n_paths, -np.inf)
        best_dst = level.copy()
        best_cash = np.zeros(n_paths)
        for m in g.moves:
            dst = level + m
            valid = (dst >= 0) & (dst < n_levels)
            dst_c = np.clip(dst, 0, n_levels - 1)
            valid &= g.feasible[i + 1, dst_c]
            cash = _cashflow(m * inst.volume_step, s, inst)
            cont = np.einsum("pk,kp->p", basis, coefs[i][:, dst_c])
            score = np.where(valid, cash + cont, -np.inf)
            better = score > best
            best = np.where(better, score, best)
            best_dst = np.where(better, dst_c, best_dst)
            best_cash = np.where(better, cash, best_cash)
        pv += g.step_df**i * best_cash
        level = best_dst
        path.append(float(g.levels[level].mean()))
    return pv, np.asarray(path)


def _simulate_spot(
    inst: GasStorage, mkt: MarketData, times: FloatArray, n_paths: int, rng: np.random.Generator
) -> FloatArray:
    kappa, sigma = mkt.mean_reversion, mkt.spot_vol
    n = times.size
    x = np.zeros((n, n_paths))
    var = np.zeros(n)
    prev_t, prev_x, prev_var = 0.0, np.zeros(n_paths), 0.0
    half = (n_paths + 1) // 2
    for i, t in enumerate(times):
        dt = t - prev_t
        decay = np.exp(-kappa * dt)
        step_var = sigma**2 * (1 - np.exp(-2 * kappa * dt)) / (2 * kappa)
        z = rng.standard_normal(half)
        z = np.concatenate([z, -z])[:n_paths]  # antithetic
        prev_x = prev_x * decay + np.sqrt(step_var) * z
        prev_var = prev_var * decay**2 + step_var
        x[i], var[i], prev_t = prev_x, prev_var, t
    fwd = mkt.curve(times)[:, None]
    return np.asarray(fwd * np.exp(x - 0.5 * var[:, None]), dtype=np.float64)


def intrinsic_value(inst: GasStorage, mkt: MarketData) -> float:
    """Optimal value against the forward curve with no optionality (exact DP)."""
    g = _grid(inst, mkt)
    fwd = mkt.curve(g.times)[:, None]
    v0, _, _ = _backward(inst, g, fwd, regress=False)
    return float(mkt.df(inst.start) * v0[0])


def lsmc_value(
    inst: GasStorage,
    mkt: MarketData,
    *,
    n_paths: int = 20_000,
    seed: int | None = 7,
    degree: int = 3,
) -> StorageResult:
    """Least-squares Monte Carlo value of the storage contract.

    The policy is fitted on one set of paths and the reported value is its out-of-sample
    (low-biased) PV on an independent set. The in-sample estimate is returned alongside.
    """
    g = _grid(inst, mkt)
    seeds = np.random.SeedSequence(seed).spawn(2)
    train = _simulate_spot(inst, mkt, g.times, n_paths, np.random.default_rng(seeds[0]))
    v_in, coefs, scales = _backward(inst, g, train, regress=True, degree=degree)
    test = _simulate_spot(inst, mkt, g.times, n_paths, np.random.default_rng(seeds[1]))
    pv, inventory = _forward(inst, g, test, coefs, scales)

    disc = float(mkt.df(inst.start))
    value = disc * float(pv.mean())
    intrinsic = intrinsic_value(inst, mkt)
    return StorageResult(
        value=value,
        intrinsic=intrinsic,
        extrinsic=value - intrinsic,
        stderr=disc * float(pv.std(ddof=1) / np.sqrt(n_paths)),
        in_sample=disc * float(v_in.mean()),
        expected_inventory=inventory,
    )
