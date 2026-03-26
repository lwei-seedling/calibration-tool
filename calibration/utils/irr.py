"""IRR computation with edge case handling."""
from __future__ import annotations

import warnings

import numpy as np
import numpy_financial as npf


def _irr_single(cashflows: np.ndarray) -> float:
    """Compute IRR for a single cashflow vector.

    Returns:
        IRR as a decimal (e.g. 0.12 for 12%).
        -1.0 sentinel for total loss (no positive inflows with negative outflow).
        NaN for undefined (no investment outflow).
        Uses numpy_financial.irr which picks root closest to zero.
    """
    c0 = cashflows[0]
    positive_inflows = np.sum(np.maximum(0.0, cashflows[1:]))

    # No investment outflow → IRR undefined
    if c0 >= 0.0:
        return float("nan")

    # No positive inflows at all → total loss
    if positive_inflows == 0.0:
        return -1.0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = npf.irr(cashflows)

    if result is None or np.isnan(result):
        # Fallback: Newton's method
        result = _irr_newton(cashflows)

    if np.isinf(result):
        return 10.0  # cap at 1000%

    return float(result)


def _irr_newton(cashflows: np.ndarray, r0: float = 0.10, max_iter: int = 100, tol: float = 1e-8) -> float:
    """Newton's method fallback for IRR."""
    t = np.arange(len(cashflows), dtype=float)
    r = r0
    for _ in range(max_iter):
        factors = (1.0 + r) ** t
        npv = np.sum(cashflows / factors)
        dnpv = np.sum(-t * cashflows / ((1.0 + r) * factors))
        if dnpv == 0.0:
            break
        r_new = r - npv / dnpv
        if abs(r_new - r) < tol:
            return float(r_new)
        r = r_new
    return float("nan")


def batch_irr(cashflows: np.ndarray) -> np.ndarray:
    """Compute IRR for each simulation path.

    Args:
        cashflows: Array of shape (n_sims, T) where axis-0 is simulation paths
                   and axis-1 is time periods. cashflows[:, 0] should be
                   negative (investment outflow).

    Returns:
        irr_vector of shape (n_sims,). Sentinels:
          -1.0  → total loss (no inflows)
          NaN   → undefined (no outflow at t=0)
          10.0  → capped at 1000% for practical outliers
    """
    cashflows = np.asarray(cashflows, dtype=float)
    n_sims = cashflows.shape[0]
    result = np.empty(n_sims, dtype=float)
    for s in range(n_sims):
        result[s] = _irr_single(cashflows[s])
    return result


def npv_loss(cashflows: np.ndarray, discount_rate: float = 0.0) -> np.ndarray:
    """NPV-based terminal loss for each simulation path.

    L[s] = max(0, -NPV(CF[s], discount_rate))

    This is the primary loss metric fed into the loss waterfall. Using NPV
    (rather than a simple undiscounted sum) accounts for the time value of
    money: a recovery that arrives 15 years from now is worth less than an
    equivalent near-term loss.

    When discount_rate=0.0 (the default for backward compatibility) the
    result is identical to max(0, -sum(CF)), preserving existing behaviour.

    Args:
        cashflows: shape (n_sims, T+1); axis-1 index 0 is the t=0 outflow.
        discount_rate: annual discount rate. 0.0 → undiscounted (sum-based).

    Returns:
        loss array of shape (n_sims,), non-negative.
    """
    cashflows = np.asarray(cashflows, dtype=float)
    T = cashflows.shape[1] - 1
    t = np.arange(T + 1, dtype=float)
    discount_factors = (1.0 + discount_rate) ** t   # shape (T+1,)
    npv = (cashflows / discount_factors).sum(axis=1)  # shape (n_sims,)
    return np.maximum(0.0, -npv)


def clean_irr(irr_vector: np.ndarray) -> np.ndarray:
    """Replace NaN with -1.0 and cap +inf at 10.0 for safe statistics.

    NaN is treated conservatively as total loss. +inf is capped at 10.0
    (1000% return) to avoid distorting aggregate statistics.
    """
    out = irr_vector.copy()
    out[np.isnan(out)] = -1.0
    out[np.isposinf(out)] = 10.0
    out[np.isneginf(out)] = -1.0
    nan_fraction = np.mean(np.isnan(irr_vector))
    if nan_fraction > 0.05:
        import warnings
        warnings.warn(
            f"{nan_fraction:.1%} of IRR paths returned NaN — check cashflow inputs.",
            RuntimeWarning,
            stacklevel=2,
        )
    return out
