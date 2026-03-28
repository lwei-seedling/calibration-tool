"""IRR computation with edge case handling.

Why brentq instead of Newton's method
--------------------------------------
Newton-Raphson explores unbounded regions of r, producing (1+r)**t overflow
for large t and extreme trial rates. brentq is a bracketed root-finder: it
*guarantees* every evaluation of NPV(r) occurs within the interval
[-0.999, 10.0], so no overflow is possible.

Why NaN is expected in some Monte Carlo paths
----------------------------------------------
In a blended-finance Monte Carlo, some simulation paths produce cashflow
sequences with no sign change (e.g. construction-phase-only losses with zero
revenue in tail scenarios). IRR is mathematically undefined for these paths.
Returning NaN is correct; clean_irr() converts NaN to the -1.0 (total-loss)
sentinel for downstream portfolio statistics.
"""
from __future__ import annotations

import dataclasses

import numpy as np
from scipy.optimize import brentq

# Bounded search interval for brentq. Evaluated at these endpoints only;
# no trial rate outside this range can trigger overflow.
_R_LO = -0.999  # near-total loss; avoids log1p(r) singularity at r=-1
_R_HI = 10.0    # 1000% return cap (matches clean_irr sentinel)


@dataclasses.dataclass
class IrrDiagnostics:
    """Lightweight diagnostics from a batch_irr call.

    Attributes:
        n_computed:      Total number of simulation paths processed.
        n_no_sign_change: Paths with no sign change in cashflows — IRR is
                          mathematically undefined; counted before solver call.
                          Includes total-loss paths (returned as -1.0 sentinel).
        n_failures:      Paths where NPV had no root in [-0.999, 10.0] or
                          brentq failed to converge — returned as NaN.
    """
    n_computed: int
    n_no_sign_change: int
    n_failures: int


def _npv_stable(r: float, cashflows: np.ndarray, t: np.ndarray) -> float:
    """NPV using log1p for numerical stability at large t.

    Replaces (1+r)**t with exp(t*log1p(r)), which avoids overflow for large t
    (e.g. 30-year projects with a trial r near the upper bracket boundary).

    Valid only for r > -1, which is guaranteed by the brentq interval.
    """
    discount = np.exp(t * np.log1p(r))
    return float(np.sum(cashflows / discount))


def _irr_single(cashflows: np.ndarray) -> float:
    """Compute IRR for a single cashflow vector.

    Returns:
        IRR as a decimal (e.g. 0.12 for 12%).
        -1.0  — total-loss sentinel (negative outflow, zero inflows).
        NaN   — IRR undefined: no sign change, or no root in [-0.999, 10.0].
    """
    has_negative = np.any(cashflows < 0.0)
    has_positive = np.any(cashflows > 0.0)

    # No investment outflow → IRR undefined
    if not has_negative:
        return float("nan")

    # No positive inflows → total-loss sentinel
    if not has_positive:
        return -1.0

    t = np.arange(len(cashflows), dtype=float)

    # brentq requires opposite signs at the bracket endpoints.
    # Compute NPV at both ends of the valid domain.
    npv_lo = _npv_stable(_R_LO, cashflows, t)
    npv_hi = _npv_stable(_R_HI, cashflows, t)

    if not np.isfinite(npv_lo) or not np.isfinite(npv_hi):
        return float("nan")

    if npv_lo * npv_hi > 0.0:
        # NPV is same sign at both ends — no root inside the interval.
        return float("nan")

    try:
        r = brentq(_npv_stable, _R_LO, _R_HI, args=(cashflows, t),
                   xtol=1e-8, maxiter=100)
    except ValueError:
        return float("nan")

    return float(r)


def batch_irr(
    cashflows: np.ndarray,
    return_diagnostics: bool = False,
) -> np.ndarray | tuple[np.ndarray, IrrDiagnostics]:
    """Compute IRR for each simulation path.

    Args:
        cashflows: Array of shape (n_sims, T) where axis-0 is simulation paths
                   and axis-1 is time periods. cashflows[:, 0] should be
                   negative (investment outflow).
        return_diagnostics: If True, also return an IrrDiagnostics dataclass
                   with counts of failures and undefined paths.

    Returns:
        irr_vector of shape (n_sims,). Sentinels:
          -1.0  → total loss (no inflows)
          NaN   → IRR undefined or no root in [-0.999, 10.0]
          10.0  → capped at 1000% (applied by clean_irr; brentq upper bound)

        If return_diagnostics=True, returns (irr_vector, IrrDiagnostics).
    """
    cashflows = np.asarray(cashflows, dtype=float)
    n_sims = cashflows.shape[0]
    result = np.empty(n_sims, dtype=float)

    n_no_sign_change = 0
    n_failures = 0

    for s in range(n_sims):
        cf = cashflows[s]
        has_negative = np.any(cf < 0.0)
        has_positive = np.any(cf > 0.0)

        if not has_negative or not has_positive:
            n_no_sign_change += 1

        val = _irr_single(cf)
        result[s] = val

        if np.isnan(val) and has_negative and has_positive:
            # Has sign change but solver found no root — bracket miss
            n_failures += 1

    if return_diagnostics:
        diag = IrrDiagnostics(
            n_computed=n_sims,
            n_no_sign_change=n_no_sign_change,
            n_failures=n_failures,
        )
        return result, diag

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
