"""IRR computation with edge case handling.

Why brentq instead of Newton's method
--------------------------------------
Newton-Raphson explores unbounded regions of r, producing (1+r)**t overflow
for large t and extreme trial rates. brentq is a bracketed root-finder: it
*guarantees* every evaluation of NPV(r) occurs within the interval
[-0.999, 10.0], so no overflow is possible.

Sentinels, and why a bracket miss is not a failure
--------------------------------------------------
Some simulation paths produce cashflow sequences with no sign change (e.g.
construction-phase-only losses with zero revenue in tail scenarios). IRR is
mathematically undefined for these, and they return the -1.0 total-loss
sentinel directly.

A separate case is a path whose true IRR lies *outside* [-0.999, 10.0] — an
outlier returning more than 1000%. NPV then has the same sign at both bracket
endpoints. That is a root outside the interval, not an undefined one, so it
reports the corresponding bound (10.0 or -0.999) rather than NaN. Reporting
NaN here would send a spectacular return through clean_irr's NaN branch and
book it as a total loss.

NaN is therefore reserved for genuine solver failure, and clean_irr() still
converts it conservatively to -1.0.
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
        n_failures:      Paths where brentq failed to converge despite a sign
                          change — returned as NaN. A true IRR outside the
                          bracket is *not* counted here: it reports the
                          corresponding bound instead.
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
        -1.0   — total-loss sentinel (negative outflow, zero inflows).
        -0.999 — the true IRR is below the bracket floor; the floor is reported.
        10.0   — the true IRR exceeds the 1000% cap.
        NaN    — no sign change, or brentq failed to converge.
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
        # NPV has the same sign at both ends, so the root lies outside the
        # bracket rather than being undefined. Which side it fell off depends on
        # NPV's direction in r: decreasing for an investment-shaped cashflow
        # (outflow first), increasing for a borrowing-shaped one (inflow first).
        # Read the direction off the endpoints instead of assuming a shape —
        # hardcoding "decreasing" reports a borrowing path's +2900% cost of
        # funds as the -0.999 floor. Returning NaN here would be worse still:
        # clean_irr would book a very high return as a total loss.
        decreasing = npv_lo > npv_hi
        if (npv_hi > 0.0) == decreasing:
            return _R_HI
        return _R_LO

    try:
        r = brentq(_npv_stable, _R_LO, _R_HI, args=(cashflows, t),
                   xtol=1e-8, maxiter=100)
    except ValueError:
        return float("nan")

    return float(r)


def _npv_vec(rates: np.ndarray, cashflows: np.ndarray, t: np.ndarray) -> np.ndarray:
    """NPV of each row of `cashflows` at its own rate in `rates`.

    Uses exp(t*log1p(r)) for the same overflow safety as the scalar path.
    """
    discount = np.exp(t[np.newaxis, :] * np.log1p(rates)[:, np.newaxis])
    return np.sum(cashflows / discount, axis=1)


def batch_irr(
    cashflows: np.ndarray,
    return_diagnostics: bool = False,
) -> np.ndarray | tuple[np.ndarray, IrrDiagnostics]:
    """Compute IRR for each simulation path.

    Vectorised bisection over all paths at once. The calibrator re-evaluates the
    waterfall — and therefore this function — for every trial alpha, so a Python
    loop calling a scalar root-finder per path dominates the runtime of the whole
    pipeline. Bisection needs more iterations than Brent's method but each one is
    a single array operation over every path, which is far cheaper overall.

    Args:
        cashflows: Array of shape (n_sims, T) where axis-0 is simulation paths
                   and axis-1 is time periods. cashflows[:, 0] should be
                   negative (investment outflow).
        return_diagnostics: If True, also return an IrrDiagnostics dataclass
                   with counts of failures and undefined paths.

    Returns:
        irr_vector of shape (n_sims,). Sentinels:
          -1.0   → total loss (no inflows)
          -0.999 → true IRR below the bracket floor (the floor is reported)
          10.0   → IRR above the 1000% cap
          NaN    → no sign change, or the bracket endpoints were not finite

        If return_diagnostics=True, returns (irr_vector, IrrDiagnostics).
    """
    cashflows = np.asarray(cashflows, dtype=float)
    n_sims, n_periods = cashflows.shape
    t = np.arange(n_periods, dtype=float)

    result = np.empty(n_sims, dtype=float)

    has_negative = np.any(cashflows < 0.0, axis=1)
    has_positive = np.any(cashflows > 0.0, axis=1)

    # No investment outflow → IRR undefined. No inflows → total loss.
    result[~has_negative] = np.nan
    result[has_negative & ~has_positive] = -1.0

    solvable = has_negative & has_positive
    n_no_sign_change = int(np.sum(~solvable))
    n_failures = 0

    if np.any(solvable):
        cf = cashflows[solvable]
        lo = np.full(cf.shape[0], _R_LO)
        hi = np.full(cf.shape[0], _R_HI)

        npv_lo = _npv_vec(lo, cf, t)
        npv_hi = _npv_vec(hi, cf, t)

        vals = np.empty(cf.shape[0], dtype=float)

        # Endpoints that could not be evaluated are genuine solver failures.
        bad = ~np.isfinite(npv_lo) | ~np.isfinite(npv_hi)
        vals[bad] = np.nan
        n_failures = int(np.sum(bad))

        # Root outside the bracket: report the bound it fell past, not NaN.
        # Direction-aware, matching _irr_single: NPV decreasing in r means both
        # endpoints positive puts the root above the cap; for an increasing NPV
        # (borrowing-shaped path) the mapping flips.
        outside = (npv_lo * npv_hi > 0.0) & ~bad
        decreasing = npv_lo > npv_hi
        above_cap = outside & ((npv_hi > 0.0) == decreasing)
        vals[above_cap] = _R_HI
        vals[outside & ~above_cap] = _R_LO

        bracketed = ~bad & ~outside
        if np.any(bracketed):
            b_cf = cf[bracketed]
            b_lo = lo[bracketed]
            b_hi = hi[bracketed]
            f_lo = npv_lo[bracketed]

            # Bisection to the same 1e-8 tolerance as the scalar solver.
            # The bracket is 11.0 wide, so 2^-n * 11 < 1e-8 needs n >= 31.
            for _ in range(60):
                mid = 0.5 * (b_lo + b_hi)
                f_mid = _npv_vec(mid, b_cf, t)
                same_side = (f_mid * f_lo) > 0.0
                b_lo = np.where(same_side, mid, b_lo)
                f_lo = np.where(same_side, f_mid, f_lo)
                b_hi = np.where(same_side, b_hi, mid)
                if np.all(b_hi - b_lo < 1e-9):
                    break

            vals[bracketed] = 0.5 * (b_lo + b_hi)

        result[solvable] = vals

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
    """Replace NaN with -1.0 and clip infinities to the sentinel bounds.

    NaN (genuine solver failure) is treated conservatively as total loss.
    Infinities are clipped to +/-10.0 and -1.0. Note that IRRs beyond the
    bracket are already reported as 10.0 / -0.999 by batch_irr, so they arrive
    here as ordinary finite values.
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
