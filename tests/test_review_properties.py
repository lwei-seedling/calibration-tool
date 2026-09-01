"""Property tests for the economics of the engine.

These assert on *behaviour that must hold for the model to mean anything*, rather
than on shapes and ranges. Each one corresponds to a finding in
docs/BUILD_REVIEW.md and fails against the pre-fix engine.
"""
from __future__ import annotations

import numpy as np
import pytest

from calibration.portfolio.models import PortfolioInputs
from calibration.portfolio.optimizer import PortfolioOptimizer
from calibration.project.models import ProjectInputs
from calibration.utils.irr import clean_irr, batch_irr
from calibration.utils.stats import cvar
from calibration.vehicle.calibration import CalibratorConfig, CatalyticCalibrator
from calibration.vehicle.capital_stack import CapitalStack
from calibration.vehicle.models import VehicleInputs
from calibration.vehicle.risk_mitigants import CoverageType, GrantReserve, Guarantee


def _stack(guarantee: float = 0.0, reserve: float = 0.0,
           mezz: float = 0.10, coupon: float = 0.08) -> CapitalStack:
    return CapitalStack(
        total_capital=10_000_000,
        grant_reserve=GrantReserve(reserve),
        guarantee=Guarantee(guarantee, CoverageType.PERCENTAGE),
        senior_coupon=coupon,
        mezzanine_coupon=0.12,
        mezzanine_fraction=mezz,
        lifetime_years=10,
    )


def _volatile_cashflows(n_sims: int = 400, seed: int = 1, vol: float = 0.45,
                        level: float = 1_800_000) -> np.ndarray:
    rng = np.random.default_rng(seed)
    cfs = np.empty((n_sims, 11))
    cfs[:, 0] = -10_000_000
    cfs[:, 1:] = level * np.exp(rng.normal(0, vol, (n_sims, 10)))
    return cfs


# ---------------------------------------------------------------------------
# F1 — a solvent vehicle must repay senior principal
# ---------------------------------------------------------------------------

def test_profitable_vehicle_repays_senior_principal():
    """A vehicle returning 3x its capital must not short its senior lenders.

    Deterministic cashflows, no volatility: -$10M at t=0 then +$3M/yr for 10y.
    Senior is owed coupon each period plus principal at maturity; there is
    ample cash to cover both, so the tranche must be made whole.
    """
    cfs = np.zeros((1, 11))
    cfs[0, 0] = -10_000_000
    cfs[0, 1:] = 3_000_000

    stack = _stack(mezz=0.0)
    alpha = 0.20
    tranche_cfs = stack._cashflow_waterfall(cfs, alpha)
    _fl, _mz, senior = stack._build_tranches(alpha)

    received = tranche_cfs["senior"][0, 1:].sum()
    coupons = senior.notional * stack.senior_coupon * 10
    principal_repaid = received - coupons

    assert principal_repaid == pytest.approx(senior.notional, rel=1e-9), (
        f"senior short by ${senior.notional - principal_repaid:,.0f} of principal "
        f"despite $30M of inflows on $10M of capital"
    )


def test_senior_irr_reaches_coupon_when_cash_is_ample():
    """With cash to spare, the senior tranche should earn its coupon."""
    cfs = np.zeros((1, 11))
    cfs[0, 0] = -10_000_000
    cfs[0, 1:] = 3_000_000

    result = _stack(mezz=0.0).waterfall(cfs, 0.20)["senior"]
    assert result.median_irr == pytest.approx(0.08, abs=1e-4)


# ---------------------------------------------------------------------------
# F2 — the correlation matrix must correlate
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("rho", [0.0, 0.5, 0.9])
def test_correlation_matrix_is_applied(rho):
    """Project cashflows must end up correlated at approximately the target rho.

    ``correlation_matrix`` is a rank (Spearman) correlation — that is what the
    Iman-Conover reordering controls. Linear correlation cannot equal it for
    skewed marginals: that is a Frechet bound, not an implementation gap.
    """
    from scipy.stats import spearmanr

    projects = [
        ProjectInputs(base_cashflows=[-1_000_000.0] + [260_000.0] * 10,
                      price_vol=0.35, lifetime_years=10)
        for _ in range(2)
    ]
    corr = [[1.0, rho], [rho, 1.0]]
    vehicle = VehicleInputs(projects=projects, correlation_matrix=corr,
                            total_capital=2_400_000)
    inputs = PortfolioInputs(vehicles=[vehicle], total_budget=2_400_000,
                             n_sims=4000, seed=7)

    opt = PortfolioOptimizer(inputs)
    per_project = opt._correlated_project_cashflows(0, seed_offset=0)
    totals = [cfs.sum(axis=1) for cfs in per_project]
    achieved = float(spearmanr(totals[0], totals[1]).statistic)

    assert achieved == pytest.approx(rho, abs=0.04), (
        f"target rho={rho}, achieved rank correlation={achieved:.4f}"
    )


def test_higher_correlation_widens_vehicle_cashflow_spread():
    """Less diversification must produce a wider vehicle-level distribution."""
    def spread(rho: float) -> float:
        projects = [
            ProjectInputs(base_cashflows=[-1_000_000.0] + [260_000.0] * 10,
                          price_vol=0.35, lifetime_years=10)
            for _ in range(3)
        ]
        corr = [[1.0 if i == j else rho for j in range(3)] for i in range(3)]
        vehicle = VehicleInputs(projects=projects, correlation_matrix=corr,
                                total_capital=3_600_000)
        inputs = PortfolioInputs(vehicles=[vehicle], total_budget=3_600_000,
                                 n_sims=3000, seed=11)
        cfs = PortfolioOptimizer(inputs)._simulate_vehicle(0, seed_offset=0)
        return float(cfs.sum(axis=1).std())

    assert spread(0.8) > spread(0.0) * 1.15


# ---------------------------------------------------------------------------
# F3 — IRR must not sign-flip
# ---------------------------------------------------------------------------

def test_very_high_irr_is_capped_not_recorded_as_total_loss():
    """A 4995% return must not be booked as -1.0 (total loss)."""
    cf = np.array([[-100_000.0] + [4_995_000.0] * 10])
    cleaned = clean_irr(batch_irr(cf))
    assert cleaned[0] == pytest.approx(10.0), (
        f"expected the 10.0 cap, got {cleaned[0]}"
    )


def test_known_irr_is_recovered():
    """A cashflow with an analytic IRR must return that IRR."""
    # $1000 invested, 10% coupon for 5 years, principal back at t=5 -> IRR = 10%.
    cf = np.array([[-1000.0, 100.0, 100.0, 100.0, 100.0, 1100.0]])
    assert clean_irr(batch_irr(cf))[0] == pytest.approx(0.10, abs=1e-6)


def test_total_loss_still_maps_to_minus_one():
    """The genuine total-loss case must be unchanged."""
    cf = np.array([[-1000.0, 0.0, 0.0, 0.0]])
    assert clean_irr(batch_irr(cf))[0] == pytest.approx(-1.0)


# ---------------------------------------------------------------------------
# F4 — CVaR must describe the tail
# ---------------------------------------------------------------------------

def test_cvar_uses_the_tail_not_every_zero_loss_path():
    """With losses in 2% of paths, CVaR(95%) is the mean of the worst 5%."""
    losses = np.zeros(1000)
    losses[:20] = np.linspace(1e6, 5e6, 20)
    expected = float(np.mean(np.sort(losses)[-50:]))
    assert cvar(losses, 0.95) == pytest.approx(expected, rel=1e-9)


def test_cvar_is_at_least_var():
    """CVaR is an expected shortfall: it can never sit below VaR."""
    rng = np.random.default_rng(3)
    for _ in range(20):
        losses = np.maximum(0.0, rng.normal(0, 1e5, 500))
        assert cvar(losses, 0.95) >= np.nanquantile(losses, 0.95) - 1e-9


# ---------------------------------------------------------------------------
# F5 — degenerate calibrations must be refused, not reported
# ---------------------------------------------------------------------------

def test_hurdle_above_senior_coupon_is_rejected():
    """Senior IRR is capped at the coupon, so this configuration is unsatisfiable."""
    cfs = _volatile_cashflows()
    with pytest.raises(ValueError, match="hurdle"):
        CatalyticCalibrator(
            _stack(), cfs, CalibratorConfig(investor_hurdle_irr=0.12)
        ).calibrate()


def test_calibration_never_returns_a_vanishing_senior_tranche():
    """A vehicle that returns nothing must raise, not report a degenerate alpha."""
    cfs = np.tile(np.array([-10_000_000.0] + [0.0] * 10), (200, 1))
    with pytest.raises(ValueError):
        CatalyticCalibrator(
            _stack(), cfs, CalibratorConfig(investor_hurdle_irr=0.07)
        ).calibrate()


def test_senior_floor_check_holds_at_large_capital():
    """The senior-notional floor comparison must not break on float rounding.

    notional and floor are computed by different float expressions whose gap
    scales with total_capital; an absolute epsilon made a feasible $5bn vehicle
    report h(alpha_hi) = -1.0 and calibrate() falsely raise infeasible.
    """
    stack = CapitalStack(
        total_capital=5e9,
        grant_reserve=GrantReserve(0),
        guarantee=Guarantee(0.0, CoverageType.PERCENTAGE),
        senior_coupon=0.08, mezzanine_coupon=0.12,
        mezzanine_fraction=0.13, lifetime_years=10,
    )
    cfg = CalibratorConfig(min_senior_fraction=0.07, investor_hurdle_irr=0.07)
    cfs = np.tile(np.array([-5e9] + [1.6e9] * 10), (50, 1))
    cal = CatalyticCalibrator(stack, cfs, cfg)
    assert cal._h(cal._max_structural_alpha()) > -1.0
    alpha = cal.calibrate()
    assert cal._h(alpha) >= -1e-9


def test_calibrated_alpha_is_actually_feasible():
    """The returned alpha must satisfy the constraints it was solved for."""
    cfs = _volatile_cashflows(level=2_400_000)
    cal = CatalyticCalibrator(_stack(), cfs, CalibratorConfig(investor_hurdle_irr=0.07))
    alpha = cal.calibrate()
    assert cal._h(alpha) >= -1e-9, f"h({alpha:.6f}) = {cal._h(alpha):+.6f} < 0"


# ---------------------------------------------------------------------------
# F6 — mitigants must reach the senior tranche
# ---------------------------------------------------------------------------

def test_guarantee_improves_senior_irr():
    """A partial credit guarantee must raise the return of the tranche it wraps.

    Uses a cash-starved vehicle: senior IRR is capped at the coupon, so the
    mitigants only have room to show up when the median path is actually short.
    """
    cfs = _volatile_cashflows(level=700_000)
    without = _stack(guarantee=0.0).waterfall(cfs, 0.30)["senior"].median_irr
    with_guar = _stack(guarantee=0.60).waterfall(cfs, 0.30)["senior"].median_irr
    assert with_guar > without, (
        f"guarantee changed senior IRR by {with_guar - without:+.6f}"
    )


def test_grant_reserve_improves_senior_irr():
    """A donor-funded cash cushion must be usable to pay senior lenders."""
    cfs = _volatile_cashflows(level=700_000)
    without = _stack(reserve=0.0).waterfall(cfs, 0.30)["senior"].median_irr
    with_res = _stack(reserve=2_000_000).waterfall(cfs, 0.30)["senior"].median_irr
    assert with_res > without


def test_more_guarantee_weakly_reduces_alpha():
    """More credit support must never require more catalytic capital."""
    cfs = _volatile_cashflows(level=900_000)
    cfg = CalibratorConfig(investor_hurdle_irr=0.07)
    low = CatalyticCalibrator(_stack(guarantee=0.0), cfs, cfg).calibrate()
    high = CatalyticCalibrator(_stack(guarantee=0.60), cfs, cfg).calibrate()
    assert high <= low + 1e-6, f"alpha rose from {low:.4f} to {high:.4f}"


# ---------------------------------------------------------------------------
# F7 — reported CVaR must be comparable to the constraint
# ---------------------------------------------------------------------------

def test_reported_cvar_respects_the_configured_limit():
    """A portfolio the LP calls optimal must report a CVaR within cvar_max."""
    projects = [ProjectInputs(base_cashflows=[-1_000_000.0] + [240_000.0] * 10,
                              price_vol=0.40, lifetime_years=10)]
    vehicle = VehicleInputs(projects=projects, correlation_matrix=[[1.0]],
                            total_capital=1_200_000, senior_coupon=0.08)
    inputs = PortfolioInputs(vehicles=[vehicle], total_budget=1_200_000,
                             catalytic_budget=400_000, cvar_max=0.20,
                             n_sims=500, seed=5,
                             calibrator_config=CalibratorConfig(investor_hurdle_irr=0.07))
    result = PortfolioOptimizer(inputs).run()
    if result.status == "optimal":
        assert result.cvar_95 <= inputs.cvar_max + 1e-6
