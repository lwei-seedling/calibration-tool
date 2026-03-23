"""Tests for the vehicle layer: waterfall and calibration."""
import numpy as np
import pytest

from calibration.vehicle.calibration import CalibratorConfig, CatalyticCalibrator
from calibration.vehicle.capital_stack import CapitalStack
from calibration.vehicle.risk_mitigants import Guarantee, GrantReserve, CoverageType


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def make_stack(
    total_capital=1_000_000,
    grant_reserve_amt=0.0,
    guarantee_coverage=0.0,
    senior_coupon=0.08,
    mezzanine_coupon=0.12,
    mezzanine_fraction=0.0,
    lifetime_years=5,
):
    return CapitalStack(
        total_capital=total_capital,
        grant_reserve=GrantReserve(grant_reserve_amt),
        guarantee=Guarantee(guarantee_coverage, CoverageType.PERCENTAGE),
        senior_coupon=senior_coupon,
        mezzanine_coupon=mezzanine_coupon,
        mezzanine_fraction=mezzanine_fraction,
        lifetime_years=lifetime_years,
    )


def zero_loss_cashflows(n_sims=100, total_capital=1_000_000, T=5):
    """Cashflows where every path has net positive lifetime cashflow (no loss).

    Structured to allow a senior debt tranche (bullet maturity) to be repaid:
    steady coupon-like payments each year, plus a large terminal principal
    recovery that ensures t=T has enough to repay senior principal.
    """
    cfs = np.zeros((n_sims, T + 1))
    cfs[:, 0] = -total_capital
    # Steady operating cashflow each year; increase terminal to repay principal
    cfs[:, 1:T] = total_capital * 0.12  # annual coupon coverage
    cfs[:, T] = total_capital * 1.2     # terminal: principal + final coupon
    return cfs


def total_loss_cashflows(n_sims=100, total_capital=1_000_000, T=5):
    """Cashflows where every path has total loss equal to total_capital."""
    cfs = np.zeros((n_sims, T + 1))
    cfs[:, 0] = -total_capital
    # No operating cashflows → total loss = capex
    return cfs


# ------------------------------------------------------------------
# Loss waterfall tests
# ------------------------------------------------------------------

class TestLossWaterfall:
    def test_zero_loss_propagation(self):
        """When there is no terminal loss, all layers absorb zero."""
        stack = make_stack()
        cfs = zero_loss_cashflows()
        results = stack.waterfall(cfs, alpha=0.30)
        for name, tr in results.items():
            np.testing.assert_array_equal(tr.loss_distribution, 0.0, err_msg=f"{name} should absorb 0 loss")

    def test_loss_absorbed_by_first_loss_before_senior(self):
        """Small losses should be absorbed by first-loss, not reach senior."""
        total = 1_000_000
        T = 5
        n_sims = 50
        # Create cashflows with a small loss (10% of total capital)
        cfs = np.zeros((n_sims, T + 1))
        cfs[:, 0] = -total
        cfs[:, 1:] = total * 0.9 / T  # recover only 90%

        alpha = 0.40  # 40% catalytic → FL = 400k, enough to absorb 100k loss
        stack = make_stack(total_capital=total)
        results = stack.waterfall(cfs, alpha=alpha)

        # Senior should have zero loss
        np.testing.assert_array_equal(
            results["senior"].loss_distribution, 0.0,
            err_msg="Senior should not absorb any loss when FL is sufficient"
        )
        # First-loss should absorb the loss
        assert np.all(results["first_loss"].loss_distribution > 0)

    def test_grant_reserve_absorbs_first(self):
        """Grant reserve should be fully depleted before first-loss is hit."""
        total = 1_000_000
        reserve = 50_000
        T = 5
        n_sims = 20
        # Loss = 30k per path (less than reserve)
        cfs = np.zeros((n_sims, T + 1))
        cfs[:, 0] = -total
        cfs[:, 1:] = (total - 30_000) / T  # recover all but 30k

        stack = make_stack(total_capital=total, grant_reserve_amt=reserve)
        results = stack.waterfall(cfs, alpha=0.30)

        # First-loss and senior should absorb nothing (reserve covers it)
        np.testing.assert_array_equal(results["first_loss"].loss_distribution, 0.0)
        np.testing.assert_array_equal(results["senior"].loss_distribution, 0.0)

    def test_guarantee_protects_senior(self):
        """Guarantee on senior should absorb losses before they reach senior tranche."""
        total = 1_000_000
        T = 5
        n_sims = 50
        alpha = 0.20  # FL = 200k
        guarantee_coverage = 0.50  # 50% of senior notional

        # Loss = 500k → FL absorbs 200k, remaining 300k hits guarantee
        # Senior notional = total * (1-alpha) = 800k
        # Guarantee cap = 0.5 * 800k = 400k → covers 300k fully
        cfs = np.zeros((n_sims, T + 1))
        cfs[:, 0] = -total
        cfs[:, 1:] = total * 0.50 / T  # recover 50% → 500k loss

        stack = make_stack(total_capital=total, guarantee_coverage=guarantee_coverage)
        results = stack.waterfall(cfs, alpha=alpha)

        # Senior should absorb nothing (guarantee covers residual after FL)
        np.testing.assert_array_equal(
            results["senior"].loss_distribution, 0.0,
            err_msg="Guarantee should fully protect senior in this scenario"
        )

    def test_total_loss_wipes_first_loss(self):
        """When total loss = total capital and no guarantee, senior is also wiped."""
        total = 1_000_000
        T = 5
        n_sims = 30
        alpha = 0.30  # FL = 300k

        cfs = total_loss_cashflows(n_sims=n_sims, total_capital=total, T=T)
        stack = make_stack(total_capital=total)
        results = stack.waterfall(cfs, alpha=alpha)

        # FL fully absorbed (300k), senior partially/fully absorbed
        assert np.all(results["first_loss"].loss_distribution == pytest.approx(alpha * total))
        assert np.all(results["senior"].loss_distribution > 0)


# ------------------------------------------------------------------
# Calibration tests
# ------------------------------------------------------------------

class TestCalibration:
    def test_calibrate_returns_float_in_bounds(self):
        """Calibration should return alpha in [alpha_lo, alpha_hi]."""
        total = 1_000_000
        T = 5
        n_sims = 200
        rng = np.random.default_rng(42)

        # Semi-realistic cashflows
        cfs = np.zeros((n_sims, T + 1))
        cfs[:, 0] = -total
        cfs[:, 1:] = rng.normal(loc=total * 0.25, scale=total * 0.1, size=(n_sims, T))

        stack = make_stack(total_capital=total, lifetime_years=T)
        cfg = CalibratorConfig(
            investor_hurdle_irr=0.05,
            max_loss_probability=0.10,
            alpha_lo=0.0,
            alpha_hi=0.99,
        )
        calibrator = CatalyticCalibrator(stack, cfs, cfg)
        alpha = calibrator.calibrate()

        assert 0.0 <= alpha <= 0.99

    def test_calibrate_no_loss_project_needs_minimal_catalytic(self):
        """A project with zero loss needs minimal catalytic capital."""
        total = 1_000_000
        T = 5
        n_sims = 100
        cfs = zero_loss_cashflows(n_sims=n_sims, total_capital=total, T=T)

        stack = make_stack(total_capital=total, lifetime_years=T)
        cfg = CalibratorConfig(
            investor_hurdle_irr=0.05,
            max_loss_probability=0.10,
            alpha_lo=0.0,
            alpha_hi=0.99,
        )
        calibrator = CatalyticCalibrator(stack, cfs, cfg)
        alpha = calibrator.calibrate()

        # With zero loss, calibration should find a feasible solution
        assert 0.0 <= alpha <= 0.99

    def test_calibrate_infeasible_raises(self):
        """When no alpha can satisfy constraints, ValueError should be raised."""
        total = 1_000_000
        T = 5
        n_sims = 100
        # Severe total loss on every path
        cfs = total_loss_cashflows(n_sims=n_sims, total_capital=total, T=T)

        stack = make_stack(total_capital=total, lifetime_years=T)
        cfg = CalibratorConfig(
            investor_hurdle_irr=0.50,   # impossibly high hurdle
            max_loss_probability=0.01,  # near-zero loss tolerance
            alpha_lo=0.0,
            alpha_hi=0.99,
        )
        calibrator = CatalyticCalibrator(stack, cfs, cfg)
        with pytest.raises(ValueError, match="infeasible"):
            calibrator.calibrate()
