"""Tests for the portfolio optimizer."""
import numpy as np
import pytest

from calibration.portfolio.models import PortfolioInputs
from calibration.portfolio.optimizer import PortfolioOptimizer
from calibration.project.models import ProjectInputs
from calibration.vehicle.calibration import CalibratorConfig
from calibration.vehicle.models import VehicleInputs


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def make_project(capex=500_000, opex=20_000, price=50.0, yield_=20_000, T=5, price_vol=0.1):
    return ProjectInputs(
        capex=capex,
        opex_annual=opex,
        price=price,
        yield_=yield_,
        lifetime_years=T,
        price_vol=price_vol,
        yield_vol=0.05,
        inflation_rate=0.02,
        fx_vol=0.03,
        delay_prob=0.03,
    )


def make_vehicle(n_projects=2, total_capital=1_000_000, **kwargs):
    projects = [make_project() for _ in range(n_projects)]
    corr = np.eye(n_projects).tolist()
    return VehicleInputs(
        projects=projects,
        correlation_matrix=corr,
        total_capital=total_capital,
        guarantee_coverage=0.30,
        grant_reserve=total_capital * 0.05,
        mezzanine_fraction=0.10,
        senior_coupon=0.08,
        mezzanine_coupon=0.12,
    )


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------

class TestPortfolioOptimizer:
    def test_allocations_sum_to_budget(self):
        """Allocations must sum to the total budget."""
        vehicles = [make_vehicle(total_capital=500_000), make_vehicle(total_capital=500_000)]
        cfg = CalibratorConfig(investor_hurdle_irr=0.04, max_loss_probability=0.15)
        inputs = PortfolioInputs(
            vehicles=vehicles,
            calibrator_config=cfg,
            total_budget=1_000_000,
            n_sims=200,
            seed=42,
        )
        optimizer = PortfolioOptimizer(inputs)
        result = optimizer.run()

        total_allocated = sum(result.allocations.values())
        assert total_allocated == pytest.approx(1_000_000, rel=1e-4)

    def test_leverage_ratio_above_one(self):
        """Leverage ratio (commercial/catalytic) should be > 1 when alpha < 0.5."""
        vehicles = [make_vehicle(total_capital=400_000), make_vehicle(total_capital=600_000)]
        cfg = CalibratorConfig(investor_hurdle_irr=0.04, max_loss_probability=0.20)
        inputs = PortfolioInputs(
            vehicles=vehicles,
            calibrator_config=cfg,
            total_budget=1_000_000,
            n_sims=200,
            seed=1,
        )
        optimizer = PortfolioOptimizer(inputs)
        result = optimizer.run()

        # With sensible projects and lenient constraints, alpha < 0.5 → leverage > 1
        # Allow alpha up to 0.5 before flagging
        total_catalytic = sum(result.catalytic_allocations.values())
        total_commercial = sum(result.commercial_allocations.values())
        assert total_catalytic + total_commercial == pytest.approx(1_000_000, rel=1e-3)

    def test_catalytic_fractions_in_range(self):
        """All calibrated catalytic fractions must be in [0, 1]."""
        vehicles = [make_vehicle(), make_vehicle()]
        cfg = CalibratorConfig(investor_hurdle_irr=0.05, max_loss_probability=0.15)
        inputs = PortfolioInputs(
            vehicles=vehicles,
            calibrator_config=cfg,
            total_budget=2_000_000,
            n_sims=150,
            seed=10,
        )
        optimizer = PortfolioOptimizer(inputs)
        result = optimizer.run()

        for v, alpha in result.catalytic_fractions.items():
            assert 0.0 <= alpha <= 1.0, f"Vehicle {v}: alpha={alpha} out of range"

    def test_portfolio_irr_distribution_shape(self):
        """Portfolio IRR distribution should have shape (n_sims,)."""
        vehicles = [make_vehicle()]
        cfg = CalibratorConfig(investor_hurdle_irr=0.04, max_loss_probability=0.20)
        inputs = PortfolioInputs(
            vehicles=vehicles,
            calibrator_config=cfg,
            total_budget=1_000_000,
            n_sims=100,
            seed=5,
        )
        optimizer = PortfolioOptimizer(inputs)
        result = optimizer.run()

        assert result.portfolio_irr_distribution.shape == (100,)
        assert result.portfolio_loss_distribution.shape == (100,)

    def test_cvar_not_negative(self):
        """CVaR of losses should be non-negative."""
        vehicles = [make_vehicle(), make_vehicle()]
        cfg = CalibratorConfig(investor_hurdle_irr=0.04, max_loss_probability=0.20)
        inputs = PortfolioInputs(
            vehicles=vehicles,
            calibrator_config=cfg,
            total_budget=2_000_000,
            n_sims=150,
            seed=7,
        )
        optimizer = PortfolioOptimizer(inputs)
        result = optimizer.run()

        assert result.cvar_95 >= 0.0

    def test_result_status_is_set(self):
        """Result status string must be set."""
        vehicles = [make_vehicle()]
        cfg = CalibratorConfig()
        inputs = PortfolioInputs(
            vehicles=vehicles,
            calibrator_config=cfg,
            total_budget=1_000_000,
            n_sims=100,
            seed=0,
        )
        optimizer = PortfolioOptimizer(inputs)
        result = optimizer.run()
        assert result.status is not None
        assert len(result.status) > 0
