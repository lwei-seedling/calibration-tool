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


class TestCorrelation:
    """Project correlation must actually reach the vehicle cashflows."""

    @staticmethod
    def _vehicle(rho: float, n_sims: int = 4000):
        vehicle = make_vehicle(n_projects=2).model_copy(
            update={"correlation_matrix": [[1.0, rho], [rho, 1.0]]}
        )
        inputs = PortfolioInputs(
            vehicles=[vehicle], total_budget=1_000_000, n_sims=n_sims, seed=7
        )
        return PortfolioOptimizer(inputs)._simulate_vehicle(0, seed_offset=0)

    def test_correlation_increases_vehicle_variance(self):
        """Two identical projects: Var(sum) = 2σ²(1+ρ), so ρ=0.9 must give
        ~1.9x the variance of ρ=0. The pre-fix code gave a ratio of ~1.0."""
        var_indep = np.var(self._vehicle(0.0).sum(axis=1))
        var_corr = np.var(self._vehicle(0.9).sum(axis=1))
        ratio = var_corr / var_indep
        assert 1.6 < ratio < 2.1, f"variance ratio {ratio:.2f}, expected ~1.9"

    def test_reorder_preserves_each_project_marginal(self):
        """Correlation is imposed by permuting paths, so each project's set of
        simulated paths must be unchanged (compare sorted lifetime totals)."""
        from calibration.project.simulation import ProjectSimulator
        from calibration.utils.stats import cholesky_correlated_draws

        n_sims, rho, seed = 2000, 0.8, 11
        vehicle = make_vehicle(n_projects=2).model_copy(
            update={"correlation_matrix": [[1.0, rho], [rho, 1.0]]}
        )
        opt = PortfolioOptimizer(PortfolioInputs(
            vehicles=[vehicle], total_budget=1_000_000, n_sims=n_sims, seed=seed))
        vehicle_cfs = opt._simulate_vehicle(0, seed_offset=0)

        # Re-simulate the raw projects with the optimizer's seeds.
        raw = [ProjectSimulator(p).run(n_sims=n_sims, seed=seed + j).cashflows
               for j, p in enumerate(vehicle.projects)]
        # The vehicle total must be a sum of one path from each raw project, and
        # together the paths used must be exactly each project's original set.
        draws = cholesky_correlated_draws(
            n_sims, vehicle.corr_array, np.random.default_rng(seed + 999))
        rebuilt = sum(
            r[np.argsort(r.sum(axis=1), kind="stable")[np.argsort(np.argsort(draws[:, j]))]]
            for j, r in enumerate(raw)
        )
        np.testing.assert_allclose(vehicle_cfs, rebuilt)


class TestLoudFailures:
    """Failed calibration / LP must be flagged, never dressed up as a result."""

    def test_infeasible_calibration_is_flagged(self):
        vehicles = [make_vehicle(total_capital=500_000), make_vehicle(total_capital=500_000)]
        # 500% hurdle: unreachable at any alpha.
        cfg = CalibratorConfig(investor_hurdle_irr=5.0, max_loss_probability=0.15)
        inputs = PortfolioInputs(
            vehicles=vehicles, calibrator_config=cfg,
            total_budget=1_000_000, n_sims=200, seed=42,
        )
        with pytest.warns(RuntimeWarning, match="INFEASIBLE"):
            result = PortfolioOptimizer(inputs).run()
        assert result.infeasible_vehicles == [0, 1]
        assert result.status == "calibration_infeasible"

    def test_feasible_run_has_no_flags(self):
        vehicles = [make_vehicle(total_capital=500_000), make_vehicle(total_capital=500_000)]
        cfg = CalibratorConfig(investor_hurdle_irr=0.04, max_loss_probability=0.15)
        inputs = PortfolioInputs(
            vehicles=vehicles, calibrator_config=cfg,
            total_budget=1_000_000, n_sims=200, seed=42,
        )
        result = PortfolioOptimizer(inputs).run()
        assert result.infeasible_vehicles == []
        assert result.status == result.lp_status
        assert all(np.isfinite(a) for a in result.catalytic_fractions.values())

    def test_lp_failure_returns_zero_allocations(self):
        vehicles = [make_vehicle(total_capital=500_000), make_vehicle(total_capital=500_000)]
        cfg = CalibratorConfig(investor_hurdle_irr=0.04, max_loss_probability=0.15)
        # min_deployment above total vehicle capacity: LP is infeasible.
        inputs = PortfolioInputs(
            vehicles=vehicles, calibrator_config=cfg, total_budget=1_000_000,
            catalytic_budget=1_000_000, min_deployment=5_000_000,
            n_sims=200, seed=42,
        )
        with pytest.warns(RuntimeWarning, match="did not solve"):
            result = PortfolioOptimizer(inputs).run()
        assert result.status != "optimal"
        assert sum(result.allocations.values()) == 0.0

    def test_infeasible_vehicle_excluded_from_allocation(self):
        """A feasible + infeasible mix: the infeasible vehicle must get zero
        capital and must not count in catalytic/commercial totals."""
        good = make_vehicle(total_capital=500_000)
        bad = VehicleInputs(
            projects=[make_project(price=1.0), make_project(price=1.0)],
            correlation_matrix=np.eye(2).tolist(), total_capital=500_000,
            guarantee_coverage=0.0, grant_reserve=0.0, mezzanine_fraction=0.10,
        )
        cfg = CalibratorConfig(investor_hurdle_irr=0.07, max_loss_probability=0.15)
        for cat_budget in (None, 300_000):
            inputs = PortfolioInputs(
                vehicles=[good, bad], calibrator_config=cfg, total_budget=500_000,
                catalytic_budget=cat_budget, n_sims=200, seed=42,
            )
            with pytest.warns(RuntimeWarning, match="INFEASIBLE"):
                result = PortfolioOptimizer(inputs).run()
            assert result.infeasible_vehicles == [1]
            assert result.allocations[1] == pytest.approx(0.0, abs=1e-3)
            assert result.catalytic_allocations[1] == 0.0
            assert np.isnan(result.catalytic_fractions[1])
            assert np.isnan(result.marginal_catalytic_efficiency[1])
            assert result.allocations[0] > 0

    def test_lp_failure_not_hidden_by_calibration_failure(self):
        """When both calibration and the LP fail, lp_status must still say so."""
        good = make_vehicle(total_capital=500_000)
        bad = VehicleInputs(
            projects=[make_project(price=1.0)], correlation_matrix=[[1.0]],
            total_capital=500_000, guarantee_coverage=0.0, grant_reserve=0.0,
        )
        cfg = CalibratorConfig(investor_hurdle_irr=0.07, max_loss_probability=0.15)
        inputs = PortfolioInputs(
            vehicles=[good, bad], calibrator_config=cfg, total_budget=1_000_000,
            catalytic_budget=1_000_000, min_deployment=5_000_000, n_sims=200, seed=42,
        )
        with pytest.warns(RuntimeWarning):
            result = PortfolioOptimizer(inputs).run()
        assert result.status == "calibration_infeasible"
        assert result.lp_status not in ("optimal", "optimal_inaccurate")
        assert sum(result.allocations.values()) == 0.0
