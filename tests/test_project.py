"""Tests for the project simulation layer."""
import numpy as np
import pytest

from calibration.project.models import ProjectInputs, ProjectResult
from calibration.project.simulation import ProjectSimulator


@pytest.fixture
def base_inputs():
    return ProjectInputs(
        capex=1_000_000,
        opex_annual=50_000,
        price=100.0,
        yield_=20_000,
        lifetime_years=10,
        price_vol=0.15,
        yield_vol=0.10,
        inflation_rate=0.02,
        fx_vol=0.05,
        delay_prob=0.05,
    )


def test_cashflow_shape(base_inputs):
    sim = ProjectSimulator(base_inputs)
    result = sim.run(n_sims=200, seed=42)
    assert result.cashflows.shape == (200, 11)  # T+1 = 11


def test_capex_at_t0(base_inputs):
    sim = ProjectSimulator(base_inputs)
    result = sim.run(n_sims=100, seed=0)
    # All t=0 cashflows should equal -capex
    np.testing.assert_allclose(result.cashflows[:, 0], -base_inputs.capex)


def test_irr_distribution_shape(base_inputs):
    sim = ProjectSimulator(base_inputs)
    result = sim.run(n_sims=500, seed=7)
    assert result.irr_distribution.shape == (500,)


def test_irr_distribution_range(base_inputs):
    sim = ProjectSimulator(base_inputs)
    result = sim.run(n_sims=500, seed=7)
    # All IRRs should be >= -1.0 (total loss sentinel) and finite or capped
    assert np.all(result.irr_distribution >= -1.0)
    assert np.all(np.isfinite(result.irr_distribution))


def test_loss_probability_in_range(base_inputs):
    sim = ProjectSimulator(base_inputs)
    result = sim.run(n_sims=500, seed=42)
    assert 0.0 <= result.loss_probability <= 1.0


def test_reproducibility(base_inputs):
    sim = ProjectSimulator(base_inputs)
    r1 = sim.run(n_sims=100, seed=99)
    r2 = sim.run(n_sims=100, seed=99)
    np.testing.assert_array_equal(r1.cashflows, r2.cashflows)
    np.testing.assert_array_equal(r1.irr_distribution, r2.irr_distribution)


def test_zero_vol_deterministic():
    """With zero volatility and no delay, all paths should be identical."""
    inputs = ProjectInputs(
        capex=500_000,
        opex_annual=20_000,
        price=50.0,
        yield_=20_000,
        lifetime_years=5,
        price_vol=0.0,
        yield_vol=0.0,
        inflation_rate=0.0,
        fx_vol=0.0,
        delay_prob=0.0,
    )
    sim = ProjectSimulator(inputs)
    result = sim.run(n_sims=50, seed=0)
    # All paths should have the same cashflows
    np.testing.assert_allclose(result.cashflows, np.tile(result.cashflows[0], (50, 1)))


def test_high_loss_project():
    """A project with huge capex and tiny revenue should have high loss probability."""
    inputs = ProjectInputs(
        capex=10_000_000,
        opex_annual=100_000,
        price=1.0,
        yield_=100,
        lifetime_years=5,
        price_vol=0.0,
        yield_vol=0.0,
        inflation_rate=0.0,
        fx_vol=0.0,
        delay_prob=0.0,
    )
    sim = ProjectSimulator(inputs)
    result = sim.run(n_sims=100, seed=0)
    assert result.loss_probability == 1.0


def test_profitable_project():
    """A very profitable project should have near-zero loss probability."""
    inputs = ProjectInputs(
        capex=100_000,
        opex_annual=5_000,
        price=100.0,
        yield_=50_000,
        lifetime_years=10,
        price_vol=0.0,
        yield_vol=0.0,
        inflation_rate=0.0,
        fx_vol=0.0,
        delay_prob=0.0,
    )
    sim = ProjectSimulator(inputs)
    result = sim.run(n_sims=100, seed=0)
    assert result.loss_probability == 0.0
