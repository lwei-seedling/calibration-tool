"""Tests for the project simulation layer."""
import numpy as np
import pytest

from calibration.project.models import ProjectInputs, ProjectResult
from calibration.project.simulation import ProjectSimulator
from calibration.utils.irr import npv_loss


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


# ---------------------------------------------------------------------------
# New tests: NPV loss, base_cashflows, structural delay, backward compat
# ---------------------------------------------------------------------------

def test_npv_loss_larger_than_sum_loss_with_discount():
    """With discount_rate > 0, NPV loss should be >= sum-based loss for typical
    project finance cashflows (large upfront capex, back-loaded recovery)."""
    # Setup: large capex, revenues only in later years
    cashflows = np.array([[-1_000_000, 0, 0, 0, 0, 200_000, 200_000, 200_000, 200_000, 200_000]], dtype=float)
    # Sum loss: -1M + 5*200k = 0 → no loss
    sum_loss = np.maximum(0.0, -cashflows.sum(axis=1))
    assert sum_loss[0] == pytest.approx(0.0)
    # NPV loss at 10%: discounted value of future 200k CFs < 1M upfront
    npv_l = npv_loss(cashflows, discount_rate=0.10)
    assert npv_l[0] > 0, "NPV loss should be positive due to discounting of back-loaded CFs"


def test_npv_loss_zero_discount_equals_sum():
    """npv_loss with discount_rate=0.0 should equal max(0, -sum(CF))."""
    rng = np.random.default_rng(42)
    cashflows = rng.normal(0, 500_000, size=(200, 11))
    cashflows[:, 0] = -1_000_000
    sum_based = np.maximum(0.0, -cashflows.sum(axis=1))
    npv_based = npv_loss(cashflows, discount_rate=0.0)
    np.testing.assert_allclose(npv_based, sum_based, rtol=1e-9)


def test_base_cashflows_mode_shape():
    """base_cashflows mode should produce cashflows of correct shape."""
    base = [150_000, 180_000, 200_000, 210_000, 220_000]  # 5 operating years
    inputs = ProjectInputs(
        capex=800_000,
        base_cashflows=base,
        price_vol=0.15,
    )
    assert inputs.lifetime_years == 5
    sim = ProjectSimulator(inputs)
    result = sim.run(n_sims=100, seed=0)
    assert result.cashflows.shape == (100, 6)  # T+1 = 6


def test_base_cashflows_capex_at_t0():
    """Capex should appear as negative outflow at t=0 in cashflow mode."""
    base = [200_000] * 8
    inputs = ProjectInputs(capex=1_000_000, base_cashflows=base, price_vol=0.0)
    sim = ProjectSimulator(inputs)
    result = sim.run(n_sims=50, seed=0)
    np.testing.assert_allclose(result.cashflows[:, 0], -1_000_000)


def test_base_cashflows_mean_preserved():
    """With price_vol=0, all paths should equal exactly the base_cashflows."""
    base = [100_000, 120_000, 140_000, 160_000]
    inputs = ProjectInputs(capex=500_000, base_cashflows=base, price_vol=0.0)
    sim = ProjectSimulator(inputs)
    result = sim.run(n_sims=30, seed=0)
    expected = np.array([-500_000] + base, dtype=float)
    np.testing.assert_allclose(
        result.cashflows,
        np.tile(expected, (30, 1)),
        rtol=1e-9,
    )


def test_base_cashflows_no_parametric_fields_required():
    """ProjectInputs in cashflow mode should not require price/yield/opex."""
    inputs = ProjectInputs(
        capex=500_000,
        base_cashflows=[100_000, 110_000, 120_000],
    )
    assert inputs.lifetime_years == 3
    assert inputs.price is None
    assert inputs.yield_ is None
    assert inputs.opex_annual is None


def test_structural_delay_shifts_revenue():
    """With delay_years_probs=[0,0,1] (always 1-year delay), the first operating
    period should always have zero revenue."""
    inputs = ProjectInputs(
        capex=500_000,
        opex_annual=0.0,
        price=50.0,
        yield_=10_000,
        lifetime_years=5,
        price_vol=0.0,
        yield_vol=0.0,
        inflation_rate=0.0,
        fx_vol=0.0,
        delay_prob=0.0,
        delay_years_probs=[0.0, 0.0, 1.0],  # always 1-year delay
    )
    sim = ProjectSimulator(inputs)
    result = sim.run(n_sims=50, seed=0)
    # t=1 should be zero (delayed by 1 year) and opex is zero so CF[t=1]=0
    np.testing.assert_allclose(result.cashflows[:, 1], 0.0)
    # t=2..T should have positive revenue (shifted from t=1..T-1)
    assert np.all(result.cashflows[:, 2] > 0)


def test_delay_prob_backward_compat():
    """delay_prob still works after refactor: setting it to 0 gives higher
    average revenue than setting it to 1.0 (all periods zeroed)."""
    base_inputs = dict(
        capex=100_000,
        opex_annual=0.0,
        price=50.0,
        yield_=10_000,
        lifetime_years=5,
        price_vol=0.0,
        yield_vol=0.0,
        inflation_rate=0.0,
        fx_vol=0.0,
    )
    no_delay = ProjectSimulator(ProjectInputs(**base_inputs, delay_prob=0.0)).run(200, seed=1)
    all_delay = ProjectSimulator(ProjectInputs(**base_inputs, delay_prob=1.0)).run(200, seed=1)
    assert no_delay.cashflows[:, 1:].sum() > all_delay.cashflows[:, 1:].sum()
