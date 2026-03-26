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
    """base_cashflows (full lifecycle) should produce cashflows of correct shape.

    base_cashflows = [CF_0, CF_1, ..., CF_T] → shape (n_sims, T+1).
    T = len(base_cashflows) - 1.
    """
    base = [-800_000, 150_000, 180_000, 200_000, 210_000, 220_000]  # t=0 + 5 operating years
    inputs = ProjectInputs(base_cashflows=base, price_vol=0.15)
    assert inputs.lifetime_years == 5
    sim = ProjectSimulator(inputs)
    result = sim.run(n_sims=100, seed=0)
    assert result.cashflows.shape == (100, 6)  # len(base) = 6


def test_base_cashflows_negative_t0_preserved():
    """CF[0] (capex outflow) must be passed through unchanged on all paths."""
    base = [-1_000_000, 200_000, 200_000, 200_000, 200_000, 200_000, 200_000, 200_000, 200_000]
    inputs = ProjectInputs(base_cashflows=base, price_vol=0.20)
    sim = ProjectSimulator(inputs)
    result = sim.run(n_sims=50, seed=0)
    # t=0 must equal base_cashflows[0] exactly on every path (no shock applied)
    np.testing.assert_allclose(result.cashflows[:, 0], base[0])


def test_base_cashflows_mean_preserved():
    """With price_vol=0, all paths should equal exactly the base_cashflows."""
    base = [-500_000, 100_000, 120_000, 140_000, 160_000]
    inputs = ProjectInputs(base_cashflows=base, price_vol=0.0)
    sim = ProjectSimulator(inputs)
    result = sim.run(n_sims=30, seed=0)
    expected = np.array(base, dtype=float)
    np.testing.assert_allclose(
        result.cashflows,
        np.tile(expected, (30, 1)),
        rtol=1e-9,
    )


def test_base_cashflows_no_parametric_fields_required():
    """ProjectInputs in cashflow mode should not require price/yield/opex."""
    inputs = ProjectInputs(
        base_cashflows=[-500_000, 100_000, 110_000, 120_000],
    )
    assert inputs.lifetime_years == 3
    assert inputs.price is None
    assert inputs.yield_ is None
    assert inputs.opex_annual is None


def test_multi_year_capex():
    """Multi-year capex (negative CF in t=0 and t=1) should be preserved unchanged."""
    base = [-500_000, -200_000, 80_000, 120_000, 150_000, 150_000]
    inputs = ProjectInputs(base_cashflows=base, price_vol=0.15)
    assert inputs.lifetime_years == 5
    sim = ProjectSimulator(inputs)
    result = sim.run(n_sims=50, seed=0)
    # Negative periods (t=0 and t=1) must be unshocked
    np.testing.assert_allclose(result.cashflows[:, 0], base[0])
    np.testing.assert_allclose(result.cashflows[:, 1], base[1])
    # Positive periods should vary across paths (stochastic)
    assert result.cashflows[:, 2].std() > 0


def test_price_index_shape_and_drift():
    """GBM price_index should have shape (n_sims, T) and correct mean at each step.

    With mu=0, sigma>0: E[price_index_t] ~ 1.0 for all t (martingale under Q).
    """
    base = [-100_000, 20_000, 20_000, 20_000, 20_000, 20_000]
    inputs = ProjectInputs(base_cashflows=base, price_vol=0.20)
    sim = ProjectSimulator(inputs)
    n_sims = 5000
    result = sim.run(n_sims=n_sims, seed=7)
    T = 5
    assert result.cashflows.shape == (n_sims, T + 1)
    # Mean of operating cashflows should be close to base (within 5%)
    mean_cf = result.cashflows[:, 1:].mean(axis=0)
    base_op = np.array(base[1:], dtype=float)
    np.testing.assert_allclose(mean_cf, base_op, rtol=0.05)


def test_revenue_cost_mode():
    """base_revenue + base_costs mode: cost is unshocked, revenue is shocked."""
    inputs = ProjectInputs(
        base_cashflows=[-300_000],
        base_revenue=[80_000, 90_000, 100_000],
        base_costs=[20_000, 20_000, 20_000],
        price_vol=0.0,
    )
    assert inputs.lifetime_years == 3
    sim = ProjectSimulator(inputs)
    result = sim.run(n_sims=20, seed=0)
    assert result.cashflows.shape == (20, 4)  # T+1 = 4
    # With price_vol=0: net CF[t] = revenue[t] - cost[t] for all paths
    expected_op = np.array([60_000, 70_000, 80_000], dtype=float)
    np.testing.assert_allclose(result.cashflows[:, 1:], np.tile(expected_op, (20, 1)), rtol=1e-9)
    np.testing.assert_allclose(result.cashflows[:, 0], -300_000)


def test_price_series_overrides_vol():
    """price_series should be used to estimate sigma (overrides price_vol)."""
    # Constant price series → zero vol → deterministic revenue
    price_series = [100.0, 100.0, 100.0, 100.0, 100.0]
    inputs = ProjectInputs(
        base_cashflows=[-200_000, 50_000, 60_000, 70_000],
        price_series=price_series,
        price_vol=0.99,  # should be overridden
    )
    sim = ProjectSimulator(inputs)
    result = sim.run(n_sims=30, seed=0)
    # sigma=0 from constant prices → all paths identical
    np.testing.assert_allclose(result.cashflows, np.tile(result.cashflows[0], (30, 1)))


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
