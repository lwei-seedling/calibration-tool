"""Monte Carlo project simulator."""
from __future__ import annotations

import numpy as np

from calibration.project.models import ProjectInputs, ProjectResult
from calibration.utils.irr import batch_irr, clean_irr


class ProjectSimulator:
    """Simulates project cashflows via Monte Carlo.

    Each simulation path draws independent shocks for price, yield, FX,
    inflation, and production delays. The resulting cashflow matrix feeds
    into IRR computation and loss probability estimation.

    Cashflow sign convention:
      t=0: CF = -capex  (outflow)
      t≥1: CF = revenue[t] - opex[t]  (net operating cashflow)

    where:
      revenue[s, t] = price[s, t] * yield[s, t] * fx[s, t] * (1 if no delay else 0)
      opex[s, t]    = opex_annual * inflation_factor[s, t]
    """

    def __init__(self, inputs: ProjectInputs) -> None:
        self.inputs = inputs

    def run(self, n_sims: int = 1000, seed: int | None = None) -> ProjectResult:
        """Run Monte Carlo simulation.

        Args:
            n_sims: Number of simulation paths.
            seed: Random seed for reproducibility.

        Returns:
            ProjectResult with cashflows (n_sims, T+1), irr_distribution, loss_probability.
        """
        rng = np.random.default_rng(seed)
        p = self.inputs
        T = p.lifetime_years

        # Shape: (n_sims, T) for operating periods t=1..T
        # --- Price shocks (lognormal) ---
        price_shocks = rng.standard_normal((n_sims, T))
        prices = p.price * np.exp(
            (p.price_vol * price_shocks) - 0.5 * p.price_vol**2
        )

        # --- Yield shocks (lognormal) ---
        yield_shocks = rng.standard_normal((n_sims, T))
        yields = p.yield_ * np.exp(
            (p.yield_vol * yield_shocks) - 0.5 * p.yield_vol**2
        )

        # --- FX shocks (lognormal, multiplicative on revenue) ---
        fx_shocks = rng.standard_normal((n_sims, T))
        fx = np.exp(
            (p.fx_vol * fx_shocks) - 0.5 * p.fx_vol**2
        )

        # --- Inflation factor for opex ---
        t_idx = np.arange(1, T + 1, dtype=float)  # shape (T,)
        inflation_factors = (1.0 + p.inflation_rate) ** t_idx  # broadcast over sims

        # --- Delay indicator (Bernoulli: 1 = delay, 0 = production) ---
        delay = (rng.uniform(0.0, 1.0, (n_sims, T)) < p.delay_prob).astype(float)
        no_delay = 1.0 - delay

        # --- Revenue: price × yield × FX × (no delay) ---
        revenue = prices * yields * fx * no_delay  # (n_sims, T)

        # --- Opex (cost, expressed as positive; sign applied below) ---
        opex = p.opex_annual * inflation_factors  # (T,), broadcast over sims

        # --- Build cashflow matrix (n_sims, T+1) ---
        cashflows = np.empty((n_sims, T + 1), dtype=float)
        cashflows[:, 0] = -p.capex                      # t=0: capex outflow
        cashflows[:, 1:] = revenue - opex               # t=1..T: net operating CF

        # --- IRR ---
        raw_irr = batch_irr(cashflows)
        irr_distribution = clean_irr(raw_irr)

        # --- Loss probability ---
        terminal_loss = np.maximum(0.0, -cashflows.sum(axis=1))
        loss_probability = float(np.mean(terminal_loss > 0.0))

        return ProjectResult(
            cashflows=cashflows,
            irr_distribution=irr_distribution,
            loss_probability=loss_probability,
        )
