"""Monte Carlo project simulator."""
from __future__ import annotations

import numpy as np

from calibration.project.models import ProjectInputs, ProjectResult
from calibration.utils.irr import batch_irr, clean_irr, npv_loss


class ProjectSimulator:
    """Simulates project cashflows via Monte Carlo.

    Supports two input modes determined by `ProjectInputs`:

    **Parametric mode**: draws independent shocks for price, yield, FX,
    inflation, and delay. Applies a structural delay (delay_years_probs) that
    shifts the revenue start date, followed by the per-period delay_prob.

    **Cashflow mode**: scales a user-supplied `base_cashflows` vector by a
    single lognormal multiplier per path (sigma=price_vol). Capex is placed
    at t=0; base_cashflows become t=1..T.

    Cashflow sign convention:
      t=0: CF = -capex  (investment outflow)
      t≥1: CF = revenue[t] - opex[t]  (net operating cashflow)

    Loss is computed using NPV at `discount_rate` (defaults to 0.0, giving
    sum-based loss for backward compatibility).
    """

    def __init__(self, inputs: ProjectInputs) -> None:
        self.inputs = inputs

    def run(self, n_sims: int = 1000, seed: int | None = None) -> ProjectResult:
        """Run Monte Carlo simulation.

        Args:
            n_sims: Number of simulation paths.
            seed: Random seed for reproducibility.

        Returns:
            ProjectResult with cashflows (n_sims, T+1), irr_distribution,
            loss_probability (NPV-based).
        """
        rng = np.random.default_rng(seed)
        p = self.inputs

        if p.base_cashflows is not None:
            cashflows = self._run_cashflow_mode(rng, n_sims, p)
        else:
            cashflows = self._run_parametric_mode(rng, n_sims, p)

        # --- IRR ---
        raw_irr = batch_irr(cashflows)
        irr_distribution = clean_irr(raw_irr)

        # --- Loss probability (NPV-based) ---
        terminal_loss = npv_loss(cashflows, p.discount_rate)
        loss_probability = float(np.mean(terminal_loss > 0.0))

        return ProjectResult(
            cashflows=cashflows,
            irr_distribution=irr_distribution,
            loss_probability=loss_probability,
        )

    # ------------------------------------------------------------------
    # Cashflow-based simulation path
    # ------------------------------------------------------------------

    def _run_cashflow_mode(
        self,
        rng: np.random.Generator,
        n_sims: int,
        p: ProjectInputs,
    ) -> np.ndarray:
        """Scale user-supplied base_cashflows with a lognormal multiplier.

        Each path s receives a scalar multiplier:
            m[s] ~ LogNormal(mu = -0.5*sigma^2, sigma = price_vol)
        so that E[m] = 1 and the base cashflows are preserved on average.
        """
        base = np.asarray(p.base_cashflows, dtype=float)  # (T,)
        T = len(base)
        sigma = p.price_vol

        shocks = rng.standard_normal(n_sims)
        multipliers = np.exp(sigma * shocks - 0.5 * sigma ** 2)  # (n_sims,)

        cashflows = np.empty((n_sims, T + 1), dtype=float)
        cashflows[:, 0] = -(p.capex or 0.0)          # t=0: investment outflow
        cashflows[:, 1:] = base * multipliers[:, None]  # (n_sims, T)
        return cashflows

    # ------------------------------------------------------------------
    # Parametric simulation path
    # ------------------------------------------------------------------

    def _run_parametric_mode(
        self,
        rng: np.random.Generator,
        n_sims: int,
        p: ProjectInputs,
    ) -> np.ndarray:
        T = p.lifetime_years  # type: ignore[assignment]

        # --- Price shocks (lognormal) ---
        price_shocks = rng.standard_normal((n_sims, T))
        prices = p.price * np.exp(p.price_vol * price_shocks - 0.5 * p.price_vol ** 2)

        # --- Yield shocks (lognormal) ---
        yield_shocks = rng.standard_normal((n_sims, T))
        yields = p.yield_ * np.exp(p.yield_vol * yield_shocks - 0.5 * p.yield_vol ** 2)

        # --- FX shocks (lognormal, multiplicative on revenue) ---
        fx_shocks = rng.standard_normal((n_sims, T))
        fx = np.exp(p.fx_vol * fx_shocks - 0.5 * p.fx_vol ** 2)

        # --- Inflation factor for opex ---
        t_idx = np.arange(1, T + 1, dtype=float)
        inflation_factors = (1.0 + p.inflation_rate) ** t_idx  # (T,)
        opex = p.opex_annual * inflation_factors  # (T,)

        # --- Base revenue (before delay application) ---
        revenue = prices * yields * fx  # (n_sims, T)

        # --- Structural delay (delay_years_probs) ---
        # Draws a single delay per path: 0y, 0.5y, or 1y.
        #   0y  : no change to revenue
        #   0.5y: first operating period gets half revenue
        #   1y  : first period zeroed; all subsequent periods shifted right
        #         (last period's revenue is lost)
        delay_choices = np.array([0.0, 0.5, 1.0])
        probs = np.array(p.delay_years_probs, dtype=float)
        delay_per_path = rng.choice(delay_choices, size=n_sims, p=probs)  # (n_sims,)

        half_mask = delay_per_path == 0.5
        full_mask = delay_per_path == 1.0

        if np.any(half_mask):
            revenue[half_mask, 0] *= 0.5

        if np.any(full_mask):
            # Shift revenue right by one period (lose the last period's revenue)
            revenue[full_mask, 1:] = revenue[full_mask, :-1].copy()
            revenue[full_mask, 0] = 0.0

        # --- Per-period delay_prob (deprecated; kept for backward compatibility) ---
        if p.delay_prob > 0.0:
            per_period_no_delay = (
                rng.uniform(0.0, 1.0, (n_sims, T)) >= p.delay_prob
            ).astype(float)
            revenue = revenue * per_period_no_delay

        # --- Build cashflow matrix (n_sims, T+1) ---
        cashflows = np.empty((n_sims, T + 1), dtype=float)
        cashflows[:, 0] = -p.capex                # t=0: capex outflow
        cashflows[:, 1:] = revenue - opex         # t=1..T: net operating CF
        return cashflows
