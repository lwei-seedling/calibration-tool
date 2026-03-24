"""Portfolio optimizer: orchestrates the full pipeline and solves the allocation LP."""
from __future__ import annotations

import warnings

import cvxpy as cp
import numpy as np

from calibration.portfolio.models import PortfolioInputs, PortfolioResult
from calibration.project.simulation import ProjectSimulator
from calibration.utils.irr import batch_irr, clean_irr
from calibration.utils.stats import cholesky_correlated_draws, cvar
from calibration.vehicle.calibration import CatalyticCalibrator
from calibration.vehicle.capital_stack import CapitalStack
from calibration.vehicle.risk_mitigants import Guarantee, GrantReserve
from calibration.vehicle.risk_mitigants import CoverageType


class PortfolioOptimizer:
    """Orchestrates the PROJECT → VEHICLE → PORTFOLIO pipeline.

    Steps:
    1. For each vehicle: simulate correlated project cashflows.
    2. Calibrate the minimum catalytic fraction per vehicle.
    3. Solve the portfolio allocation LP to minimise total catalytic capital
       subject to budget, return, and CVaR constraints.

    LP formulation (Rockafellar-Uryasev CVaR linearization):

      MINIMIZE    sum_v c_v * w_v           (total catalytic capital)
      s.t.
        sum_v w_v = B
        w_v_min <= w_v <= w_v_max
        mean(R_portfolio) >= R_min
        zeta + 1/(S*(1-beta)) * sum_s u_s <= CVaR_max
        u_s >= sum_v l_{v,s} * w_v - zeta    for all s
        u_s >= 0, w_v >= 0
    """

    def __init__(self, inputs: PortfolioInputs) -> None:
        self.inputs = inputs

    # ------------------------------------------------------------------
    # Step 1: Simulate vehicle cashflows
    # ------------------------------------------------------------------

    def _simulate_vehicle(
        self,
        vehicle_idx: int,
        seed_offset: int,
    ) -> np.ndarray:
        """Run correlated Monte Carlo for all projects in a vehicle.

        Returns:
            vehicle_cashflows: shape (n_sims, T+1) — sum of project cashflows
              after applying Cholesky-correlated shocks at vehicle level.
        """
        inputs = self.inputs
        vehicle = inputs.vehicles[vehicle_idx]
        n_sims = inputs.n_sims
        base_seed = (inputs.seed or 0) + seed_offset * 1000

        # Simulate each project independently first
        project_cashflows_raw = []
        for j, proj_inputs in enumerate(vehicle.projects):
            sim = ProjectSimulator(proj_inputs)
            result = sim.run(n_sims=n_sims, seed=base_seed + j)
            project_cashflows_raw.append(result.cashflows)  # (n_sims, T_j+1)

        # Pad all cashflow arrays to the maximum lifetime in this vehicle so
        # they can be summed into a single (n_sims, T_max+1) matrix.
        T = max(cfs.shape[1] - 1 for cfs in project_cashflows_raw)
        project_cashflows = []
        for cfs in project_cashflows_raw:
            if cfs.shape[1] < T + 1:
                pad_cols = T + 1 - cfs.shape[1]
                cfs = np.concatenate([cfs, np.zeros((n_sims, pad_cols))], axis=1)
            project_cashflows.append(cfs)

        # Re-correlate project cashflows at vehicle level using Cholesky
        # We correlate per-period cashflow innovations across projects.
        J = len(project_cashflows)
        corr = vehicle.corr_array  # (J, J)
        rng = np.random.default_rng(base_seed + 999)

        # Generate J-dimensional correlated draws (n_sims, J)
        corr_draws = cholesky_correlated_draws(n_sims, corr, rng)  # (n_sims, J)

        # Apply correlation adjustment: replace project returns with correlated ordering.
        # Method: rank-based reordering (Iman-Conover style). For each project j,
        # reorder its simulation paths to match the rank ordering from corr_draws[:, j].
        correlated_cfs = []
        for j in range(J):
            cfs = project_cashflows[j]  # (n_sims, T+1)
            target_ranks = np.argsort(np.argsort(corr_draws[:, j]))  # rank of each path
            source_ranks = np.argsort(np.argsort(cfs.sum(axis=1)))   # rank by total CF
            # Reorder: path with rank k in source gets the position of rank k in target
            reorder_idx = np.empty(n_sims, dtype=int)
            reorder_idx[target_ranks] = np.arange(n_sims)
            cfs_reordered = cfs[reorder_idx]
            correlated_cfs.append(cfs_reordered)

        # Sum across projects → vehicle cashflows
        vehicle_cashflows = np.sum(correlated_cfs, axis=0)  # (n_sims, T+1)
        return vehicle_cashflows

    # ------------------------------------------------------------------
    # Step 2: Build CapitalStack and calibrate alpha* per vehicle
    # ------------------------------------------------------------------

    def _build_capital_stack(self, vehicle_idx: int) -> CapitalStack:
        vehicle = self.inputs.vehicles[vehicle_idx]
        T = max(p.lifetime_years for p in vehicle.projects)
        return CapitalStack(
            total_capital=vehicle.total_capital,
            grant_reserve=GrantReserve(vehicle.grant_reserve),
            guarantee=Guarantee(
                coverage_limit=vehicle.guarantee_coverage,
                coverage_type=CoverageType.PERCENTAGE,
            ),
            senior_coupon=vehicle.senior_coupon,
            mezzanine_coupon=vehicle.mezzanine_coupon,
            mezzanine_fraction=vehicle.mezzanine_fraction,
            lifetime_years=T,
            discount_rate=vehicle.discount_rate,
        )

    def _calibrate_vehicle(
        self,
        vehicle_idx: int,
        vehicle_cashflows: np.ndarray,
    ) -> float:
        capital_stack = self._build_capital_stack(vehicle_idx)
        calibrator = CatalyticCalibrator(
            capital_stack=capital_stack,
            vehicle_cashflows=vehicle_cashflows,
            config=self.inputs.calibrator_config,
        )
        try:
            return calibrator.calibrate()
        except ValueError as exc:
            warnings.warn(
                f"Vehicle {vehicle_idx} calibration failed: {exc}. "
                "Using alpha=0.99 as fallback.",
                RuntimeWarning,
                stacklevel=3,
            )
            return 0.99

    # ------------------------------------------------------------------
    # Step 3: Extract per-vehicle loss and return distributions
    # ------------------------------------------------------------------

    def _vehicle_distributions(
        self,
        vehicle_idx: int,
        vehicle_cashflows: np.ndarray,
        alpha: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (loss_rate, return_rate) arrays of shape (n_sims,).

        loss_rate[s]   = senior absorbed loss / vehicle total capital
        return_rate[s] = senior IRR (clean) per path
        """
        capital_stack = self._build_capital_stack(vehicle_idx)
        tranche_results = capital_stack.waterfall(vehicle_cashflows, alpha)
        total_capital = self.inputs.vehicles[vehicle_idx].total_capital

        if "senior" in tranche_results:
            senior = tranche_results["senior"]
            loss_rate = senior.loss_distribution / max(total_capital, 1e-9)
            return_rate = senior.irr_distribution
        else:
            n_sims = vehicle_cashflows.shape[0]
            loss_rate = np.zeros(n_sims)
            return_rate = np.full(n_sims, self.inputs.calibrator_config.investor_hurdle_irr)

        return loss_rate, return_rate

    # ------------------------------------------------------------------
    # Step 4: Solve allocation LP
    # ------------------------------------------------------------------

    def _solve_lp(
        self,
        catalytic_fractions: list[float],
        loss_rates: list[np.ndarray],
        return_rates: list[np.ndarray],
    ) -> tuple[np.ndarray, str]:
        """Solve the portfolio allocation LP.

        Variables:
          w: (N_v,) allocations
          zeta: scalar VaR threshold
          u: (S,) excess loss auxiliary variables

        Returns:
            (w_optimal, status)
        """
        cfg = self.inputs
        N_v = len(cfg.vehicles)
        S = cfg.n_sims
        B = cfg.total_budget
        beta = cfg.cvar_confidence

        # Decision variables
        w = cp.Variable(N_v, nonneg=True, name="w")
        zeta = cp.Variable(name="zeta")
        u = cp.Variable(S, nonneg=True, name="u")

        # c_v: catalytic fraction (objective coefficient)
        c = np.array(catalytic_fractions)

        # l_vs: loss rate matrix (N_v, S)
        L = np.stack(loss_rates, axis=0)  # (N_v, S)

        # r_vs: return rate matrix (N_v, S)
        R = np.stack(return_rates, axis=0)  # (N_v, S)

        # Objective: minimize total catalytic capital deployed
        objective = cp.Minimize(c @ w)

        constraints = [
            # Budget equality
            cp.sum(w) == B,
            # Per-vehicle bounds
            w >= cfg.min_allocation,
            w <= cfg.max_allocation_fraction * B,
        ]

        # Minimum expected return constraint
        if cfg.min_expected_return > 0:
            # mean portfolio return = (1/S) * sum_s sum_v w_v * R_v[s] / B
            portfolio_return_sum = cp.sum(R @ w)  # shape scalar: sum over s of (sum_v R_{v,s} * w_v)
            # But R has shape (N_v, S), so R @ w gives (S,) portfolio returns per path
            # Actually R is (N_v, S), so R.T @ w gives (S,) portfolio-level returns weighted by w
            # Mean return = (1/S) * sum_s (R.T @ w)[s] / B
            constraints.append(
                (1.0 / (S * B)) * cp.sum(R.T @ w) >= cfg.min_expected_return
            )

        # CVaR constraint (Rockafellar-Uryasev)
        # Portfolio loss rate = (1/B) * sum_v w_v * l_v[s] = (L.T @ w) / B
        portfolio_loss = (L.T @ w) / B  # shape (S,)
        constraints += [
            zeta + (1.0 / (S * (1.0 - beta))) * cp.sum(u) <= cfg.cvar_max,
            u >= portfolio_loss - zeta,
        ]

        problem = cp.Problem(objective, constraints)

        # Try CLARABEL first, fall back to ECOS/SCS
        for solver in [cp.CLARABEL, cp.ECOS, cp.SCS]:
            try:
                problem.solve(solver=solver, verbose=False)
                if problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                    break
            except Exception:
                continue

        if w.value is None:
            return np.full(N_v, B / N_v), problem.status or "failed"

        return np.maximum(0.0, w.value), problem.status or "optimal"

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self) -> PortfolioResult:
        """Execute the full pipeline and return optimization results."""
        inputs = self.inputs
        N_v = len(inputs.vehicles)
        n_sims = inputs.n_sims

        # Per-vehicle results
        all_vehicle_cashflows: list[np.ndarray] = []
        all_alphas: list[float] = []
        all_loss_rates: list[np.ndarray] = []
        all_return_rates: list[np.ndarray] = []

        for v_idx in range(N_v):
            vehicle_cfs = self._simulate_vehicle(v_idx, seed_offset=v_idx)
            alpha = self._calibrate_vehicle(v_idx, vehicle_cfs)
            loss_rate, return_rate = self._vehicle_distributions(v_idx, vehicle_cfs, alpha)

            all_vehicle_cashflows.append(vehicle_cfs)
            all_alphas.append(alpha)
            all_loss_rates.append(loss_rate)
            all_return_rates.append(return_rate)

        # Solve allocation LP
        w_opt, status = self._solve_lp(all_alphas, all_loss_rates, all_return_rates)

        # Build result
        allocations = {v: float(w_opt[v]) for v in range(N_v)}
        catalytic_allocs = {v: float(w_opt[v] * all_alphas[v]) for v in range(N_v)}
        commercial_allocs = {v: float(w_opt[v] * (1.0 - all_alphas[v])) for v in range(N_v)}

        total_catalytic = sum(catalytic_allocs.values())
        total_commercial = sum(commercial_allocs.values())
        leverage = total_commercial / max(total_catalytic, 1e-9)

        marginal_eff = {
            v: (1.0 - all_alphas[v]) / max(all_alphas[v], 1e-9)
            for v in range(N_v)
        }

        # Portfolio-level IRR distribution (weighted by allocation)
        portfolio_return_paths = np.zeros(n_sims)
        portfolio_loss_paths = np.zeros(n_sims)
        for v in range(N_v):
            weight = w_opt[v] / max(inputs.total_budget, 1e-9)
            portfolio_return_paths += weight * all_return_rates[v]
            portfolio_loss_paths += weight * all_loss_rates[v]

        portfolio_cvar = cvar(portfolio_loss_paths, inputs.cvar_confidence)

        return PortfolioResult(
            allocations=allocations,
            catalytic_allocations=catalytic_allocs,
            commercial_allocations=commercial_allocs,
            catalytic_fractions={v: all_alphas[v] for v in range(N_v)},
            leverage_ratio=leverage,
            marginal_catalytic_efficiency=marginal_eff,
            portfolio_irr_distribution=portfolio_return_paths,
            portfolio_loss_distribution=portfolio_loss_paths,
            cvar_95=portfolio_cvar,
            status=status,
        )
