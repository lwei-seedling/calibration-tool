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
    3. Solve the portfolio allocation LP to MAXIMIZE total commercial capital
       mobilized, subject to catalytic budget, return, and CVaR constraints.

    LP formulation (Rockafellar-Uryasev CVaR linearization):

      MAXIMIZE    sum_v (1 - c_v) * w_v     (total commercial capital mobilized)
      s.t.
        [if catalytic_budget set]  sum_v c_v * w_v <= B_cat
        [else legacy]              sum_v w_v == B_total
        [if total_budget set]      sum_v w_v <= B_total
        [if min_deployment set]    sum_v w_v >= min_deployment
        w_v >= w_v_min
        w_v <= total_capital_v     (per-vehicle capacity)
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

    def _correlated_project_cashflows(
        self,
        vehicle_idx: int,
        seed_offset: int,
    ) -> list[np.ndarray]:
        """Simulate each project, then reorder paths to induce the target correlation.

        Each project is simulated independently and then its simulation paths are
        permuted (Iman-Conover) so that the *rank* of a project's total lifetime
        cashflow tracks the rank of a correlated normal draw. Because the draws
        carry the target correlation, the reordered projects do too.

        The permutation must map the project's own cashflow ordering onto the
        target ordering: the path with the k-th smallest total cashflow has to
        land where the k-th smallest draw sits. A permutation built from the
        draws alone reshuffles paths without regard to their magnitude, which
        leaves the projects independent no matter what correlation was asked for.

        Returns:
            List of J arrays, each (n_sims, T_max+1), zero-padded to a common
            horizon and reordered in place.
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

        J = len(project_cashflows)
        rng = np.random.default_rng(base_seed + 999)

        # Rank reordering reproduces the *rank* correlation of the draws, and the
        # rank correlation of a bivariate normal with linear correlation r is
        # (6/pi)*arcsin(r/2) — measurably below r. Inverting that here means the
        # matrix the user supplies is delivered as the achieved rank correlation
        # rather than landing ~0.10 low at mid-range values.
        target = np.asarray(vehicle.corr_array, dtype=float)
        draw_corr = 2.0 * np.sin(np.pi * np.clip(target, -1.0, 1.0) / 6.0)
        np.fill_diagonal(draw_corr, 1.0)

        corr_draws = cholesky_correlated_draws(n_sims, draw_corr, rng)  # (n_sims, J)

        correlated_cfs = []
        for j in range(J):
            cfs = project_cashflows[j]                               # (n_sims, T+1)
            # target_ranks[i] = rank the path at output position i should have
            target_ranks = np.argsort(np.argsort(corr_draws[:, j]))
            # sorted_idx[k] = index of the path with the k-th smallest total CF
            sorted_idx = np.argsort(cfs.sum(axis=1))
            # Output position i receives the path whose CF rank matches the draw
            # rank at i, so CF ranks now co-move exactly as the draws do.
            correlated_cfs.append(cfs[sorted_idx[target_ranks]])

        return correlated_cfs

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
        correlated_cfs = self._correlated_project_cashflows(vehicle_idx, seed_offset)
        return np.sum(correlated_cfs, axis=0)  # (n_sims, T+1)

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
            # No silent fallback. The old alpha=0.99 fallback reported a vehicle
            # as needing 99% catalytic capital when what actually happened is
            # that no catalytic fraction works — a very different answer, and one
            # the caller cannot distinguish from a real result.
            name = self.inputs.vehicles[vehicle_idx]
            raise ValueError(
                f"Vehicle {vehicle_idx} (capital ${name.total_capital:,.0f}) "
                f"cannot be calibrated: {exc}"
            ) from exc

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
        vehicle_capacities: np.ndarray,
    ) -> tuple[np.ndarray, str]:
        """Solve the portfolio allocation LP.

        Variables:
          w: (N_v,) allocations (total capital per vehicle)
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

        # c_v: catalytic fraction per vehicle
        c = np.array(catalytic_fractions)

        # l_vs: loss rate matrix (N_v, S)
        L = np.stack(loss_rates, axis=0)  # (N_v, S)

        # r_vs: return rate matrix (N_v, S)
        R = np.stack(return_rates, axis=0)  # (N_v, S)

        # Objective: MAXIMIZE total commercial capital mobilized
        objective = cp.Maximize((1.0 - c) @ w)

        constraints = [
            # Per-vehicle minimum allocation
            w >= cfg.min_allocation,
            # Per-vehicle capacity (natural upper bound from vehicle total_capital)
            w <= vehicle_capacities,
        ]

        # Per-vehicle concentration limit, as a fraction of the total budget.
        if cfg.max_allocation_fraction < 1.0:
            constraints.append(w <= cfg.max_allocation_fraction * B)

        if cfg.catalytic_budget is not None:
            # New mode: catalytic budget constraint (foundation's catalytic capital limit)
            constraints.append(c @ w <= cfg.catalytic_budget)
            # Total capital upper bound (optional)
            constraints.append(cp.sum(w) <= B)
        else:
            # Legacy mode: total budget equality
            constraints.append(cp.sum(w) == B)

        # Minimum total deployment (prevents degenerate all-zero solution)
        if cfg.min_deployment > 0.0:
            constraints.append(cp.sum(w) >= cfg.min_deployment)

        # Minimum expected return constraint.
        # The quantity of interest is the capital-weighted mean return,
        #   mean_s (R.T @ w)[s] / sum(w) >= R_min,
        # whose denominator is a decision variable. Multiplying through by
        # sum(w) >= 0 keeps it linear and, unlike dividing by the fixed budget B,
        # keeps the constraint meaning the same thing whatever fraction of the
        # budget is actually deployed.
        if cfg.min_expected_return > 0:
            constraints.append(
                (1.0 / S) * cp.sum(R.T @ w) >= cfg.min_expected_return * cp.sum(w)
            )

        # CVaR constraint (Rockafellar-Uryasev), on dollar losses.
        # CVaR is positively homogeneous, so constraining the portfolio *loss
        # rate* (dollar loss / capital deployed) is the same as
        #   CVaR(dollar loss) <= cvar_max * sum(w),
        # which is linear. This is the basis the result is reported on, so the
        # number the caller reads is directly comparable to cvar_max.
        portfolio_loss = L.T @ w  # shape (S,), dollars
        constraints += [
            zeta + (1.0 / (S * (1.0 - beta))) * cp.sum(u)
            <= cfg.cvar_max * cp.sum(w),
            u >= portfolio_loss - zeta,
        ]

        problem = cp.Problem(objective, constraints)

        # Try CLARABEL first, fall back to ECOS/SCS. A problem the solver proves
        # infeasible or unbounded will not become feasible under a different
        # solver, so only retry on solver *failure*.
        for solver in [cp.CLARABEL, cp.ECOS, cp.SCS]:
            try:
                problem.solve(solver=solver, verbose=False)
            except Exception:
                continue
            if problem.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
                break
            if problem.status in (cp.INFEASIBLE, cp.UNBOUNDED):
                break

        status = problem.status or "failed"

        if w.value is None:
            # Do not invent an allocation. An equal split looks like an answer
            # while satisfying none of the constraints, and callers that only
            # render the numbers will never notice.
            warnings.warn(
                f"Portfolio LP did not solve (status: {status}). Returning zero "
                "allocations — check cvar_max, min_deployment and the catalytic "
                "budget against the calibrated vehicles.",
                RuntimeWarning,
                stacklevel=2,
            )
            return np.zeros(N_v), status

        return np.maximum(0.0, w.value), status

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
        vehicle_capacities = np.array([v.total_capital for v in inputs.vehicles])
        w_opt, status = self._solve_lp(
            all_alphas, all_loss_rates, all_return_rates, vehicle_capacities
        )

        # Build result
        allocations = {v: float(w_opt[v]) for v in range(N_v)}
        catalytic_allocs = {v: float(w_opt[v] * all_alphas[v]) for v in range(N_v)}
        commercial_allocs = {v: float(w_opt[v] * (1.0 - all_alphas[v])) for v in range(N_v)}

        total_catalytic = sum(catalytic_allocs.values())
        total_commercial = sum(commercial_allocs.values())
        # No catalytic capital required is a real outcome, not a divide-by-zero to
        # paper over: max(x, 1e-9) turned it into a meaningless 1e16x leverage.
        leverage = (
            float("inf") if total_catalytic <= 0.0
            else total_commercial / total_catalytic
        )

        # Marginal catalytic efficiency per vehicle = (1-alpha)/alpha (leverage at min feasible alpha).
        # This is the commercial capital mobilized per unit of catalytic capital deployed.
        marginal_eff = {
            v: (float("inf") if all_alphas[v] <= 0.0
                else (1.0 - all_alphas[v]) / all_alphas[v])
            for v in range(N_v)
        }

        # Portfolio-level IRR/loss distributions (weighted by deployed capital)
        total_deployed = max(float(w_opt.sum()), 1e-9)
        portfolio_return_paths = np.zeros(n_sims)
        portfolio_loss_paths = np.zeros(n_sims)
        for v in range(N_v):
            weight = w_opt[v] / total_deployed
            portfolio_return_paths += weight * all_return_rates[v]
            portfolio_loss_paths += weight * all_loss_rates[v]

        # Reported on the same basis the LP constrains: loss per dollar deployed.
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
