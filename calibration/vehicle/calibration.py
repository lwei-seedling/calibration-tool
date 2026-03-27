"""Catalytic capital calibrator: solves for minimum alpha meeting investor constraints."""
from __future__ import annotations

import warnings

import numpy as np
from pydantic import BaseModel, Field
from scipy.optimize import brentq

from calibration.vehicle.capital_stack import CapitalStack


class CalibratorConfig(BaseModel):
    """Configuration for the catalytic calibration algorithm."""

    investor_hurdle_irr: float = Field(
        0.08, description="Minimum median senior IRR required by commercial investors."
    )
    max_loss_probability: float = Field(
        0.05, ge=0.0, le=1.0, description="Maximum acceptable senior tranche loss probability."
    )
    alpha_lo: float = Field(0.0, ge=0.0, le=1.0, description="Lower bound for catalytic fraction search.")
    alpha_hi: float = Field(0.99, ge=0.0, le=1.0, description="Upper bound for catalytic fraction search.")
    brentq_xtol: float = Field(1e-4, description="Absolute tolerance for Brent's method.")
    monotonicity_n_check: int = Field(20, description="Points to use for monotonicity verification.")
    monotonicity_epsilon: float = Field(0.005, description="Tolerance for monotonicity check.")
    grid_n_coarse: int = Field(50, description="Points in coarse grid fallback.")
    grid_n_fine: int = Field(50, description="Points in fine grid fallback.")


class CatalyticCalibrator:
    """Finds the minimum catalytic fraction alpha* such that:

      median_s { IRR_senior(alpha, s) } >= hurdle_irr
      P_s { loss_senior(alpha, s) > 0 } <= max_loss_prob

    **Vehicle-level risk**: investor constraints (hurdle IRR, max loss probability)
    are evaluated on the VEHICLE-LEVEL senior tranche, not on individual projects.
    Individual projects may have negative NPV or high loss probability. What matters
    is whether the vehicle's waterfall structure (diversification + subordination
    layers) produces a senior tranche that meets investor thresholds.
    Catalytic capital is allocated at the vehicle level and absorbed via the capital
    stack (Grant Reserve → First-Loss → Mezzanine → Guarantee → Senior).

    Uses Brent's method when the objective is monotone, falls back to a
    two-phase grid search when it is not (typically due to simulation noise).

    Common-random-number technique: simulation paths are pre-generated once
    and reused across all alpha evaluations to keep h(alpha) smooth.
    """

    def __init__(
        self,
        capital_stack: CapitalStack,
        vehicle_cashflows: np.ndarray,
        config: CalibratorConfig | None = None,
    ) -> None:
        self.capital_stack = capital_stack
        self.vehicle_cashflows = vehicle_cashflows
        self.config = config or CalibratorConfig()

    def _h(self, alpha: float) -> float:
        """Objective function: negative when infeasible, non-negative when feasible.

        h(alpha) = min(g1(alpha), g2(alpha))

        where:
            g1(alpha) = median_IRR_senior(alpha) - hurdle_irr
            g2(alpha) = max_loss_prob - loss_prob_senior(alpha)

        Note on smoothness: min(g1, g2) is non-smooth at the boundary where
        g1 == g2 (the kink). For well-conditioned calibrations this is rarely
        the root and Brent's method handles it correctly (it only requires a
        sign change, not differentiability). If the kink causes convergence
        issues, consider replacing min() with a smooth penalty such as:
            h(alpha) = g1(alpha) + g2(alpha) - |g1(alpha) - g2(alpha)|  (LogSumExp)
        or evaluating the two constraints sequentially (find min alpha s.t.
        g1>=0, then verify g2>=0 at that point).
        """
        cfg = self.config
        results = self.capital_stack.waterfall(self.vehicle_cashflows, alpha)

        if "senior" not in results:
            # Degenerate: no senior tranche (alpha ≈ 1.0)
            return 1.0

        senior = results["senior"]
        g1 = float(np.nanmedian(senior.irr_distribution)) - cfg.investor_hurdle_irr
        g2 = cfg.max_loss_probability - senior.loss_probability
        return min(g1, g2)

    def _check_monotonicity(self) -> bool:
        """Evaluate h at N evenly-spaced alpha points and verify non-decreasing."""
        cfg = self.config
        alphas = np.linspace(cfg.alpha_lo, cfg.alpha_hi, cfg.monotonicity_n_check)
        values = np.array([self._h(a) for a in alphas])

        # Check that values are non-decreasing within epsilon tolerance
        diffs = np.diff(values)
        return bool(np.all(diffs >= -cfg.monotonicity_epsilon))

    def _find_bracket(self, alphas: np.ndarray, values: np.ndarray) -> tuple[float, float] | None:
        """Find a bracket [alpha_lo, alpha_hi] where h changes sign."""
        neg_mask = values < 0
        pos_mask = values >= 0
        if not np.any(neg_mask) or not np.any(pos_mask):
            return None
        # Last negative index, first positive index after it
        neg_indices = np.where(neg_mask)[0]
        pos_indices = np.where(pos_mask)[0]
        last_neg = neg_indices[-1]
        # Find first positive index after last negative
        after = pos_indices[pos_indices > last_neg]
        if len(after) == 0:
            return None
        first_pos = after[0]
        return float(alphas[last_neg]), float(alphas[first_pos])

    def _grid_search(self) -> float:
        """Two-phase grid search fallback when h is not monotone."""
        cfg = self.config

        # Phase 1: coarse grid
        coarse_alphas = np.linspace(cfg.alpha_lo, cfg.alpha_hi, cfg.grid_n_coarse)
        coarse_values = np.array([self._h(a) for a in coarse_alphas])
        feasible_mask = coarse_values >= 0

        if not np.any(feasible_mask):
            raise ValueError(
                "No feasible catalytic fraction found on the coarse grid. "
                "Consider relaxing hurdle IRR or max_loss_probability constraints."
            )

        alpha_coarse = float(coarse_alphas[np.where(feasible_mask)[0][0]])

        # Phase 2: fine grid around alpha_coarse
        step = (cfg.alpha_hi - cfg.alpha_lo) / cfg.grid_n_coarse
        fine_lo = max(cfg.alpha_lo, alpha_coarse - 2 * step)
        fine_alphas = np.linspace(fine_lo, alpha_coarse, cfg.grid_n_fine)
        fine_values = np.array([self._h(a) for a in fine_alphas])
        feasible_fine = fine_values >= 0

        if not np.any(feasible_fine):
            return alpha_coarse

        return float(fine_alphas[np.where(feasible_fine)[0][0]])

    def calibrate(self) -> float:
        """Solve for the minimum feasible catalytic fraction.

        Returns:
            alpha*: minimum catalytic fraction in [alpha_lo, alpha_hi] satisfying
                    both the IRR and loss probability constraints.

        Raises:
            ValueError: if no feasible solution exists.
        """
        cfg = self.config

        # Quick feasibility check at boundaries
        h_lo = self._h(cfg.alpha_lo)
        h_hi = self._h(cfg.alpha_hi)

        if h_hi < 0:
            raise ValueError(
                f"Constraints infeasible even at alpha={cfg.alpha_hi}. "
                "Consider relaxing hurdle IRR or max_loss_probability."
            )

        if h_lo >= 0:
            # Already feasible at alpha_lo — return immediately
            return cfg.alpha_lo

        # Check monotonicity
        is_monotone = self._check_monotonicity()

        if is_monotone:
            try:
                alpha_star = brentq(
                    self._h,
                    cfg.alpha_lo,
                    cfg.alpha_hi,
                    xtol=cfg.brentq_xtol,
                    rtol=cfg.brentq_xtol,
                )
            except ValueError as exc:
                warnings.warn(
                    f"Brent's method failed ({exc}); falling back to grid search.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                alpha_star = self._grid_search()
        else:
            warnings.warn(
                "Objective function is not monotone — using grid search fallback.",
                RuntimeWarning,
                stacklevel=2,
            )
            alpha_star = self._grid_search()

        return float(alpha_star)
