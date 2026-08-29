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
        0.07,
        description="Minimum median senior IRR required by commercial investors. Must be "
                    "strictly below the vehicle's senior_coupon (default 8%): the waterfall "
                    "never pays senior more than coupon plus principal, so a hurdle at or "
                    "above the coupon is unreachable at any catalytic fraction.",
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
    min_senior_fraction: float = Field(
        0.05, ge=0.0, lt=1.0,
        description="Minimum senior tranche size, as a fraction of total capital, for a "
                    "calibration to count as economically meaningful. Above this alpha the "
                    "senior tranche is so thin that its constraints are satisfied "
                    "vacuously rather than by the structure working."
    )


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

        # A senior tranche that has been shrunk to nothing satisfies any hurdle
        # trivially — its IRR converges to the coupon and it cannot take a loss.
        # That is an artefact of the tranching arithmetic, not a structure that
        # works, so it is reported as infeasible rather than as a solution.
        floor = cfg.min_senior_fraction * self.capital_stack.total_capital
        # Strictly below, so that alpha exactly at _max_structural_alpha() — where
        # the senior tranche sits precisely on the floor — is still evaluated.
        if "senior" not in results or results["senior"].notional < floor - 1e-9:
            return -1.0

        senior = results["senior"]
        g1 = float(np.nanmedian(senior.irr_distribution)) - cfg.investor_hurdle_irr
        g2 = cfg.max_loss_probability - senior.loss_probability
        return min(g1, g2)

    def _probe(self, alpha_lo: float, alpha_hi: float) -> tuple[bool, np.ndarray, np.ndarray]:
        """Sample h across the search range and report whether it is non-decreasing.

        Returns (is_monotone, alphas, values). The sampled values are handed back
        so the caller can reuse them to seed a tighter bracket instead of
        throwing away the ~20 waterfall evaluations they cost.
        """
        cfg = self.config
        alphas = np.linspace(alpha_lo, alpha_hi, cfg.monotonicity_n_check)
        values = np.array([self._h(a) for a in alphas])
        is_monotone = bool(np.all(np.diff(values) >= -cfg.monotonicity_epsilon))
        return is_monotone, alphas, values

    @staticmethod
    def _bracket_from_probe(
        alphas: np.ndarray, values: np.ndarray, fallback: tuple[float, float]
    ) -> tuple[float, float]:
        """Narrow the root bracket to the last sign change seen in the probe."""
        neg = np.where(values < 0)[0]
        pos = np.where(values >= 0)[0]
        if neg.size == 0 or pos.size == 0:
            return fallback
        last_neg = int(neg[-1])
        after = pos[pos > last_neg]
        if after.size == 0:
            return fallback
        return float(alphas[last_neg]), float(alphas[int(after[0])])

    def _grid_search(self, alpha_lo: float, alpha_hi: float) -> float:
        """Two-phase grid search fallback when h is not monotone."""
        cfg = self.config

        # Phase 1: coarse grid
        coarse_alphas = np.linspace(alpha_lo, alpha_hi, cfg.grid_n_coarse)
        coarse_values = np.array([self._h(a) for a in coarse_alphas])
        feasible_mask = coarse_values >= 0

        if not np.any(feasible_mask):
            raise ValueError(
                "No feasible catalytic fraction found on the coarse grid — "
                "constraints infeasible. Consider relaxing hurdle IRR or "
                "max_loss_probability, or adding protective mitigants."
            )

        alpha_coarse = float(coarse_alphas[np.where(feasible_mask)[0][0]])

        # Phase 2: fine grid between the last infeasible point and alpha_coarse
        step = (alpha_hi - alpha_lo) / cfg.grid_n_coarse
        fine_lo = max(alpha_lo, alpha_coarse - 2 * step)
        fine_alphas = np.linspace(fine_lo, alpha_coarse, cfg.grid_n_fine)
        fine_values = np.array([self._h(a) for a in fine_alphas])
        feasible_fine = fine_values >= 0

        if not np.any(feasible_fine):
            return alpha_coarse

        return float(fine_alphas[np.where(feasible_fine)[0][0]])

    # ------------------------------------------------------------------
    # Structural guards
    # ------------------------------------------------------------------

    def _validate_structure(self) -> None:
        """Reject configurations no alpha can satisfy.

        The cashflow waterfall never pays the senior tranche more than its
        coupon plus principal, so senior IRR is capped at ``senior_coupon``. A
        hurdle at or above the coupon is therefore unreachable at any alpha:
        the search can only "satisfy" it by shrinking the senior tranche to
        nothing. Catching that here turns a confidently-reported nonsense alpha
        into an explicit error.
        """
        cfg = self.config
        coupon = self.capital_stack.senior_coupon
        if cfg.investor_hurdle_irr >= coupon:
            raise ValueError(
                f"Constraints infeasible: investor_hurdle_irr "
                f"({cfg.investor_hurdle_irr:.2%}) must be below senior_coupon "
                f"({coupon:.2%}). Senior IRR is capped at the coupon rate, so no "
                f"catalytic fraction can meet this hurdle. Raise senior_coupon or "
                f"lower the hurdle."
            )

    def _max_structural_alpha(self) -> float:
        """Largest alpha that still leaves a senior tranche worth calibrating."""
        return (
            1.0
            - self.capital_stack.mezzanine_fraction
            - self.config.min_senior_fraction
        )

    def _ensure_feasible(self, alpha: float, alpha_hi: float) -> float:
        """Nudge a root that landed marginally on the infeasible side.

        brentq converges to within ``brentq_xtol`` of the root, and the root is
        the boundary of the feasible set — so the returned alpha can sit just
        below it. Since the caller treats the result as "the minimum alpha that
        satisfies the constraints", step up until it actually does.
        """
        cfg = self.config
        if self._h(alpha) >= 0.0:
            return alpha
        step = max(cfg.brentq_xtol, 1e-6)
        for _ in range(16):
            alpha = min(alpha + step, alpha_hi)
            if self._h(alpha) >= 0.0:
                return alpha
            if alpha >= alpha_hi:
                break
        raise ValueError(
            "Constraints infeasible: no catalytic fraction in "
            f"[{cfg.alpha_lo}, {alpha_hi:.4f}] satisfies both the hurdle IRR and "
            "the loss probability limit."
        )

    def calibrate(self) -> float:
        """Solve for the minimum feasible catalytic fraction.

        Returns:
            alpha*: minimum catalytic fraction satisfying both the IRR and loss
                    probability constraints, with a senior tranche large enough
                    for those constraints to mean something.

        Raises:
            ValueError: if no feasible solution exists.
        """
        cfg = self.config
        self._validate_structure()

        alpha_lo = cfg.alpha_lo
        alpha_hi = min(cfg.alpha_hi, self._max_structural_alpha())
        if alpha_hi <= alpha_lo:
            raise ValueError(
                "Constraints infeasible: mezzanine_fraction "
                f"({self.capital_stack.mezzanine_fraction:.2%}) leaves no room for "
                f"a senior tranche of at least {cfg.min_senior_fraction:.2%} of "
                "capital. Reduce mezzanine_fraction or min_senior_fraction."
            )

        # Quick feasibility check at boundaries
        if self._h(alpha_lo) >= 0:
            # Already feasible with no catalytic capital at all.
            return alpha_lo

        if self._h(alpha_hi) < 0:
            raise ValueError(
                f"Constraints infeasible even at alpha={alpha_hi:.4f} (the largest "
                "catalytic fraction that still leaves a meaningful senior tranche). "
                "Consider relaxing hurdle IRR or max_loss_probability, or adding "
                "protective mitigants."
            )

        is_monotone, alphas, values = self._probe(alpha_lo, alpha_hi)

        if is_monotone:
            lo, hi = self._bracket_from_probe(alphas, values, (alpha_lo, alpha_hi))
            try:
                alpha_star = brentq(
                    self._h, lo, hi,
                    xtol=cfg.brentq_xtol,
                    rtol=cfg.brentq_xtol,
                )
            except ValueError as exc:
                warnings.warn(
                    f"Brent's method failed ({exc}); falling back to grid search.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                alpha_star = self._grid_search(alpha_lo, alpha_hi)
        else:
            warnings.warn(
                "Objective function is not monotone — using grid search fallback.",
                RuntimeWarning,
                stacklevel=2,
            )
            alpha_star = self._grid_search(alpha_lo, alpha_hi)

        return float(self._ensure_feasible(float(alpha_star), alpha_hi))
