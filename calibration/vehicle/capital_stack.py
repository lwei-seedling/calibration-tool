"""Dual waterfall: per-period cashflow distribution + cumulative terminal loss absorption."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from calibration.utils.irr import batch_irr, clean_irr
from calibration.utils.stats import var, cvar
from calibration.vehicle.models import TrancheResult
from calibration.vehicle.risk_mitigants import Guarantee, GrantReserve


@dataclass
class _Tranche:
    name: str
    notional: float
    coupon: float   # annual coupon rate


class CapitalStack:
    """Implements the dual waterfall for a blended-finance vehicle.

    Loss waterfall (cumulative terminal, junior → senior):
      1. Grant Reserve
      2. First-Loss / Equity
      3. Mezzanine
      4. Guarantee (wrapping Senior)
      5. Senior

    Cashflow waterfall (per-period, senior → junior):
      1. Senior (coupon + principal)
      2. Mezzanine (coupon + principal)
      3. First-Loss / Equity (residual)
      Grant Reserve receives no distributions.

    Args:
        total_capital: Total vehicle capital.
        grant_reserve: GrantReserve object (amount may be 0).
        guarantee: Guarantee object wrapping senior tranche (coverage may be 0).
        senior_coupon: Annual coupon on senior tranche.
        mezzanine_coupon: Annual coupon on mezzanine tranche.
        mezzanine_fraction: Fraction of total capital in mezzanine tranche.
        lifetime_years: Project lifetime (number of operating periods).
    """

    def __init__(
        self,
        total_capital: float,
        grant_reserve: GrantReserve,
        guarantee: Guarantee,
        senior_coupon: float,
        mezzanine_coupon: float,
        mezzanine_fraction: float,
        lifetime_years: int,
    ) -> None:
        self.total_capital = total_capital
        self.grant_reserve = grant_reserve
        self.guarantee = guarantee
        self.senior_coupon = senior_coupon
        self.mezzanine_coupon = mezzanine_coupon
        self.mezzanine_fraction = mezzanine_fraction
        self.lifetime_years = lifetime_years

    def _build_tranches(self, alpha: float) -> tuple[_Tranche, _Tranche, _Tranche]:
        """Build tranche notionals given catalytic fraction alpha.

        alpha = (grant_reserve + first_loss) / total_capital
        Remaining capital is split between mezzanine and senior.
        Grant reserve is a fixed input; first-loss is solved as the residual
        of catalytic capital minus the reserve.

        Returns:
            (first_loss_tranche, mezzanine_tranche, senior_tranche)
        """
        total = self.total_capital
        reserve = self.grant_reserve.amount

        catalytic_total = alpha * total
        first_loss_notional = max(0.0, catalytic_total - reserve)
        commercial_total = total - catalytic_total
        mezzanine_notional = self.mezzanine_fraction * total
        senior_notional = max(0.0, commercial_total - mezzanine_notional)

        fl = _Tranche("first_loss", first_loss_notional, 0.0)  # equity: no coupon
        mezz = _Tranche("mezzanine", mezzanine_notional, self.mezzanine_coupon)
        senior = _Tranche("senior", senior_notional, self.senior_coupon)
        return fl, mezz, senior

    # ------------------------------------------------------------------
    # Loss waterfall (cumulative terminal)
    # ------------------------------------------------------------------

    def _loss_waterfall(
        self,
        terminal_loss: np.ndarray,
        alpha: float,
    ) -> dict[str, np.ndarray]:
        """Distribute terminal losses across layers.

        Args:
            terminal_loss: shape (n_sims,), L[s] = max(0, -sum_t CF[s,t])
            alpha: catalytic fraction

        Returns:
            dict mapping layer name → absorbed loss per path (n_sims,)
        """
        fl, mezz, senior = self._build_tranches(alpha)

        absorbed = {}
        remaining = terminal_loss.copy()

        # Layer 1: Grant Reserve
        gr_abs, remaining = self.grant_reserve.absorb(remaining)
        absorbed["grant_reserve"] = gr_abs

        # Layer 2: First-Loss
        fl_abs = np.minimum(remaining, fl.notional)
        remaining = remaining - fl_abs
        absorbed["first_loss"] = fl_abs

        # Layer 3: Mezzanine
        mezz_abs = np.minimum(remaining, mezz.notional)
        remaining = remaining - mezz_abs
        absorbed["mezzanine"] = mezz_abs

        # Layer 4: Guarantee (wrapping Senior)
        guar_abs, remaining = self.guarantee.absorb(remaining, senior.notional)
        absorbed["guarantee"] = guar_abs

        # Layer 5: Senior
        senior_abs = np.minimum(remaining, senior.notional)
        remaining = remaining - senior_abs
        absorbed["senior"] = senior_abs

        return absorbed

    # ------------------------------------------------------------------
    # Cashflow waterfall (per-period distribution)
    # ------------------------------------------------------------------

    def _cashflow_waterfall(
        self,
        vehicle_cashflows: np.ndarray,
        alpha: float,
    ) -> dict[str, np.ndarray]:
        """Distribute per-period positive cashflows across tranches.

        Uses a simplified bullet-maturity approximation:
          - Coupons are paid annually on outstanding notionals.
          - Principal returns entirely at terminal period T.
          - Shortfalls accumulate in arrears and are paid first when cash is available.

        Args:
            vehicle_cashflows: shape (n_sims, T+1)
            alpha: catalytic fraction

        Returns:
            dict mapping tranche name → cashflow array (n_sims, T+1)
              where axis-1 index 0 is the investment outflow (negative)
              and indices 1..T are received cashflows.
        """
        fl, mezz, senior = self._build_tranches(alpha)
        n_sims, T_plus_1 = vehicle_cashflows.shape
        T = T_plus_1 - 1

        # Initialize tranche cashflow matrices (investment outflow at t=0)
        tranche_cfs: dict[str, np.ndarray] = {
            "senior": np.zeros((n_sims, T_plus_1)),
            "mezzanine": np.zeros((n_sims, T_plus_1)),
            "first_loss": np.zeros((n_sims, T_plus_1)),
        }
        tranche_cfs["senior"][:, 0] = -senior.notional
        tranche_cfs["mezzanine"][:, 0] = -mezz.notional
        tranche_cfs["first_loss"][:, 0] = -(fl.notional + self.grant_reserve.amount)

        # Track arrears per tranche (n_sims,)
        senior_arrears = np.zeros(n_sims)
        mezz_arrears = np.zeros(n_sims)

        senior_outstanding = np.full(n_sims, senior.notional)
        mezz_outstanding = np.full(n_sims, mezz.notional)

        for t in range(1, T + 1):
            available = np.maximum(0.0, vehicle_cashflows[:, t])

            # --- Senior ---
            is_terminal = (t == T)
            senior_coupon_due = senior_outstanding * senior.coupon
            senior_principal_due = senior_outstanding if is_terminal else np.zeros(n_sims)
            senior_due = senior_arrears + senior_coupon_due + senior_principal_due

            senior_paid = np.minimum(available, senior_due)
            available -= senior_paid
            senior_arrears = senior_due - senior_paid
            # Update outstanding principal
            principal_paid_sr = np.minimum(senior_paid, senior_principal_due)
            senior_outstanding = np.maximum(0.0, senior_outstanding - principal_paid_sr)
            tranche_cfs["senior"][:, t] = senior_paid

            # --- Mezzanine ---
            mezz_coupon_due = mezz_outstanding * mezz.coupon
            mezz_principal_due = mezz_outstanding if is_terminal else np.zeros(n_sims)
            mezz_due = mezz_arrears + mezz_coupon_due + mezz_principal_due

            mezz_paid = np.minimum(available, mezz_due)
            available -= mezz_paid
            mezz_arrears = mezz_due - mezz_paid
            principal_paid_mz = np.minimum(mezz_paid, mezz_principal_due)
            mezz_outstanding = np.maximum(0.0, mezz_outstanding - principal_paid_mz)
            tranche_cfs["mezzanine"][:, t] = mezz_paid

            # --- First-loss / equity gets residual ---
            tranche_cfs["first_loss"][:, t] = available

        return tranche_cfs

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def waterfall(
        self,
        vehicle_cashflows: np.ndarray,
        alpha: float,
    ) -> dict[str, TrancheResult]:
        """Run both waterfalls and return TrancheResult for each tranche.

        Args:
            vehicle_cashflows: shape (n_sims, T+1); vehicle-level aggregated CFs.
            alpha: catalytic fraction in [0, 1].

        Returns:
            dict with keys: 'senior', 'mezzanine', 'first_loss'
        """
        fl, mezz, senior = self._build_tranches(alpha)

        # --- Terminal loss per path ---
        terminal_loss = np.maximum(0.0, -vehicle_cashflows.sum(axis=1))

        # --- Loss waterfall ---
        loss_absorbed = self._loss_waterfall(terminal_loss, alpha)

        # --- Cashflow waterfall ---
        tranche_cfs = self._cashflow_waterfall(vehicle_cashflows, alpha)

        results: dict[str, TrancheResult] = {}
        tranche_meta = {
            "senior": senior,
            "mezzanine": mezz,
            "first_loss": fl,
        }
        for name, tranche in tranche_meta.items():
            if tranche.notional == 0.0:
                # Skip zero-notional tranches
                continue

            cfs = tranche_cfs[name]
            raw_irr = batch_irr(cfs)
            irr_dist = clean_irr(raw_irr)

            loss_dist = loss_absorbed[name]
            loss_prob = float(np.mean(loss_dist > 0.0))
            var_95 = var(loss_dist, 0.95)
            cvar_95 = cvar(loss_dist, 0.95)
            median_irr = float(np.nanmedian(irr_dist))

            results[name] = TrancheResult(
                name=name,
                notional=tranche.notional,
                irr_distribution=irr_dist,
                loss_distribution=loss_dist,
                loss_probability=loss_prob,
                var_95=var_95,
                cvar_95=cvar_95,
                median_irr=median_irr,
            )

        return results
