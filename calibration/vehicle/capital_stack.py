"""Dual waterfall: per-period cashflow distribution + cumulative terminal loss absorption."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from calibration.utils.irr import batch_irr, clean_irr, npv_loss
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
      4. Guarantee (applied to senior tranche losses after all subordination
                    layers are exhausted: senior_loss_gross → guarantee →
                    senior_loss_net)
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
        discount_rate: Annual discount rate for NPV-based terminal loss.
                       Default 0.0 gives undiscounted (sum-based) loss for
                       backward compatibility.
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
        discount_rate: float = 0.0,
    ) -> None:
        self.total_capital = total_capital
        self.grant_reserve = grant_reserve
        self.guarantee = guarantee
        self.senior_coupon = senior_coupon
        self.mezzanine_coupon = mezzanine_coupon
        self.mezzanine_fraction = mezzanine_fraction
        self.lifetime_years = lifetime_years
        self.discount_rate = discount_rate

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
            terminal_loss: shape (n_sims,), L[s] = max(0, -NPV(CF[s], r))
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
        """Distribute per-period cashflows across tranches.

        Bullet maturity, with the two mechanisms that make it solvent:

        **Cash retention (sinking fund).** Principal is contractually repaid at
        T, so cash has to be held against it. After each period's coupons are
        paid, cash is retained up to a target of ``(senior + mezzanine
        outstanding) * t / T`` and only the excess is released to equity. The
        retained balance is fully liquid: it is added back to available cash at
        the start of the next period, so it also absorbs coupon shortfalls the
        way a debt service reserve account does.

        Without this the structure leaks every spare dollar to equity each
        period and then cannot meet the bullet: a vehicle returning 3x its
        capital still defaults on senior principal.

        **Risk mitigants reach the tranche they protect.** The grant reserve
        seeds the retained balance (it is donor cash, available to pay senior),
        and the guarantee is drawn to make senior whole when cash falls short,
        up to ``coverage * senior notional`` over the life of the vehicle. Both
        are also layers in the loss waterfall; these are two views of the same
        protection, not two separate pots.

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
        # The catalytic bucket funds both the first-loss tranche and the grant
        # reserve, so it carries both as its t=0 outflow and receives whatever
        # of the reserve is left over at maturity.
        tranche_cfs["first_loss"][:, 0] = -(fl.notional + self.grant_reserve.amount)

        # Track arrears per tranche (n_sims,)
        senior_arrears = np.zeros(n_sims)
        mezz_arrears = np.zeros(n_sims)

        senior_outstanding = np.full(n_sims, senior.notional)
        mezz_outstanding = np.full(n_sims, mezz.notional)

        # Retained cash, seeded by the donor-funded grant reserve.
        retained = np.full(n_sims, self.grant_reserve.amount)

        # Guarantee capacity available to top up senior payments over the life
        # of the vehicle (0.0 when no guarantee is attached).
        guarantee_remaining = np.full(
            n_sims, self.guarantee.effective_cap(senior.notional)
        )

        for t in range(1, T + 1):
            is_terminal = (t == T)

            # Retained cash is liquid: it funds this period's obligations before
            # being topped back up below.
            available = np.maximum(0.0, vehicle_cashflows[:, t]) + retained
            retained = np.zeros(n_sims)

            # --- Senior (coupon always; principal at maturity) ---
            senior_coupon_due = senior_outstanding * senior.coupon
            senior_principal_due = senior_outstanding if is_terminal else np.zeros(n_sims)
            senior_current_due = senior_arrears + senior_coupon_due
            senior_due = senior_current_due + senior_principal_due

            paid_cash = np.minimum(available, senior_due)
            available -= paid_cash

            # The guarantee makes the senior noteholder whole on any residual
            # shortfall, up to its remaining capacity.
            shortfall = senior_due - paid_cash
            guarantee_draw = np.minimum(shortfall, guarantee_remaining)
            guarantee_remaining -= guarantee_draw

            senior_received = paid_cash + guarantee_draw
            senior_arrears = shortfall - guarantee_draw

            # Receipts settle arrears, then coupon, then principal.
            principal_paid_sr = np.clip(
                senior_received - senior_current_due, 0.0, senior_principal_due
            )
            senior_outstanding = np.maximum(0.0, senior_outstanding - principal_paid_sr)
            tranche_cfs["senior"][:, t] = senior_received

            # --- Mezzanine (unguaranteed) ---
            mezz_coupon_due = mezz_outstanding * mezz.coupon
            mezz_principal_due = mezz_outstanding if is_terminal else np.zeros(n_sims)
            mezz_current_due = mezz_arrears + mezz_coupon_due
            mezz_due = mezz_current_due + mezz_principal_due

            mezz_paid = np.minimum(available, mezz_due)
            available -= mezz_paid
            mezz_arrears = mezz_due - mezz_paid
            principal_paid_mz = np.clip(
                mezz_paid - mezz_current_due, 0.0, mezz_principal_due
            )
            mezz_outstanding = np.maximum(0.0, mezz_outstanding - principal_paid_mz)
            tranche_cfs["mezzanine"][:, t] = mezz_paid

            # --- Retain cash against the bullet before releasing to equity ---
            if not is_terminal:
                target = (senior_outstanding + mezz_outstanding) * (t / T)
                retained = np.minimum(available, target)
                available -= retained

            # --- First-loss / equity gets what is genuinely surplus ---
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

        # --- Terminal loss per path (NPV-based; discount_rate=0.0 → sum-based) ---
        terminal_loss = npv_loss(vehicle_cashflows, self.discount_rate)

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
