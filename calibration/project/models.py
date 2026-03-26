"""Pydantic input models and result dataclasses for the project layer."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pydantic import BaseModel, Field, model_validator


class ProjectInputs(BaseModel):
    """Validated inputs for a single project.

    Two input modes are supported:

    **Parametric mode** (default): supply `capex`, `opex_annual`, `price`,
    `yield_`, and `lifetime_years`. The simulator builds stochastic cashflows
    from these fundamentals.

    **Cashflow mode**: supply `base_cashflows` as a full-lifecycle array
    [CF_0, CF_1, ..., CF_T] where CF_0 is the upfront outflow (negative for
    capex) and CF_1..CF_T are operating cashflows. Multi-year capex
    (negative values in early periods) is fully supported.

    Optionally, supply `base_revenue` and `base_costs` (per-period, t=1..T)
    to separate revenue from cost; price shocks are then applied to revenue
    only via a GBM path. When `base_revenue` is provided, `base_cashflows`
    must be a length-1 list containing CF[0] (the t=0 outflow).
    """

    # ------------------------------------------------------------------ #
    # Parametric inputs (required in parametric mode)                     #
    # ------------------------------------------------------------------ #
    capex: float | None = Field(
        None, description="Capital expenditure (positive value; sign applied internally). "
                          "Required in parametric mode. Ignored when base_cashflows is provided "
                          "(capex is embedded as CF[0] of base_cashflows in cashflow mode)."
    )
    opex_annual: float | None = Field(
        None, description="Annual operating expenditure (positive). Ignored in cashflow mode."
    )
    price: float | None = Field(
        None, description="Base output price per unit. Ignored in cashflow mode."
    )
    yield_: float | None = Field(
        None, description="Base annual output quantity. Ignored in cashflow mode."
    )
    lifetime_years: int | None = Field(
        None, ge=1, le=50,
        description="Project lifetime in years. Inferred from base_cashflows length "
                    "when in cashflow mode (lifetime_years = len(base_cashflows) - 1)."
    )

    # ------------------------------------------------------------------ #
    # Cashflow-based input (alternative to parametric)                    #
    # ------------------------------------------------------------------ #
    base_cashflows: list[float] | None = Field(
        None,
        description="Full-lifecycle net cashflows [CF_0, CF_1, ..., CF_T]. "
                    "CF_0 is typically negative (capex/construction outflow). "
                    "Multi-year capex is supported: supply negative values for "
                    "early construction years. When provided, used directly — "
                    "CF[0] is never overridden. Must have len >= 2 unless "
                    "base_revenue is also provided (then len == 1 for CF[0] only)."
    )
    base_revenue: list[float] | None = Field(
        None,
        description="Per-period revenue for t=1..T. When provided together with "
                    "base_costs, price shocks (GBM path) are applied to revenue only. "
                    "base_cashflows must be provided with exactly one element (CF[0])."
    )
    base_costs: list[float] | None = Field(
        None,
        description="Per-period costs for t=1..T (unshocked, pass-through). "
                    "Used together with base_revenue."
    )

    # ------------------------------------------------------------------ #
    # Price process parameters                                            #
    # ------------------------------------------------------------------ #
    price_series: list[float] | None = Field(
        None,
        description="Historical price observations. Log returns are computed and "
                    "used to estimate drift (mu) and volatility (sigma) for the GBM "
                    "price path, overriding price_vol and price_drift."
    )
    price_drift: float | None = Field(
        None,
        description="Annualised log-price drift for the GBM price path. "
                    "If None and price_series is not provided, defaults to 0.0."
    )

    # ------------------------------------------------------------------ #
    # Shared risk / volatility parameters                                 #
    # ------------------------------------------------------------------ #
    discount_rate: float = Field(
        0.0, ge=0.0,
        description="Annual discount rate used to compute NPV-based terminal loss. "
                    "Default 0.0 gives undiscounted (sum-based) loss for backward "
                    "compatibility. Set to e.g. 0.10 for a 10% discount rate."
    )
    price_vol: float = Field(
        0.15, ge=0.0, le=1.0,
        description="Annual price volatility (lognormal sigma). In cashflow mode, "
                    "used as the GBM sigma for the price index path."
    )
    yield_vol: float = Field(0.10, ge=0.0, le=1.0, description="Annual yield volatility (lognormal sigma).")
    inflation_rate: float = Field(0.02, ge=0.0, le=0.5, description="Annual inflation rate for opex.")
    fx_vol: float = Field(0.05, ge=0.0, le=1.0, description="Annual FX volatility applied to revenue.")

    # ------------------------------------------------------------------ #
    # Delay parameters                                                    #
    # ------------------------------------------------------------------ #
    delay_years_probs: list[float] = Field(
        default_factory=lambda: [1.0, 0.0, 0.0],
        description="Probability weights for structural production delay of "
                    "[0 years, 0.5 years, 1 year] respectively. Must have exactly "
                    "3 elements and sum to 1 (normalised automatically). Default "
                    "[1.0, 0.0, 0.0] means no structural delay. Use e.g. "
                    "[0.85, 0.10, 0.05] to model 15% chance of delay."
    )
    # Deprecated: per-period delay probability.  Use delay_years_probs instead.
    delay_prob: float = Field(
        0.05, ge=0.0, le=1.0,
        description="[DEPRECATED] Per-period probability of a production delay. "
                    "Use delay_years_probs for structural delay modelling instead. "
                    "Kept for backward compatibility."
    )

    @model_validator(mode="after")
    def _validate_and_normalise(self) -> "ProjectInputs":
        # ---- Input mode validation -------------------------------------- #
        if self.base_cashflows is not None:
            # Cashflow mode
            if self.base_revenue is not None:
                # Revenue+cost sub-mode: base_cashflows must be exactly [CF0]
                if len(self.base_cashflows) != 1:
                    raise ValueError(
                        "When base_revenue is provided, base_cashflows must contain "
                        "exactly one element: [CF_0] (the t=0 outflow)."
                    )
                T = len(self.base_revenue)
                if T == 0:
                    raise ValueError("base_revenue must have at least one element.")
                if self.base_costs is not None and len(self.base_costs) != T:
                    raise ValueError(
                        f"base_costs length {len(self.base_costs)} must match "
                        f"base_revenue length {T}."
                    )
                if self.lifetime_years is None:
                    object.__setattr__(self, "lifetime_years", T)
                elif self.lifetime_years != T:
                    raise ValueError(
                        f"lifetime_years={self.lifetime_years} does not match "
                        f"len(base_revenue)={T}."
                    )
            else:
                # Standard cashflow mode: base_cashflows is full lifecycle [CF0, CF1, ..., CFT]
                if len(self.base_cashflows) < 2:
                    raise ValueError(
                        "base_cashflows must have at least 2 elements: "
                        "[CF_0, CF_1, ..., CF_T] (full lifecycle including t=0)."
                    )
                T = len(self.base_cashflows) - 1
                if self.lifetime_years is None:
                    object.__setattr__(self, "lifetime_years", T)
                elif self.lifetime_years != T:
                    raise ValueError(
                        f"lifetime_years={self.lifetime_years} does not match "
                        f"len(base_cashflows)-1={T}."
                    )
        else:
            # Parametric mode: all core fields required
            missing = [
                f for f in ("capex", "opex_annual", "price", "yield_", "lifetime_years")
                if getattr(self, f) is None
            ]
            if missing:
                raise ValueError(
                    f"Fields required in parametric mode: {missing}. "
                    "Provide base_cashflows to use cashflow-based input instead."
                )

        # ---- Delay probs normalisation ---------------------------------- #
        if len(self.delay_years_probs) != 3:
            raise ValueError("delay_years_probs must have exactly 3 elements [P(0y), P(0.5y), P(1y)].")
        total = sum(self.delay_years_probs)
        if total <= 0:
            raise ValueError("delay_years_probs must sum to a positive number.")
        if abs(total - 1.0) > 1e-6:
            normalised = [p / total for p in self.delay_years_probs]
            object.__setattr__(self, "delay_years_probs", normalised)

        return self


@dataclass
class ProjectResult:
    """Output from ProjectSimulator.run()."""

    cashflows: np.ndarray         # shape (n_sims, T+1); t=0 is capex outflow
    irr_distribution: np.ndarray  # shape (n_sims,); IRR per path
    loss_probability: float       # fraction of paths with positive terminal loss (NPV-based)
