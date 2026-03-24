"""Pydantic input models and result dataclasses for the project layer."""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from pydantic import BaseModel, Field, model_validator


class ProjectInputs(BaseModel):
    """Validated inputs for a single project.

    Two input modes are supported:

    **Parametric mode** (default): supply `capex`, `opex_annual`, `price`,
    `yield_`, and `lifetime_years`. The simulator builds stochastic cashflows
    from these fundamentals.

    **Cashflow mode**: supply `base_cashflows` (a list of operating-period
    net cashflows for t=1..T) plus `capex`. The simulator scales these base
    cashflows with a single lognormal multiplier per path (using `price_vol`
    as the overall volatility). All parametric revenue/opex fields are ignored.
    """

    # ------------------------------------------------------------------ #
    # Parametric inputs (required in parametric mode)                     #
    # ------------------------------------------------------------------ #
    capex: float | None = Field(
        None, description="Capital expenditure (positive value; sign applied internally). "
                          "Required in parametric mode. Optional in cashflow mode "
                          "(defaults to 0 when base_cashflows is provided)."
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
                    "when in cashflow mode."
    )

    # ------------------------------------------------------------------ #
    # Cashflow-based input (alternative to parametric)                    #
    # ------------------------------------------------------------------ #
    base_cashflows: list[float] | None = Field(
        None,
        description="Operating-period net cashflows for t=1..T (capex outflow at t=0 "
                    "is always specified separately via `capex`). When provided, the "
                    "simulator applies a single lognormal multiplier per path "
                    "(sigma=price_vol) to scale these base cashflows stochastically."
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
                    "used as the overall cashflow multiplier volatility."
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
            if len(self.base_cashflows) == 0:
                raise ValueError("base_cashflows must have at least one element.")
            if self.lifetime_years is None:
                object.__setattr__(self, "lifetime_years", len(self.base_cashflows))
            elif self.lifetime_years != len(self.base_cashflows):
                raise ValueError(
                    f"lifetime_years={self.lifetime_years} does not match "
                    f"len(base_cashflows)={len(self.base_cashflows)}."
                )
            if self.capex is None:
                object.__setattr__(self, "capex", 0.0)
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
            # Economic viability hint
            base_revenue = self.price * self.yield_ * self.lifetime_years  # type: ignore[operator]
            if base_revenue < self.capex:  # type: ignore[operator]
                warnings.warn(
                    "Base-case lifetime revenue is less than capex — project may have negative expected NPV.",
                    UserWarning,
                    stacklevel=2,
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
