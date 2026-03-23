"""Pydantic input models and result dataclasses for the project layer."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pydantic import BaseModel, Field, model_validator


class ProjectInputs(BaseModel):
    """Validated inputs for a single project."""

    # Core financials
    capex: float = Field(..., gt=0, description="Capital expenditure (positive value; sign applied internally).")
    opex_annual: float = Field(..., gt=0, description="Annual operating expenditure (positive value).")
    price: float = Field(..., gt=0, description="Base output price per unit.")
    yield_: float = Field(..., gt=0, description="Base annual output quantity (yield).")
    lifetime_years: int = Field(..., ge=1, le=50, description="Project lifetime in years.")

    # Volatility and risk parameters
    price_vol: float = Field(0.15, ge=0.0, le=1.0, description="Annual price volatility (lognormal sigma).")
    yield_vol: float = Field(0.10, ge=0.0, le=1.0, description="Annual yield volatility (lognormal sigma).")
    inflation_rate: float = Field(0.02, ge=0.0, le=0.5, description="Annual inflation rate for opex.")
    fx_vol: float = Field(0.05, ge=0.0, le=1.0, description="Annual FX volatility applied to revenue.")
    delay_prob: float = Field(0.05, ge=0.0, le=1.0, description="Per-period probability of a production delay (zero revenue).")

    @model_validator(mode="after")
    def check_economics(self) -> "ProjectInputs":
        base_revenue = self.price * self.yield_ * self.lifetime_years
        if base_revenue < self.capex:
            import warnings
            warnings.warn(
                "Base-case lifetime revenue is less than capex — project may have negative expected NPV.",
                UserWarning,
                stacklevel=2,
            )
        return self


@dataclass
class ProjectResult:
    """Output from ProjectSimulator.run()."""

    cashflows: np.ndarray       # shape (n_sims, T+1); t=0 is capex outflow
    irr_distribution: np.ndarray  # shape (n_sims,); IRR per path
    loss_probability: float     # fraction of paths with positive terminal loss
