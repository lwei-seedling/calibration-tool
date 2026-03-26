"""Pydantic input models and result dataclasses for the portfolio layer."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from pydantic import BaseModel, Field

from calibration.vehicle.calibration import CalibratorConfig
from calibration.vehicle.models import VehicleInputs


class PortfolioInputs(BaseModel):
    """Validated inputs for portfolio optimization."""

    vehicles: list[VehicleInputs] = Field(
        ..., min_length=1, description="List of blended-finance vehicles to optimize across."
    )
    calibrator_config: CalibratorConfig = Field(
        default_factory=CalibratorConfig,
        description="Calibration constraints applied to each vehicle.",
    )
    total_budget: float = Field(..., gt=0, description="Total portfolio capital budget.")
    min_allocation: float = Field(
        0.0, ge=0.0, description="Minimum capital allocation to any single vehicle."
    )
    max_allocation_fraction: float = Field(
        1.0, ge=0.0, le=1.0, description="Maximum fraction of budget to any single vehicle."
    )
    min_expected_return: float = Field(
        0.0, description="Minimum acceptable expected portfolio return (weighted average IRR)."
    )
    cvar_confidence: float = Field(
        0.95, ge=0.5, lt=1.0, description="Confidence level for CVaR constraint."
    )
    cvar_max: float = Field(
        0.30, ge=0.0, description="Maximum acceptable CVaR of portfolio loss rate at cvar_confidence."
    )
    n_sims: int = Field(1000, ge=100, description="Number of Monte Carlo paths per vehicle.")
    seed: int | None = Field(None, description="Random seed for reproducibility.")


@dataclass
class PortfolioResult:
    """Output from PortfolioOptimizer.run()."""

    allocations: dict[int, float]            # vehicle index → capital allocation
    catalytic_allocations: dict[int, float]  # vehicle index → catalytic capital
    commercial_allocations: dict[int, float] # vehicle index → commercial capital
    catalytic_fractions: dict[int, float]    # vehicle index → calibrated alpha*
    leverage_ratio: float                    # total_commercial / total_catalytic
    marginal_catalytic_efficiency: dict[int, float]  # commercial per unit catalytic per vehicle
    portfolio_irr_distribution: np.ndarray   # shape (n_sims,); portfolio-level IRR
    portfolio_loss_distribution: np.ndarray  # shape (n_sims,); portfolio-level loss rate
    cvar_95: float
    status: str  # 'optimal', 'infeasible', 'unbounded', etc.
