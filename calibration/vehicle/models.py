"""Pydantic input models and result dataclasses for the vehicle layer."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from pydantic import BaseModel, Field, model_validator

from calibration.project.models import ProjectInputs


class VehicleInputs(BaseModel):
    """Validated inputs for a blended-finance vehicle (portfolio of projects)."""

    projects: list[ProjectInputs] = Field(
        ..., min_length=1, description="List of projects in this vehicle (typically 3–5)."
    )
    correlation_matrix: list[list[float]] = Field(
        ..., description="(J×J) project-level cashflow correlation matrix."
    )
    total_capital: float = Field(..., gt=0, description="Total vehicle capital (catalytic + commercial).")
    guarantee_coverage: float = Field(
        0.0, ge=0.0, le=1.0, description="Guarantee coverage as fraction of senior tranche notional."
    )
    grant_reserve: float = Field(
        0.0, ge=0.0, description="Grant reserve amount (absolute, sits ahead of first-loss in waterfall)."
    )
    mezzanine_fraction: float = Field(
        0.0, ge=0.0, le=0.5, description="Fraction of total capital allocated to mezzanine tranche."
    )
    senior_coupon: float = Field(
        0.08, ge=0.0, description="Annual coupon rate for senior tranche."
    )
    mezzanine_coupon: float = Field(
        0.12, ge=0.0, description="Annual coupon rate for mezzanine tranche."
    )
    discount_rate: float = Field(
        0.0, ge=0.0,
        description="Annual discount rate for NPV-based terminal loss computation "
                    "in the loss waterfall. Default 0.0 gives undiscounted (sum-based) "
                    "loss for backward compatibility."
    )

    @model_validator(mode="after")
    def validate_correlation_matrix(self) -> "VehicleInputs":
        J = len(self.projects)
        mat = self.correlation_matrix
        if len(mat) != J or any(len(row) != J for row in mat):
            raise ValueError(f"correlation_matrix must be {J}×{J} to match {J} projects.")
        return self

    @property
    def n_projects(self) -> int:
        return len(self.projects)

    @property
    def corr_array(self) -> np.ndarray:
        return np.array(self.correlation_matrix, dtype=float)


@dataclass
class TrancheResult:
    """Statistics for a single tranche."""

    name: str
    notional: float
    irr_distribution: np.ndarray   # shape (n_sims,)
    loss_distribution: np.ndarray  # shape (n_sims,); absorbed loss per path
    loss_probability: float
    var_95: float
    cvar_95: float
    median_irr: float


@dataclass
class VehicleResult:
    """Output from vehicle calibration."""

    catalytic_capital: float         # solved catalytic fraction × total_capital
    commercial_capital: float        # total_capital − catalytic_capital
    catalytic_fraction: float        # alpha*
    tranche_results: dict[str, TrancheResult] = field(default_factory=dict)

    @property
    def leverage_ratio(self) -> float:
        """Commercial capital mobilized per unit of catalytic capital."""
        if self.catalytic_capital == 0.0:
            return float("inf")
        return self.commercial_capital / self.catalytic_capital

    @property
    def marginal_catalytic_efficiency(self) -> float:
        """Commercial capital mobilized per unit of catalytic deployed at minimum feasible alpha.

        Defined as (1 - alpha*) / alpha* — the leverage ratio evaluated at the
        minimum catalytic fraction required to satisfy investor constraints.
        Higher values indicate more capital-efficient deployment.
        """
        return self.leverage_ratio
