"""Risk mitigants: Guarantee and GrantReserve."""
from __future__ import annotations

from enum import Enum

import numpy as np


class CoverageType(str, Enum):
    ABSOLUTE = "ABSOLUTE"
    PERCENTAGE = "PERCENTAGE"


class Guarantee:
    """Partial credit guarantee that wraps a specific tranche (typically senior).

    The guarantee does NOT sit as a sequential layer before first-loss capital.
    It intercepts losses that have already passed through grant reserve,
    first-loss, and mezzanine tranches, before they reach the wrapped tranche.

    In real DFI structures (MIGA, DFC, GuarantCo), guarantees protect senior
    noteholders, not subordinated catalytic capital.
    """

    def __init__(
        self,
        coverage_limit: float,
        coverage_type: CoverageType = CoverageType.PERCENTAGE,
    ) -> None:
        self.coverage_limit = coverage_limit
        self.coverage_type = coverage_type

    def effective_cap(self, tranche_notional: float) -> float:
        """Maximum loss the guarantee will absorb."""
        if self.coverage_type == CoverageType.ABSOLUTE:
            return self.coverage_limit
        return self.coverage_limit * tranche_notional

    def absorb(self, loss: np.ndarray, tranche_notional: float) -> tuple[np.ndarray, np.ndarray]:
        """Apply guarantee to a loss distribution.

        Args:
            loss: Per-path loss amounts (n_sims,) presented to the guarantee.
            tranche_notional: Notional of the wrapped tranche.

        Returns:
            (absorbed, residual) arrays of the same shape as loss.
        """
        cap = self.effective_cap(tranche_notional)
        absorbed = np.minimum(loss, cap)
        residual = loss - absorbed
        return absorbed, residual


class GrantReserve:
    """Cash buffer that absorbs the first dollars of project loss.

    Sits ahead of all capital tranches in the loss waterfall. Has zero
    return expectation — it is a donor-funded loss cushion.
    """

    def __init__(self, amount: float) -> None:
        if amount < 0:
            raise ValueError("Grant reserve amount must be non-negative.")
        self.amount = amount

    def absorb(self, loss: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Apply grant reserve to a loss distribution.

        Args:
            loss: Per-path terminal loss (n_sims,).

        Returns:
            (absorbed, residual) arrays. absorbed <= amount per path.
        """
        absorbed = np.minimum(loss, self.amount)
        residual = loss - absorbed
        return absorbed, residual
