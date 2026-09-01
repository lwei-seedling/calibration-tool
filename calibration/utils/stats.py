"""Statistical utilities: VaR, CVaR, and correlated Monte Carlo draws."""
from __future__ import annotations

import numpy as np


def var(losses: np.ndarray, confidence: float = 0.95) -> float:
    """Value at Risk at the given confidence level."""
    return float(np.nanquantile(losses, confidence))


def cvar(losses: np.ndarray, confidence: float = 0.95) -> float:
    """Conditional Value at Risk (Expected Shortfall) at the given confidence level.

    Averages the worst ``ceil((1 - confidence) * n)`` losses by sort order.

    Selecting the tail by rank rather than by ``losses >= VaR`` matters whenever
    the loss distribution has an atom at its quantile — which is the *normal*
    case for a senior tranche, where losses are zero in all but a few percent of
    scenarios. There VaR is exactly 0.0, a ``>=`` comparison admits every
    zero-loss path into the "tail", and the result collapses toward the mean of
    the whole distribution instead of describing the tail.
    """
    losses = np.asarray(losses, dtype=float)
    finite = losses[~np.isnan(losses)]
    if finite.size == 0:
        return float("nan")

    # The epsilon absorbs binary representation error: (1 - 0.95) * 1000 is
    # 50.00000000000004, which would otherwise round the tail up to 51 paths.
    k = max(1, int(np.ceil((1.0 - confidence) * finite.size - 1e-9)))
    tail = np.partition(finite, -k)[-k:]
    return float(np.mean(tail))


def nearest_positive_definite(matrix: np.ndarray) -> np.ndarray:
    """Project a symmetric matrix to a nearby positive-definite correlation matrix.

    Symmetrises, clips negative eigenvalues to a small positive floor, then
    rescales the diagonal back to 1. This is eigenvalue clipping, *not* Higham's
    (2002) alternating-projection algorithm: it is a single projection and does
    not claim to find the true nearest correlation matrix. It is cheap, stable,
    and adequate for repairing the small J x J matrices used here.
    """
    # Symmetrize
    B = (matrix + matrix.T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(B)
    # Clip negative eigenvalues to a small positive value
    eigenvalues = np.maximum(eigenvalues, 1e-8)
    pd = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    # Re-symmetrize and normalize diagonal to 1 (correlation matrix)
    pd = (pd + pd.T) / 2.0
    d = np.sqrt(np.diag(pd))
    pd = pd / np.outer(d, d)
    return pd


def cholesky_correlated_draws(
    n_sims: int,
    corr_matrix: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate correlated standard-normal draws via Cholesky decomposition.

    Args:
        n_sims: Number of simulation paths.
        corr_matrix: (D, D) correlation matrix (symmetric, PD).
        rng: NumPy random Generator for reproducibility.

    Returns:
        Array of shape (n_sims, D) with correlated standard-normal draws
        having covariance structure given by corr_matrix.
    """
    corr_matrix = np.asarray(corr_matrix, dtype=float)
    D = corr_matrix.shape[0]

    # Validate symmetry
    if not np.allclose(corr_matrix, corr_matrix.T, atol=1e-8):
        raise ValueError("Correlation matrix must be symmetric.")

    # Check positive definiteness; apply nearest-PD if needed. The tolerance
    # matters: a matrix whose smallest eigenvalue is a hair above zero passes a
    # bare `> 0` test and then still fails Cholesky.
    eigenvalues = np.linalg.eigvalsh(corr_matrix)
    tol = 1e-8 * max(1.0, float(np.max(np.abs(eigenvalues))))
    if np.any(eigenvalues <= tol):
        corr_matrix = nearest_positive_definite(corr_matrix)

    L = np.linalg.cholesky(corr_matrix)  # shape (D, D), lower triangular

    # Draw iid standard normals, shape (D, n_sims)
    U = rng.standard_normal(size=(D, n_sims))

    # Correlated draws: Z = L @ U, shape (D, n_sims)
    Z = L @ U

    return Z.T  # shape (n_sims, D)
