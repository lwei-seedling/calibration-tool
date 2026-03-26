"""Statistical utilities: VaR, CVaR, and correlated Monte Carlo draws."""
from __future__ import annotations

import numpy as np


def var(losses: np.ndarray, confidence: float = 0.95) -> float:
    """Value at Risk at the given confidence level."""
    return float(np.nanquantile(losses, confidence))


def cvar(losses: np.ndarray, confidence: float = 0.95) -> float:
    """Conditional Value at Risk (Expected Shortfall) at the given confidence level."""
    threshold = var(losses, confidence)
    tail = losses[losses >= threshold]
    if len(tail) == 0:
        return float(threshold)
    return float(np.nanmean(tail))


def nearest_positive_definite(matrix: np.ndarray) -> np.ndarray:
    """Project a symmetric matrix to the nearest positive-definite matrix.

    Uses Higham's (2002) algorithm via eigenvalue clipping.
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

    # Check positive definiteness; apply nearest-PD if needed
    eigenvalues = np.linalg.eigvalsh(corr_matrix)
    if np.any(eigenvalues <= 0):
        corr_matrix = nearest_positive_definite(corr_matrix)

    L = np.linalg.cholesky(corr_matrix)  # shape (D, D), lower triangular

    # Draw iid standard normals, shape (D, n_sims)
    U = rng.standard_normal(size=(D, n_sims))

    # Correlated draws: Z = L @ U, shape (D, n_sims)
    Z = L @ U

    return Z.T  # shape (n_sims, D)
