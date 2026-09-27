"""Geometry of occupied subspaces: Löwdin orthonormalisation and Grassmann log/exp maps.

Occupied orbitals C (AO basis, C^T S C = I) map to an orthonormal representation
Y = S^{1/2} C. Only span(Y) is physical, and it is invariant to rotations among
occupied orbitals, which is why extrapolating on the Grassmann manifold is
well-posed and an SVD of raw C matrices is not.
"""
from __future__ import annotations

import numpy as np


class GrassmannLogError(ValueError):
    """span(Y1) contains a direction orthogonal to span(Y0), e.g. after an orbital swap."""


def lowdin_pair(S: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    w, v = np.linalg.eigh(S)
    if w.min() <= 0.0:
        raise ValueError(f"overlap matrix is not positive definite (min eigenvalue {w.min():.3e})")
    return (v * np.sqrt(w)) @ v.T, (v / np.sqrt(w)) @ v.T


def grassmann_log(Y0: np.ndarray, Y1: np.ndarray, tol: float = 1e-6) -> np.ndarray:
    if Y0.shape != Y1.shape:
        raise ValueError(f"shape mismatch {Y0.shape} vs {Y1.shape}")
    if Y0.shape[1] == 0:
        return np.zeros_like(Y0)
    M = Y0.T @ Y1
    if np.linalg.svd(M, compute_uv=False).min() < tol:
        raise GrassmannLogError("occupied subspaces are (near-)orthogonal in some direction")
    A = (Y1 - Y0 @ M) @ np.linalg.inv(M)
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    return (U * np.arctan(s)) @ Vt


def grassmann_exp(Y0: np.ndarray, gamma: np.ndarray) -> np.ndarray:
    if Y0.shape[1] == 0:
        return Y0.copy()
    U, s, Vt = np.linalg.svd(gamma, full_matrices=False)
    Y = ((Y0 @ Vt.T) * np.cos(s)) @ Vt + (U * np.sin(s)) @ Vt
    Q, _ = np.linalg.qr(Y)  # re-orthonormalise; QR preserves the column span
    return Q
