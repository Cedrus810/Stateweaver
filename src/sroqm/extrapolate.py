"""Initial-guess extrapolators: predict the density at a new geometry from lineage history.

`history` is ordered most recent first. Each extrapolator returns a density matrix in
the new geometry's AO basis ((nao, nao) or (2, nao, nao)), or None to let PySCF build
its default guess.
"""
from __future__ import annotations

from math import comb
from typing import Protocol, Sequence

import numpy as np

from sroqm.frame import Frame, occupied_blocks
from sroqm.grassmann import GrassmannLogError, grassmann_exp, grassmann_log, lowdin_pair


class Extrapolator(Protocol):
    def guess(self, history: Sequence[Frame], mol) -> np.ndarray | None: ...


def aspc_coefficients(n: int) -> np.ndarray:
    """Kolafa's always-stable predictor coefficients for n history points (J. Comput. Chem. 2004)."""
    if n < 1:
        raise ValueError("need at least one history point")
    return np.array([(-1) ** (j + 1) * j * comb(2 * n, n - j) / comb(2 * n - 2, n - 1)
                     for j in range(1, n + 1)])


def polynomial_coefficients(n: int) -> np.ndarray:
    """Extrapolation to the next equally spaced point; exact for polynomials of degree n-1."""
    if n < 1:
        raise ValueError("need at least one history point")
    return np.array([(-1) ** j * comb(n, j + 1) for j in range(n)], dtype=float)


class ScratchGuess:
    def guess(self, history, mol):
        return None


class ReadGuess:
    def guess(self, history, mol):
        return history[0].dm.copy() if history else None


class ASPCGuess:
    """ASPC predictor on AO density matrices; ignores basis motion, as most BOMD codes do."""

    def __init__(self, order: int = 4):
        if order < 1:
            raise ValueError("order must be >= 1")
        self.order = order

    def guess(self, history, mol):
        if not history:
            return None
        k = min(self.order, len(history))
        return sum(c * f.dm for c, f in zip(aspc_coefficients(k), history[:k]))


class GrassmannGuess:
    """Extrapolate occupied subspaces on the Grassmann manifold (Polack et al., JCTC 2021).

    Per spin channel: Y_i = S_i^{1/2} C_occ,i; tangent vectors Log_{Y_0}(Y_i) are combined
    with extrapolation coefficients, mapped back with Exp, and C = S_new^{-1/2} Y.
    With one history frame, or if a Log fails (orbital swap in the history), it returns
    the latest density: a bare Löwdin transfer is a worse guess than plain reuse.
    """

    _SCHEMES = {"polynomial": polynomial_coefficients, "aspc": aspc_coefficients}

    def __init__(self, order: int = 3, scheme: str = "polynomial"):
        if order < 1:
            raise ValueError("order must be >= 1")
        if scheme not in self._SCHEMES:
            raise ValueError(f"unknown scheme {scheme!r}; expected one of {sorted(self._SCHEMES)}")
        self.order = order
        self.scheme = scheme

    def guess(self, history, mol):
        if not history:
            return None
        k = min(self.order, len(history))
        if k == 1:
            return history[0].dm.copy()
        coeffs = self._SCHEMES[self.scheme](k)
        _, s_invhalf = lowdin_pair(mol.intor_symmetric("int1e_ovlp"))
        per_frame = [[(f.ovlp_half @ C, occv) for C, occv in occupied_blocks(f.mo_coeff, f.mo_occ)]
                     for f in history[:k]]
        dms = []
        for s, (ref, occv) in enumerate(per_frame[0]):
            try:
                gamma = sum(c * grassmann_log(ref, per_frame[j][s][0])
                            for j, c in enumerate(coeffs) if j > 0)
            except GrassmannLogError:
                return history[0].dm.copy()
            C = s_invhalf @ grassmann_exp(ref, gamma)
            dms.append(occv * C @ C.T)
        return dms[0] if len(dms) == 1 else np.stack(dms)


_FACTORY = {"scratch": ScratchGuess, "read": ReadGuess, "aspc": ASPCGuess, "grassmann": GrassmannGuess}


def make_extrapolator(name: str) -> Extrapolator:
    try:
        return _FACTORY[name]()
    except KeyError:
        raise ValueError(f"unknown extrapolator {name!r}; expected one of {sorted(_FACTORY)}") from None
