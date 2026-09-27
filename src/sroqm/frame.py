"""One converged electronic snapshot at one geometry."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pyscf import gto

from sroqm.grassmann import lowdin_pair


def occupied_blocks(mo_coeff, mo_occ) -> list[tuple[np.ndarray, float]]:
    """[(C_occ, occupation value)] per spin channel: one block if restricted, two if unrestricted."""
    C = np.asarray(mo_coeff)
    occ = np.asarray(mo_occ)
    if C.ndim == 2:
        return [(C[:, occ > 0], 2.0)]
    return [(C[s][:, occ[s] > 0], 1.0) for s in range(2)]


@dataclass(frozen=True, eq=False)
class Frame:
    mol: gto.Mole
    mo_coeff: np.ndarray
    mo_occ: np.ndarray
    dm: np.ndarray
    energy: float
    ovlp: np.ndarray
    ovlp_half: np.ndarray

    @property
    def unrestricted(self) -> bool:
        return self.mo_coeff.ndim == 3

    @classmethod
    def from_orbitals(cls, mol, mo_coeff, mo_occ, energy: float = float("nan"), dm=None) -> "Frame":
        mo_coeff = np.asarray(mo_coeff, dtype=float)
        mo_occ = np.asarray(mo_occ, dtype=float)
        if dm is None:
            dms = [occv * C @ C.T for C, occv in occupied_blocks(mo_coeff, mo_occ)]
            dm = dms[0] if mo_coeff.ndim == 2 else np.stack(dms)
        ovlp = mol.intor_symmetric("int1e_ovlp")
        half, _ = lowdin_pair(ovlp)
        return cls(mol, mo_coeff, mo_occ, np.asarray(dm, dtype=float), float(energy), ovlp, half)

    @classmethod
    def from_outcome(cls, mol, outcome) -> "Frame":
        return cls.from_orbitals(mol, outcome.mo_coeff, outcome.mo_occ, outcome.energy, outcome.dm)
