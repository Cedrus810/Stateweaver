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


def _density(mo_coeff, mo_occ):
    dms = [occv * C @ C.T for C, occv in occupied_blocks(mo_coeff, mo_occ)]
    return dms[0] if mo_coeff.ndim == 2 else np.stack(dms)


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
            dm = _density(mo_coeff, mo_occ)
        ovlp = mol.intor_symmetric("int1e_ovlp")
        half, _ = lowdin_pair(ovlp)
        return cls(mol, mo_coeff, mo_occ, np.asarray(dm, dtype=float), float(energy), ovlp, half)

    def with_occupations(self, mo_occ) -> "Frame":
        """A guess on these orbitals and geometry; reuse the geometry's overlap and Löwdin factor."""
        mo_occ = np.asarray(mo_occ, dtype=float).copy()
        if mo_occ.shape != self.mo_occ.shape:
            raise ValueError(f"occupation shape {mo_occ.shape} != {self.mo_occ.shape}")
        return Frame(self.mol, self.mo_coeff, mo_occ, _density(self.mo_coeff, mo_occ),
                     float("nan"), self.ovlp, self.ovlp_half)

    @classmethod
    def from_outcome(cls, mol, outcome) -> "Frame":
        return cls.from_orbitals(mol, outcome.mo_coeff, outcome.mo_occ, outcome.energy, outcome.dm)
