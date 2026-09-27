"""PySCF backend: build an SCF object, run it from a supplied guess, return plain arrays."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from pyscf import dft, qmmm, scf

_REFERENCES = ("rhf", "uhf", "rks", "uks")

OccHook = Callable[[object], None]


@dataclass(frozen=True)
class MethodSpec:
    method: str  # "rhf", "uhf", "rks/<xc>", "uks/<xc>"
    conv_tol: float = 1e-9
    max_cycle: int = 100
    density_fit: bool = False
    grid_level: int = 3
    verbose: int = 0

    def __post_init__(self):
        if self.reference not in _REFERENCES:
            raise ValueError(f"unknown reference {self.reference!r}; expected one of {_REFERENCES}")
        if self.reference in ("rks", "uks") and not self.xc:
            raise ValueError(f"{self.reference} needs a functional, e.g. '{self.reference}/b3lyp'")
        if self.reference in ("rhf", "uhf") and self.xc:
            raise ValueError(f"{self.reference} takes no functional; use rks/uks for DFT")

    @property
    def reference(self) -> str:
        return self.method.split("/", 1)[0].lower()

    @property
    def xc(self) -> str | None:
        parts = self.method.split("/", 1)
        return parts[1] if len(parts) == 2 and parts[1] else None

    @property
    def unrestricted(self) -> bool:
        return self.reference in ("uhf", "uks")


@dataclass(frozen=True, eq=False)
class SCFOutcome:
    energy: float
    gradient: np.ndarray | None
    mo_coeff: np.ndarray
    mo_occ: np.ndarray
    mo_energy: np.ndarray
    dm: np.ndarray
    cycles: int
    converged: bool
    mf: object


class PySCFBackend:
    def __init__(self, spec: MethodSpec):
        self.spec = spec

    def build(self, mol, point_charges=None):
        s = self.spec
        if mol.spin != 0 and not s.unrestricted:
            raise ValueError(f"open-shell molecule (spin={mol.spin}) needs uhf/uks, got {s.method!r}")
        ref = s.reference
        if ref == "rhf":
            mf = scf.RHF(mol)
        elif ref == "uhf":
            mf = scf.UHF(mol)
        elif ref == "rks":
            mf = dft.RKS(mol, xc=s.xc)
        else:
            mf = dft.UKS(mol, xc=s.xc)
        if ref in ("rks", "uks"):
            mf.grids.level = s.grid_level
        if s.density_fit:
            mf = mf.density_fit()
        if point_charges is not None:
            coords, charges = point_charges
            mf = qmmm.mm_charge(mf, np.asarray(coords, float), np.asarray(charges, float), unit="Bohr")
        mf.conv_tol = s.conv_tol
        mf.max_cycle = s.max_cycle
        mf.verbose = s.verbose
        return mf

    def run(self, mol, dm0=None, occ_hook: OccHook | None = None, point_charges=None,
            with_gradient: bool = True) -> SCFOutcome:
        mf = self.build(mol, point_charges)
        if occ_hook is not None:
            occ_hook(mf)
        mf.kernel(dm0=dm0)
        gradient = np.asarray(mf.nuc_grad_method().kernel()) if with_gradient else None
        return SCFOutcome(
            energy=float(mf.e_tot),
            gradient=gradient,
            mo_coeff=np.asarray(mf.mo_coeff),
            mo_occ=np.asarray(mf.mo_occ),
            mo_energy=np.asarray(mf.mo_energy),
            dm=np.asarray(mf.make_rdm1()),
            cycles=int(mf.cycles),
            converged=bool(mf.converged),
            mf=mf,
        )
