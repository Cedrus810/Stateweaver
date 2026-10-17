"""PySCF backend: build an SCF object, run it from a supplied guess, return plain arrays."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
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
    energies: tuple = ()                        # e_tot after every SCF cycle
    seconds: dict = field(default_factory=dict)  # wall time: build, df, grids, scf, gradient


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
        mf.chkfile = None  # PySCF's default temporary chkfile is dumped every cycle and never read here
        return mf

    def run(self, mol, dm0=None, occ_hook: OccHook | None = None, point_charges=None,
            with_gradient: bool = True, max_cycle: int | None = None, init_guess: str | None = None) -> SCFOutcome:
        """`max_cycle` overrides the spec's for this call; `init_guess` ("atom", "huckel", ...) applies only without dm0."""
        t0 = time.perf_counter()
        mf = self.build(mol, point_charges)
        if max_cycle is not None:
            mf.max_cycle = max_cycle
        if occ_hook is not None:
            occ_hook(mf)
        if dm0 is None and init_guess is not None:
            dm0 = mf.get_init_guess(key=init_guess)
        energies: list[float] = []
        mf.callback = lambda envs: energies.append(float(envs["e_tot"]))
        t1 = time.perf_counter()
        # Build DF integrals and grids here, exactly as the first get_jk / get_veff would, only to time them.
        # Grids pruned by the initial density (small_rho_cutoff > 0) are left to the kernel.
        with_df = getattr(mf, "with_df", None)
        if with_df is not None and getattr(with_df, "_cderi", None) is None:
            with_df.build()
            t2 = time.perf_counter()
        else:
            t2 = t1
        if getattr(mf, "grids", None) is not None and mf.grids.coords is None and mf.small_rho_cutoff <= 1e-20:
            mf.initialize_grids(mol)
            t3 = time.perf_counter()
        else:
            t3 = t2
        mf.kernel(dm0=dm0)
        t4 = time.perf_counter()
        gradient = np.asarray(mf.nuc_grad_method().kernel()) if with_gradient else None
        t5 = time.perf_counter() if with_gradient else t4
        # Kept outcomes must not keep their DF tensors: ~30 of them pushed RSS past max_memory and PySCF then put the
        # DF integrals of every later SCF on disk, ~10x slower per cycle (search-performance design 2026-10-11 §4).
        # A later DF use rebuilds lazily; the DF gradient does not read _cderi.
        if with_df is not None:
            with_df.reset()
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
            energies=tuple(energies),
            seconds=dict(build=t1 - t0, df=t2 - t1, grids=t3 - t2, scf=t4 - t3, gradient=t5 - t4),
        )

    def gradient(self, outcome: SCFOutcome) -> np.ndarray:
        """Nuclear gradient of this already converged solution; no new SCF or occupation selection."""
        if not outcome.converged:
            raise RuntimeError("cannot compute a reused gradient from an unconverged SCF")
        return np.asarray(outcome.mf.nuc_grad_method().kernel())
