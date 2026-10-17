"""Every tolerance of the Phase-3b beam in one place (spec §7); beam, search and acceptance share one instance."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Tolerances:
    conv_tol: float = 1e-7       # Hartree; the orbital-gradient threshold stays at PySCF's sqrt(conv_tol)
    same_state_sv: float = 0.9   # occupied-space sigma_min at or above which two frames are one state
    admit_de: float = 1e-5       # Hartree; admission and the gradient re-seat must re-converge within this
    deg_band: float = 1e-4       # Hartree; two lowest lineages closer than this -> DEGENERATE
    accept_de: float = 5e-5      # Hartree; acceptance energy criterion when frames cannot be compared

    def __post_init__(self):
        for name in ("conv_tol", "admit_de", "deg_band", "accept_de"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be > 0, got {getattr(self, name)}")
        if not 0 < self.same_state_sv <= 1:
            raise ValueError(f"same_state_sv must be in (0, 1], got {self.same_state_sv}")

    def same_state(self, a, b) -> bool:
        from sroqm.overlap import occupied_overlap

        return occupied_overlap(a, b).min_sv >= self.same_state_sv
