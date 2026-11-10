"""Small benchmark systems, built in code (Angstrom input)."""
from __future__ import annotations

import numpy as np
from pyscf import gto

WATER = "O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587"

WATER_DIMER = """
O -1.551007 -0.114520  0.000000
H -1.934259  0.762503  0.000000
H -0.599677  0.040712  0.000000
O  1.350625  0.111469  0.000000
H  1.680398 -0.373741 -0.758561
H  1.680398 -0.373741  0.758561
"""

# fe3_hexaaqua: high-spin d5, 6A1g, orbitally non-degenerate -> the Phase-1 transition-metal benchmark.
# fe2_hexaaqua: high-spin d6, 5T2g, three near-degenerate t2g occupations -> a state-tracking stress test
#               (continuation sat up to 1.75 mHa above another occupation without any overlap alarm).
# fe2_pentaaqua_ammine: [Fe(H2O)5(NH3)]2+, high-spin d6 with one water swapped for ammonia -> the held-out L2 system
#               of the paper-1 protocol (docs/2026-11-06-paper1-validation-protocol.zh-CN.md §2.4): same metal, charge and
#               spin sector as fe2_hexaaqua, one new ligand, the t2g near-degeneracy partly lifted.
DEFAULT_METHOD = {"water": "rks/b3lyp", "water_dimer": "rks/b3lyp",
                  "fe2_hexaaqua": "uks/b3lyp", "fe3_hexaaqua": "uks/b3lyp", "fe2_pentaaqua_ammine": "uks/b3lyp"}


def fe_hexaaqua_atoms(fe_o: float = 2.12, o_h: float = 0.96, hoh_deg: float = 104.5):
    """[Fe(H2O)6]2+: octahedral; each water is planar with its HOH bisector along the Fe-O axis."""
    half = np.radians(hoh_deg) / 2
    axes = np.eye(3)
    atoms = [("Fe", (0.0, 0.0, 0.0))]
    for i in range(3):
        w = axes[(i + 1) % 3]  # the H atoms lie in the (u, w) plane
        for sign in (1.0, -1.0):
            u = sign * axes[i]
            o = fe_o * u
            atoms.append(("O", tuple(o)))
            for t in (1.0, -1.0):
                atoms.append(("H", tuple(o + o_h * (np.cos(half) * u + t * np.sin(half) * w))))
    return atoms


def fe_pentaaqua_ammine_atoms(fe_o: float = 2.12, fe_n: float = 2.20, n_h: float = 1.015, fenh_deg: float = 111.0):
    """[Fe(H2O)5(NH3)]2+: fe_hexaaqua_atoms with the +x water replaced by NH3 (N on the axis, its three H at
    Fe-N-H = fenh_deg, 120 degrees apart). Atom order: Fe, N, 3 H, then the five waters (O, H, H) as in fe_hexaaqua."""
    off = np.radians(180.0 - fenh_deg)  # angle of each N-H bond to the outward Fe-N axis
    u, v, w = np.eye(3)
    n = fe_n * u
    nh3 = [("N", tuple(n))] + [("H", tuple(n + n_h * (np.cos(off) * u + np.sin(off) * (np.cos(p) * v + np.sin(p) * w))))
                               for p in np.radians([0.0, 120.0, 240.0])]
    water = fe_hexaaqua_atoms(fe_o)
    return water[:1] + nh3 + water[4:]  # water[1:4] is the +x water (O, H, H)


def build(name: str, basis: str) -> gto.Mole:
    if name == "water":
        return gto.M(atom=WATER, basis=basis, verbose=0)
    if name == "water_dimer":
        return gto.M(atom=WATER_DIMER, basis=basis, verbose=0)
    if name == "fe2_hexaaqua":
        return gto.M(atom=fe_hexaaqua_atoms(), basis=basis, charge=2, spin=4, verbose=0)
    if name == "fe3_hexaaqua":
        return gto.M(atom=fe_hexaaqua_atoms(fe_o=2.00), basis=basis, charge=3, spin=5, verbose=0)
    if name == "fe2_pentaaqua_ammine":
        return gto.M(atom=fe_pentaaqua_ammine_atoms(), basis=basis, charge=2, spin=4, verbose=0)
    raise ValueError(f"unknown system {name!r}; expected one of {sorted(DEFAULT_METHOD)}")
