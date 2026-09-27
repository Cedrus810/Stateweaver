"""Cross-geometry overlap of occupied spaces: the continuity measure between frames."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pyscf import gto

from sroqm.frame import Frame, occupied_blocks


def cross_overlap(mol_a, mol_b) -> np.ndarray:
    return gto.intor_cross("int1e_ovlp", mol_a, mol_b)


@dataclass(frozen=True)
class OccupiedOverlap:
    det: float     # |<Phi_a|Phi_b>|; decays with electron count, reported as metadata
    min_sv: float  # cosine of the largest principal angle; size-intensive, ~0 on one orbital swap


def occupied_overlap(a: Frame, b: Frame) -> OccupiedOverlap:
    blocks_a = occupied_blocks(a.mo_coeff, a.mo_occ)
    blocks_b = occupied_blocks(b.mo_coeff, b.mo_occ)
    if len(blocks_a) != len(blocks_b):
        raise ValueError("cannot compare a restricted frame with an unrestricted one")
    s_ab = cross_overlap(a.mol, b.mol)
    det, min_sv = 1.0, 1.0
    for (ca, occv), (cb, _) in zip(blocks_a, blocks_b):
        if ca.shape[1] != cb.shape[1]:
            raise ValueError(f"occupied counts differ ({ca.shape[1]} vs {cb.shape[1]})")
        if ca.shape[1] == 0:
            continue
        sv = np.linalg.svd(ca.T @ s_ab @ cb, compute_uv=False)
        det *= float(np.prod(sv)) ** (2 if occv == 2.0 else 1)
        min_sv = min(min_sv, float(sv.min()))
    return OccupiedOverlap(det=det, min_sv=min_sv)
