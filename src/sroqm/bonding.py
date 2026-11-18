"""Per-lineage bonding descriptors for metal-ligand pairs (Stateweaver design
docs/coordination-electronic-lineage-research-design.zh-CN.md §5, §7; WP1).

Stateless: from one electronic Frame (any lineage, any step) and a fixed list of atom pairs, the distance, the
open-shell Mayer bond order and the two atoms' Mulliken charge and spin population. Nothing here changes energies,
forces, state selection or search.

Mayer bond order (design §5.3; Mayer, Chem. Phys. Lett. 97, 270 (1983); Int. J. Quantum Chem. 29, 477 (1986)), real AO
basis, with P = D_alpha + D_beta and W = D_alpha - D_beta (W = 0 for a restricted frame):

    M_AB = sum_{mu in A, nu in B} (PS)_{mu nu} (PS)_{nu mu} + (WS)_{mu nu} (WS)_{nu mu}

It is not confined to [0, 1] and is not a probability: values are kept raw; a negative one is only flagged.
Mulliken populations: q_A = Z_A - sum_{mu in A} (PS)_{mu mu}, m_A = sum_{mu in A} (WS)_{mu mu} (signed spin
population, not a spin quantum number). One population scheme per observation, recorded with it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sroqm.frame import Frame

BOHR_TO_ANGSTROM = 0.52917721092
SCHEME = "mulliken"


@dataclass(frozen=True)
class PairSpec:
    """A metal-donor pair to observe, by atom index in the (fixed) QM atom list."""
    pair_id: str
    atoms: tuple[int, int]
    donor_class: str = ""          # e.g. "Fe-O", "Fe-N": calibration is per class (design §5.1)


@dataclass(frozen=True)
class BondObservation:
    pair_id: str
    lineage_id: int | None
    step: int | None
    distance_angstrom: float
    mayer_order: float
    charges: tuple[float, float]   # Mulliken charges of the pair's two atoms
    spins: tuple[float, float]     # Mulliken signed spin populations of the pair's two atoms
    population_scheme: str = SCHEME
    quality: tuple[str, ...] = ()  # e.g. ("negative_mayer",): to review, never rewritten


def _ps_ws(frame: Frame) -> tuple[np.ndarray, np.ndarray | None]:
    dm, S = np.asarray(frame.dm), frame.ovlp
    if dm.ndim == 2:
        return dm @ S, None
    return (dm[0] + dm[1]) @ S, (dm[0] - dm[1]) @ S


def _atom_slices(mol) -> list[slice]:
    return [slice(int(a[2]), int(a[3])) for a in mol.aoslice_by_atom()]


def mayer_bond_orders(frame: Frame, pairs) -> dict[str, float]:
    """Open-shell Mayer bond order of every pair, sharing PS / WS across pairs (design §5.3)."""
    PS, WS = _ps_ws(frame)
    sl = _atom_slices(frame.mol)
    out = {}
    for p in pairs:
        a, b = sl[p.atoms[0]], sl[p.atoms[1]]
        m = float(np.sum(PS[a, b] * PS[b, a].T))
        if WS is not None:
            m += float(np.sum(WS[a, b] * WS[b, a].T))
        out[p.pair_id] = m
    return out


def mulliken(frame: Frame) -> tuple[np.ndarray, np.ndarray]:
    """Per-atom Mulliken charge and signed spin population (zero spin for a restricted frame)."""
    PS, WS = _ps_ws(frame)
    sl = _atom_slices(frame.mol)
    Z = frame.mol.atom_charges()  # effective nuclear charges (ECP cores removed)
    q = np.array([Z[i] - np.trace(PS[s, s]) for i, s in enumerate(sl)])
    m = np.zeros(len(sl)) if WS is None else np.array([np.trace(WS[s, s]) for s in sl])
    return q, m


def observe(frame: Frame, pairs, lineage_id=None, step=None) -> list[BondObservation]:
    """One BondObservation per pair for this frame."""
    xyz = frame.mol.atom_coords() * BOHR_TO_ANGSTROM
    orders = mayer_bond_orders(frame, pairs)
    q, m = mulliken(frame)
    out = []
    for p in pairs:
        i, j = p.atoms
        mo = orders[p.pair_id]
        out.append(BondObservation(p.pair_id, lineage_id, step, float(np.linalg.norm(xyz[i] - xyz[j])), mo,
                                   (float(q[i]), float(q[j])), (float(m[i]), float(m[j])), SCHEME,
                                   ("negative_mayer",) if mo < 0 else ()))
    return out


def observe_beam(result, pairs, step=None) -> dict[int, list[BondObservation]]:
    """Every ACTIVE lineage of one BeamResult (its `frames`), by lineage id: the bypass observer's input (design §7)."""
    return {lid: observe(f, pairs, lid, step) for lid, f in result.frames.items()}
