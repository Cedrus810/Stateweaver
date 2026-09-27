"""Rotation-invariant electronic-state fingerprints (spec §6.3) and frontier gap."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class Fingerprint:
    s2: float
    fragment_spins: dict[str, float]
    fragment_charges: dict[str, float]


@dataclass(frozen=True)
class FingerprintChange:
    ds2: float
    max_dspin: float
    max_dcharge: float


def default_fragments(mol) -> dict[str, list[int]]:
    return {f"{mol.atom_symbol(i)}{i}": [i] for i in range(mol.natm)}


def _validate(fragments: dict[str, Sequence[int]], natm: int) -> None:
    seen: set[int] = set()
    for name, idx in fragments.items():
        for i in idx:
            if not 0 <= i < natm:
                raise ValueError(f"fragment {name!r}: atom index {i} out of range 0..{natm - 1}")
            if i in seen:
                raise ValueError(f"atom {i} assigned to more than one fragment")
            seen.add(i)


def compute_fingerprint(mf, fragments: dict[str, Sequence[int]] | None = None) -> Fingerprint:
    mol = mf.mol
    frags = fragments if fragments is not None else default_fragments(mol)
    _validate(frags, mol.natm)
    _, atom_charges = mf.mulliken_meta(verbose=0)
    if mf.istype("UHF"):
        _, atom_spins = mf.mulliken_meta_spin(verbose=0)
        s2 = float(mf.spin_square()[0])
    else:
        atom_spins = np.zeros(mol.natm)
        s2 = 0.0
    return Fingerprint(
        s2=s2,
        fragment_spins={n: float(np.sum(np.asarray(atom_spins)[list(i)])) for n, i in frags.items()},
        fragment_charges={n: float(np.sum(np.asarray(atom_charges)[list(i)])) for n, i in frags.items()},
    )


def compare(a: Fingerprint, b: Fingerprint) -> FingerprintChange:
    if a.fragment_spins.keys() != b.fragment_spins.keys():
        raise ValueError("fingerprints use different fragments")
    return FingerprintChange(
        ds2=abs(b.s2 - a.s2),
        max_dspin=max((abs(b.fragment_spins[k] - a.fragment_spins[k]) for k in a.fragment_spins), default=0.0),
        max_dcharge=max((abs(b.fragment_charges[k] - a.fragment_charges[k]) for k in a.fragment_charges),
                        default=0.0),
    )


def frontier_gap(mo_energy, mo_occ) -> float:
    e = np.asarray(mo_energy)
    occ = np.asarray(mo_occ)
    if e.ndim == 1:
        e, occ = e[None], occ[None]
    gaps = [float(es[os == 0].min() - es[os > 0].max())
            for es, os in zip(e, occ) if (os > 0).any() and (os == 0).any()]
    return min(gaps, key=abs) if gaps else float("inf")
