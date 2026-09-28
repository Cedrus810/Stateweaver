"""STATE_SEARCH (spec §10.2, §10.5): find the low-lying SCF solutions at one geometry.

Candidates are an independent scratch/aufbau SCF plus every way of placing each spin
channel's metal-d electrons among that channel's metal-d-dominated orbitals of a base
state, each converged with MOM from the base orbitals. A single guess is not enough on an
orbitally degenerate metal: on high-spin [Fe(H2O)6]2+ minao landed on a t2g occupation
0.37 mHa above the lowest, which only this enumeration recovered.
"""
from __future__ import annotations

import itertools

import numpy as np
from pyscf import lo

from sroqm.frame import Frame
from sroqm.imom import make_imom_hook
from sroqm.overlap import occupied_overlap


def _metal_d_rows(mol, metal_atoms) -> list[int]:
    metals = set(metal_atoms)
    return [i for i, (atom, _, nl, _) in enumerate(mol.ao_labels(fmt=False)) if atom in metals and nl.endswith("d")]


def d_occupation_patterns(base: Frame, metal_atoms, min_pop: float = 0.5, window: int = 15,
                          max_candidates: int = 24) -> list[np.ndarray]:
    """Occupation arrays that redistribute each channel's metal-d electrons among its d-dominated MOs.

    A MO counts as d-dominated if its meta-Löwdin population on the metal d AOs exceeds `min_pop`;
    only MOs within `window` of the channel's HOMO are considered. Patterns are ordered by how many
    electrons they move relative to `base` (the base pattern itself first) and capped at `max_candidates`.
    """
    mol = base.mol
    rows = _metal_d_rows(mol, metal_atoms)
    unrestricted = base.unrestricted
    coeffs = base.mo_coeff if unrestricted else base.mo_coeff[None]
    occs = base.mo_occ if unrestricted else base.mo_occ[None]
    orth_s = lo.orth_ao(mol, "meta_lowdin").T @ base.ovlp

    per_channel = []  # per channel: list of (occupation array, number of moved electrons)
    for C, occ in zip(coeffs, occs):
        occupied = np.flatnonzero(occ > 0)
        if not rows or occupied.size == 0:
            per_channel.append([(occ.copy(), 0)])
            continue
        pop = ((orth_s @ C)[rows] ** 2).sum(axis=0)
        homo = occupied[-1]
        dset = [i for i in range(max(0, homo - window), min(len(occ), homo + window + 1)) if pop[i] > min_pop]
        filled = [i for i in dset if occ[i] > 0]
        options = []
        for combo in itertools.combinations(dset, len(filled)):
            new = occ.copy()
            new[dset] = 0.0
            new[list(combo)] = occ[filled[0]] if filled else 0.0
            options.append((new, len(set(filled) - set(combo))))
        per_channel.append(options or [(occ.copy(), 0)])

    patterns = sorted(itertools.product(*per_channel), key=lambda chans: sum(moved for _, moved in chans))
    out = [np.stack([o for o, _ in chans]) if unrestricted else chans[0][0] for chans in patterns]
    return out[:max_candidates]


def _dedupe(frames: list[Frame], same_state_sv: float) -> list[Frame]:
    kept: list[Frame] = []
    for f in sorted(frames, key=lambda f: f.energy):
        if all(occupied_overlap(k, f).min_sv < same_state_sv for k in kept):
            kept.append(f)
    return kept


def search_states(backend, mol, base: Frame | None, metal_atoms, point_charges=None, min_pop: float = 0.5,
                  max_candidates: int = 24, same_state_sv: float = 0.9) -> list[Frame]:
    """Distinct converged states at `mol`'s geometry, lowest first.

    `base` supplies the orbitals to enumerate from (e.g. the current lowest lineage at this geometry);
    with None, the scratch/aufbau orbitals are used, converged or not: MOM from them often converges
    where the scratch SCF itself does not.
    """
    found = []
    scratch = backend.run(mol, point_charges=point_charges, with_gradient=False)
    if scratch.converged:
        found.append(Frame.from_outcome(mol, scratch))
    if base is None:
        base = Frame.from_outcome(mol, scratch)
    for occ in d_occupation_patterns(base, metal_atoms, min_pop=min_pop, max_candidates=max_candidates):
        guess = Frame.from_orbitals(mol, base.mo_coeff, occ)
        out = backend.run(mol, dm0=guess.dm, occ_hook=make_imom_hook(guess, mol), point_charges=point_charges,
                          with_gradient=False)
        if out.converged:
            found.append(Frame.from_outcome(mol, out))
    return _dedupe(found, same_state_sv)
