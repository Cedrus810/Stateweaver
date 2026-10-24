"""STATE_SEARCH (spec §10.2, §10.5): find the low-lying SCF solutions at one geometry.

Candidates are an independent scratch/aufbau SCF plus every way of placing each spin
channel's metal-d electrons among that channel's metal-d-dominated orbitals of a base
state, each converged with MOM from the base orbitals. A single guess is not enough on an
orbitally degenerate metal: on high-spin [Fe(H2O)6]2+ minao landed on a t2g occupation
0.37 mHa above the lowest, which only this enumeration recovered.
"""
from __future__ import annotations

import dataclasses
import heapq
from dataclasses import dataclass

import numpy as np
from pyscf import lo

from sroqm.executor import SCFTask
from sroqm.frame import Frame
from sroqm.imom import make_imom_hook
from sroqm.overlap import occupied_overlap


def _metal_d_rows(mol, metal_atoms) -> list[int]:
    metals = set(metal_atoms)
    return [i for i, (atom, _, nl, _) in enumerate(mol.ao_labels(fmt=False)) if atom in metals and nl.endswith("d")]


def _occupation_combinations(dset, filled, moved):
    """Lexicographic combinations with exactly `moved` substitutions, without scanning other distances."""
    filled = set(filled)
    n = len(dset)
    suffix_filled = [0] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix_filled[i] = suffix_filled[i + 1] + (dset[i] in filled)

    def choose(start, nfilled, nempty, prefix):
        if nfilled < 0 or nempty < 0 or suffix_filled[start] < nfilled or \
                n - start - suffix_filled[start] < nempty:
            return
        if nfilled + nempty == 0:
            yield prefix
            return
        for i in range(start, n - nfilled - nempty + 1):
            is_filled = dset[i] in filled
            yield from choose(i + 1, nfilled - is_filled, nempty - (not is_filled), prefix + (dset[i],))

    yield from choose(0, len(filled) - moved, moved, ())


def _channel_combinations(channels, distances, index=0, prefix=()):
    """Lazy Cartesian product: itertools.product would first exhaust its input generators."""
    if index == len(channels):
        yield prefix
        return
    dset, filled = channels[index]
    for combo in _occupation_combinations(dset, filled, distances[index]):
        yield from _channel_combinations(channels, distances, index + 1, prefix + (combo,))


def _d_channels(base: Frame, metal_atoms, min_pop: float, window: int):
    """Per spin channel, (d-dominated MO indices within `window` of the HOMO, the occupied ones among them);
    None without metal d AOs or electrons."""
    mol = base.mol
    rows = _metal_d_rows(mol, metal_atoms)
    coeffs = base.mo_coeff if base.unrestricted else base.mo_coeff[None]
    occs = base.mo_occ if base.unrestricted else base.mo_occ[None]
    if not rows or not np.any(occs > 0):
        return None
    orth_s = lo.orth_ao(mol, "meta_lowdin").T @ base.ovlp
    channels = []
    for C, occ in zip(coeffs, occs):
        occupied = np.flatnonzero(occ > 0)
        if occupied.size == 0:
            channels.append(([], []))
            continue
        pop = ((orth_s @ C)[rows] ** 2).sum(axis=0)
        homo = occupied[-1]
        dset = [i for i in range(max(0, homo - window), min(len(occ), homo + window + 1)) if pop[i] > min_pop]
        channels.append((dset, [i for i in dset if occ[i] > 0]))
    return channels


def d_occupation_patterns(base: Frame, metal_atoms, min_pop: float = 0.5, window: int = 15,
                          max_candidates: int = 24) -> list[np.ndarray]:
    """Occupation arrays that redistribute each channel's metal-d electrons among its d-dominated MOs.

    A MO counts as d-dominated if its meta-Löwdin population on the metal d AOs exceeds `min_pop`;
    only MOs within `window` of the channel's HOMO are considered. Patterns are ordered by how many
    electrons they move relative to `base` (the base pattern itself first) and capped at `max_candidates`.
    Equal-distance patterns retain the lexicographic order of the original exhaustive enumeration.
    """
    if max_candidates < 0:
        raise ValueError("max_candidates must be >= 0")
    if max_candidates == 0:
        return []
    unrestricted = base.unrestricted
    occs = base.mo_occ if unrestricted else base.mo_occ[None]
    channels = _d_channels(base, metal_atoms, min_pop, window)
    if channels is None:
        return [base.mo_occ.copy()]

    limits = [min(len(filled), len(dset) - len(filled)) for dset, filled in channels]
    out = []
    for total in range(sum(limits) + 1):
        distances = ((total,),) if not unrestricted else \
            ((a, total - a) for a in range(max(0, total - limits[1]), min(limits[0], total) + 1))
        streams = [_channel_combinations(channels, d) for d in distances]
        for combos in heapq.merge(*streams):
            pattern = occs.copy()
            for s, ((dset, filled), combo) in enumerate(zip(channels, combos)):
                pattern[s, dset] = 0.0
                pattern[s, list(combo)] = occs[s, filled[0]] if filled else 0.0
            out.append(pattern if unrestricted else pattern[0])
            if len(out) == max_candidates:
                return out
    return out


def _dedupe(frames: list[Frame], same_state_sv: float, merged: list | None = None, stage: str = "") -> list[Frame]:
    """Distinct frames, lowest first. `merged` collects (stage, sigma_min, E_dropped - E_kept) of every frame dropped as
    the same state as a kept one (design memo 2026-10-23 §2.2 2a: is sigma >= 0.9 too loose near B?)."""
    kept: list[Frame] = []
    for f in sorted(frames, key=lambda f: f.energy):
        svs = [occupied_overlap(k, f).min_sv for k in kept]
        if all(sv < same_state_sv for sv in svs):
            kept.append(f)
        elif merged is not None:
            i = int(np.argmax(svs))
            merged.append((stage, svs[i], f.energy - kept[i].energy))
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
    elif base.mol is not mol:
        base = Frame.from_orbitals(mol, base.mo_coeff, base.mo_occ)
    for occ in d_occupation_patterns(base, metal_atoms, min_pop=min_pop, max_candidates=max_candidates):
        guess = base.with_occupations(occ)
        out = backend.run(mol, dm0=guess.dm, occ_hook=make_imom_hook(guess, mol), point_charges=point_charges,
                          with_gradient=False)
        if out.converged:
            found.append(Frame.from_outcome(mol, out))
    return _dedupe(found, same_state_sv)


@dataclass(frozen=True, eq=False)
class SearchResult:
    states: list            # distinct converged Frames, lowest first
    level1: int             # distinct states after the first level
    level2: int             # states the second level added
    guesses: tuple          # scratch initial guesses tried, in order ("minao" = PySCF's default)
    seeds: tuple = ()       # level-1 states the second level started from
    pending: int = 0        # unconverged enumerations within keep_window of the lowest: unknown, not absent
    screened: int = 0       # candidates the short-segment screen dropped
    screened_de: tuple = () # (task tag, E_k - anchor in Ha) of each dropped candidate, in drop order
    pending_frames: tuple = ()  # the pending candidates' last (unconverged) frames, to continue later (design §5.2)
    merged: tuple = ()      # (stage, sigma_min, dE in Ha) of every candidate the dedupe dropped as a duplicate
    window_frames: tuple = ()  # every converged candidate within keep_window of the lowest, before the dedupe


def enumeration_tasks(mol, base: Frame, metal_atoms, include_own: bool, max_cycle=None, point_charges=None,
                      tag: str = ""):
    """One IMOM SCF per metal-d occupation of `base`'s orbitals, re-read at `mol` (spec §5.1, §5.3)."""
    if base.mol is not mol:
        base = Frame.from_orbitals(mol, base.mo_coeff, base.mo_occ)
    patterns = d_occupation_patterns(base, metal_atoms)
    guesses = [base.with_occupations(occ) for occ in (patterns if include_own else patterns[1:])]
    return [SCFTask(mol=mol, dm0=g.dm, imom_ref=g, max_cycle=max_cycle, point_charges=point_charges, tag=tag)
            for g in guesses]


def orientation_blocks(base: Frame, metal_atoms, size: int = 3, min_pop: float = 0.5, window: int = 15) -> list:
    """Per spin channel, the MOs whose orientation random_orientation_tasks rotates: the channel's occupied metal-d MOs
    plus its lowest virtual metal-d MOs, `size` in all (MO order = orbital-energy order of canonical orbitals); []
    for a channel with no occupied d MO or at least `size` of them. size = 3 is the t2g block of the minority channel
    of high-spin octahedral d6, the only case calibrated (state.md §4.21, §4.23)."""
    channels = _d_channels(base, metal_atoms, min_pop, window)
    if channels is None:
        return []
    blocks = []
    for dset, filled in channels:
        empty = [i for i in dset if i not in filled]
        ok = 0 < len(filled) < size and len(filled) + len(empty) >= size
        blocks.append(sorted(filled + empty[:size - len(filled)]) if ok else [])
    return blocks


def random_orientation_tasks(mol, base: Frame, metal_atoms, n: int, rng, point_charges=None, size: int = 3,
                             tag: str = "random") -> list[SCFTask]:
    """n IMOM SCFs from `base` with each orientation block (orientation_blocks) turned by an independent Haar-random
    rotation: the occupied d electron(s) of the block land in a random direction of the block's subspace.

    After a symmetric DEGENERATE start the lowest t2g orientation cannot be constructed (E4, state.md §4.21); about 1
    in 9 random orientations reaches it (E5, §4.23). Design memo docs/2026-10-23-phase3b-open-design-questions.zh-CN.md
    option 1a."""
    if base.mol is not mol:
        base = Frame.from_orbitals(mol, base.mo_coeff, base.mo_occ)
    blocks = orientation_blocks(base, metal_atoms, size)
    if not any(blocks):
        return []
    tasks = []
    for _ in range(n):
        C = np.array(base.mo_coeff, copy=True)
        chans = C if base.unrestricted else C[None]
        for s, blk in enumerate(blocks):
            if blk:
                q, r = np.linalg.qr(rng.standard_normal((len(blk), len(blk))))
                chans[s][:, blk] = chans[s][:, blk] @ (q * np.sign(np.diag(r)))
        g = Frame.from_orbitals(mol, C, base.mo_occ)
        tasks.append(SCFTask(mol=mol, dm0=g.dm, imom_ref=g, point_charges=point_charges, tag=tag))
    return tasks


def _lowest(anchor, results):
    """The lowest converged energy among `anchor` (None: unknown) and `results`."""
    energies = ([anchor] if anchor is not None else []) + [r.outcome.energy for r in results if r.converged]
    return min(energies, default=None)


def _screened(executor, tasks, anchor, keep_window, screen):
    """Two-stage enumeration (search-performance design 2026-10-11 §5.1-5.2). With `screen = (k, margin)` and an
    `anchor` (a converged energy at this geometry), every IMOM task first runs k cycles; one still unconverged and
    more than keep_window + margin above the anchor is dropped, the others continue from their short-segment density
    with the short-segment frame as IMOM reference (as refresh stage 2 does). Scratch tasks run in full. Returns
    the results in task order (a dropped task's is its short segment) and which tasks were kept."""
    if screen is None or anchor is None:
        return list(executor.map(tasks)), [True] * len(tasks)
    k, margin = screen
    out = list(executor.map([dataclasses.replace(t, max_cycle=k, tag=f"{t.tag}:s1") if t.imom_ref is not None else t
                             for t in tasks]))
    keep, cont, where = [True] * len(tasks), [], []
    for i, (t, r) in enumerate(zip(tasks, out)):
        if t.imom_ref is None or r.converged:
            continue
        if r.outcome.energy - anchor > keep_window + margin:
            keep[i] = False
        else:
            cont.append(dataclasses.replace(t, dm0=r.outcome.dm, imom_ref=r.frame, tag=f"{t.tag}:s2"))
            where.append(i)
    for i, r in zip(where, executor.map(cont)):
        out[i] = r
    return out, keep


def _pending(results, keep, lowest, keep_window):
    """Kept, unconverged candidates within keep_window of the lowest state: unknown, not absent (spec 10-05 §5.3)."""
    return [r for r, k in zip(results, keep) if k and not r.converged and r.outcome.energy - lowest <= keep_window]


def full_search(executor, mol, bases, metal_atoms, keep_window, tol, point_charges=None, retry_guesses=False,
                guesses=("minao", "atom", "huckel"), merge_guesses=False, anchor=None, screen=None) -> SearchResult:
    """Two-level full search (spec §5.3-5.4).

    Level 1: scratch SCF + enumeration from every frame in `bases` (the lowest lineage, the lineages lost this
    step), or from the scratch orbitals when `bases` is empty. Level 2: enumeration again from every converged
    level-1 state within `keep_window` of the level-1 lowest. On traj2 step 0 level 1 alone missed the lowest
    state under three thread settings; two levels found it every time (state.md §4.9).

    With `retry_guesses`, an empty level 1 is retried with the next scratch guess, enumerating from that scratch's
    orbitals; the enumeration from `bases` does not depend on the guess and runs only with the first.

    With `merge_guesses`, level 1 runs the scratch SCF of every guess and enumerates from every scratch's orbitals
    as well as from `bases`, all merged; `retry_guesses` is then moot. On traj2 step 0, minao alone found the state
    0.70 mHa above the lowest 3 times in 4 thread settings, the merged guesses 4 / 4 (spec 2026-10-05 §7.1).

    With `screen = (k, margin)`, every enumeration is screened as in `_screened` against the lowest converged energy
    known so far, starting from `anchor` (the lowest ACTIVE lineage at this geometry, or None: no screen until a
    converged state is known). Scratch SCFs are never screened.
    """
    enum, dropped, dropped_de = [], 0, []  # every enumeration result with whether the screen kept it
    merged, converged = [], []             # dedupe records; every converged candidate frame

    def run(tasks, anchor_now):
        nonlocal dropped
        out, keep = _screened(executor, tasks, anchor_now, keep_window, screen)
        enum.extend((r, k) for t, r, k in zip(tasks, out, keep) if t.imom_ref is not None)
        dropped += keep.count(False)
        dropped_de.extend((t.tag, r.outcome.energy - anchor_now) for t, r, k in zip(tasks, out, keep) if not k)
        return out

    if merge_guesses:
        scratch_tasks = [SCFTask(mol=mol, init_guess=None if g == "minao" else g, point_charges=point_charges,
                                 tag=f"scratch:{g}") for g in guesses]
        base_tasks = [t for b in bases
                      for t in enumeration_tasks(mol, b, metal_atoms, True, point_charges=point_charges, tag="l1:base")]
        out = run(scratch_tasks + base_tasks, anchor)
        scratches, found = out[:len(scratch_tasks)], list(out[len(scratch_tasks):])
        found += run([t for s in scratches
                      for t in enumeration_tasks(mol, s.frame, metal_atoms, True, point_charges=point_charges,
                                                 tag="l1:scratch")], _lowest(anchor, out))
        tried = list(guesses)
        converged += [s.frame for s in scratches if s.converged] + [r.frame for r in found if r.converged]
        level1 = _dedupe([s.frame for s in scratches if s.converged] + [r.frame for r in found if r.converged],
                         tol.same_state_sv, merged, "level1")
        guesses = ()  # level 1 is done; skip the sequential loop below
    else:
        tried, level1 = [], []
    for guess in guesses:
        tried.append(guess)
        scratch_task = SCFTask(mol=mol, init_guess=None if guess == "minao" else guess, point_charges=point_charges,
                               tag=f"scratch:{guess}")
        if bases and len(tried) == 1:  # scratch and the enumeration are independent: one batch
            tasks = [t for b in bases
                     for t in enumeration_tasks(mol, b, metal_atoms, True, point_charges=point_charges, tag="l1:base")]
            scratch, *found = run([scratch_task] + tasks, anchor)
        else:  # no bases, or a retry: the enumeration starts from the scratch orbitals
            (scratch,) = executor.map([scratch_task])
            found = run(enumeration_tasks(mol, scratch.frame, metal_atoms, True, point_charges=point_charges,
                                          tag="l1:scratch"), _lowest(anchor, [scratch]))
        converged += ([scratch.frame] if scratch.converged else []) + [r.frame for r in found if r.converged]
        level1 = _dedupe(([scratch.frame] if scratch.converged else []) + [r.frame for r in found if r.converged],
                         tol.same_state_sv, merged, "level1")
        if level1 or not retry_guesses:
            break
    if not level1:
        # Nothing converged. Candidates within keep_window of a known energy (the anchor) are still unknown, not
        # absent: count them and hand them on to be continued (design 2026-10-11 §5.2; review 2026-10-23 #2).
        pending = [] if anchor is None else _pending([r for r, _ in enum], [k for _, k in enum], anchor, keep_window)
        return SearchResult([], 0, 0, tuple(tried), (), len(pending), dropped, tuple(dropped_de),
                            tuple(r.frame for r in pending), tuple(merged))
    seeds = [f for f in level1 if f.energy - level1[0].energy <= keep_window]
    found2 = run([t for f in seeds
                  for t in enumeration_tasks(mol, f, metal_atoms, False, point_charges=point_charges, tag="l2")],
                 min(e for e in (anchor, level1[0].energy) if e is not None))
    converged += [r.frame for r in found2 if r.converged]
    states = _dedupe(level1 + [r.frame for r in found2 if r.converged], tol.same_state_sv, merged, "level2")
    pending = _pending([r for r, _ in enum], [k for _, k in enum], states[0].energy, keep_window)  # spec 10-05 §5.3
    window = tuple(sorted((f for f in converged if f.energy - states[0].energy <= keep_window), key=lambda f: f.energy))
    return SearchResult(states, len(level1), len(states) - len(level1), tuple(tried), tuple(seeds), len(pending),
                        dropped, tuple(dropped_de), tuple(r.frame for r in pending), tuple(merged), window)
