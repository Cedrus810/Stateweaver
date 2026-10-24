import dataclasses
import itertools
from types import SimpleNamespace

import numpy as np
import pytest

from sroqm.backend import MethodSpec, PySCFBackend
from sroqm.executor import SCFTask, SerialExecutor
from sroqm.frame import Frame
from sroqm.overlap import occupied_overlap
from sroqm.search import _screened, d_occupation_patterns, enumeration_tasks, full_search, search_states
from sroqm.tolerances import Tolerances


@pytest.fixture(scope="module")
def fe2_backend():
    return PySCFBackend(MethodSpec("uks/b3lyp", conv_tol=1e-7, density_fit=True))


@pytest.fixture(scope="module")
def fe2_base(fe2_distorted, fe2_backend):
    return Frame.from_outcome(fe2_distorted, fe2_backend.run(fe2_distorted, with_gradient=False))


@pytest.fixture(scope="module")
def fe2_states(fe2_distorted, fe2_backend):
    return search_states(fe2_backend, fe2_distorted, None, metal_atoms=[0])


def test_d_patterns_put_the_minority_electron_in_each_d_orbital(fe2_base):
    patterns = d_occupation_patterns(fe2_base, metal_atoms=[0])
    assert len({tuple(p[1]) for p in patterns}) == 5  # high-spin d6: one beta electron, five d orbitals
    for p in patterns:
        assert p[0].sum() == fe2_base.mo_occ[0].sum()
        assert p[1].sum() == fe2_base.mo_occ[1].sum()


def test_d_patterns_without_a_metal_are_just_the_base(fe2_base):
    (only,) = d_occupation_patterns(fe2_base, metal_atoms=[])
    assert np.array_equal(only, fe2_base.mo_occ)


def test_search_finds_the_three_t2g_states_sorted_and_distinct(fe2_states):
    energies = [f.energy for f in fe2_states]
    assert energies == sorted(energies)
    assert sum(e - energies[0] < 5e-3 for e in energies) == 3  # 5T2g: three t2g occupations
    for a, b in itertools.combinations(fe2_states, 2):
        assert occupied_overlap(a, b).min_sv < 0.9


def test_search_from_a_high_state_still_reaches_the_lowest(fe2_distorted, fe2_backend, fe2_states):
    low = [f for f in fe2_states if f.energy - fe2_states[0].energy < 5e-3]
    again = search_states(fe2_backend, fe2_distorted, low[-1], metal_atoms=[0])
    assert again[0].energy == pytest.approx(fe2_states[0].energy, abs=1e-6)


def test_search_without_a_metal_returns_the_aufbau_state(water):
    (state,) = search_states(PySCFBackend(MethodSpec("rks/b3lyp")), water, None, metal_atoms=[])
    assert np.isfinite(state.energy)


class ScratchNeverConverges(PySCFBackend):
    """Real SCF, but the scratch (no-guess) run reports failure, as on a hard metal geometry."""

    def run(self, mol, dm0=None, occ_hook=None, **kw):
        out = super().run(mol, dm0=dm0, occ_hook=occ_hook, **kw)
        return dataclasses.replace(out, converged=False) if dm0 is None else out


def test_search_enumerates_from_an_unconverged_scratch_solution(water):
    (state,) = search_states(ScratchNeverConverges(MethodSpec("rks/b3lyp")), water, None, metal_atoms=[])
    assert np.isfinite(state.energy)


def test_max_candidates_caps_the_enumeration(fe2_base):
    assert len(d_occupation_patterns(fe2_base, metal_atoms=[0], max_candidates=2)) == 2


class RecordingExecutor(SerialExecutor):
    def __init__(self, backend):
        super().__init__(backend)
        self.batches = []

    def map(self, tasks):
        self.batches.append(list(tasks))
        return super().map(tasks)


class FailUntilAtomGuess(PySCFBackend):
    """Every SCF reports failure until a scratch SCF with the 'atom' initial guess has run."""

    ok = False

    def run(self, mol, dm0=None, occ_hook=None, init_guess=None, **kw):
        if init_guess == "atom":
            type(self).ok = True
        out = super().run(mol, dm0=dm0, occ_hook=occ_hook, init_guess=init_guess, **kw)
        return out if type(self).ok else dataclasses.replace(out, converged=False)


def test_full_search_without_a_metal_returns_the_aufbau_state(water):
    be = PySCFBackend(MethodSpec("rks/b3lyp"))
    res = full_search(SerialExecutor(be), water, [], metal_atoms=[], keep_window=2e-3, tol=Tolerances())
    (state,) = res.states
    assert state.energy == pytest.approx(be.run(water, with_gradient=False).energy, abs=1e-7)
    assert (res.level1, res.level2, res.guesses) == (1, 0, ("minao",))


def test_full_search_enumerates_from_bases_at_another_geometry(water, water_step):
    be = PySCFBackend(MethodSpec("rks/b3lyp"))
    moved = water.set_geom_(water.atom_coords() + water_step, unit="Bohr", inplace=False)
    old = Frame.from_outcome(water, be.run(water, with_gradient=False))
    ex = RecordingExecutor(be)
    res = full_search(ex, moved, [old], metal_atoms=[], keep_window=2e-3, tol=Tolerances())
    (first, *rest) = ex.batches
    assert first[0].imom_ref is None and first[1].imom_ref is not None  # scratch + one enumeration task, one batch
    assert first[1].mol is moved and first[1].imom_ref.mol is moved
    assert res.states[0].energy == pytest.approx(be.run(moved, with_gradient=False).energy, abs=1e-7)


def test_full_search_retries_initial_guesses_only_when_asked(water):
    FailUntilAtomGuess.ok = False
    none = full_search(SerialExecutor(FailUntilAtomGuess(MethodSpec("rks/b3lyp"))), water, [], [], 2e-3, Tolerances())
    assert none.states == [] and none.guesses == ("minao",)
    FailUntilAtomGuess.ok = False
    res = full_search(SerialExecutor(FailUntilAtomGuess(MethodSpec("rks/b3lyp"))), water, [], [], 2e-3, Tolerances(),
                      retry_guesses=True)
    assert res.guesses == ("minao", "atom")
    assert len(res.states) == 1


def test_a_guess_retry_does_not_rerun_the_enumeration_from_the_bases(water, water_step):
    # regression: every retried guess re-ran the identical enumeration from the bases
    moved = water.set_geom_(water.atom_coords() + water_step, unit="Bohr", inplace=False)
    old = Frame.from_outcome(water, PySCFBackend(MethodSpec("rks/b3lyp")).run(water, with_gradient=False))
    FailUntilAtomGuess.ok = False
    ex = RecordingExecutor(FailUntilAtomGuess(MethodSpec("rks/b3lyp")))
    res = full_search(ex, moved, [old], [], 2e-3, Tolerances(), retry_guesses=True)
    from_base = [t for b in ex.batches for t in b
                 if t.imom_ref is not None and np.allclose(t.imom_ref.mo_coeff, old.mo_coeff)]
    assert len(from_base) == 1
    assert res.guesses == ("minao", "atom") and len(res.states) == 1


def test_enumeration_tasks_respect_include_own(fe2_base):
    with_own = enumeration_tasks(fe2_base.mol, fe2_base, [0], include_own=True, max_cycle=7)
    without = enumeration_tasks(fe2_base.mol, fe2_base, [0], include_own=False)
    assert len(with_own) == len(without) + 1 == len(d_occupation_patterns(fe2_base, [0]))
    assert {t.max_cycle for t in with_own} == {7}


def test_full_search_second_level_starts_from_every_low_state(fe2_distorted, fe2_backend, fe2_states):
    ex = RecordingExecutor(fe2_backend)
    res = full_search(ex, fe2_distorted, [], [0], keep_window=5e-3, tol=Tolerances())
    assert res.states[0].energy <= fe2_states[0].energy + 1e-6
    assert res.seeds and max(f.energy for f in res.seeds) - min(f.energy for f in res.seeds) <= 5e-3
    level2 = ex.batches[-1]  # one task per (seed, pattern other than the seed's own)
    assert len(level2) == sum(len(d_occupation_patterns(f, [0])) - 1 for f in res.seeds)


def test_merged_guesses_run_every_scratch_and_the_bases(water, water_step):
    # spec 2026-10-05 §7.1 (E1): minao alone found state B at traj2 step 0 3 / 4 times, the merged guesses 4 / 4
    be = PySCFBackend(MethodSpec("rks/b3lyp"))
    moved = water.set_geom_(water.atom_coords() + water_step, unit="Bohr", inplace=False)
    old = Frame.from_outcome(water, be.run(water, with_gradient=False))
    ex = RecordingExecutor(be)
    res = full_search(ex, moved, [old], metal_atoms=[], keep_window=2e-3, tol=Tolerances(), merge_guesses=True)
    first, second = ex.batches[0], ex.batches[1]
    assert [t.init_guess for t in first[:3]] == [None, "atom", "huckel"] and all(t.imom_ref is None for t in first[:3])
    assert len(first) == 4 and np.allclose(first[3].imom_ref.mo_coeff, old.mo_coeff)  # + the base's enumeration
    assert len(second) == 3 and all(t.imom_ref is not None and t.mol is moved for t in second)  # one per scratch
    assert res.guesses == ("minao", "atom", "huckel")
    assert len(res.states) == 1  # one state on water, found many times
    assert res.states[0].energy == pytest.approx(be.run(moved, with_gradient=False).energy, abs=1e-7)


class EnumerationNeverConverges(PySCFBackend):
    def run(self, mol, dm0=None, occ_hook=None, **kw):
        out = super().run(mol, dm0=dm0, occ_hook=occ_hook, **kw)
        return dataclasses.replace(out, converged=False) if occ_hook is not None else out


def test_unconverged_enumerations_inside_the_window_are_pending_not_absent(water, water_step):
    # spec 2026-10-05 §5.3: an NC candidate within keep_window is "unknown", reported as pending
    moved = water.set_geom_(water.atom_coords() + water_step, unit="Bohr", inplace=False)
    old = Frame.from_outcome(water, PySCFBackend(MethodSpec("rks/b3lyp")).run(water, with_gradient=False))
    ex = SerialExecutor(EnumerationNeverConverges(MethodSpec("rks/b3lyp")))
    res = full_search(ex, moved, [old], metal_atoms=[], keep_window=2e-3, tol=Tolerances(), merge_guesses=True)
    assert len(res.states) == 1 and res.pending == 4  # the base's enumeration + one per scratch, all NC
    # their last frames come back too, at this geometry, so the beam can continue them (design 10-11 §5.2)
    assert len(res.pending_frames) == 4 and all(f.mol is moved for f in res.pending_frames)
    clean = full_search(SerialExecutor(PySCFBackend(MethodSpec("rks/b3lyp"))), moved, [old], [], 2e-3, Tolerances(),
                        merge_guesses=True)
    assert clean.pending == 0 and clean.pending_frames == ()


def test_search_tasks_are_tagged_by_source(water, water_step):
    # search-performance design 2026-10-11 §4: the ledger tells scratch, level-1 and level-2 tasks apart
    be = PySCFBackend(MethodSpec("rks/b3lyp"))
    moved = water.set_geom_(water.atom_coords() + water_step, unit="Bohr", inplace=False)
    old = Frame.from_outcome(water, be.run(water, with_gradient=False))
    ex = RecordingExecutor(be)
    full_search(ex, moved, [old], metal_atoms=[], keep_window=2e-3, tol=Tolerances(), merge_guesses=True)
    tags = [[t.tag for t in b] for b in ex.batches]
    assert tags[0] == ["scratch:minao", "scratch:atom", "scratch:huckel", "l1:base"]
    assert tags[1] == ["l1:scratch"] * 3
    assert all(t == "l2" for b in tags[2:] for t in b)
    ex = RecordingExecutor(be)
    full_search(ex, moved, [old], metal_atoms=[], keep_window=2e-3, tol=Tolerances())
    assert [t.tag for t in ex.batches[0]] == ["scratch:minao", "l1:base"]


class Scripted:
    """Answers each task by tag with (energy, converged); records the tag batches."""

    def __init__(self, script):
        self.script, self.batches = script, []

    def map(self, tasks):
        self.batches.append([t.tag for t in tasks])
        return [SimpleNamespace(converged=self.script[t.tag][1], frame=t.imom_ref,
                                outcome=SimpleNamespace(energy=self.script[t.tag][0], dm=None)) for t in tasks]


def _tasks(water):
    ref = object()  # any IMOM reference: the screen only looks at energies
    return [SCFTask(mol=water, imom_ref=ref, tag=t) for t in ("near", "far", "done")] + \
           [SCFTask(mol=water, tag="scratch")]


def test_the_screen_drops_only_far_unconverged_candidates(water):
    ex = Scripted({"near:s1": (-0.995, False), "far:s1": (-0.9929, False), "done:s1": (-0.999, True),
                   "scratch": (-0.9, False), "near:s2": (-1.0, True)})
    out, keep = _screened(ex, _tasks(water), anchor=-1.0, keep_window=2e-3, screen=(4, 5e-3))
    assert ex.batches == [["near:s1", "far:s1", "done:s1", "scratch"], ["near:s2"]]
    assert keep == [True, False, True, True]
    assert out[0].converged and out[0].outcome.energy == -1.0 and not out[1].converged


def test_the_screen_is_off_without_an_anchor(water):
    for anchor, screen in ((None, (4, 5e-3)), (-1.0, None)):
        ex = Scripted({"near": (-1.0, True), "far": (-1.0, True), "done": (-1.0, True), "scratch": (-1.0, True)})
        out, keep = _screened(ex, _tasks(water), anchor=anchor, keep_window=2e-3, screen=screen)
        assert ex.batches == [["near", "far", "done", "scratch"]] and all(keep)


def test_screened_full_search_finds_the_same_states(water, water_step):
    be = PySCFBackend(MethodSpec("rks/b3lyp"))
    moved = water.set_geom_(water.atom_coords() + water_step, unit="Bohr", inplace=False)
    old = Frame.from_outcome(water, be.run(water, with_gradient=False))
    plain = full_search(SerialExecutor(be), moved, [old], [], 2e-3, Tolerances(), merge_guesses=True)
    ex = RecordingExecutor(be)
    screened = full_search(ex, moved, [old], [], 2e-3, Tolerances(), merge_guesses=True,
                           anchor=plain.states[0].energy, screen=(4, 5e-3))
    assert [t.tag for t in ex.batches[0]] == ["scratch:minao", "scratch:atom", "scratch:huckel", "l1:base:s1"]
    assert screened.screened == 0
    assert [f.energy for f in screened.states] == pytest.approx([f.energy for f in plain.states], abs=1e-6)


def test_dropped_candidates_are_not_pending(water):
    # a dropped candidate's short segment can sit within keep_window of a lowest state that is above the anchor
    ex = Scripted({"near:s1": (-0.995, False), "far:s1": (-0.9929, False), "done:s1": (-0.999, True),
                   "scratch": (-0.9, False), "near:s2": (-1.0, False)})
    out, keep = _screened(ex, _tasks(water), anchor=-1.0, keep_window=2e-3, screen=(4, 5e-3))
    from sroqm.search import _pending
    assert len(_pending(out, keep, lowest=-0.9935, keep_window=2e-3)) == 1  # "near" (NC, kept) only, not "far"


class Inflating(SerialExecutor):
    """Real SCFs, but every unconverged short segment reads 1 Ha higher: the screen must drop all of them."""

    def map(self, tasks):
        out = super().map(tasks)
        return [dataclasses.replace(r, outcome=dataclasses.replace(r.outcome, energy=r.outcome.energy + 1.0))
                if t.tag.endswith(":s1") and not r.converged else r for t, r in zip(tasks, out)]


def test_full_search_drops_far_short_segments_and_does_not_count_them_pending(water, water_step):
    be = PySCFBackend(MethodSpec("rks/b3lyp", conv_tol=1e-12))  # tight: no IMOM task converges in 4 cycles
    moved = water.set_geom_(water.atom_coords() + water_step, unit="Bohr", inplace=False)
    old = Frame.from_outcome(water, be.run(water, with_gradient=False))
    anchor = Frame.from_outcome(moved, be.run(moved, with_gradient=False)).energy
    sr = full_search(Inflating(be), moved, [old], [], 2e-3, Tolerances(conv_tol=1e-12), merge_guesses=True,
                     anchor=anchor, screen=(4, 5e-3))
    assert sr.screened >= 1 and sr.pending == 0
    assert len(sr.screened_de) == sr.screened and all(de > 2e-3 + 5e-3 for _, de in sr.screened_de)
    assert all(isinstance(tag, str) and tag.startswith("l") for tag, _ in sr.screened_de)
    assert sr.states and abs(sr.states[0].energy - anchor) < 1e-6  # the unscreened scratch SCFs still find it


class NothingConverges(PySCFBackend):
    def run(self, mol, dm0=None, occ_hook=None, **kw):
        return dataclasses.replace(super().run(mol, dm0=dm0, occ_hook=occ_hook, **kw), converged=False)


def test_an_empty_search_still_hands_on_its_pending_candidates(water, water_step):
    # review 2026-10-23 #2: with nothing converged the search returned early and dropped the in-window NC candidates,
    # although the beam had a converged anchor at this geometry
    moved = water.set_geom_(water.atom_coords() + water_step, unit="Bohr", inplace=False)
    old = Frame.from_outcome(water, PySCFBackend(MethodSpec("rks/b3lyp")).run(water, with_gradient=False))
    anchor = PySCFBackend(MethodSpec("rks/b3lyp")).run(moved, with_gradient=False).energy
    ex = SerialExecutor(NothingConverges(MethodSpec("rks/b3lyp", max_cycle=50)))
    res = full_search(ex, moved, [old], [], 2e-3, Tolerances(), merge_guesses=True, anchor=anchor)
    assert res.states == [] and res.pending >= 1 and len(res.pending_frames) == res.pending
    assert all(f.mol is moved for f in res.pending_frames)
    blind = full_search(ex, moved, [old], [], 2e-3, Tolerances(), merge_guesses=True, anchor=None)
    assert blind.pending == 0  # no known energy: no window to be "within"


def test_the_search_records_what_its_dedupe_merged_and_keeps_the_window(water, water_step):
    # design memo 2026-10-23 2a: every candidate dropped as a duplicate, with sigma and dE; every converged candidate
    # within keep_window, before the dedupe
    moved = water.set_geom_(water.atom_coords() + water_step, unit="Bohr", inplace=False)
    old = Frame.from_outcome(water, PySCFBackend(MethodSpec("rks/b3lyp")).run(water, with_gradient=False))
    res = full_search(SerialExecutor(PySCFBackend(MethodSpec("rks/b3lyp"))), moved, [old], [], 2e-3, Tolerances(),
                      merge_guesses=True)
    assert len(res.states) == 1 and res.merged  # three scratch guesses and the base all land on the ground state
    assert all(stage in ("level1", "level2") and sv >= 0.9 and abs(de) < 1e-6 for stage, sv, de in res.merged)
    assert len(res.window_frames) == len(res.merged) + 1 and res.states[0] in res.window_frames


@pytest.fixture(scope="module")
def fe_ion():
    """A bare high-spin Fe2+ (d6) in STO-3G: a metal d shell with one minority-spin d electron, in seconds."""
    from pyscf import gto
    mol = gto.M(atom="Fe 0 0 0", charge=2, spin=4, basis="sto-3g", verbose=0)
    return Frame.from_outcome(mol, PySCFBackend(MethodSpec("uhf")).run(mol, with_gradient=False))


def test_orientation_blocks_take_the_minority_d_electron_and_its_two_lowest_empty_d(fe_ion):
    from sroqm.search import orientation_blocks
    alpha, beta = orientation_blocks(fe_ion, [0])
    assert alpha == []  # five alpha d electrons: nothing to orient
    occ = [i for i in beta if fe_ion.mo_occ[1][i] > 0]
    assert len(beta) == 3 and len(occ) == 1 and occ[0] < min(i for i in beta if i not in occ)
    assert orientation_blocks(fe_ion, []) == []


def test_random_orientations_keep_the_electron_count_and_the_block_subspace(fe_ion):
    from sroqm.search import orientation_blocks, random_orientation_tasks
    blk = orientation_blocks(fe_ion, [0])[1]
    tasks = random_orientation_tasks(fe_ion.mol, fe_ion, [0], 4, np.random.default_rng(0))
    assert len(tasks) == 4 and all(t.tag == "random" and t.imom_ref is not None for t in tasks)
    S = fe_ion.ovlp
    p_block = fe_ion.mo_coeff[1][:, blk] @ fe_ion.mo_coeff[1][:, blk].T
    hole_dirs = []
    for t in tasks:
        g = t.imom_ref
        C = g.mo_coeff[1]
        np.testing.assert_allclose(C.T @ S @ C, np.eye(C.shape[1]), atol=1e-8)  # still orthonormal
        assert np.einsum("ij,ji->", g.dm[1], S) == pytest.approx(fe_ion.mol.nelec[1], abs=1e-8)
        np.testing.assert_allclose(g.mo_coeff[0], fe_ion.mo_coeff[0])           # alpha untouched
        d = C[:, [i for i in blk if g.mo_occ[1][i] > 0]][:, 0]
        np.testing.assert_allclose(p_block @ S @ d, d, atol=1e-8)                # stays in the block's span
        hole_dirs.append(d)
    assert abs(hole_dirs[0] @ S @ hole_dirs[1]) < 0.999  # different directions
    assert random_orientation_tasks(fe_ion.mol, fe_ion, [], 4, np.random.default_rng(0)) == []
