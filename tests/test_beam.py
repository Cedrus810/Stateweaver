"""BeamEngine rules (Phase-3b spec §4-§6). Water ground and MOM-excited states stand in for two lineages:
they are 0.3 Ha apart, so the windows are widened accordingly; discovery is scripted by monkeypatching
full_search / refresh in sroqm.beam, as the Phase-3a tests did with search_states."""
import dataclasses

import numpy as np
import pytest

from sroqm.backend import MethodSpec, PySCFBackend
from sroqm.beam import BeamEngine, _Member
from sroqm.beamrules import BeamParams
from sroqm.engine import Engine
from sroqm.executor import SerialExecutor
from sroqm.frame import Frame
from sroqm.lineage import LineageStatus
from sroqm.schedule import RefreshResult
from sroqm.search import SearchResult
from sroqm.tolerances import Tolerances

TOL = Tolerances(conv_tol=1e-9)
WIDE = dict(keep_window=0.5, prune_window=1.0, near_gap=1e-3, search_every=None)


def uks():
    return PySCFBackend(MethodSpec("uks/b3lyp", conv_tol=1e-9))


def beam(water, executor=None, policy="adiabatic_min", tol=TOL, **kw):
    return BeamEngine(water, uks(), policy=policy, params=BeamParams(**{**WIDE, **kw}), tol=tol, executor=executor)


def at(mol, f):
    return Frame.from_orbitals(mol, f.mo_coeff, f.mo_occ, f.energy, f.dm)


@pytest.fixture(scope="module")
def ground(water):
    return Frame.from_outcome(water, uks().run(water, with_gradient=False))


@pytest.fixture(scope="module")
def ground_outcome(water):
    return uks().run(water, with_gradient=False)


class Discovery:
    def __init__(self, monkeypatch, search=(), refresh=()):
        self.search, self.refresh = list(search), list(refresh)
        self.search_calls, self.refresh_calls = [], []
        monkeypatch.setattr("sroqm.beam.full_search", self._search)
        monkeypatch.setattr("sroqm.beam.refresh", self._refresh)

    def _search(self, executor, mol, bases, metal_atoms, keep_window, tol, **kw):
        self.search_calls.append(dict(bases=list(bases), **kw))
        states = self.search.pop(0) if self.search else []
        return SearchResult([at(mol, f) for f in states], len(states), 0, ("minao",), ())

    def _refresh(self, executor, mol, base, patterns, cap, **kw):
        self.refresh_calls.append(dict(cap=cap, **kw))
        sibs = self.refresh.pop(0) if self.refresh else []
        return RefreshResult([at(mol, f) for f in sibs], 0, ())


class Sabotage(SerialExecutor):
    """Runs every task, then replaces the result of tasks IMOM-pinned to `target` (object identity) with how(r)."""

    def __init__(self, backend, how, once=False):
        super().__init__(backend)
        self.how, self.once, self.target, self.hits = how, once, None, 0
        self.tasks = []

    def map(self, tasks):
        self.tasks += tasks
        out = super().map(tasks)
        for i, t in enumerate(tasks):
            if self.target is not None and t.imom_ref is self.target and not (self.once and self.hits):
                self.hits += 1
                out[i] = self.how(out[i])
        return out


def unconverged(r):
    return dataclasses.replace(r, outcome=dataclasses.replace(r.outcome, converged=False))


def kinds(res):
    return [k for k, _ in res.events]


def test_first_step_admits_the_searched_states_and_selects_the_lowest(water, ground, excited_water, monkeypatch):
    Discovery(monkeypatch, search=[[excited_water, ground]])
    res = beam(water).compute(water.atom_coords())
    assert kinds(res).count("spawn") == 2
    assert res.energy == pytest.approx(ground.energy, abs=1e-8)
    assert TOL.same_state(res.frame, ground)
    assert [i for i, _ in res.manifold][0] == res.lineage_id and res.manifold[0][1] == 0.0
    assert res.manifold[1][1] == pytest.approx(excited_water.energy - ground.energy, abs=1e-6)
    assert res.status == "MULTISTATE"
    assert res.forces.shape == (3, 3)


def test_admission_rejects_a_state_that_does_not_reconverge(water, ground, excited_water, monkeypatch):
    fake = dataclasses.replace(excited_water, energy=excited_water.energy - 0.01)
    Discovery(monkeypatch, search=[[ground, fake]])
    res = beam(water).compute(water.atom_coords())
    assert kinds(res).count("spawn") == 1 and ("unstable", None) in res.events


def test_only_the_selected_lineage_gets_forces_and_a_full_budget(water, ground, excited_water, monkeypatch):
    Discovery(monkeypatch, search=[[ground, excited_water]])
    ex = Sabotage(uks(), how=lambda r: r)
    b = beam(water, executor=ex, near_gap=1.0)  # the excited lineage counts as near the lowest
    b.compute(water.atom_coords())
    ex.tasks.clear()
    b.compute(water.atom_coords())
    advance = ex.tasks[:2]
    sel = [t for t in advance if t.with_gradient]
    other = [t for t in advance if not t.with_gradient]
    assert len(sel) == 1 and sel[0].max_cycle is None and sel[0].allow_fallback
    assert other[0].max_cycle == 70 and not other[0].allow_fallback
    b2 = beam(water, executor=ex, near_gap=1e-3)
    Discovery(monkeypatch, search=[[ground, excited_water]])
    b2.compute(water.atom_coords())
    ex.tasks.clear()
    b2.compute(water.atom_coords())
    assert [t.max_cycle for t in ex.tasks[:2] if not t.with_gradient] == [30]


@pytest.mark.parametrize("how", ["unconverged", "fallback", "foreign"])
def test_each_kind_of_loss_is_detected(water, ground, excited_water, ground_outcome, monkeypatch, how):
    # the revive attempt runs in the same step; only "unconverged" also breaks it, so only then it stays dormant
    breakers = {"unconverged": unconverged,
                "fallback": lambda r: dataclasses.replace(r, fallback=True),
                "foreign": lambda r: dataclasses.replace(r, outcome=ground_outcome)}
    Discovery(monkeypatch, search=[[ground, excited_water]])
    ex = Sabotage(uks(), breakers[how])
    b = beam(water, executor=ex, revive_steps=0)
    b.compute(water.atom_coords())
    excited = next(m for m in b.members if m.id != b._selected)
    ex.target = excited.frame
    res = b.compute(water.atom_coords())
    assert ("lost", excited.id) in res.events
    if how == "unconverged":
        assert res.lineage_status[excited.id] == "dormant"


def test_a_lost_lineage_is_revived_with_its_id(water, ground, excited_water, monkeypatch):
    Discovery(monkeypatch, search=[[ground, excited_water]])
    ex = Sabotage(uks(), unconverged, once=True)  # only the advance is broken; the revive attempt succeeds
    b = beam(water, executor=ex)
    b.compute(water.atom_coords())
    excited = next(m for m in b.members if m.id != b._selected)
    ex.target = excited.frame
    res = b.compute(water.atom_coords())
    assert ("lost", excited.id) in res.events and ("revive", excited.id) in res.events
    assert res.lineage_status[excited.id] == "active"


def test_full_beam_evicts_its_highest_lineage_even_if_it_was_selected(water, ground, excited_water, monkeypatch):
    # regression: the first v5 draft protected the selected lineage and silently dropped the lowest state
    Discovery(monkeypatch, search=[[excited_water], [ground]])
    b = beam(water, max_lineages=1, search_every=1)
    b.compute(water.atom_coords())
    res = b.compute(water.atom_coords())
    assert ("evict", 0) in res.events and ("switch", 1) in res.events
    assert res.energy == pytest.approx(ground.energy, abs=1e-8)


def test_reseated_gradient_matches_a_full_step(water, ground, excited_water, monkeypatch):
    Discovery(monkeypatch, search=[[excited_water], [ground]])
    b = beam(water, max_lineages=1, search_every=1)
    b.compute(water.atom_coords())
    res = b.compute(water.atom_coords())  # selection moved to a freshly admitted lineage: forces by re-seat
    ref = Engine(water, uks()).compute(water.atom_coords())
    assert np.allclose(res.forces, ref.forces, atol=1e-5)


def test_gradient_reseat_retries_with_the_full_budget(water, ground, excited_water, monkeypatch):
    class ShortReseatFails(SerialExecutor):
        def map(self, tasks):
            out = super().map(tasks)
            return [unconverged(r) if t.with_gradient and t.max_cycle == 10 else r for t, r in zip(tasks, out)]

    Discovery(monkeypatch, search=[[excited_water], [ground]])
    b = beam(water, executor=ShortReseatFails(uks()), max_lineages=1, search_every=1)
    b.compute(water.atom_coords())
    res = b.compute(water.atom_coords())
    assert res.forces is not None and np.all(np.isfinite(res.forces))


def test_follow_protects_the_followed_lineage(water, ground, excited_water, monkeypatch):
    Discovery(monkeypatch, search=[[excited_water], [ground]])
    b = beam(water, policy="follow", max_lineages=1, search_every=1)
    b.compute(water.atom_coords())
    res = b.compute(water.atom_coords())
    assert ("full", None) in res.events and res.lineage_id == 0 and not res.switched
    assert res.energy == pytest.approx(excited_water.energy, abs=1e-8)


def test_follow_falls_back_when_the_followed_lineage_is_lost(water, ground, excited_water, monkeypatch):
    Discovery(monkeypatch, search=[[ground, excited_water]])
    ex = Sabotage(uks(), unconverged)  # persistent: the revive attempt fails too
    b = beam(water, executor=ex, policy="follow")
    first = b.compute(water.atom_coords())
    ex.target = b._member(first.lineage_id).frame
    res = b.compute(water.atom_coords())
    assert ("follow_lost", first.lineage_id) in res.events
    assert res.lineage_id != first.lineage_id and res.lineage_status[res.lineage_id] == "active"


def test_lineages_on_one_state_are_merged_keeping_the_lower(water, ground, monkeypatch):
    Discovery(monkeypatch, search=[[ground]])
    b = beam(water)
    b.compute(water.atom_coords())
    twin = Engine(water, uks(), initial_state=ground)
    twin.lineage.id = 1
    b.members.append(_Member(id=1, engine=twin, seed=ground))
    b._next_id = 2
    res = b.compute(water.atom_coords())
    assert ("merge", 1) in res.events or ("merge", 0) in res.events
    assert sorted(res.lineage_status.values()) == ["active", "pruned"]


def test_prune_after_n_prune_steps_above_the_window(water, ground, excited_water, monkeypatch):
    Discovery(monkeypatch, search=[[ground]])
    b = beam(water, keep_window=0.1, prune_window=0.2, n_prune=2)
    b.compute(water.atom_coords())
    eng = Engine(water, uks(), initial_state=excited_water)
    eng.lineage.id = 1
    b.members.append(_Member(id=1, engine=eng, seed=excited_water))
    b._next_id = 2
    first = b.compute(water.atom_coords())
    second = b.compute(water.atom_coords())
    assert ("prune", 1) not in first.events and ("prune", 1) in second.events
    assert second.lineage_status[1] == "dormant"


def test_selected_loss_searches_from_its_seed_and_revives_by_matching(water, ground, excited_water, monkeypatch):
    disc = Discovery(monkeypatch, search=[[ground, excited_water], [ground]])
    ex = Sabotage(uks(), unconverged)  # persistent: only the search candidate can bring the state back
    b = beam(water, executor=ex)
    first = b.compute(water.atom_coords())
    seed = b._member(first.lineage_id).frame
    ex.target = seed
    res = b.compute(water.atom_coords())
    assert any(base is seed for base in disc.search_calls[1]["bases"])
    assert len(disc.search_calls[1]["bases"]) == 2  # the lowest remaining lineage + the lost seed
    assert ("revive", first.lineage_id) in res.events and res.lineage_id == first.lineage_id


def test_an_empty_beam_admits_the_best_candidate_by_force(water, ground, excited_water, monkeypatch):
    fake = dataclasses.replace(excited_water, energy=excited_water.energy - 0.01)
    disc = Discovery(monkeypatch, search=[[ground], [fake]])
    ex = Sabotage(uks(), unconverged)
    b = beam(water, executor=ex)
    b.compute(water.atom_coords())
    ex.target = b.members[0].frame
    res = b.compute(water.atom_coords())
    assert disc.search_calls[1]["retry_guesses"] is True
    assert ("unstable", None) in res.events and ("forced", 1) in res.events
    assert res.energy == pytest.approx(excited_water.energy, abs=1e-6)


def test_an_empty_search_keeps_the_current_lineages(water, ground, monkeypatch):
    Discovery(monkeypatch, search=[[ground], []])
    b = beam(water, search_every=1)
    b.compute(water.atom_coords())
    res = b.compute(water.atom_coords())
    assert ("search_empty", None) in res.events and res.lineage_status == {0: "active"}


def test_no_state_at_all_raises(water, monkeypatch):
    Discovery(monkeypatch, search=[[]])
    with pytest.raises(RuntimeError, match="no converged electronic state"):
        beam(water).compute(water.atom_coords())


def test_degenerate_status(water, ground, excited_water, monkeypatch):
    Discovery(monkeypatch, search=[[ground, excited_water]])
    res = beam(water, tol=Tolerances(conv_tol=1e-9, deg_band=0.5)).compute(water.atom_coords())
    assert res.status == "DEGENERATE"


def test_refresh_runs_on_schedule_with_the_scheduler_cap(water, ground, excited_water, monkeypatch):
    disc = Discovery(monkeypatch, search=[[ground]], refresh=[[excited_water]])
    b = beam(water)
    b.compute(water.atom_coords())                # step 0: search only
    res = b.compute(water.atom_coords())          # step 1: first refresh; one lineage -> cap_near
    assert disc.refresh_calls[0]["cap"] == 70
    assert kinds(res).count("spawn") == 1         # the refreshed sibling was admitted
    b.compute(water.atom_coords())                # step 2: refresh again (the beam changed at step 1)
    assert disc.refresh_calls[1]["cap"] == 30     # two lineages, 0.3 Ha apart, nothing unknown


def test_point_charges_reach_every_scf(water, ground, excited_water, monkeypatch):
    pc = (np.array([[0.0, 0.0, 6.0]]), np.array([-0.5]))
    disc = Discovery(monkeypatch, search=[[excited_water], [ground]])
    ex = Sabotage(uks(), how=lambda r: r)
    b = beam(water, executor=ex, max_lineages=1, search_every=1)
    b.compute(water.atom_coords(), point_charges=pc)
    b.compute(water.atom_coords(), point_charges=pc)   # advance, admission, eviction, re-seat gradient
    assert ex.tasks and all(t.point_charges is pc for t in ex.tasks)
    assert all(c["point_charges"] is pc for c in disc.search_calls + disc.refresh_calls)


def test_tolerances_are_the_single_source(water):
    with pytest.raises(ValueError, match="conv_tol"):
        BeamEngine(water, PySCFBackend(MethodSpec("uks/b3lyp", conv_tol=1e-7)), tol=Tolerances(conv_tol=1e-9))


def test_beam_validates_inputs(water):
    with pytest.raises(ValueError, match="policy"):
        BeamEngine(water, uks(), policy="magic", tol=TOL)
    with pytest.raises(ValueError, match="shape"):
        BeamEngine(water, uks(), tol=TOL).compute(np.zeros((2, 3)))


def test_a_failed_admission_does_not_use_up_the_next_candidates_chance(water, ground, excited_water, monkeypatch):
    # regression: candidates were cut to max_lineages before admission, so a failure blocked the ones behind it
    fake = dataclasses.replace(excited_water, energy=ground.energy - 0.01)  # sorts first, cannot reconverge
    Discovery(monkeypatch, search=[[fake, ground]])
    res = beam(water, max_lineages=1).compute(water.atom_coords())
    assert ("unstable", None) in res.events and "forced" not in kinds(res)
    assert kinds(res).count("spawn") == 1
    assert res.energy == pytest.approx(ground.energy, abs=1e-8)


def test_a_candidate_that_cannot_win_a_slot_is_not_converged(water, ground, excited_water, monkeypatch):
    Discovery(monkeypatch, search=[[ground], [excited_water]])
    ex = Sabotage(uks(), how=lambda r: r)
    b = beam(water, executor=ex, max_lineages=1, search_every=1)
    b.compute(water.atom_coords())
    ex.tasks.clear()
    res = b.compute(water.atom_coords())
    assert ("full", None) in res.events
    assert len(ex.tasks) == 1  # only the advance of the one lineage


def test_result_ids_and_steps_match_the_beam(water, ground, excited_water, monkeypatch):
    # regression: a lineage's first Result carried lineage_id 0 and counted steps from its own birth
    Discovery(monkeypatch, search=[[ground]], refresh=[[excited_water]])
    b = beam(water)
    b.compute(water.atom_coords())
    for k in (1, 2):
        res = b.compute(water.atom_coords())
        assert len(res.results) == 2
        assert all(r.lineage_id == i and r.step == k for i, r in res.results.items())


@pytest.mark.parametrize("how", ["energy", "state"])
def test_a_reseat_on_another_state_is_refused(water, ground, excited_water, monkeypatch, how):
    from sroqm.imom import make_imom_hook

    other = uks().run(water, dm0=excited_water.dm, occ_hook=make_imom_hook(excited_water, water))
    assert other.converged

    class WrongReseat(SerialExecutor):
        on = False

        def map(self, tasks):
            out = super().map(tasks)
            for i, t in enumerate(tasks):
                if self.on and t.with_gradient and t.imom_ref is not None and t.dm0 is t.imom_ref.dm:  # a re-seat
                    o = out[i].outcome
                    out[i] = dataclasses.replace(out[i], outcome=(
                        dataclasses.replace(o, energy=o.energy + 1e-3) if how == "energy" else other))
            return out

    Discovery(monkeypatch, search=[[excited_water], [ground]])
    ex = WrongReseat(uks())
    b = beam(water, executor=ex, max_lineages=1, search_every=1)
    b.compute(water.atom_coords())  # step 0 re-seats its new lineage too: sabotage only step 1's
    ex.on = True
    with pytest.raises(RuntimeError, match="re-seat"):
        b.compute(water.atom_coords())


@pytest.mark.parametrize("density_fit,embedded", [(False, False), (True, True)])
def test_default_serial_executor_reuses_the_admitted_scf_for_forces(water, monkeypatch, density_fit, embedded):
    be = PySCFBackend(MethodSpec("uks/b3lyp", conv_tol=TOL.conv_tol, density_fit=density_fit))
    pc = (np.array([[0., 0., 6.]]), np.array([-.5])) if embedded else None
    found = Frame.from_outcome(water, be.run(water, point_charges=pc, with_gradient=False))
    Discovery(monkeypatch, search=[[found]])
    b = BeamEngine(water, be, params=BeamParams(**WIDE), tol=TOL)
    run, gradient = be.run, be.gradient
    calls, grad_calls = [], []

    def counted(*args, **kwargs):
        calls.append(kwargs)
        return run(*args, **kwargs)

    def counted_gradient(outcome):
        grad_calls.append(outcome)
        return gradient(outcome)

    monkeypatch.setattr(be, "run", counted)
    monkeypatch.setattr(be, "gradient", counted_gradient)
    result = b.compute(water.atom_coords(), point_charges=pc)
    assert len(calls) == 1 and calls[0]["with_gradient"] is False  # admission, no gradient re-seat SCF
    assert len(grad_calls) == 1 and grad_calls[0].converged
    reference = run(water, point_charges=pc)
    np.testing.assert_allclose(result.forces, -reference.gradient, atol=1e-6)
    np.testing.assert_array_equal(result.results[result.lineage_id].forces, result.forces)
    assert all(m.scf_result is None for m in b.members)


def test_a_valid_reseat_updates_energy_frame_and_history_once(water, ground, monkeypatch):
    class RefineEnergy(SerialExecutor):
        reseated = None

        def map(self, tasks):
            out = super().map(tasks)
            for i, task in enumerate(tasks):
                if task.with_gradient and task.max_cycle == 10:
                    refined = dataclasses.replace(out[i].outcome, energy=out[i].outcome.energy + 1e-6)
                    out[i] = dataclasses.replace(out[i], outcome=refined)
                    self.reseated = out[i]
            return out

    Discovery(monkeypatch, search=[[ground]])
    ex = RefineEnergy(uks())
    b = beam(water, executor=ex)
    result = b.compute(water.atom_coords())
    engine = b.members[0].engine
    assert result.energy == ex.reseated.outcome.energy == result.frame.energy
    assert result.lineage_energies[result.lineage_id] == result.energy
    np.testing.assert_array_equal(result.results[result.lineage_id].forces, result.forces)
    assert engine.lineage.energies == [result.energy]
    assert len(engine.lineage.frames) == len(engine.lineage.fingerprints) == len(engine.lineage.gaps) == 1
    assert engine.step == b.step == 1 and result.results[result.lineage_id].step == 0


def test_reseat_energy_refinement_reselects_the_lowest(water, ground, excited_water, monkeypatch):
    near = dataclasses.replace(excited_water, energy=ground.energy + 1e-6)
    Discovery(monkeypatch, search=[[ground, near]])

    class NearCrossing(SerialExecutor):
        gradients = 0

        def map(self, tasks):
            out = super().map(tasks)
            for i, task in enumerate(tasks):
                if task.imom_ref is not None:
                    energy = task.imom_ref.energy
                    if task.with_gradient:
                        self.gradients += 1
                        if TOL.same_state(task.imom_ref, ground):
                            energy += 2e-6  # initially lowest, now above the other member, still within admit_de
                    out[i] = dataclasses.replace(out[i], outcome=dataclasses.replace(out[i].outcome, energy=energy))
            return out

    ex = NearCrossing(uks())
    b = beam(water, executor=ex)
    result = b.compute(water.atom_coords())
    assert ex.gradients == 2
    assert TOL.same_state(result.frame, excited_water)
    assert result.energy == min(result.lineage_energies.values())
    assert result.manifold[0] == (result.lineage_id, 0.)


def test_failed_reused_gradient_releases_step_resources(water, ground, monkeypatch):
    Discovery(monkeypatch, search=[[ground]])
    b = beam(water)

    def broken(*args):
        raise RuntimeError("gradient failed")

    monkeypatch.setattr(b.backend, "gradient", broken)
    with pytest.raises(RuntimeError, match="gradient failed"):
        b.compute(water.atom_coords())
    assert all(m.scf_result is None for m in b.members)


@dataclasses.dataclass(frozen=True)
class SplitTwins(Tolerances):
    """Real same_state, except that the two candidate frames tagged by `twins` (energies) count as distinct:
    a stand-in for the non-transitive case sigma(c1, X), sigma(c2, X) >= 0.9 > sigma(c1, c2)."""
    twins: tuple = ()

    def same_state(self, a, b):
        if {a.energy, b.energy} == set(self.twins):
            return False
        return super().same_state(a, b)


def test_twins_converging_to_one_state_do_not_evict_a_distinct_lineage(water, ground, excited_water, monkeypatch):
    # spec 2026-10-03 §4.1 step 4 / §12.1 #2: both twins re-converge onto the ground state in one admission batch;
    # the second must not evict the excited lineage and then be merged away
    c1 = dataclasses.replace(ground, energy=ground.energy + 2e-6)
    c2 = dataclasses.replace(ground, energy=ground.energy + 4e-6)
    tol = SplitTwins(conv_tol=1e-9, twins=(c1.energy, c2.energy))
    Discovery(monkeypatch, search=[[excited_water], [c1, c2]])
    b = beam(water, tol=tol, max_lineages=2, search_every=1)
    first = b.compute(water.atom_coords())
    res = b.compute(water.atom_coords())
    assert ("duplicate", None) in res.events
    assert ("evict", first.lineage_id) not in res.events and "merge" not in kinds(res)
    assert kinds(res).count("spawn") == 1
    assert res.lineage_status[first.lineage_id] == "active"
    assert len(res.manifold) == 2 and res.energy == pytest.approx(ground.energy, abs=1e-8)


def test_every_full_search_merges_the_initial_guesses(water, ground, monkeypatch):
    disc = Discovery(monkeypatch, search=[[ground], [ground]])
    b = beam(water, search_every=1)
    b.compute(water.atom_coords())
    b.compute(water.atom_coords())
    assert len(disc.search_calls) == 2 and all(c["merge_guesses"] is True for c in disc.search_calls)


def test_a_failed_reseat_falls_back_to_the_next_lowest_lineage(water, ground, excited_water, monkeypatch):
    # spec 2026-10-05 §6.2: diag r0 step 15 crashed when the newly selected lineage could not be re-seated
    class FailFirstReseat(SerialExecutor):
        failed = 0

        def map(self, tasks):
            out = super().map(tasks)
            for i, t in enumerate(tasks):
                if t.with_gradient and t.imom_ref is not None and t.dm0 is t.imom_ref.dm and self.failed < 2:
                    self.failed += 1  # the short re-seat and its full-budget retry
                    out[i] = unconverged(out[i])
            return out

    disc = Discovery(monkeypatch, search=[[excited_water, ground]])
    b = beam(water, executor=FailFirstReseat(uks()))
    res = b.compute(water.atom_coords())
    failed = [i for k, i in res.events if k == "reseat_failed"]
    assert len(failed) == 1 and res.lineage_status[failed[0]] == "dormant"
    assert TOL.same_state(res.frame, excited_water) and res.lineage_id != failed[0]
    assert res.forces.shape == (3, 3) and np.all(np.isfinite(res.forces))
    assert res.energy == pytest.approx(excited_water.energy, abs=1e-6)
    b.compute(water.atom_coords())
    assert len(disc.search_calls) == 2  # search_every=None: the failure itself asks for the next search


def test_trace_records_budgets_search_and_refresh(water, ground, excited_water, monkeypatch):
    Discovery(monkeypatch, search=[[ground, excited_water]], refresh=[[]])
    b = beam(water)
    first = b.compute(water.atom_coords())
    assert first.trace["search"]["n_states"] == 2 and first.trace["advance"] == []
    res = b.compute(water.atom_coords())
    adv = {a["id"]: a for a in res.trace["advance"]}
    assert adv[first.lineage_id]["max_cycle"] is None  # the selected lineage: full budget
    other = next(i for i in adv if i != first.lineage_id)
    assert adv[other]["max_cycle"] == BeamParams().cap and adv[other]["converged"] is True
    assert res.trace["search"] is None and res.trace["refresh"] == dict(log=[], unknown=0)


def test_a_search_with_pending_candidates_says_so(water, ground, monkeypatch):
    def search(executor, mol, bases, metal_atoms, keep_window, tol, **kw):
        return SearchResult([at(mol, ground)], 1, 0, ("minao",), (), pending=2)

    monkeypatch.setattr("sroqm.beam.full_search", search)
    res = beam(water).compute(water.atom_coords())
    assert ("pending", None) in res.events and res.trace["search"]["pending"] == 2


def test_every_lineage_frame_is_returned(water, ground, excited_water, monkeypatch):
    Discovery(monkeypatch, search=[[ground, excited_water]])
    res = beam(water).compute(water.atom_coords())
    assert set(res.frames) == set(res.lineage_energies) and res.frames[res.lineage_id] is res.frame


class EnergyOnlyBackend(PySCFBackend):
    """Fails on any gradient request; finalize() marks its energy so the test can see which SCF it finished."""

    def __init__(self, spec):
        super().__init__(spec)
        self.finalized = []

    def run(self, mol, dm0=None, occ_hook=None, point_charges=None, with_gradient=True, **kw):
        assert not with_gradient, "an energy_only beam asked for a gradient"
        return super().run(mol, dm0, occ_hook, point_charges, with_gradient=False, **kw)

    def gradient(self, outcome):
        raise AssertionError("an energy_only beam asked for a gradient")

    def finalize(self, outcome):
        self.finalized.append(outcome)
        return outcome.energy - 1.0, True, 3


def test_energy_only_computes_no_gradient_and_finalizes_the_selected(water, ground, excited_water, monkeypatch):
    """r2scan_jax spec 2026-10-11 §3.6: replay without forces; the selected lineage's SCF gets the backend's finish."""
    Discovery(monkeypatch, search=[[excited_water, ground]])
    be = EnergyOnlyBackend(MethodSpec("uks/b3lyp", conv_tol=1e-9))
    b = BeamEngine(water, be, params=BeamParams(**WIDE), tol=TOL, energy_only=True)
    for _ in range(2):
        res = b.compute(water.atom_coords())
        assert res.forces is None and TOL.same_state(res.frame, ground)
        assert res.energy_final == pytest.approx(res.energy - 1.0, abs=1e-12)
        assert res.trace["final"]["id"] == res.lineage_id
    assert len(be.finalized) == 2
    assert beam(water).energy_only is False


def test_a_rejected_candidate_does_not_pull_the_next_refresh(water, ground, excited_water, monkeypatch):
    # spec 2026-09-30 §5.2: only spawn/revive/lost/merge/prune/evict/switch change the beam; a candidate
    # turned away (`full`) leaves it as it was and must not force a refresh on the next step
    disc = Discovery(monkeypatch, search=[[ground], [excited_water]])
    b = beam(water, max_lineages=1)
    for _ in range(3):                              # step 0 search, steps 1-2 refresh; one lineage: gap is inf
        b.compute(water.atom_coords())
    assert len(disc.refresh_calls) == 2 and b.scheduler.next_step == 6
    b._search_next = "reseat"
    res = b.compute(water.atom_coords())            # step 3: the searched excited state finds no slot
    assert ("full", None) in res.events and b.scheduler.next_step == 6
    b.compute(water.atom_coords())                  # step 4: still no refresh
    assert len(disc.refresh_calls) == 2


def test_an_unknown_search_result_still_pulls_the_next_refresh(water, ground, monkeypatch):
    disc = Discovery(monkeypatch, search=[[ground]])
    b = beam(water)
    for _ in range(3):
        b.compute(water.atom_coords())
    assert b.scheduler.next_step == 6

    def pending_search(executor, mol, bases, metal_atoms, keep_window, tol, **kw):
        return SearchResult([at(mol, ground)], 1, 0, ("minao",), (), pending=1)

    monkeypatch.setattr("sroqm.beam.full_search", pending_search)
    b._search_next = "reseat"
    res = b.compute(water.atom_coords())            # step 3: a candidate inside the window did not converge
    assert ("pending", None) in res.events and b.scheduler.next_step == 4


def test_every_scf_of_a_step_lands_in_the_task_ledger(water, ground, excited_water, monkeypatch):
    # search-performance design 2026-10-11 §4: one ledger row per physical SCF, tagged by source
    Discovery(monkeypatch, search=[[ground]], refresh=[[excited_water]])
    b = beam(water)
    first = b.compute(water.atom_coords())
    assert [r["tag"] for r in first.trace["tasks"]] == ["admit"]
    second = b.compute(water.atom_coords())
    tags = [r["tag"] for r in second.trace["tasks"]]
    assert tags[0] == "advance" and tags.count("admit") == 1  # the refreshed sibling's admission
    row = second.trace["tasks"][0]
    assert set(row) == {"tag", "max_cycle", "imom", "cycles", "converged", "fallback", "energy", "energies", "seconds"}
    assert row["imom"] and row["converged"] and len(row["energies"]) == row["cycles"]
    assert row["energy"] == pytest.approx(second.energy, abs=1e-8) and row["seconds"]["scf"] > 0


def test_a_search_says_why_it_ran(water, ground, monkeypatch):
    Discovery(monkeypatch, search=[[ground], [ground]])
    b = beam(water, search_every=2)
    assert b.compute(water.atom_coords()).trace["search"]["reason"] == "start"
    assert b.compute(water.atom_coords()).trace["search"] is None
    assert b.compute(water.atom_coords()).trace["search"]["reason"] == "periodic"
    b._search_next = "reseat"
    assert b.compute(water.atom_coords()).trace["search"]["reason"] == "reseat"
    assert b._search_next is None


class UnknownRefresh(Discovery):
    """Every refresh leaves one candidate unconverged inside the window."""

    def _refresh(self, executor, mol, base, patterns, cap, **kw):
        self.refresh_calls.append(dict(cap=cap, **kw))
        return RefreshResult([], 1, ("nc",))


DEG = Tolerances(conv_tol=1e-9, deg_band=0.5)  # ground and excited water count as degenerate


def test_a_degenerate_start_searches_the_next_steps_from_every_lineage(water, ground, excited_water, monkeypatch):
    disc = Discovery(monkeypatch, search=[[ground, excited_water]])
    b = beam(water, tol=DEG, densify_steps=2)
    first = b.compute(water.atom_coords())
    assert first.status == "DEGENERATE" and first.trace["densify"] == dict(event="start", until=2)
    res = [b.compute(water.atom_coords()) for _ in range(3)]
    assert [r.trace["search"] and r.trace["search"]["reason"] for r in res] == ["densify", "densify", None]
    assert res[1].trace["densify"] == dict(event="end", until=2, unresolved=False)
    assert len(disc.search_calls[1]["bases"]) == 2  # every ACTIVE lineage, not only the lowest


def test_a_clear_start_or_densify_off_adds_no_searches(water, ground, excited_water, monkeypatch):
    disc = Discovery(monkeypatch, search=[[ground, excited_water]])
    b = beam(water, tol=DEG)                         # degenerate start, densify_steps = 0
    for _ in range(3):
        b.compute(water.atom_coords())
    assert len(disc.search_calls) == 1
    disc = Discovery(monkeypatch, search=[[ground, excited_water]])
    b = beam(water, densify_steps=2)                 # 0.3 Ha apart: MULTISTATE, not DEGENERATE
    for _ in range(3):
        b.compute(water.atom_coords())
    assert len(disc.search_calls) == 1


def test_densified_discovery_extends_while_candidates_stay_unknown(water, ground, excited_water, monkeypatch):
    UnknownRefresh(monkeypatch, search=[[ground, excited_water]])
    b = beam(water, tol=DEG, densify_steps=1, search_every=4)
    res = [b.compute(water.atom_coords()) for _ in range(5)]
    # step 1 is the window; unknown keeps extending it to 2 and 3; step 4 is the periodic search, so no further
    assert [r.trace["search"]["reason"] for r in res] == ["start", "densify", "densify", "densify", "periodic"]
    assert res[1].trace["densify"] == dict(event="extend", until=2)
    assert res[3].trace["densify"] == dict(event="end", until=3, unresolved=True)  # the periodic search takes over


def test_persistent_unknowns_escalate_to_a_full_search(water, ground, monkeypatch):
    disc = UnknownRefresh(monkeypatch, search=[[ground]])
    b = beam(water, escalate_unknown=2)
    res = [b.compute(water.atom_coords()) for _ in range(5)]
    # step 0 search; steps 1-2 refresh with an unknown each -> escalate after step 2; step 3 searches (and its
    # refresh is unknown again: the count restarts at 1); step 4 is the second unknown -> escalate again
    assert [("escalate", None) in r.events for r in res] == [False, False, True, False, True]
    assert res[3].trace["search"]["reason"] == "escalate" and len(disc.search_calls) == 2
    disc = UnknownRefresh(monkeypatch, search=[[ground]])
    b = beam(water)                                  # escalate_unknown = None: unknowns only pull refreshes
    for _ in range(5):
        b.compute(water.atom_coords())
    assert len(disc.search_calls) == 1


def test_every_search_inside_the_densify_window_starts_from_every_lineage(water, ground, excited_water, monkeypatch):
    disc = Discovery(monkeypatch, search=[[ground, excited_water]])
    b = beam(water, tol=DEG, densify_steps=2, search_every=1)   # every step is also a periodic search
    for _ in range(4):
        b.compute(water.atom_coords())
    assert [len(c["bases"]) for c in disc.search_calls] == [0, 2, 2, 1]  # step 3 is past the window


def test_an_escalated_search_starts_from_every_active_lineage(water, ground, excited_water, monkeypatch):
    # design §6: recovery / coverage searches start from the existing frames, not only the lowest
    disc = UnknownRefresh(monkeypatch, search=[[ground, excited_water]])
    b = beam(water, escalate_unknown=2)
    res = [b.compute(water.atom_coords()) for _ in range(4)]
    assert res[3].trace["search"]["reason"] == "escalate" and len(res[3].lineage_status) == 2
    assert all(s == "active" for s in res[3].lineage_status.values())
    assert len(disc.search_calls[0]["bases"]) == 0 and len(disc.search_calls[1]["bases"]) == 2
    assert res[3].trace["search"]["n_bases"] == 2 and res[3].trace["search"]["wide"] is True
    assert res[0].trace["search"]["n_bases"] == 0 and res[0].trace["search"]["wide"] is False


def test_a_densified_search_also_starts_from_a_revivable_dormant_seed(water, ground, excited_water, monkeypatch):
    # design §6: symmetry-breaking stage -> every ACTIVE representative plus the still-legal lost-state seeds
    disc = Discovery(monkeypatch, search=[[ground, excited_water]])
    b = beam(water, tol=DEG, densify_steps=2)
    b.compute(water.atom_coords())
    gone = b.members[1]
    gone.status, gone.lost_step = LineageStatus.DORMANT, 0  # lost last step: not "this step", still revivable
    seed = gone.seed = gone.frame  # a loss keeps the last frame as the seed
    res = b.compute(water.atom_coords())
    assert res.trace["search"]["reason"] == "densify"
    bases = disc.search_calls[1]["bases"]
    assert len(bases) == 2 and any(base is seed for base in bases)  # 1 ACTIVE frame + 1 seed
    assert res.trace["search"]["n_bases"] == 2


def test_search_gets_the_lowest_lineage_as_anchor_and_the_screen(water, ground, monkeypatch):
    disc = Discovery(monkeypatch, search=[[ground], [ground]])
    b = beam(water, search_every=1, screen_margin=5e-3)
    b.compute(water.atom_coords())
    second = b.compute(water.atom_coords())
    assert disc.search_calls[0]["anchor"] is None and disc.search_calls[0]["screen"] == (4, 5e-3)
    assert disc.search_calls[1]["anchor"] == pytest.approx(second.energy, abs=1e-8)
    assert beam(water).params.screen_margin is None
