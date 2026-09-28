import pytest

from helpers import ethylene
from sroqm.backend import MethodSpec, PySCFBackend
from sroqm.beam import BeamEngine, _Member
from sroqm.engine import Engine
from sroqm.lineage import LineageStatus

TWIST = (0, 30, 60, 85, 95, 120, 150, 180)  # pi and pi* swap between 95 and 120 degrees


def rks():
    return PySCFBackend(MethodSpec("rks/b3lyp", conv_tol=1e-10))


def run_twist(**kw):
    beam = BeamEngine(ethylene(0), rks(), **kw)
    return beam, [beam.compute(ethylene(th).atom_coords()) for th in TWIST]


@pytest.fixture(scope="module")
def scheduled_twist():
    return run_twist(search_every=1, n_prune=2)


@pytest.fixture(scope="module")
def fe2_first_step(fe2_distorted):
    beam = BeamEngine(fe2_distorted, PySCFBackend(MethodSpec("uks/b3lyp", conv_tol=1e-7, density_fit=True)),
                      metal_atoms=[0])
    return beam, beam.compute(fe2_distorted.atom_coords())


def test_first_step_spawns_a_lineage_per_low_t2g_state(fe2_first_step):
    beam, res = fe2_first_step
    assert len(res.lineage_energies) == 3
    assert [kind for kind, _ in res.events] == ["spawn"] * 3
    assert res.status == "MULTISTATE"
    assert res.energy == pytest.approx(min(res.lineage_energies.values()))
    assert res.lineage_id == min(res.lineage_energies, key=res.lineage_energies.get)


def test_scheduled_search_brings_the_beam_back_to_the_ground_state(scheduled_twist):
    _, results = scheduled_twist
    assert results[-1].energy == pytest.approx(results[0].energy, abs=1e-6)  # 180 deg is the 0 deg molecule
    (switch,) = [i for i, r in enumerate(results) if r.switched]
    assert ("spawn", results[switch].lineage_id) in results[switch].events


def test_excited_lineage_goes_dormant_after_the_crossing(scheduled_twist):
    _, results = scheduled_twist
    assert sorted(results[-1].lineage_status.values()) == ["active", "dormant"]
    assert any(kind == "dormant" for r in results for kind, _ in r.events)


def test_alarm_alone_triggers_the_search():
    # no schedule: the gap_sign_flip at the crossing must start the search by itself
    _, results = run_twist(search_every=None)
    assert results[-1].energy == pytest.approx(results[0].energy, abs=1e-6)


def test_follow_policy_stays_on_the_first_lineage():
    _, results = run_twist(search_every=1, policy="follow")
    assert results[-1].energy - results[0].energy > 0.4  # stays on the diabatic (pi*)^2 state
    assert not any(r.switched for r in results)
    assert len({r.lineage_id for r in results}) == 1


def test_full_beam_replaces_its_highest_lineage_with_a_lower_state():
    # with room for one lineage, the ground state found after the crossing must still take over
    _, results = run_twist(search_every=1, max_lineages=1)
    assert results[-1].energy == pytest.approx(results[0].energy, abs=1e-6)
    assert any(("dormant", 0) in r.events for r in results)


def test_duplicate_lineages_are_merged(water, water_step):
    # a lineage that loses its state (e.g. after imom_fallback) can land on another lineage's state
    backend = PySCFBackend(MethodSpec("rks/b3lyp", conv_tol=1e-10))
    beam = BeamEngine(water, backend)
    beam.compute(water.atom_coords())
    twin = Engine(water, backend, initial_state=beam.members[0].engine.lineage.last_frame)
    twin.lineage.id = 1
    beam.members.append(_Member(id=1, engine=twin))
    res = beam.compute(water.atom_coords() + water_step)
    assert ("merge", 1) in res.events
    assert res.lineage_status == {0: "active", 1: "pruned"}


def test_revive_respects_max_lineages(water, excited_water, monkeypatch):
    # search returns the active state and a dormant lineage's state; a full beam must not grow past its cap
    backend = PySCFBackend(MethodSpec("uks/b3lyp"))
    beam = BeamEngine(water, backend, max_lineages=1, keep_window=0.5, prune_window=1.0, search_every=1)
    x = water.atom_coords()
    beam.compute(x)
    excited = Engine(water, backend, initial_state=excited_water)
    excited.lineage.id = 1
    excited.compute(x)
    beam.members.append(_Member(id=1, engine=excited, status=LineageStatus.DORMANT))
    found = [beam.members[0].engine.lineage.last_frame, excited.lineage.last_frame]
    monkeypatch.setattr("sroqm.beam.search_states", lambda *a, **k: found)
    res = beam.compute(x)
    assert res.lineage_status == {0: "active", 1: "dormant"}
    assert not res.events


@pytest.mark.parametrize("kw", [dict(policy="magic"), dict(max_lineages=0), dict(n_prune=0),
                                dict(keep_window=1e-2, prune_window=5e-3), dict(search_every=0)])
def test_beam_validates_inputs(kw):
    with pytest.raises(ValueError):
        BeamEngine(ethylene(0), rks(), **kw)
