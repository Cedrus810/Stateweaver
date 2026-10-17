"""Real-SCF beam runs: the Phase-3a ethylene twist (RKS, pi/pi* crossing between 95 and 120 degrees) under the
Phase-3b rules, and the first step on distorted high-spin [Fe(H2O)6]2+ (slow: a two-level search)."""
import pytest

from helpers import ethylene
from sroqm.backend import MethodSpec, PySCFBackend
from sroqm.beam import BeamEngine
from sroqm.beamrules import BeamParams
from sroqm.executor import SerialExecutor
from sroqm.search import full_search
from sroqm.tolerances import Tolerances

TWIST = (0, 30, 60, 85, 95, 120, 150, 180)
RKS_TOL = Tolerances(conv_tol=1e-10)


def rks():
    return PySCFBackend(MethodSpec("rks/b3lyp", conv_tol=1e-10))


def run_twist(policy="adiabatic_min", **kw):
    beam = BeamEngine(ethylene(0), rks(), policy=policy, params=BeamParams(**kw), tol=RKS_TOL)
    return beam, [beam.compute(ethylene(th).atom_coords()) for th in TWIST]


@pytest.fixture(scope="module")
def scheduled_twist():
    return run_twist(search_every=1)


def test_scheduled_search_brings_the_beam_back_to_the_ground_state(scheduled_twist):
    _, results = scheduled_twist
    assert results[-1].energy == pytest.approx(results[0].energy, abs=1e-6)  # 180 deg is the 0 deg molecule
    switches = [r for r in results if r.switched]
    assert switches and all(("switch", r.lineage_id) in r.events for r in switches)


def test_the_excited_lineage_leaves_the_beam_after_the_crossing(scheduled_twist):
    _, results = scheduled_twist
    assert "dormant" in results[-1].lineage_status.values()
    assert any(k in ("prune", "lost", "evict") for r in results for k, _ in r.events)


def test_alarm_alone_triggers_the_search():
    _, results = run_twist(search_every=None)
    assert results[-1].energy == pytest.approx(results[0].energy, abs=1e-6)


def test_follow_policy_stays_on_the_first_lineage():
    _, results = run_twist(search_every=1, policy="follow")
    assert results[-1].energy - results[0].energy > 0.4  # stays on the diabatic (pi*)^2 state
    assert not any(r.switched for r in results)
    assert len({r.lineage_id for r in results}) == 1


def test_one_lineage_beam_still_ends_on_the_ground_state():
    _, results = run_twist(search_every=1, max_lineages=1)
    assert results[-1].energy == pytest.approx(results[0].energy, abs=1e-6)
    assert any(k == "evict" for r in results for k, _ in r.events)


def test_first_step_on_distorted_fe2(fe2_distorted):
    be = PySCFBackend(MethodSpec("uks/b3lyp", conv_tol=1e-7, density_fit=True))
    res = BeamEngine(fe2_distorted, be, metal_atoms=[0]).compute(fe2_distorted.atom_coords())
    ref = full_search(SerialExecutor(be), fe2_distorted, [], [0], 2e-3, Tolerances())
    assert res.energy == pytest.approx(ref.states[0].energy, abs=5e-5)
    assert 1 <= len(res.manifold) <= 3 and all(de <= 2e-3 for _, de in res.manifold)
