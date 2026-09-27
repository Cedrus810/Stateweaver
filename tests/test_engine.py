import numpy as np
import pytest

from helpers import geometry_series
from sroqm.backend import MethodSpec, PySCFBackend
from sroqm.engine import Engine, Status
from sroqm.extrapolate import GrassmannGuess, ScratchGuess


def rks(**kw):
    return PySCFBackend(MethodSpec("rks/b3lyp", conv_tol=1e-10, **kw))


def at(mol, x):
    return mol.set_geom_(x, unit="Bohr", inplace=False)


def test_engine_matches_scratch_scf_along_a_trajectory(water, water_step):
    be = rks()
    eng = Engine(water, be)
    for x in geometry_series(water, water_step, 5):
        res = eng.compute(x)
        ref = be.run(at(water, x))
        assert res.energy == pytest.approx(ref.energy, abs=1e-8)
        assert np.allclose(res.forces, -ref.gradient, atol=1e-4)
        assert res.status is Status.FAST, res.alarms
    assert eng.step == 5
    assert res.overlap_min_sv > 0.99


def test_extrapolation_reduces_scf_cycles(water, water_step):
    coords = geometry_series(water, water_step, 6)

    def mean_cycles(extrapolator):
        eng = Engine(water, rks(), extrapolator=extrapolator)
        return np.mean([eng.compute(x).scf_iterations for x in coords][3:])

    assert mean_cycles(GrassmannGuess()) < mean_cycles(ScratchGuess())  # prototype: 3 vs 7


def test_imom_lineage_keeps_an_excited_state(water, water_step, excited_water):
    be = PySCFBackend(MethodSpec("uks/b3lyp"))
    eng = Engine(water, be, initial_state=(excited_water.mo_coeff, excited_water.mo_occ))
    for x in geometry_series(water, water_step, 3):
        res = eng.compute(x)
        assert res.energy - be.run(at(water, x), with_gradient=False).energy > 0.2
        assert "overlap_drop" not in {a.name for a in res.alarms}


def test_aufbau_root_switch_is_flagged(water, water_step, excited_water):
    eng = Engine(water, PySCFBackend(MethodSpec("uks/b3lyp")), continuation="aufbau",
                 initial_state=(excited_water.mo_coeff, excited_water.mo_occ))
    res = eng.compute(water.atom_coords() + water_step)
    assert res.status is Status.ALARM
    assert "overlap_drop" in {a.name for a in res.alarms}


def test_moving_point_charges_match_scratch(water, water_step):
    be = rks()
    eng = Engine(water, be)
    for k, x in enumerate(geometry_series(water, water_step, 4)):
        pc = (np.array([[6.0 + 0.05 * k, 0.0, 0.0]]), np.array([0.5]))
        res = eng.compute(x, point_charges=pc)
        assert res.energy == pytest.approx(be.run(at(water, x), point_charges=pc, with_gradient=False).energy,
                                           abs=1e-8)


def test_unconverged_scf_is_reported_not_raised(water):
    res = Engine(water, rks(max_cycle=2)).compute(water.atom_coords())
    assert not res.converged
    assert res.status is Status.ALARM
    assert "scf_not_converged" in {a.name for a in res.alarms}


def test_engine_validates_inputs(water, oxygen):
    with pytest.raises(ValueError, match="continuation"):
        Engine(water, rks(), continuation="magic")
    with pytest.raises(ValueError, match="open-shell"):
        Engine(oxygen, rks())
    with pytest.raises(ValueError, match="shape"):
        Engine(water, rks()).compute(np.zeros((2, 3)))


def test_lineage_bookkeeping(water, water_step):
    eng = Engine(water, rks(), history_length=2)
    for x in geometry_series(water, water_step, 4):
        eng.compute(x)
    assert len(eng.lineage.history) == 2
    assert len(eng.lineage.energies) == 4
    assert eng.lineage.multiplicity == 1
