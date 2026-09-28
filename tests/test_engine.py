import dataclasses

import numpy as np
import pytest

from helpers import ethylene, geometry_series
from sroqm.alarms import AlarmThresholds
from sroqm.backend import MethodSpec, PySCFBackend
from sroqm.engine import Engine, Status
from sroqm.extrapolate import GrassmannGuess, ScratchGuess
from sroqm.frame import Frame


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


def test_imom_drift_onto_an_excited_state_is_flagged():
    # twisting through 90 degrees swaps pi and pi*; IMOM keeps following the old pi and ends on (pi*)^2
    eng = Engine(ethylene(0), rks())
    results = [eng.compute(ethylene(th).atom_coords()) for th in (0, 30, 60, 85, 95, 120, 150, 180)]
    flips = [i for i, r in enumerate(results) if "gap_sign_flip" in {a.name for a in r.alarms}]
    assert flips, [(r.frontier_gap, r.status.value) for r in results]
    assert results[flips[0]].frontier_gap < 0 < results[flips[0] - 1].frontier_gap
    assert results[flips[0]].status is Status.ALARM


def test_history_is_cut_after_a_root_switch():
    eng = Engine(ethylene(0), rks(), continuation="aufbau")
    for th in (0, 30, 60, 85, 95):
        eng.compute(ethylene(th).atom_coords())
    assert len(eng.lineage.history) == 5
    res = eng.compute(ethylene(120).atom_coords())  # aufbau swaps pi/pi* here
    assert "overlap_drop" in {a.name for a in res.alarms}
    assert len(eng.lineage.history) == 1  # frames of the old state must not feed the next extrapolation


class IMOMNeverConverges(PySCFBackend):
    """Real SCF, but reports failure whenever an occupation hook (IMOM) is installed.

    Stands in for IMOM pinned to a state that stopped being an SCF minimum, which on
    [Fe(H2O)6]2+ hit max_cycle on 4 of 10 MD steps; too expensive for a unit test.
    """

    def run(self, mol, dm0=None, occ_hook=None, **kw):
        out = super().run(mol, dm0=dm0, occ_hook=occ_hook, **kw)
        return dataclasses.replace(out, converged=False) if occ_hook is not None else out


def test_unconverged_imom_falls_back_to_aufbau(water, water_step):
    eng = Engine(water, IMOMNeverConverges(MethodSpec("rks/b3lyp", conv_tol=1e-10)))
    eng.compute(water.atom_coords())
    res = eng.compute(water.atom_coords() + water_step)
    names = {a.name for a in res.alarms}
    assert res.converged
    assert "imom_fallback" in names
    assert "scf_not_converged" not in names


def test_probe_finds_the_ground_state_below_an_excited_lineage(water, excited_water):
    # stands in for [Fe(H2O)6]2+, where continuation sat 1.75 mHa above another t2g occupation
    # with smooth overlap and no gap flip: only an independent SCF can see such a state
    eng = Engine(water, PySCFBackend(MethodSpec("uks/b3lyp")), probe_every=1,
                 initial_state=(excited_water.mo_coeff, excited_water.mo_occ))
    res = eng.compute(water.atom_coords())
    (probe_alarm,) = [a for a in res.alarms if a.name == "lower_state_found"]
    assert probe_alarm.value > 0.2
    assert res.probe_energy == pytest.approx(res.energy - probe_alarm.value)


def test_probe_runs_on_schedule_and_is_quiet_on_the_ground_state(water, water_step):
    eng = Engine(water, rks(), probe_every=2)
    results = [eng.compute(x) for x in geometry_series(water, water_step, 3)]
    assert [r.probe_energy is not None for r in results] == [True, False, True]
    assert all("lower_state_found" not in {a.name for a in r.alarms} for r in results)
    assert results[2].probe_energy == pytest.approx(results[2].energy, abs=1e-7)


def test_probe_every_must_be_positive(water):
    with pytest.raises(ValueError, match="probe_every"):
        Engine(water, rks(), probe_every=0)


def test_engine_can_be_seeded_with_a_frame_at_another_geometry(water, water_step):
    # a beam spawns a lineage from a state found at the current geometry, not at the template's
    be = rks()
    mol = at(water, water.atom_coords() + water_step)
    seed = Frame.from_outcome(mol, be.run(mol, with_gradient=False))
    res = Engine(water, be, initial_state=seed).compute(mol.atom_coords())
    assert res.scf_iterations <= 2
    assert res.energy == pytest.approx(seed.energy, abs=1e-8)


def test_first_step_without_history_does_not_judge_scf_cycles(water, water_step):
    eng = Engine(water, rks(), thresholds=AlarmThresholds(max_scf_cycles=1))
    first = eng.compute(water.atom_coords())
    second = eng.compute(water.atom_coords() + water_step)
    assert "scf_cycles" not in {a.name for a in first.alarms}
    assert "scf_cycles" in {a.name for a in second.alarms}


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
