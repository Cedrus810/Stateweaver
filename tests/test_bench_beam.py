import json

import numpy as np
import pytest

from bench_beam import judge_step, load_reference, main, replay
from sroqm.backend import MethodSpec, PySCFBackend
from sroqm.beam import BeamEngine
from sroqm.beamrules import BeamParams
from sroqm.frame import Frame
from sroqm.tolerances import Tolerances


def rks():
    return PySCFBackend(MethodSpec("rks/b3lyp", conv_tol=1e-10))


@pytest.fixture(scope="module")
def ground(water):
    return Frame.from_outcome(water, rks().run(water, with_gradient=False))


def test_judge_same_state_passes_whatever_the_energy(ground):
    v = judge_step(ground, False, Frame.from_orbitals(ground.mol, ground.mo_coeff, ground.mo_occ, ground.energy - 1.0),
                   Tolerances())
    assert v.passed and v.reason == "same-state" and v.sigma_min > 0.999


def test_judge_other_state(water, excited_water):
    uks = PySCFBackend(MethodSpec("uks/b3lyp"))
    g = Frame.from_outcome(water, uks.run(water, with_gradient=False))
    t = Tolerances(accept_de=5e-5, deg_band=1.0)
    assert judge_step(excited_water, False, g, t).reason == "miss"
    assert judge_step(excited_water, True, g, t).reason == "degenerate-band"
    assert judge_step(g, False, excited_water, t).reason == "energy"  # selected is lower than the reference


def test_replay_and_cli_on_a_short_water_trajectory(tmp_path, water, water_step):
    coords = [water.atom_coords() + k * water_step for k in range(3)]
    be = rks()
    for k, x in enumerate(coords):
        mol = water.set_geom_(x, unit="Bohr", inplace=False)
        f = Frame.from_outcome(mol, be.run(mol, with_gradient=False))
        np.savez(tmp_path / f"w_{k:03d}.npz", energies=np.array([f.energy]), mo_coeff=f.mo_coeff, mo_occ=f.mo_occ,
                 coords=x)
    assert load_reference(tmp_path / "w_001.npz", water).energy == pytest.approx(
        np.load(tmp_path / "w_001.npz")["energies"][0])
    beam = BeamEngine(water, be, params=BeamParams(search_every=None), tol=Tolerances(conv_tol=1e-10))
    rows = replay(beam, coords, tmp_path, "w", Tolerances(conv_tol=1e-10))
    assert [r["passed"] for r in rows] == [True] * 3
    np.save(tmp_path / "coords.npy", np.array(coords))
    out = tmp_path / "result.json"
    code = main(["--coords", str(tmp_path / "coords.npy"), "--ref", str(tmp_path), "--prefix", "w",
                 "--out", str(out), "--system", "water", "--basis", "6-31g", "--method", "rks/b3lyp",
                 "--conv-tol", "1e-10", "--metal-atoms", ""])
    assert code == 0
    assert json.loads(out.read_text())["summary"]["miss"] == 0
    summary = json.loads(out.read_text())["summary"]
    assert summary["warn"] == 0 and summary["warned"] == []
    assert all("advance" in r["trace"] for r in json.loads(out.read_text())["rows"])
    for r in json.loads(out.read_text())["rows"]:
        (lin,) = r["lineages"]  # one state on water
        assert lin["sv_ref"] > 0.999 and abs(lin["de_ref_mha"]) < 1e-3 and r["e_ref"] < 0
    assert json.loads(out.read_text())["params"]["densify_steps"] == 0
    assert all(r["peak_rss_mb"] > 0 and r["rss_mb"] > 0 for r in json.loads(out.read_text())["rows"])


def test_an_energy_verdict_above_the_noise_is_a_warning():
    from bench_beam import StepVerdict, is_warn
    assert is_warn(StepVerdict(True, "energy", 0.55, 0.045))       # traj2 step 5 of the runs that had lost B
    assert not is_warn(StepVerdict(True, "energy", 0.50, 0.0000))  # degenerate orientation, same energy
    assert not is_warn(StepVerdict(True, "same-state", 1.0, 0.045))


def test_cli_passes_the_discovery_switches_and_records_the_params(tmp_path, water, monkeypatch):
    import bench_beam
    seen = {}

    class Stop(Exception):
        pass

    def spy(*args, **kw):
        seen.update(kw)
        raise Stop

    monkeypatch.setattr(bench_beam, "BeamEngine", spy)
    np.save(tmp_path / "coords.npy", np.array([water.atom_coords()]))
    with pytest.raises(Stop):
        main(["--coords", str(tmp_path / "coords.npy"), "--prefix", "w", "--out", str(tmp_path / "r.json"),
              "--system", "water", "--basis", "6-31g", "--method", "rks/b3lyp", "--metal-atoms", "",
              "--densify-steps", "4", "--escalate-unknown", "2", "--screen-margin-mha", "5"])
    p = seen["params"]
    assert (p.densify_steps, p.escalate_unknown, p.screen_margin) == (4, 2, pytest.approx(5e-3))
