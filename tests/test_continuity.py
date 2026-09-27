import numpy as np
import pytest
from pyscf import gto

from helpers import converged_frames, geometry_series
from sroqm.backend import MethodSpec, PySCFBackend
from sroqm.frame import Frame
from sroqm.imom import make_imom_hook
from sroqm.overlap import cross_overlap, occupied_overlap


@pytest.fixture(scope="module")
def water_frames(water, water_step):
    return converged_frames(water, geometry_series(water, water_step, 2))


def test_cross_overlap_at_one_geometry_is_the_overlap_matrix(water):
    assert np.allclose(cross_overlap(water, water), water.intor("int1e_ovlp"))


def test_overlap_of_a_frame_with_itself_is_one(water_frames):
    ov = occupied_overlap(water_frames[0], water_frames[0])
    assert ov.det == pytest.approx(1.0, abs=1e-10)
    assert ov.min_sv == pytest.approx(1.0, abs=1e-10)


def test_overlap_across_one_md_step_stays_near_one(water_frames):
    ov = occupied_overlap(water_frames[0], water_frames[1])
    assert ov.min_sv > 0.99
    assert ov.det > 0.99


def test_orbital_swap_drives_min_sv_to_zero(water_frames):
    f = water_frames[0]
    occ = f.mo_occ.copy()
    homo = np.flatnonzero(occ)[-1]
    occ[homo], occ[homo + 1] = 0.0, 2.0
    ov = occupied_overlap(f, Frame.from_orbitals(f.mol, f.mo_coeff, occ))
    assert ov.min_sv < 1e-8
    assert ov.det < 1e-8


def test_overlap_rejects_restricted_vs_unrestricted(water_frames, excited_water):
    with pytest.raises(ValueError, match="restricted"):
        occupied_overlap(water_frames[0], excited_water)


def test_overlap_with_empty_beta_channel():
    h = gto.M(atom="H 0 0 0", basis="6-31g", spin=1, verbose=0)
    frames = converged_frames(h, geometry_series(h, np.array([[0.0, 0.0, 0.01]]), 2), method="uks/b3lyp")
    ov = occupied_overlap(frames[0], frames[1])
    assert ov.min_sv > 0.99


def test_imom_follows_an_excited_state_across_a_step(water, water_step, excited_water):
    mol = water.set_geom_(water.atom_coords() + water_step, unit="Bohr", inplace=False)
    be = PySCFBackend(MethodSpec("uks/b3lyp"))
    ground = be.run(mol, with_gradient=False)
    imom = be.run(mol, dm0=excited_water.dm, occ_hook=make_imom_hook(excited_water, mol), with_gradient=False)
    aufbau = be.run(mol, dm0=excited_water.dm, with_gradient=False)
    assert imom.converged
    assert imom.energy - ground.energy > 0.2
    assert abs(aufbau.energy - ground.energy) < 1e-6
    assert occupied_overlap(excited_water, Frame.from_outcome(mol, imom)).min_sv > 0.95
    assert occupied_overlap(excited_water, Frame.from_outcome(mol, aufbau)).min_sv < 0.1
