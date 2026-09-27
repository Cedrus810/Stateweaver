import numpy as np
import pytest
from pyscf import gto

from helpers import converged_frames, geometry_series, lowdin_density_error
from sroqm.extrapolate import (ASPCGuess, GrassmannGuess, ReadGuess, ScratchGuess, aspc_coefficients,
                               make_extrapolator, polynomial_coefficients)
from sroqm.frame import Frame


@pytest.fixture(scope="module")
def water_frames(water, water_step):
    return converged_frames(water, geometry_series(water, water_step, 5))


def test_frame_from_orbitals_rebuilds_density(water_frames):
    f = water_frames[0]
    g = Frame.from_orbitals(f.mol, f.mo_coeff, f.mo_occ)
    assert np.allclose(g.dm, f.dm, atol=1e-8)
    assert np.isnan(g.energy)
    assert np.allclose(f.ovlp_half @ f.ovlp_half, f.ovlp)
    assert not f.unrestricted


def test_coefficient_tables():
    assert np.allclose(aspc_coefficients(4), [2.8, -2.8, 1.2, -0.2])
    assert np.allclose(polynomial_coefficients(3), [3.0, -3.0, 1.0])
    for n in range(1, 7):
        assert aspc_coefficients(n).sum() == pytest.approx(1.0)
        assert polynomial_coefficients(n).sum() == pytest.approx(1.0)
    with pytest.raises(ValueError):
        aspc_coefficients(0)


def test_empty_history_gives_no_guess(water):
    for ex in (ScratchGuess(), ReadGuess(), ASPCGuess(), GrassmannGuess()):
        assert ex.guess((), water) is None


def test_extrapolators_beat_reading_the_previous_density(water_frames):
    history = tuple(reversed(water_frames[:4]))
    target = water_frames[4]
    e_read = lowdin_density_error(ReadGuess().guess(history, target.mol), target)
    e_aspc = lowdin_density_error(ASPCGuess(order=4).guess(history, target.mol), target)
    e_gr = lowdin_density_error(GrassmannGuess(order=3).guess(history, target.mol), target)
    assert e_aspc < 0.1 * e_read   # prototype: ~0.008x
    assert e_gr < 0.01 * e_read    # prototype: ~0.001x


def test_grassmann_guess_conserves_electrons(water_frames):
    history = tuple(reversed(water_frames[:4]))
    target = water_frames[4]
    dm = GrassmannGuess().guess(history, target.mol)
    assert np.trace(dm @ target.ovlp) == pytest.approx(10.0, abs=1e-8)


def test_grassmann_with_one_frame_reads_the_density(water_frames):
    dm = GrassmannGuess().guess((water_frames[0],), water_frames[1].mol)
    assert np.allclose(dm, water_frames[0].dm)


def test_grassmann_falls_back_to_read_on_orbital_swap(water_frames):
    f = water_frames[1]
    occ = f.mo_occ.copy()
    homo = np.flatnonzero(occ)[-1]
    occ[homo], occ[homo + 1] = 0.0, 2.0
    swapped = Frame.from_orbitals(f.mol, f.mo_coeff, occ)
    history = (water_frames[2], swapped, water_frames[0])
    dm = GrassmannGuess(order=3).guess(history, water_frames[3].mol)
    assert np.allclose(dm, water_frames[2].dm)


def test_unrestricted_guess_with_empty_beta_channel():
    h = gto.M(atom="H 0 0 0", basis="6-31g", spin=1, verbose=0)
    step = np.array([[0.0, 0.0, 0.01]])
    frames = converged_frames(h, geometry_series(h, step, 4), method="uks/b3lyp")
    dm = GrassmannGuess(order=3).guess(tuple(reversed(frames[:3])), frames[3].mol)
    assert dm.shape == (2, h.nao, h.nao)
    assert np.trace(dm[0] @ frames[3].ovlp) == pytest.approx(1.0, abs=1e-8)
    assert np.allclose(dm[1], 0.0)


def test_make_extrapolator_by_name():
    assert isinstance(make_extrapolator("scratch"), ScratchGuess)
    assert isinstance(make_extrapolator("read"), ReadGuess)
    assert isinstance(make_extrapolator("aspc"), ASPCGuess)
    assert isinstance(make_extrapolator("grassmann"), GrassmannGuess)
    with pytest.raises(ValueError, match="unknown"):
        make_extrapolator("magic")
