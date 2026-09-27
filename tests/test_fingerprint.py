import numpy as np
import pytest

from sroqm.backend import MethodSpec, PySCFBackend
from sroqm.fingerprint import Fingerprint, compare, compute_fingerprint, default_fragments, frontier_gap


@pytest.fixture(scope="module")
def water_mf(water):
    return PySCFBackend(MethodSpec("rks/b3lyp")).run(water, with_gradient=False).mf


@pytest.fixture(scope="module")
def oxygen_mf(oxygen):
    return PySCFBackend(MethodSpec("uks/b3lyp")).run(oxygen, with_gradient=False).mf


def test_default_fragments_are_one_per_atom(water):
    assert default_fragments(water) == {"O0": [0], "H1": [1], "H2": [2]}


def test_closed_shell_fingerprint_has_no_spin(water_mf):
    fp = compute_fingerprint(water_mf)
    assert fp.s2 == 0.0
    assert all(v == 0.0 for v in fp.fragment_spins.values())
    assert sum(fp.fragment_charges.values()) == pytest.approx(0.0, abs=1e-6)


def test_custom_fragments_sum_their_atoms(water_mf):
    per_atom = compute_fingerprint(water_mf)
    grouped = compute_fingerprint(water_mf, {"O": [0], "H2": [1, 2]})
    assert grouped.fragment_charges["H2"] == pytest.approx(
        per_atom.fragment_charges["H1"] + per_atom.fragment_charges["H2"])


def test_triplet_oxygen_fingerprint(oxygen_mf):
    fp = compute_fingerprint(oxygen_mf)
    assert fp.s2 == pytest.approx(2.0, abs=0.05)
    assert sum(fp.fragment_spins.values()) == pytest.approx(2.0, abs=1e-6)
    assert fp.fragment_spins["O0"] == pytest.approx(1.0, abs=0.05)


@pytest.mark.parametrize("frags", [{"X": [5]}, {"A": [0, 1], "B": [1]}])
def test_invalid_fragments_raise(water_mf, frags):
    with pytest.raises(ValueError):
        compute_fingerprint(water_mf, frags)


def test_compare_reports_absolute_changes():
    a = Fingerprint(2.0, {"Fe": 3.0, "O": 1.0}, {"Fe": 0.5, "O": -0.5})
    b = Fingerprint(2.3, {"Fe": 2.5, "O": 1.1}, {"Fe": 0.6, "O": -0.5})
    ch = compare(a, b)
    assert ch.ds2 == pytest.approx(0.3)
    assert ch.max_dspin == pytest.approx(0.5)
    assert ch.max_dcharge == pytest.approx(0.1)
    with pytest.raises(ValueError):
        compare(a, Fingerprint(2.0, {"Fe": 3.0}, {"Fe": 0.5}))


def test_frontier_gap_cases():
    assert frontier_gap(np.array([-1.0, -0.5, 0.2, 0.6]), np.array([2, 2, 0, 0])) == pytest.approx(0.7)
    e = np.array([[-1.0, -0.5, 0.0], [-1.0, -0.3, 0.0]])
    occ = np.array([[1, 1, 0], [1, 0, 0]])
    assert frontier_gap(e, occ) == pytest.approx(0.5)            # alpha 0.5, beta 0.7
    assert frontier_gap(np.array([-1.0, -0.5, 0.2]), np.array([1, 0, 1])) == pytest.approx(-0.7)  # non-aufbau
    assert frontier_gap(np.array([-1.0]), np.array([2])) == float("inf")
