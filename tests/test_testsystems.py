import numpy as np
import pytest

from sroqm.testsystems import DEFAULT_METHOD, build, fe_hexaaqua_atoms, fe_pentaaqua_ammine_atoms


def test_fe_hexaaqua_geometry():
    atoms = fe_hexaaqua_atoms()
    xyz = {i: np.array(p) for i, (_, p) in enumerate(atoms)}
    sym = [s for s, _ in atoms]
    assert sym.count("Fe") == 1 and sym.count("O") == 6 and sym.count("H") == 12
    o_idx = [i for i, s in enumerate(sym) if s == "O"]
    for i in o_idx:
        assert np.linalg.norm(xyz[i]) == pytest.approx(2.12)
        assert np.linalg.norm(xyz[i + 1] - xyz[i]) == pytest.approx(0.96)
        assert np.linalg.norm(xyz[i + 2] - xyz[i]) == pytest.approx(0.96)
    h = np.array([xyz[i] for i, s in enumerate(sym) if s == "H"])
    hh = np.linalg.norm(h[:, None] - h[None], axis=-1)[np.triu_indices(12, 1)]
    assert hh.min() > 1.4  # no H-H clashes (intra-water H-H is ~1.52 A)


def test_fe_pentaaqua_ammine_geometry():
    # the +x water of fe_hexaaqua swapped for NH3; the other five waters unchanged
    atoms = fe_pentaaqua_ammine_atoms()
    sym = [s for s, _ in atoms]
    xyz = np.array([p for _, p in atoms])
    assert sym[:5] == ["Fe", "N", "H", "H", "H"] and sym.count("O") == 5 and sym.count("H") == 13
    assert np.allclose(xyz[1], [2.20, 0, 0])
    for h in xyz[2:5]:
        assert np.linalg.norm(h - xyz[1]) == pytest.approx(1.015)
        cos = np.dot(xyz[0] - xyz[1], h - xyz[1]) / (np.linalg.norm(xyz[0] - xyz[1]) * np.linalg.norm(h - xyz[1]))
        assert np.degrees(np.arccos(cos)) == pytest.approx(111.0)
    assert [p for s, p in fe_hexaaqua_atoms() if s != "Fe"][3:] == [p for _, p in atoms[5:]]
    h = xyz[[i for i, s in enumerate(sym) if s == "H"]]
    assert np.linalg.norm(h[:, None] - h[None], axis=-1)[np.triu_indices(len(h), 1)].min() > 1.4


@pytest.mark.parametrize("name, natm, charge, spin", [
    ("water", 3, 0, 0), ("water_dimer", 6, 0, 0), ("fe2_hexaaqua", 19, 2, 4), ("fe3_hexaaqua", 19, 3, 5),
    ("fe2_pentaaqua_ammine", 20, 2, 4)])
def test_build(name, natm, charge, spin):
    mol = build(name, "sto-3g")
    assert (mol.natm, mol.charge, mol.spin) == (natm, charge, spin)
    assert name in DEFAULT_METHOD


def test_fe3_hexaaqua_uses_the_shorter_fe_o_bond():
    mol = build("fe3_hexaaqua", "sto-3g")
    fe_o = np.linalg.norm(mol.atom_coords(unit="Angstrom")[1] - mol.atom_coords(unit="Angstrom")[0])
    assert fe_o == pytest.approx(2.00)


def test_unknown_system_raises():
    with pytest.raises(ValueError, match="unknown system"):
        build("benzene", "sto-3g")
