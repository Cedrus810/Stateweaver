"""bonding.py (Stateweaver design §5, V0 checks §9): Mayer bond orders and Mulliken populations per Frame."""
from types import SimpleNamespace

import numpy as np
import pytest
from pyscf import gto, scf

from sroqm.bonding import PairSpec, mayer_bond_orders, mulliken, observe, observe_beam
from sroqm.frame import Frame
from sroqm.testsystems import WATER_DIMER

DIMER = [(s, tuple(float(v) for v in xyz)) for s, *xyz in (l.split() for l in WATER_DIMER.strip().splitlines())]
PAIRS = [PairSpec("hbond", (3, 2)), PairSpec("oo", (0, 3)), PairSpec("oh", (0, 1))]


def converged(mf) -> Frame:
    mf.kernel()
    assert mf.converged
    return Frame.from_orbitals(mf.mol, mf.mo_coeff, mf.mo_occ, mf.e_tot)


def mol(atoms, basis="6-31g", spin=0):
    return gto.M(atom=atoms, basis=basis, spin=spin, verbose=0)


def explicit_mayer(frame, i, j):
    """Element-by-element sum, independent of the vectorised one."""
    dm = np.asarray(frame.dm)
    P, W = (dm, np.zeros_like(dm)) if dm.ndim == 2 else (dm[0] + dm[1], dm[0] - dm[1])
    PS, WS = P @ frame.ovlp, W @ frame.ovlp
    s = frame.mol.aoslice_by_atom()
    return sum(PS[m, n] * PS[n, m] + WS[m, n] * WS[n, m] for m in range(s[i][2], s[i][3]) for n in range(s[j][2], s[j][3]))


def test_textbook_closed_shell_bond_orders():
    p = [PairSpec("x", (0, 1))]
    assert mayer_bond_orders(converged(scf.RHF(mol("H 0 0 0; H 0 0 0.74", "sto-3g"))), p)["x"] == pytest.approx(1.0)
    assert mayer_bond_orders(converged(scf.RHF(mol("N 0 0 0; N 0 0 1.10", "sto-3g"))), p)["x"] == pytest.approx(3.0)
    # a closed-shell UHF has W = 0: the same as RHF
    assert mayer_bond_orders(converged(scf.UHF(mol("H 0 0 0; H 0 0 0.74", "sto-3g"))), p)["x"] == pytest.approx(1.0)


def test_open_shell_order_matches_an_explicit_sum_and_spins_add_up(oxygen):
    f = converged(scf.UHF(oxygen))
    assert mayer_bond_orders(f, [PairSpec("x", (0, 1))])["x"] == pytest.approx(explicit_mayer(f, 0, 1), rel=1e-12)
    q, m = mulliken(f)
    assert q.sum() == pytest.approx(oxygen.charge, abs=1e-8) and m.sum() == pytest.approx(oxygen.spin, abs=1e-8)


def test_orders_do_not_depend_on_atom_order_or_rigid_motion():
    base = mayer_bond_orders(converged(scf.RHF(mol(DIMER))), PAIRS)
    perm = [5, 3, 1, 0, 4, 2]  # new position k holds old atom perm[k]
    where = {old: new for new, old in enumerate(perm)}
    shuffled = converged(scf.RHF(mol([DIMER[i] for i in perm])))
    moved = mayer_bond_orders(shuffled, [PairSpec(p.pair_id, (where[p.atoms[0]], where[p.atoms[1]])) for p in PAIRS])
    th = 0.7
    R = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])
    turned = [(s, tuple(R @ np.array(x) + [1.0, -2.0, 0.5])) for s, x in DIMER]
    rotated = mayer_bond_orders(converged(scf.RHF(mol(turned))), PAIRS)
    for k in base:
        assert moved[k] == pytest.approx(base[k], abs=1e-6) and rotated[k] == pytest.approx(base[k], abs=1e-6)


def test_a_separated_fragment_has_no_bond_order():
    near = mayer_bond_orders(converged(scf.RHF(mol(DIMER))), PAIRS)
    far = [(s, (x[0] + (10.0 if i >= 3 else 0.0), x[1], x[2])) for i, (s, x) in enumerate(DIMER)]
    apart = mayer_bond_orders(converged(scf.RHF(mol(far))), PAIRS)
    assert near["hbond"] > 0.03 and abs(apart["hbond"]) < 1e-6 and abs(apart["oo"]) < 1e-6
    assert apart["oh"] == pytest.approx(near["oh"], abs=0.05)  # the intramolecular bond stays


def test_observe_reports_distance_charges_spins_and_flags_a_negative_order(oxygen, monkeypatch):
    f = converged(scf.UHF(oxygen))
    (o,) = observe(f, [PairSpec("oo", (0, 1), "O-O")], lineage_id=3, step=7)
    assert o.distance_angstrom == pytest.approx(np.linalg.norm(oxygen.atom_coords(unit="Angstrom")[1]
                                                               - oxygen.atom_coords(unit="Angstrom")[0]))
    assert (o.lineage_id, o.step, o.population_scheme, o.quality) == (3, 7, "mulliken", ())
    assert o.spins[0] + o.spins[1] == pytest.approx(2.0, abs=1e-6)
    monkeypatch.setattr("sroqm.bonding.mayer_bond_orders", lambda frame, pairs: {"oo": -0.01})
    (o,) = observe(f, [PairSpec("oo", (0, 1))])
    assert o.mayer_order == -0.01 and o.quality == ("negative_mayer",)  # kept raw, only flagged


def test_observe_beam_covers_every_active_lineage(oxygen):
    f = converged(scf.UHF(oxygen))
    obs = observe_beam(SimpleNamespace(frames={2: f, 5: f}), [PairSpec("oo", (0, 1))], step=1)
    assert sorted(obs) == [2, 5] and obs[5][0].lineage_id == 5 and obs[5][0].step == 1
