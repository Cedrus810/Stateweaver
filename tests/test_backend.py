import numpy as np
import pytest
from pyscf import dft

from sroqm.backend import MethodSpec, PySCFBackend


def test_method_spec_parses_reference_and_xc():
    s = MethodSpec("uks/b3lyp")
    assert s.reference == "uks"
    assert s.xc == "b3lyp"
    assert s.unrestricted
    assert MethodSpec("rhf").xc is None


@pytest.mark.parametrize("bad", ["ccsd", "rks", "uks", "rhf/b3lyp"])
def test_method_spec_rejects_bad_strings(bad):
    with pytest.raises(ValueError):
        MethodSpec(bad)


def test_rks_energy_matches_plain_pyscf(water):
    ref = dft.RKS(water, xc="b3lyp")
    ref.conv_tol = 1e-10
    ref.kernel()
    out = PySCFBackend(MethodSpec("rks/b3lyp", conv_tol=1e-10)).run(water)
    assert out.converged
    assert out.energy == pytest.approx(ref.e_tot, abs=1e-8)
    assert out.gradient.shape == (3, 3)
    assert out.dm.shape == (water.nao, water.nao)


def test_converged_guess_needs_few_cycles(water):
    be = PySCFBackend(MethodSpec("rks/b3lyp", conv_tol=1e-10))
    first = be.run(water, with_gradient=False)
    again = be.run(water, dm0=first.dm, with_gradient=False)
    assert again.gradient is None
    assert again.cycles <= 2 < first.cycles


def test_uks_returns_spin_resolved_arrays(oxygen):
    out = PySCFBackend(MethodSpec("uks/b3lyp")).run(oxygen, with_gradient=False)
    assert out.dm.shape == (2, oxygen.nao, oxygen.nao)
    assert out.mo_coeff.shape[0] == 2
    assert out.mo_occ.shape[0] == 2


def test_point_charges_change_energy(water):
    be = PySCFBackend(MethodSpec("rks/b3lyp"))
    bare = be.run(water, with_gradient=False)
    emb = be.run(water, point_charges=(np.array([[6.0, 0.0, 0.0]]), np.array([0.5])))
    assert abs(emb.energy - bare.energy) > 1e-4
    assert emb.gradient.shape == (3, 3)


def test_open_shell_molecule_with_restricted_method_raises(oxygen):
    with pytest.raises(ValueError, match="open-shell"):
        PySCFBackend(MethodSpec("rks/b3lyp")).run(oxygen)


def test_occ_hook_runs_on_the_final_scf_object(water):
    seen = []
    be = PySCFBackend(MethodSpec("rhf"))
    be.run(water, occ_hook=lambda mf: seen.append(mf), point_charges=(np.array([[6.0, 0, 0]]), np.array([0.1])),
           with_gradient=False)
    assert len(seen) == 1
    assert "QMMM" in type(seen[0]).__name__  # hook sees the embedded object, not the bare one
