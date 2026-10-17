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


def test_run_max_cycle_overrides_the_spec(water):
    out = PySCFBackend(MethodSpec("rks/b3lyp", conv_tol=1e-12)).run(water, with_gradient=False, max_cycle=1)
    assert out.cycles <= 1
    assert not out.converged


def test_run_init_guess_is_used_without_dm0(water):
    be = PySCFBackend(MethodSpec("rks/b3lyp", conv_tol=1e-10))
    minao = be.run(water, with_gradient=False)
    atom = be.run(water, with_gradient=False, init_guess="atom")
    assert atom.converged
    assert atom.energy == pytest.approx(minao.energy, abs=1e-8)


def test_reused_gradient_rejects_unconverged_outcomes(water):
    be = PySCFBackend(MethodSpec("rks/b3lyp", conv_tol=1e-12))
    out = be.run(water, with_gradient=False, max_cycle=1)
    assert not out.converged
    with pytest.raises(RuntimeError, match="unconverged"):
        be.gradient(out)


def test_scf_objects_write_no_checkpoint_file(water):
    # nothing reads PySCF's per-SCF temporary chkfile; it was dumped every cycle of every candidate
    assert PySCFBackend(MethodSpec("rks/b3lyp")).build(water).chkfile is None
    assert PySCFBackend(MethodSpec("uks/b3lyp", density_fit=True)).build(water).chkfile is None


def test_run_records_every_cycle_energy_and_where_the_time_went(water):
    # search-performance design 2026-10-11 §4: the per-task ledger needs per-cycle energies (to calibrate the
    # short-segment screen) and setup / SCF / gradient wall times (to see what the fixed per-SCF cost is)
    out = PySCFBackend(MethodSpec("uks/b3lyp", conv_tol=1e-9, density_fit=True)).run(water)
    assert out.converged and len(out.energies) == out.cycles >= 2
    assert out.energies[-1] == pytest.approx(out.energy, abs=1e-6)
    assert set(out.seconds) == {"build", "df", "grids", "scf", "gradient"}
    assert all(t >= 0 for t in out.seconds.values()) and out.seconds["df"] > 0 and out.seconds["grids"] > 0
    nograd = PySCFBackend(MethodSpec("rks/b3lyp")).run(water, with_gradient=False)
    assert nograd.seconds["gradient"] == 0 and nograd.seconds["df"] == 0


def test_kept_outcomes_do_not_hold_df_integrals(water):
    # full_search keeps ~30 outcomes alive; each held its mf's DF tensor (116 MB for Fe2+ def2-svp), so RSS passed
    # max_memory and PySCF moved the DF integrals of the next SCFs to disk: 7-8 s/cycle instead of 0.7 (ledger 10-11)
    be = PySCFBackend(MethodSpec("uks/b3lyp", conv_tol=1e-10, density_fit=True))
    out = be.run(water)
    assert out.mf.with_df._cderi is None
    assert np.allclose(be.gradient(out), out.gradient, atol=1e-7)  # the reused gradient does not need _cderi


def test_prebuilt_df_and_grids_leave_the_energy_unchanged(water):
    ref = dft.UKS(water, xc="b3lyp").density_fit()
    ref.conv_tol = 1e-10
    ref.kernel()
    out = PySCFBackend(MethodSpec("uks/b3lyp", conv_tol=1e-10, density_fit=True)).run(water, with_gradient=False)
    assert out.converged and out.energy == pytest.approx(ref.e_tot, abs=1e-9)
