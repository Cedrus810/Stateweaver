import dataclasses

import numpy as np
import pytest

from sroqm.backend import MethodSpec, PySCFBackend
from sroqm.executor import SCFTask, SerialExecutor, run_task
from sroqm.frame import Frame


def rks():
    return PySCFBackend(MethodSpec("rks/b3lyp", conv_tol=1e-10))


class IMOMNeverConverges(PySCFBackend):
    def run(self, mol, dm0=None, occ_hook=None, **kw):
        out = super().run(mol, dm0=dm0, occ_hook=occ_hook, **kw)
        return dataclasses.replace(out, converged=False) if occ_hook is not None else out


def test_scratch_task_matches_the_backend(water):
    be = rks()
    r = run_task(be, SCFTask(mol=water))
    assert r.converged and not r.fallback
    assert r.outcome.energy == pytest.approx(be.run(water, with_gradient=False).energy, abs=1e-9)
    assert r.outcome.gradient is None
    assert isinstance(r.frame, Frame) and r.frame.energy == r.outcome.energy


def test_imom_task_falls_back_only_when_allowed(water):
    be = IMOMNeverConverges(MethodSpec("rks/b3lyp", conv_tol=1e-10))
    ref = Frame.from_outcome(water, be.run(water, with_gradient=False))
    pinned = SCFTask(mol=water, dm0=ref.dm, imom_ref=ref)
    assert not run_task(be, pinned).converged
    r = run_task(be, dataclasses.replace(pinned, allow_fallback=True))
    assert r.converged and r.fallback
    assert r.cycles >= 2


def test_task_options_reach_the_scf(water):
    r = run_task(PySCFBackend(MethodSpec("rks/b3lyp", conv_tol=1e-12)), SCFTask(mol=water, max_cycle=1, with_gradient=True))
    assert r.cycles <= 1 and not r.converged
    assert r.outcome.gradient.shape == (3, 3)


@pytest.mark.parametrize("kw", [dict(max_cycle=0), dict(allow_fallback=True)])
def test_task_validates(water, kw):
    with pytest.raises(ValueError):
        SCFTask(mol=water, **kw)


def test_serial_executor_keeps_task_order(water, water_step):
    moved = water.set_geom_(water.atom_coords() + water_step, unit="Bohr", inplace=False)
    results = SerialExecutor(rks()).map([SCFTask(mol=water), SCFTask(mol=moved)])
    assert [r.mol for r in results] == [water, moved]
    assert results[0].outcome.energy != pytest.approx(results[1].outcome.energy, abs=1e-6)


def test_frame_access_does_not_rebuild_geometry_factors(water, monkeypatch):
    r = run_task(rks(), SCFTask(mol=water))
    original = Frame.from_outcome.__func__
    calls = []

    def counted(cls, *args):
        calls.append(args)
        return original(cls, *args)

    monkeypatch.setattr(Frame, "from_outcome", classmethod(counted))
    first = r.frame
    assert r.frame is first and len(calls) == 1


def test_occupation_guesses_reuse_geometry_but_have_independent_density(water):
    base = run_task(rks(), SCFTask(mol=water)).frame
    occ = base.mo_occ.copy()
    filled = np.flatnonzero(occ > 0)
    empty = np.flatnonzero(occ == 0)
    occ[filled[-1]], occ[empty[0]] = 0., 2.
    guess = base.with_occupations(occ)
    assert guess.mol is base.mol and guess.ovlp is base.ovlp and guess.ovlp_half is base.ovlp_half
    np.testing.assert_allclose(guess.dm, (base.mo_coeff * occ) @ base.mo_coeff.T)
    assert not np.shares_memory(guess.dm, base.dm)
    occ[:] = 0
    assert guess.mo_occ.sum() == base.mo_occ.sum() and not np.array_equal(guess.mo_occ, base.mo_occ)
    assert np.isnan(guess.energy)
    with pytest.raises(ValueError, match="occupation shape"):
        base.with_occupations(np.zeros(1))


# ---- ProcessExecutor (performance design 2026-10-03 §9) ----------------------------------------------------------

@pytest.fixture(scope="module")
def pool():
    from sroqm.executor import ProcessExecutor
    with ProcessExecutor(MethodSpec("rks/b3lyp", conv_tol=1e-10), n_workers=2, threads=1, pin=False) as ex:
        yield ex


def test_process_executor_matches_serial_in_task_order(pool, water, water_step):
    moved = water.set_geom_(water.atom_coords() + water_step, unit="Bohr", inplace=False)
    ref = run_task(rks(), SCFTask(mol=water)).frame
    tasks = [SCFTask(mol=water, tag="a"), SCFTask(mol=moved, tag="b"),
             SCFTask(mol=moved, dm0=ref.dm, imom_ref=ref, max_cycle=50, tag="c")]
    serial = SerialExecutor(rks()).map(tasks)
    par = pool.map(tasks)
    assert [r.mol for r in par] == [water, moved, moved]  # the driver's own molecules, in task order
    for s, p in zip(serial, par):
        assert p.converged == s.converged and p.outcome.energy == pytest.approx(s.outcome.energy, abs=1e-8)
        assert p.outcome.mf is None and p.handle is not None  # the SCF object stays on the worker
    assert pool.map([]) == []
    pool.release()


def test_a_molecule_crosses_to_a_worker_intact(pool, water):
    # PySCF's own pickle (JSON) flattens the empty (0, 8) _ecpbas; the atom guess then crashed on Fe (2026-10-22)
    from multiprocessing.reduction import ForkingPickler
    back = ForkingPickler.loads(ForkingPickler.dumps(water))
    for k in ("_atm", "_bas", "_env", "_ecpbas"):
        assert getattr(back, k).shape == getattr(water, k).shape
    (r,) = pool.map([SCFTask(mol=water, init_guess="atom")])
    assert r.converged and r.outcome.energy == pytest.approx(
        run_task(rks(), SCFTask(mol=water, init_guess="atom")).outcome.energy, abs=1e-8)
    pool.release()


def test_process_executor_gradient_and_fingerprint_come_from_the_kept_scf(pool, water):
    from sroqm.fingerprint import compute_fingerprint
    (p,) = pool.map([SCFTask(mol=water)])
    s = run_task(rks(), SCFTask(mol=water))
    np.testing.assert_allclose(p.handle.gradient(), rks().gradient(s.outcome), atol=1e-6)
    fp, ref = p.fingerprint(), compute_fingerprint(s.outcome.mf)
    assert fp.s2 == pytest.approx(ref.s2, abs=1e-8)
    assert fp.fragment_charges == pytest.approx(ref.fragment_charges, abs=1e-6)
    pool.release()
    with pytest.raises(RuntimeError, match="stale handle"):
        p.handle.gradient()


def test_process_executor_raises_a_worker_error_and_stays_usable(pool, water):
    with pytest.raises(RuntimeError, match="task 1"):
        pool.map([SCFTask(mol=water), SCFTask(mol=water, dm0=np.zeros((2, 2)), tag="bad")])
    (r,) = pool.map([SCFTask(mol=water)])
    assert r.converged
    pool.release()


def test_split_cpus_gives_each_worker_its_own_block():
    from sroqm.executor import split_cpus
    assert split_cpus(4, 10, range(40)) == [list(range(i, i + 10)) for i in (0, 10, 20, 30)]
    with pytest.raises(ValueError):
        split_cpus(5, 10, range(40))
