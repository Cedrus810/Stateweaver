import dataclasses

import numpy as np
import pytest

from sroqm.backend import MethodSpec, PySCFBackend
from sroqm.executor import TaskResult
from sroqm.frame import Frame
from sroqm.imom import make_imom_hook
from sroqm.schedule import RefreshScheduler, refresh


@pytest.fixture(scope="module")
def outcomes(water, excited_water):
    be = PySCFBackend(MethodSpec("uks/b3lyp"))
    ground = be.run(water, with_gradient=False)
    excited = be.run(water, dm0=excited_water.dm, occ_hook=make_imom_hook(excited_water, water), with_gradient=False)
    return ground, excited


class ScriptedExecutor:
    """Returns prepared TaskResults batch by batch and records the tasks it was given."""

    def __init__(self, batches):
        self.batches, self.seen = list(batches), []

    def map(self, tasks):
        self.seen.append(list(tasks))
        out = self.batches.pop(0)
        assert len(out) == len(tasks)
        return out


def result(mol, outcome, converged=True, energy=None):
    out = dataclasses.replace(outcome, converged=converged, energy=outcome.energy if energy is None else energy)
    return TaskResult(mol, out, out.cycles, False)


def test_refresh_sorts_screens_and_counts_unknowns(water, outcomes):
    ground, excited = outcomes
    base = Frame.from_outcome(water, ground)
    patterns = [base.mo_occ] * 4
    ex = ScriptedExecutor([
        [result(water, excited),                                   # converged in stage 1: a sibling
         result(water, ground),                                    # converged onto the base: not a sibling
         result(water, ground, False, ground.energy + 0.020),      # unconverged, 20 mHa up: screened
         result(water, ground, False, ground.energy + 0.001)],     # unconverged, 1 mHa up: goes to stage 2
        [result(water, excited, False)],                           # still unconverged: unknown
    ])
    r = refresh(ex, water, base, patterns, cap=30, screen_cycles=4, screen_window=5e-3)
    assert len(r.siblings) == 1 and r.siblings[0].energy == pytest.approx(excited.energy)
    assert r.unknown == 1
    assert {t.max_cycle for t in ex.seen[0]} == {4} and [t.max_cycle for t in ex.seen[1]] == [30]
    assert "nc" in r.log and any(e.startswith("x+20") for e in r.log)


def test_refresh_without_patterns_runs_nothing(water, outcomes):
    base = Frame.from_outcome(water, outcomes[0])
    r = refresh(ScriptedExecutor([[], []]), water, base, [], cap=30)
    assert (r.siblings, r.unknown) == ([], 0)


def test_scheduler_forces_the_next_refresh():
    s = RefreshScheduler()
    assert s.due(0)
    assert s.record(0, gap=1e-3, unknown=0, degenerate=False, beam_changed=False) == 1  # first refresh
    for kw in (dict(unknown=1), dict(degenerate=True), dict(beam_changed=True)):
        base = dict(gap=1e-3, unknown=0, degenerate=False, beam_changed=False)
        assert s.record(1, **{**base, **kw}) == 1


def test_scheduler_trend():
    s = RefreshScheduler(max_interval=4)
    s.record(0, 2.0e-3, 0, False, False)
    assert s.record(1, 1.6e-3, 0, False, False) == 2   # closing at 0.4 mHa/step: crossing in 4 steps -> refresh in 2
    s.record(3, 1.6e-3, 0, False, False)
    assert s.record(7, 2.0e-3, 0, False, False) == 4   # opening: max_interval
    assert not s.due(10) and s.due(11)


def test_scheduler_handles_a_single_lineage():
    s = RefreshScheduler(max_interval=4)
    s.record(0, np.inf, 0, False, False)
    assert s.record(1, np.inf, 0, False, False) == 4
    assert s.record(5, 1e-3, 0, False, False) == 1     # a second state appeared: closing from infinity


def test_scheduler_poke():
    s = RefreshScheduler(max_interval=4)
    s.record(0, 1e-3, 0, False, False)
    s.record(1, 2e-3, 0, False, False)
    assert s.next_step == 5
    s.poke(2)
    assert s.next_step == 3


def test_stage2_cap():
    s = RefreshScheduler(cap=30, cap_near=70, near_gap=5e-4)
    assert s.stage2_cap(gap_prev=1e-3, n_active=2) == 30
    assert s.stage2_cap(gap_prev=1e-4, n_active=2) == 70
    assert s.stage2_cap(gap_prev=1e-3, n_active=1) == 70
    s.record(0, 1e-3, unknown=2, degenerate=False, beam_changed=False)
    assert s.stage2_cap(gap_prev=1e-3, n_active=2) == 70


@pytest.mark.parametrize("kw", [dict(max_interval=0), dict(cap=0), dict(cap=80, cap_near=70), dict(near_gap=0)])
def test_scheduler_validates(kw):
    with pytest.raises(ValueError):
        RefreshScheduler(**kw)


def test_refresh_tasks_are_tagged_by_stage(water, outcomes):
    from sroqm.executor import SerialExecutor

    class Recording(SerialExecutor):
        def __init__(self, backend):
            super().__init__(backend)
            self.tags = []

        def map(self, tasks):
            self.tags.append([t.tag for t in tasks])
            return super().map(tasks)

    ground, _ = outcomes
    base = Frame.from_outcome(water, ground)
    ex = Recording(PySCFBackend(MethodSpec("uks/b3lyp", conv_tol=1e-13)))  # one screening cycle cannot converge
    refresh(ex, water, base, [base.mo_occ], cap=30, screen_cycles=1, screen_window=1.0)
    assert ex.tags == [["refresh:s1"], ["refresh:s2"]]
