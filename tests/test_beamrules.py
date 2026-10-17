from types import SimpleNamespace

import pytest

from sroqm.alarms import Alarm
from sroqm.beamrules import (BeamParams, admission_slot, advance_budget, is_lost, next_strikes, revive_eligible,
                             status_of, search_reason)
from sroqm.engine import Budget

P = BeamParams()


def test_defaults_are_the_spec_values():
    assert (P.max_lineages, P.keep_window, P.prune_window, P.n_prune) == (3, 2e-3, 4e-3, 2)
    assert (P.cap, P.cap_near, P.near_gap, P.revive_steps, P.search_every) == (30, 70, 5e-4, 3, 10)
    assert (P.screen_cycles, P.screen_window, P.max_interval, P.grad_reseat_cycles) == (4, 5e-3, 4, 10)
    assert (P.densify_steps, P.escalate_unknown, P.screen_margin) == (0, None, None)


@pytest.mark.parametrize("kw", [dict(max_lineages=0), dict(n_prune=0), dict(keep_window=0),
                                dict(keep_window=5e-3, prune_window=4e-3), dict(cap=0), dict(cap=80),
                                dict(near_gap=0), dict(revive_steps=-1), dict(search_every=0),
                                dict(screen_cycles=0), dict(screen_window=0), dict(max_interval=0),
                                dict(grad_reseat_cycles=0), dict(densify_steps=-1), dict(escalate_unknown=0),
                                dict(screen_margin=-1e-3)])
def test_params_validate(kw):
    with pytest.raises(ValueError):
        BeamParams(**kw)


def test_advance_budget_by_selection_and_gap():
    assert advance_budget(True, 0.0, P) == Budget()
    assert advance_budget(False, 1e-4, P) == Budget(max_cycle=70, allow_fallback=False, with_gradient=False)
    assert advance_budget(False, 1e-3, P) == Budget(max_cycle=30, allow_fallback=False, with_gradient=False)


@pytest.mark.parametrize("converged, alarm, lost", [(False, None, True), (True, "overlap_drop", True),
                                                     (True, "imom_fallback", True), (True, "scf_cycles", False),
                                                     (True, None, False)])
def test_is_lost(converged, alarm, lost):
    alarms = (Alarm(0, alarm, 1.0, 0.0),) if alarm else ()
    assert is_lost(SimpleNamespace(converged=converged, alarms=alarms)) is lost



@pytest.mark.parametrize("converged, alarm, lost", [(False, None, True), (True, "overlap_drop", False),
                                                     (True, "imom_fallback", True), (True, None, False)])
def test_the_followed_lineage_is_lost_only_off_its_state(converged, alarm, lost):
    # follow tracks a diabatic state through a crossing, where a large occupied-space rotation is the normal case;
    # an aufbau fallback still means the SCF left the state
    alarms = (Alarm(0, alarm, 1.0, 0.0),) if alarm else ()
    assert is_lost(SimpleNamespace(converged=converged, alarms=alarms), followed=True) is lost

def test_search_reason():
    assert search_reason(0, 3, False, (), None) == "start"
    assert search_reason(5, 0, False, (), None) == "empty"
    assert search_reason(5, 2, True, (), None) == "lost"
    assert search_reason(5, 2, False, ("gap_sign_flip",), None) == "alarm"
    assert search_reason(10, 2, False, (), 10) == "periodic" and search_reason(11, 2, False, (), 10) is None
    assert search_reason(5, 2, False, ("scf_cycles",), None) is None
    assert search_reason(0, 0, True, ("overlap_drop",), 10) == "start"  # the first reason that applies


def test_admission_slot():
    assert admission_slot(-1.0, {0: -1.1}, 2, None) == ("admit", None)
    assert admission_slot(-1.0, {0: -1.1, 1: -0.9}, 2, None) == ("evict", 1)
    assert admission_slot(-1.0, {0: -1.1, 1: -0.9}, 2, protected=1) == ("full", None)  # follow keeps it
    assert admission_slot(-0.8, {0: -1.1, 1: -0.9}, 2, None) == ("full", None)
    assert admission_slot(-1.2, {0: -1.1}, 1, None) == ("evict", 0)                   # even the selected one


def test_next_strikes():
    assert next_strikes(1, -1.0, -1.01, 4e-3) == 2
    assert next_strikes(1, -1.0, -1.001, 4e-3) == 0


def test_revive_eligible():
    assert revive_eligible(5, 5, 3) and revive_eligible(5, 8, 3)
    assert not revive_eligible(5, 9, 3) and not revive_eligible(None, 5, 3)


def test_status_precedence():
    assert status_of([-1.0, -1.0 + 5e-5], 2e-3, 1e-4, [("spawn", 1)], ("scf_cycles",)) == "DEGENERATE"
    assert status_of([-1.0, -1.0 + 1e-3], 2e-3, 1e-4, [("spawn", 1)], ()) == "MULTISTATE"
    assert status_of([-1.0, -0.99], 2e-3, 1e-4, [("lost", 2)], ()) == "ALARM"
    assert status_of([-1.0], 2e-3, 1e-4, [], ("scf_cycles",)) == "ALARM"
    assert status_of([-1.0], 2e-3, 1e-4, [], ()) == "FAST"
