"""coordination.py (Stateweaver design §6-7; V0 event semantics §9): synthetic observations, no SCF."""
import json

import pytest

from sroqm.bonding import BondObservation, PairSpec
from sroqm.coordination import BondThresholds, CoordinationMonitor, PairMonitor, region

TH = BondThresholds(r_on=2.4, r_off=3.0, m_on=0.20, m_off=0.05, n_confirm=2)
PAIRS = [PairSpec("FeO1", (0, 1), "Fe-O")]


def ob(r, m, lid=0, step=0, pid="FeO1"):
    return BondObservation(pid, lid, step, r, m, (1.0, -0.3), (3.8, 0.02))


def run(monitor, path, lid=0, start=0, **kw):
    events = []
    for k, (r, m) in enumerate(path, start):
        events += monitor.step(k, {lid: [ob(r, m, lid, k)]}, selected=lid, **kw)
    return events


def kinds(events):
    return [e["event_type"] for e in events]


def test_thresholds_must_leave_a_gap_between_the_two_regions():
    with pytest.raises(ValueError):
        BondThresholds(r_on=3.0, r_off=2.4, m_on=0.2, m_off=0.05)
    with pytest.raises(ValueError):
        BondThresholds(r_on=2.4, r_off=3.0, m_on=0.05, m_off=0.2)
    with pytest.raises(ValueError):
        CoordinationMonitor(PAIRS, {"Fe-N": TH})  # every pair's donor class needs thresholds


def test_regions():
    assert [region(ob(*x), TH) for x in ((2.1, 0.3), (3.5, 0.01), (2.7, 0.1), (2.1, 0.01), (3.5, 0.3))] == \
        ["on", "off", "mid", "conflict", "conflict"]


def test_pull_away_and_return_gives_one_dissociation_and_one_formation_with_onset_and_confirmation():
    away = [(2.1, 0.33), (2.15, 0.32), (2.5, 0.18), (2.8, 0.10), (3.1, 0.04), (3.4, 0.02), (3.6, 0.01)]
    mon = CoordinationMonitor(PAIRS, {"Fe-O": TH})
    ev = run(mon, away + away[::-1])
    assert kinds(ev) == ["INITIALIZED", "DISSOCIATED", "FORMED"]
    init, diss, form = ev
    assert (init["new_label"], init["onset_step"], init["confirmed_at"]) == ("BONDED", 0, 1)
    assert (diss["previous_label"], diss["new_label"], diss["onset_step"], diss["confirmed_at"]) == \
        ("BONDED", "UNBOUND", 2, 5)  # left the bonded region at step 2, second step in the unbound region is 5
    assert (form["new_label"], form["onset_step"], form["confirmed_at"]) == ("BONDED", 10, 13)


def test_a_new_observation_is_not_assumed_bonded():
    mon = CoordinationMonitor(PAIRS, {"Fe-O": TH})
    assert run(mon, [(2.1, 0.33)]) == []
    assert mon.snapshot()["0:FeO1"]["stable"] is None and mon.snapshot()["0:FeO1"]["transition"] == "AMBIGUOUS"


def test_small_oscillations_keep_the_label_and_a_departure_that_comes_back_is_withdrawn():
    mon = CoordinationMonitor(PAIRS, {"Fe-O": TH})
    ev = run(mon, [(2.1, 0.33), (2.2, 0.30), (2.3, 0.28), (2.2, 0.30), (2.35, 0.26), (2.25, 0.29)])
    assert kinds(ev) == ["INITIALIZED"]
    ev = run(mon, [(2.6, 0.15), (2.8, 0.09), (2.3, 0.28)], start=6)  # into the middle and back
    assert kinds(ev) == ["ATTEMPT_WITHDRAWN"] and ev[0]["onset_step"] == 6
    assert mon.snapshot()["0:FeO1"]["stable"] == "BONDED" and mon.snapshot()["0:FeO1"]["transition"] is None


def test_the_middle_without_a_trend_or_with_conflicting_descriptors_is_ambiguous_not_breaking():
    mon = CoordinationMonitor(PAIRS, {"Fe-O": TH})
    run(mon, [(2.1, 0.33), (2.1, 0.33)])
    run(mon, [(2.6, 0.15)], start=2)
    assert mon.snapshot()["0:FeO1"]["transition"] == "BREAKING"  # r up, M down
    run(mon, [(2.6, 0.15)], start=3)
    assert mon.snapshot()["0:FeO1"]["transition"] == "AMBIGUOUS"  # no trend
    run(mon, [(3.5, 0.30)], start=4)
    s = mon.snapshot()["0:FeO1"]
    assert s["transition"] == "AMBIGUOUS" and s["quality"] == "DESCRIPTOR_CONFLICT" and s["stable"] == "BONDED"


def test_an_unconverged_frame_interrupts_the_confirmation_count():
    mon = CoordinationMonitor(PAIRS, {"Fe-O": TH})
    run(mon, [(2.1, 0.33), (2.1, 0.33)])
    ev = mon.step(2, {0: [ob(3.4, 0.02, 0, 2)]}, selected=0)
    ev += mon.step(3, {0: [ob(3.5, 0.01, 0, 3)]}, selected=0, invalid={0})
    assert ev == [] and mon.snapshot()["0:FeO1"]["quality"] == "UNCONVERGED"
    ev = mon.step(4, {0: [ob(3.5, 0.01, 0, 4)]}, selected=0)
    assert ev == []  # 1 valid in the unbound region since the interruption
    ev = mon.step(5, {0: [ob(3.6, 0.01, 0, 5)]}, selected=0)
    assert kinds(ev) == ["DISSOCIATED"] and ev[0]["onset_step"] == 2


def test_switching_the_selected_branch_at_one_geometry_is_electronic_not_a_bond_event():
    mon = CoordinationMonitor(PAIRS, {"Fe-O": TH})
    for k in range(3):  # lineage 0 bonded, lineage 1 unbound (an electronic difference), same geometry
        ev = mon.step(k, {0: [ob(2.6, 0.30, 0, k)], 1: [ob(2.6, 0.02, 1, k)]}, selected=0)
    ev = mon.step(3, {0: [ob(2.6, 0.30, 0, 3)], 1: [ob(2.6, 0.02, 1, 3)]}, selected=1)
    assert kinds(ev) == ["SELECTED_LINEAGE_SWITCH"]
    d = ev[0]["differences"]["FeO1"]
    assert d["dmayer"] == pytest.approx(-0.28) and ev[0]["source_type"] == "ELECTRONIC"
    assert "DISSOCIATED" not in kinds(ev)


def test_branches_that_disagree_on_a_label_raise_one_state_dependent_event():
    th = BondThresholds(r_on=2.7, r_off=3.0, m_on=0.20, m_off=0.05, n_confirm=1)
    mon = CoordinationMonitor(PAIRS, {"Fe-O": th})
    ev = []
    for k in range(3):
        ev += mon.step(k, {0: [ob(2.6, 0.30, 0, k)], 1: [ob(3.1, 0.02, 1, k)]}, selected=0)
    assert kinds(ev).count("STATE_DEPENDENT_BONDING_CHANGE") == 1


def test_a_revived_lineage_keeps_its_identity_but_must_reconfirm():
    mon = CoordinationMonitor(PAIRS, {"Fe-O": TH})
    run(mon, [(2.1, 0.33), (2.1, 0.33)])
    mon.step(2, {}, selected=0)  # dormant: no observation
    assert mon.snapshot()["0:FeO1"]["quality"] == "MISSING"
    ev = mon.step(3, {0: [ob(2.1, 0.33, 0, 3)]}, selected=0, events=[("revive", 0)])
    assert kinds(ev) == ["SEGMENT_RESTART"] and ev[0]["previous_label"] == "BONDED" and ev[0]["segment"] == 1
    assert mon.snapshot()["0:FeO1"]["stable"] is None
    ev = mon.step(4, {0: [ob(2.1, 0.33, 0, 4)]}, selected=0)
    assert kinds(ev) == ["INITIALIZED"] and ev[0]["segment"] == 1


def test_a_merged_lineage_is_closed_and_an_uncertain_one_restarts():
    mon = CoordinationMonitor(PAIRS, {"Fe-O": TH})
    for k in range(2):
        mon.step(k, {0: [ob(2.1, 0.33, 0, k)], 1: [ob(2.1, 0.33, 1, k)]}, selected=0)
    ev = mon.step(2, {0: [ob(2.1, 0.33, 0, 2)]}, selected=0, events=[("merge", 1)], uncertain={0})
    assert sorted(kinds(ev)) == ["MONITOR_CLOSED", "SEGMENT_RESTART"]
    assert "1:FeO1" not in mon.snapshot() and mon.snapshot()["0:FeO1"]["segment"] == 1


def test_restoring_a_saved_state_continues_with_the_same_events():
    path = [(2.1, 0.33), (2.15, 0.32), (2.5, 0.18), (2.8, 0.10), (3.1, 0.04), (3.4, 0.02), (3.6, 0.01),
            (3.0, 0.06), (2.5, 0.17), (2.2, 0.30), (2.1, 0.33)]
    whole = run(CoordinationMonitor(PAIRS, {"Fe-O": TH}), path)
    first = CoordinationMonitor(PAIRS, {"Fe-O": TH})
    head = run(first, path[:4])
    state = json.loads(json.dumps(first.to_state()))  # through JSON, as a file would
    resumed = CoordinationMonitor.from_state(PAIRS, {"Fe-O": TH}, state)
    assert head + run(resumed, path[4:], start=4) == whole
    assert resumed.snapshot() == run_snapshot(path)


def run_snapshot(path):
    mon = CoordinationMonitor(PAIRS, {"Fe-O": TH})
    run(mon, path)
    return mon.snapshot()


def test_pair_monitor_without_observations_only_marks_itself_stale():
    m = PairMonitor(0, "FeO1", stable="BONDED", transition=None)
    assert m.update(None, False, 5, TH) == [] and (m.stable, m.quality) == ("BONDED", "MISSING")


def test_beam_step_marks_unconverged_and_continuity_doubtful_lineages(monkeypatch):
    from types import SimpleNamespace

    from sroqm import coordination
    seen = {}

    def fake_observe(result, pairs, step):
        return {lid: [ob(2.1, 0.33, lid, step)] for lid in result.frames}

    def spy(self, step, obs, selected, events=(), invalid=(), uncertain=()):
        seen.update(step=step, lids=sorted(obs), selected=selected, events=list(events), invalid=set(invalid),
                    uncertain=set(uncertain))
        return []

    monkeypatch.setattr(coordination, "observe_beam", fake_observe)
    monkeypatch.setattr(CoordinationMonitor, "step", spy)
    alarm = SimpleNamespace(name="overlap_drop")
    res = SimpleNamespace(frames={0: None, 2: None, 5: None}, lineage_id=2, events=(("switch", 2),),
                          results={0: SimpleNamespace(converged=True, alarms=()),
                                   2: SimpleNamespace(converged=False, alarms=()),
                                   5: SimpleNamespace(converged=True, alarms=(alarm,)),
                                   9: SimpleNamespace(converged=False, alarms=())})  # 9: not ACTIVE, ignored
    coordination.beam_step(CoordinationMonitor(PAIRS, {"Fe-O": TH}), res, 4)
    assert seen == dict(step=4, lids=[0, 2, 5], selected=2, events=[("switch", 2)], invalid={2}, uncertain={5})
