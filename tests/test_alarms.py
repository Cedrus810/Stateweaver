import pytest

from sroqm.alarms import AlarmThresholds, evaluate_alarms
from sroqm.fingerprint import FingerprintChange
from sroqm.overlap import OccupiedOverlap

QUIET = dict(converged=True, cycles=4, gap=0.3,
             change=FingerprintChange(0.0, 0.0, 0.0), overlap=OccupiedOverlap(0.99, 0.999))


def names(**over):
    return {a.name for a in evaluate_alarms(**{**QUIET, **over}, thresholds=AlarmThresholds())}


def test_quiet_step_raises_nothing():
    assert names() == set()


def test_first_step_without_reference_checks_only_scf_and_gap():
    assert names(change=None, overlap=None) == set()
    assert names(change=None, overlap=None, converged=False) == {"scf_not_converged"}


@pytest.mark.parametrize("override, expected", [
    (dict(converged=False), "scf_not_converged"),
    (dict(cycles=40), "scf_cycles"),
    (dict(gap=0.001), "small_gap"),
    (dict(gap=-0.005), "small_gap"),
    (dict(change=FingerprintChange(0.5, 0.0, 0.0)), "s2_jump"),
    (dict(change=FingerprintChange(0.0, 0.8, 0.0)), "spin_jump"),
    (dict(change=FingerprintChange(0.0, 0.0, 0.8)), "charge_jump"),
    (dict(overlap=OccupiedOverlap(0.1, 0.05)), "overlap_drop"),
])
def test_each_trigger(override, expected):
    assert names(**override) == {expected}


def test_overlap_alarm_uses_min_sv_not_det():
    # a large system can have a small determinant on a perfectly smooth step
    assert names(overlap=OccupiedOverlap(0.3, 0.98)) == set()


def test_alarm_records_tier_value_and_threshold():
    (a,) = evaluate_alarms(**{**QUIET, "overlap": OccupiedOverlap(0.1, 0.05)}, thresholds=AlarmThresholds())
    assert (a.tier, a.value, a.threshold) == (1, 0.05, 0.9)
