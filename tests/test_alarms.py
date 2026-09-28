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


def test_gap_sign_flip_in_any_spin_channel():
    assert names(gaps=(0.05, -0.02), prev_gaps=(0.04, 0.03)) == {"gap_sign_flip"}
    assert names(gaps=(-0.3,), prev_gaps=(0.01,)) == {"gap_sign_flip"}


def test_no_gap_sign_flip_without_a_sign_change_or_a_previous_step():
    assert names(gaps=(0.05, 0.02), prev_gaps=(0.04, 0.03)) == set()
    assert names(gaps=(-0.2,), prev_gaps=(-0.1,)) == set()        # an excited seed stays non-aufbau
    assert names(gaps=(float("inf"), 0.2), prev_gaps=(float("inf"), 0.3)) == set()
    assert names(gaps=(-0.2,), prev_gaps=None) == set()


def test_lower_state_found_by_a_probe():
    (a,) = evaluate_alarms(**QUIET, probe_delta=5e-4, thresholds=AlarmThresholds())
    assert (a.tier, a.name, a.value, a.threshold) == (2, "lower_state_found", 5e-4, 1e-4)


def test_probe_at_or_above_the_lineage_energy_is_quiet():
    assert names(probe_delta=1e-6) == set()
    assert names(probe_delta=-0.3) == set()
    assert names(probe_delta=None) == set()


def test_imom_fallback_is_reported():
    assert names(imom_fallback=True) == {"imom_fallback"}


def test_scf_cycles_are_skipped_when_no_history_guess_was_used():
    assert names(cycles=None) == set()


def test_overlap_alarm_uses_min_sv_not_det():
    # a large system can have a small determinant on a perfectly smooth step
    assert names(overlap=OccupiedOverlap(0.3, 0.98)) == set()


def test_alarm_records_tier_value_and_threshold():
    (a,) = evaluate_alarms(**{**QUIET, "overlap": OccupiedOverlap(0.1, 0.05)}, thresholds=AlarmThresholds())
    assert (a.tier, a.value, a.threshold) == (1, 0.05, 0.9)
