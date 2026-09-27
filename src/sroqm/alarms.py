"""Tier-0 (per-step SCF/population) and Tier-1 (cross-geometry overlap) alarms, spec §8."""
from __future__ import annotations

from dataclasses import dataclass

from sroqm.fingerprint import FingerprintChange
from sroqm.overlap import OccupiedOverlap


@dataclass(frozen=True)
class AlarmThresholds:
    max_scf_cycles: int = 15
    max_ds2: float = 0.1
    max_dspin: float = 0.3
    max_dcharge: float = 0.3
    min_abs_gap: float = 0.01  # Hartree
    min_overlap_sv: float = 0.9


@dataclass(frozen=True)
class Alarm:
    tier: int
    name: str
    value: float
    threshold: float


def evaluate_alarms(*, converged: bool, cycles: int, gap: float, change: FingerprintChange | None,
                    overlap: OccupiedOverlap | None, thresholds: AlarmThresholds) -> tuple[Alarm, ...]:
    t = thresholds
    alarms: list[Alarm] = []
    if not converged:
        alarms.append(Alarm(0, "scf_not_converged", 1.0, 0.0))
    if cycles > t.max_scf_cycles:
        alarms.append(Alarm(0, "scf_cycles", float(cycles), float(t.max_scf_cycles)))
    if abs(gap) < t.min_abs_gap:
        alarms.append(Alarm(0, "small_gap", gap, t.min_abs_gap))
    if change is not None:
        if change.ds2 > t.max_ds2:
            alarms.append(Alarm(0, "s2_jump", change.ds2, t.max_ds2))
        if change.max_dspin > t.max_dspin:
            alarms.append(Alarm(0, "spin_jump", change.max_dspin, t.max_dspin))
        if change.max_dcharge > t.max_dcharge:
            alarms.append(Alarm(0, "charge_jump", change.max_dcharge, t.max_dcharge))
    if overlap is not None and overlap.min_sv < t.min_overlap_sv:
        alarms.append(Alarm(1, "overlap_drop", overlap.min_sv, t.min_overlap_sv))
    return tuple(alarms)
