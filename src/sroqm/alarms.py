"""Tier-0 (per-step SCF/population) and Tier-1 (cross-geometry overlap) alarms, spec §8."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

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
    probe_tol: float = 1e-4  # Hartree; a probe this far below the lineage means a lower state exists


@dataclass(frozen=True)
class Alarm:
    tier: int
    name: str
    value: float
    threshold: float


def evaluate_alarms(*, converged: bool, cycles: int | None, gap: float, change: FingerprintChange | None,
                    overlap: OccupiedOverlap | None, thresholds: AlarmThresholds,
                    gaps: tuple[float, ...] = (), prev_gaps: tuple[float, ...] | None = None,
                    imom_fallback: bool = False, probe_delta: float | None = None) -> tuple[Alarm, ...]:
    """`cycles` is None when the SCF started without a history guess (nothing to judge continuation by).

    `gaps` / `prev_gaps` are per-spin-channel frontier gaps of this and the previous step. A sign change
    means the followed state crossed an orbital crossing: IMOM stays diabatic (now non-aufbau), aufbau
    switched roots. Overlap cannot see the first case, and a crossing between two steps can skip the
    small-gap window entirely.

    `imom_fallback` is True when IMOM failed to converge and the step was redone with aufbau, so the
    returned state may not be the followed one.

    `probe_delta` is E(lineage) - E(independent probe SCF) on a scheduled probe step (Tier 2). A
    continuation stays in its own SCF minimum, so a lower state that appears elsewhere is invisible
    to every other check here.
    """
    t = thresholds
    alarms: list[Alarm] = []
    if imom_fallback:
        alarms.append(Alarm(0, "imom_fallback", 1.0, 0.0))
    if prev_gaps is not None:
        for now, before in zip(gaps, prev_gaps):
            if np.isfinite(now) and np.isfinite(before) and np.sign(now) != np.sign(before):
                alarms.append(Alarm(0, "gap_sign_flip", now, 0.0))
                break
    if not converged:
        alarms.append(Alarm(0, "scf_not_converged", 1.0, 0.0))
    if cycles is not None and cycles > t.max_scf_cycles:
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
    if probe_delta is not None and probe_delta > t.probe_tol:
        alarms.append(Alarm(2, "lower_state_found", probe_delta, t.probe_tol))
    return tuple(alarms)
