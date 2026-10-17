"""Refresh (cheap discovery inside the current manifold) and when to run it (Phase-3b spec §5.1-5.2)."""
from __future__ import annotations

import math
from dataclasses import dataclass

from sroqm.executor import SCFTask
from sroqm.frame import Frame
from sroqm.overlap import occupied_overlap


@dataclass(frozen=True, eq=False)
class RefreshResult:
    siblings: list          # converged Frames distinct from the base and each other, lowest first
    unknown: int            # still unconverged after stage 2: "unknown", not "absent"
    log: tuple              # "x+37" screened (mHa), "nc" unknown, "+0.123" sibling (mHa above base)


def refresh(executor, mol, base: Frame, patterns, cap: int, screen_cycles: int = 4, screen_window: float = 5e-3,
            same_state_sv: float = 0.9, point_charges=None) -> RefreshResult:
    """Try every occupation in `patterns` on `base`'s orbitals (base at `mol`): stage 1 screens, stage 2 converges."""
    guess_base = base if base.mol is mol else Frame.from_orbitals(mol, base.mo_coeff, base.mo_occ)
    guesses = [guess_base.with_occupations(occ) for occ in patterns]
    stage1 = executor.map([SCFTask(mol=mol, dm0=g.dm, imom_ref=g, max_cycle=screen_cycles,
                                   point_charges=point_charges, tag="refresh:s1") for g in guesses])
    log, done, stage2 = [], [], []
    for r in stage1:
        de = r.outcome.energy - base.energy
        if r.converged:
            done.append(r.frame)
        elif de > screen_window:
            log.append(f"x{1e3 * de:+.0f}")
        else:
            stage2.append(SCFTask(mol=mol, dm0=r.outcome.dm, imom_ref=r.frame, max_cycle=cap,
                                  point_charges=point_charges, tag="refresh:s2"))
    unknown = 0
    for r in executor.map(stage2):
        if r.converged:
            done.append(r.frame)
        else:
            unknown += 1
            log.append("nc")
    siblings = []
    for f in sorted(done, key=lambda f: f.energy):
        if occupied_overlap(base, f).min_sv < same_state_sv and \
                all(occupied_overlap(s, f).min_sv < same_state_sv for s in siblings):
            siblings.append(f)
    log += [f"{1e3 * (f.energy - base.energy):+.3f}" for f in siblings]
    return RefreshResult(siblings, unknown, tuple(log))


class RefreshScheduler:
    """When the next refresh runs, and how much its second stage may spend (spec §5.1-5.2)."""

    def __init__(self, max_interval: int = 4, cap: int = 30, cap_near: int = 70, near_gap: float = 5e-4):
        if max_interval < 1 or cap < 1 or cap_near < cap or near_gap <= 0:
            raise ValueError("need max_interval >= 1, 1 <= cap <= cap_near, near_gap > 0")
        self.max_interval, self.cap, self.cap_near, self.near_gap = max_interval, cap, cap_near, near_gap
        self.next_step = 0
        self._last: tuple[int, float] | None = None
        self._unknown_last = False

    def due(self, step: int) -> bool:
        return step >= self.next_step

    def stage2_cap(self, gap_prev: float, n_active: int) -> int:
        near = gap_prev < self.near_gap or self._unknown_last or n_active <= 1
        return self.cap_near if near else self.cap

    def record(self, step: int, gap: float, unknown: int, degenerate: bool, beam_changed: bool) -> int:
        self._unknown_last = unknown > 0
        if unknown or degenerate or beam_changed or self._last is None:
            interval = 1
        elif not math.isfinite(gap):
            interval = self.max_interval
        elif not math.isfinite(self._last[1]):
            interval = 1  # a second state just appeared: the gap closed from infinity
        else:
            rate = (gap - self._last[1]) / (step - self._last[0])
            interval = (self.max_interval if rate >= 0
                        else min(self.max_interval, max(1, math.floor(gap / -rate / 2))))
        self._last = (step, gap)
        self.next_step = step + interval
        return interval

    def poke(self, step: int) -> None:
        self.next_step = min(self.next_step, step + 1)
