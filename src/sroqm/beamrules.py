"""Beam parameters and the pure decision rules of a beam step (Phase-3b spec §4, §6)."""
from __future__ import annotations

from dataclasses import dataclass

from sroqm.engine import Budget

LOST_ALARMS = frozenset({"scf_not_converged", "overlap_drop", "imom_fallback"})
SEARCH_ALARMS = frozenset({"gap_sign_flip", "overlap_drop", "imom_fallback"})


@dataclass(frozen=True)
class BeamParams:
    max_lineages: int = 3
    keep_window: float = 2e-3        # Hartree; admission window above the lowest
    prune_window: float = 4e-3       # Hartree
    n_prune: int = 2
    cap: int = 30                    # SCF cycles: non-selected advance, admission, refresh stage 2
    cap_near: int = 70               # ... when close to the lowest (or unknown); revive attempts
    near_gap: float = 5e-4           # Hartree
    revive_steps: int = 3
    search_every: int | None = 10
    screen_cycles: int = 4
    screen_window: float = 5e-3      # Hartree
    max_interval: int = 4
    grad_reseat_cycles: int = 10
    densify_steps: int = 0                 # full search on each of the next n steps after a DEGENERATE start
    escalate_unknown: int | None = None    # n discovery steps in a row with unknowns -> full search next step
    screen_margin: float | None = None     # Hartree; screen search candidates (search design 2026-10-11 §5.1)
    densify_random: int = 0                # random t2g-orientation SCFs on each densify-window step (memo 10-23 1a)

    def __post_init__(self):
        if self.max_lineages < 1 or self.n_prune < 1:
            raise ValueError("max_lineages and n_prune must be >= 1")
        if not 0 < self.keep_window < self.prune_window:
            raise ValueError("need 0 < keep_window < prune_window")
        if not 1 <= self.cap <= self.cap_near:
            raise ValueError("need 1 <= cap <= cap_near")
        if self.near_gap <= 0 or self.screen_window <= 0:
            raise ValueError("near_gap and screen_window must be > 0")
        if self.revive_steps < 0:
            raise ValueError("revive_steps must be >= 0")
        if self.search_every is not None and self.search_every < 1:
            raise ValueError("search_every must be >= 1 or None")
        if self.screen_cycles < 1 or self.max_interval < 1 or self.grad_reseat_cycles < 1:
            raise ValueError("screen_cycles, max_interval and grad_reseat_cycles must be >= 1")
        if self.densify_steps < 0 or self.densify_random < 0:
            raise ValueError("densify_steps and densify_random must be >= 0")
        if self.escalate_unknown is not None and self.escalate_unknown < 1:
            raise ValueError("escalate_unknown must be >= 1 or None")
        if self.screen_margin is not None and self.screen_margin < 0:
            raise ValueError("screen_margin must be >= 0 or None")


def advance_budget(selected: bool, gap_last: float, p: BeamParams) -> Budget:
    """Spec §4.2. A lineage close to the lowest may be about to cross below it: it gets cap_near, not cap."""
    if selected:
        return Budget()
    return Budget(max_cycle=p.cap_near if gap_last < p.near_gap else p.cap, allow_fallback=False, with_gradient=False)


def is_lost(result, followed: bool = False) -> bool:
    """`followed`: the lineage the follow policy tracks. It follows a diabatic state through crossings, where
    overlap_drop is the expected rotation of its orbitals, so only leaving the state (no convergence, fallback)
    loses it; overlap_drop stays an alarm and triggers a search."""
    lost_alarms = LOST_ALARMS - {"overlap_drop"} if followed else LOST_ALARMS
    return not result.converged or any(a.name in lost_alarms for a in result.alarms)


def search_reason(step: int, n_active: int, selected_lost: bool, selected_alarms, search_every) -> str | None:
    """Why a full search runs this step (spec 2026-09-30 §5.3), or None. The label goes into the trace so the cost
    report can split start, recovery and periodic searches (search-performance design 2026-10-11 §6)."""
    if step == 0:
        return "start"
    if n_active == 0:
        return "empty"
    if selected_lost:
        return "lost"
    if SEARCH_ALARMS.intersection(selected_alarms):
        return "alarm"
    if search_every is not None and step % search_every == 0:
        return "periodic"
    return None


def admission_slot(energy: float, active: dict, max_lineages: int, protected: int | None):
    """Spec §4.5: room -> admit; full -> evict the highest unprotected lineage if it is above `energy`."""
    if len(active) < max_lineages:
        return "admit", None
    candidates = {i: e for i, e in active.items() if i != protected}
    if not candidates:
        return "full", None
    worst = max(candidates, key=candidates.get)
    return ("evict", worst) if candidates[worst] > energy else ("full", None)


def next_strikes(strikes: int, energy: float, emin: float, prune_window: float) -> int:
    return strikes + 1 if energy - emin > prune_window else 0


def revive_eligible(lost_step: int | None, step: int, revive_steps: int) -> bool:
    return lost_step is not None and step - lost_step <= revive_steps


def status_of(energies_sorted, keep_window: float, deg_band: float, events, selected_alarms) -> str:
    e = list(energies_sorted)
    if len(e) >= 2 and e[1] - e[0] < deg_band:
        return "DEGENERATE"
    if sum(x - e[0] <= keep_window for x in e) >= 2:
        return "MULTISTATE"
    if events or selected_alarms:
        return "ALARM"
    return "FAST"
