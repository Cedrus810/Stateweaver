"""Multi-lineage beam (spec §9–§11): keep every low-lying electronic state alive along the path.

Each lineage is a single-lineage `Engine`. On the first step, and whenever a lineage raises a
state-change alarm or the schedule says so, `search_states` runs at the current geometry;
states within `keep_window` of the lowest that match no existing lineage are spawned (or revive
a dormant lineage they match); with the beam full, a lower state replaces the highest lineage.
Active lineages that have become the same state (e.g. after an imom_fallback) are merged,
keeping the lower one. A lineage more than `prune_window` above the lowest for `n_prune`
consecutive steps goes dormant. E and F come from the surface policy:
`adiabatic_min` (lowest lineage; `switched` marks the cusp) or `follow` (the first lineage).
Every step's result carries all active lineages' results regardless of policy.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from sroqm.alarms import AlarmThresholds
from sroqm.engine import Engine, Result
from sroqm.frame import Frame
from sroqm.lineage import LineageStatus
from sroqm.overlap import occupied_overlap
from sroqm.search import search_states

_POLICIES = ("adiabatic_min", "follow")
_SEARCH_TRIGGERS = frozenset({"gap_sign_flip", "overlap_drop", "imom_fallback"})


@dataclass
class _Member:
    id: int
    engine: Engine
    status: LineageStatus = LineageStatus.ACTIVE
    strikes: int = 0  # consecutive steps above the prune window


@dataclass(frozen=True, eq=False)
class BeamResult:
    step: int
    energy: float
    forces: np.ndarray
    policy: str
    lineage_id: int
    switched: bool  # adiabatic_min changed lineage this step: E/F are continuous only up to a cusp
    status: str  # FAST | MULTISTATE | ALARM
    lineage_energies: dict[int, float]
    lineage_status: dict[int, str]
    events: tuple[tuple[str, int], ...]  # ("spawn" | "revive" | "dormant" | "merge", lineage id)
    results: dict[int, Result] = field(repr=False)


class BeamEngine:
    def __init__(self, template, backend, metal_atoms=(), policy: str = "adiabatic_min",
                 keep_window: float = 8e-3, prune_window: float = 16e-3, n_prune: int = 3,
                 max_lineages: int = 4, search_every: int | None = None, fragments=None,
                 thresholds: AlarmThresholds | None = None, same_state_sv: float = 0.9):
        """keep_window / prune_window in Hartree (defaults ~5 / ~10 kcal/mol, spec §9.1, §10.3)."""
        if policy not in _POLICIES:
            raise ValueError(f"policy must be one of {_POLICIES}, got {policy!r}")
        if max_lineages < 1 or n_prune < 1:
            raise ValueError("max_lineages and n_prune must be >= 1")
        if not 0 < keep_window < prune_window:
            raise ValueError("need 0 < keep_window < prune_window")
        if search_every is not None and search_every < 1:
            raise ValueError(f"search_every must be >= 1 or None, got {search_every}")
        self.template = template
        self.backend = backend
        self.metal_atoms = list(metal_atoms)
        self.policy = policy
        self.keep_window = keep_window
        self.prune_window = prune_window
        self.n_prune = n_prune
        self.max_lineages = max_lineages
        self.search_every = search_every
        self.fragments = fragments
        self.thresholds = thresholds
        self.same_state_sv = same_state_sv
        self.members: list[_Member] = []
        self.step = 0
        self._selected: int | None = None
        self._followed: int | None = None

    def compute(self, coords_bohr, point_charges=None) -> BeamResult:
        coords = np.asarray(coords_bohr, dtype=float)
        mol = self.template.set_geom_(coords, unit="Bohr", inplace=False)
        results: dict[int, Result] = {}
        events: list[tuple[str, int]] = []

        if not self.members:
            states = search_states(self.backend, mol, None, self.metal_atoms, point_charges,
                                   same_state_sv=self.same_state_sv)
            if not states:
                raise RuntimeError("no converged electronic state at the first geometry")
            self._admit(states, states[0].energy, coords, point_charges, results, events)
        else:
            for m in self._active():
                results[m.id] = m.engine.compute(coords, point_charges)
            scheduled = self.search_every is not None and self.step % self.search_every == 0
            alarmed = any(a.name in _SEARCH_TRIGGERS for r in results.values() for a in r.alarms)
            if scheduled or alarmed:
                lowest = min(results, key=lambda i: results[i].energy)
                base = self._member(lowest).engine.lineage.last_frame
                states = search_states(self.backend, mol, base, self.metal_atoms, point_charges,
                                       same_state_sv=self.same_state_sv)
                self._admit(states, min(r.energy for r in results.values()), coords, point_charges, results, events)

        self._merge(results, events)
        emin = min(r.energy for r in results.values())
        self._prune(results, emin, events)
        if self._followed is None:
            self._followed = min(results, key=lambda i: results[i].energy)
        selected = (self._followed if self.policy == "follow"
                    else min(results, key=lambda i: results[i].energy))
        switched = self._selected is not None and selected != self._selected
        self._selected = selected

        near = sum(r.energy - emin <= self.keep_window for r in results.values())
        status = ("MULTISTATE" if near >= 2
                  else "ALARM" if events or results[selected].alarms else "FAST")
        out = BeamResult(
            step=self.step,
            energy=results[selected].energy,
            forces=results[selected].forces,
            policy=self.policy,
            lineage_id=selected,
            switched=switched,
            status=status,
            lineage_energies={i: r.energy for i, r in results.items()},
            lineage_status={m.id: m.status.value for m in self.members},
            events=tuple(events),
            results=results,
        )
        self.step += 1
        return out

    def _active(self) -> list[_Member]:
        return [m for m in self.members if m.status is LineageStatus.ACTIVE]

    def _member(self, lineage_id: int) -> _Member:
        return next(m for m in self.members if m.id == lineage_id)

    def _new_engine(self, seed: Frame) -> Engine:
        return Engine(self.template, self.backend, fragments=self.fragments, thresholds=self.thresholds,
                      initial_state=seed)

    def _same_state(self, m: _Member, state: Frame) -> bool:
        last = m.engine.lineage.last_frame
        return last is not None and occupied_overlap(last, state).min_sv >= self.same_state_sv

    def _admit(self, states, emin, coords, point_charges, results, events) -> None:
        """Spawn or revive lineages for found states within keep_window of the lowest known energy."""
        ref = min(emin, states[0].energy) if states else emin
        for state in states:  # lowest first
            if state.energy - ref > self.keep_window:
                break
            match = next((m for m in self.members if self._same_state(m, state)), None)
            if match is not None and match.status is LineageStatus.ACTIVE:
                continue
            room = len(self._active()) < self.max_lineages or self._retire_highest_above(state.energy, results, events)
            if not room:
                continue
            if match is not None:  # dormant or merged: restart it from the state just found
                match.engine = self._new_engine(state)
                match.status, match.strikes = LineageStatus.ACTIVE, 0
                member, kind = match, "revive"
            else:
                member, kind = _Member(id=len(self.members), engine=self._new_engine(state)), "spawn"
                member.engine.lineage.id = member.id
                member.engine.lineage.birth_step = self.step
                self.members.append(member)
            results[member.id] = member.engine.compute(coords, point_charges)
            events.append((kind, member.id))

    def _retire_highest_above(self, energy, results, events) -> bool:
        """Full beam: make room for a state at `energy` by retiring the highest active lineage above it."""
        candidates = [m for m in self._active() if m.id in results
                      and not (self.policy == "follow" and m.id == self._followed)]
        if not candidates:
            return False
        highest = max(candidates, key=lambda m: results[m.id].energy)
        if results[highest.id].energy <= energy:
            return False
        highest.status = LineageStatus.DORMANT
        events.append(("dormant", highest.id))
        return True

    def _merge(self, results, events) -> None:
        """Retire active lineages that sit on the same state as another; keep the lower (ties: lower id)."""
        kept: list[_Member] = []
        for m in self._active():
            frame = m.engine.lineage.last_frame
            twin = next((k for k in kept if self._same_state(k, frame)), None)
            if twin is None:
                kept.append(m)
                continue
            loser = m if results[m.id].energy >= results[twin.id].energy - 1e-6 else twin
            if loser is twin:
                kept[kept.index(twin)] = m
            loser.status = LineageStatus.PRUNED
            results.pop(loser.id)
            events.append(("merge", loser.id))

    def _prune(self, results, emin, events) -> None:
        for m in self._active():
            if m.id not in results or (self.policy == "follow" and m.id == self._followed):
                continue
            if results[m.id].energy - emin > self.prune_window:
                m.strikes += 1
                if m.strikes >= self.n_prune:
                    m.status = LineageStatus.DORMANT
                    events.append(("dormant", m.id))
            else:
                m.strikes = 0
