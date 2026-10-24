"""Multi-lineage beam with the Phase-3b hygiene rules (spec docs/superpowers/specs/2026-09-30-phase3b-beam-hygiene-design.md).

One step: every ACTIVE lineage advances with a budget set by its gap to the lowest; lost lineages go DORMANT
and, if they were within keep_window, keep their last converged frame as a revive seed; discovery = a two-level
full search when due, a refresh from the lowest lineage when scheduled, and revive attempts; every candidate
must pass admission (IMOM from its own density returns to the same state) and must not land on an ACTIVE
lineage; a full beam evicts its highest lineage for a lower state; lineages on one state merge; the lowest lineage
is selected (adiabatic_min) and only it gets forces; lineages far above the lowest for n_prune steps go DORMANT.

energy_only=True (replay of a fixed-geometry trajectory, r2scan_jax spec 2026-10-11 §3.6): no SCF computes a
gradient and no re-seat runs, so BeamResult.forces is None; every beam decision is unchanged. If the backend has
finalize(outcome) -> (energy, converged, cycles) (r2scan_jax.R2scanBackend: FP64 finish of an FP32-stage SCF),
it runs on the selected lineage's SCF and fills energy_final; BeamResult.energy stays the energy the beam
selected on.
"""
from __future__ import annotations

import dataclasses
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from sroqm.alarms import AlarmThresholds
from sroqm.beamrules import (BeamParams, admission_slot, advance_budget, is_lost, next_strikes, revive_eligible,
                             status_of, search_reason)
from sroqm.engine import Budget, Engine, Result
from sroqm.executor import SCFTask, SerialExecutor, TaskResult
from sroqm.extrapolate import GrassmannGuess
from sroqm.frame import Frame
from sroqm.lineage import LineageStatus
from sroqm.overlap import occupied_overlap
from sroqm.schedule import RefreshScheduler, refresh
from sroqm.search import d_occupation_patterns, full_search, random_orientation_tasks
from sroqm.tolerances import Tolerances

class ReseatError(RuntimeError):
    """A gradient re-seat did not return this lineage's state (spec 2026-10-05 §6.2); the step can fall back."""


_POLICIES = ("adiabatic_min", "follow")
# Events that change the beam and so force the next refresh (spec 2026-09-30 §5.2); `forced` is an admission,
# `reseat_failed` drops a lineage, `follow_lost` changes the selection. A rejected candidate (`full`, `duplicate`,
# `unstable`) leaves the beam as it was; an unknown (`pending`) forces the refresh on its own.
_CHANGES = ("spawn", "revive", "lost", "merge", "prune", "evict", "switch", "forced", "reseat_failed", "follow_lost")


class _Ledger:
    """Wraps the executor for one step and keeps one row per physical SCF (search-performance design 2026-10-11 §4)."""

    def __init__(self, executor):
        self.executor, self.rows = executor, []

    def map(self, tasks):
        out = self.executor.map(tasks)
        for t, r in zip(tasks, out):
            o = r.outcome
            self.rows.append(dict(tag=t.tag, max_cycle=t.max_cycle, imom=t.imom_ref is not None, cycles=r.cycles,
                                  converged=r.converged, fallback=r.fallback, energy=float(o.energy),
                                  energies=[float(e) for e in getattr(o, "energies", ())],
                                  seconds={k: float(v) for k, v in getattr(o, "seconds", {}).items()}))
        return out


@dataclass
class _Member:
    id: int
    engine: Engine
    status: LineageStatus = LineageStatus.ACTIVE
    strikes: int = 0
    gap_last: float = 0.0            # energy above the lowest ACTIVE lineage at the end of the previous step
    seed: Frame | None = None        # last converged frame: revive seed after a loss, match key while DORMANT
    lost_step: int | None = None     # set on a loss within keep_window: revive attempts run from here
    result: Result | None = None
    forces: np.ndarray | None = None
    scf_result: TaskResult | None = field(default=None, repr=False)  # retained only during this beam step

    @property
    def frame(self) -> Frame:
        return self.engine.lineage.last_frame


@dataclass(frozen=True, eq=False)
class _Carry:
    """A search candidate still unconverged inside the window: continued on later steps (design 2026-10-11 §5.2)."""
    frame: Frame    # its last frame (unconverged), at the geometry where it last ran
    born: int       # the step whose search left it pending


@dataclass(frozen=True, eq=False)
class BeamResult:
    step: int
    energy: float
    forces: np.ndarray
    policy: str
    lineage_id: int
    switched: bool                                  # adiabatic_min changed lineage: E/F continuous only up to a cusp
    status: str                                     # DEGENERATE | MULTISTATE | ALARM | FAST
    manifold: tuple[tuple[int, float], ...]         # ACTIVE lineages as (id, E - E_min), lowest first
    lineage_energies: dict[int, float]
    lineage_status: dict[int, str]
    events: tuple[tuple[str, int | None], ...]
    frame: Frame = field(repr=False)                # the selected lineage's frame (state-identity checks)
    results: dict[int, Result] = field(repr=False)
    trace: dict = field(default_factory=dict, repr=False)  # per-step diagnostics for benchmarks (spec 2026-10-05 §3.3)
    frames: dict = field(default_factory=dict, repr=False)  # every ACTIVE lineage's frame, by id
    energy_final: float | None = None               # energy_only + backend.finalize: the selected state's finish
    search_frames: tuple = field(default=(), repr=False)  # start search: converged candidates in the window (2a)


class BeamEngine:
    def __init__(self, template, backend, metal_atoms=(), policy: str = "adiabatic_min",
                 params: BeamParams | None = None, tol: Tolerances | None = None, executor=None,
                 fragments=None, thresholds: AlarmThresholds | None = None, energy_only: bool = False):
        if policy not in _POLICIES:
            raise ValueError(f"policy must be one of {_POLICIES}, got {policy!r}")
        self.params = params if params is not None else BeamParams()
        self.tol = tol if tol is not None else Tolerances()
        if backend.spec.conv_tol != self.tol.conv_tol:
            raise ValueError(f"backend conv_tol {backend.spec.conv_tol} != Tolerances.conv_tol {self.tol.conv_tol}: "
                             "every SCF of the beam uses the one Tolerances instance (spec §7)")
        self.template, self.backend, self.metal_atoms, self.policy = template, backend, list(metal_atoms), policy
        self.executor = executor if executor is not None else SerialExecutor(backend)
        self._ledger = _Ledger(self.executor)
        self.fragments, self.thresholds = fragments, thresholds
        self.energy_only = energy_only
        p = self.params
        self.scheduler = RefreshScheduler(p.max_interval, p.cap, p.cap_near, p.near_gap)
        self.members: list[_Member] = []
        self.step = 0
        self._selected: int | None = None
        self._followed: int | None = None
        self._next_id = 0
        self._gap_prev = np.inf
        self._search_next: str | None = None  # "reseat" / "escalate": search next step
        self._densify_until = -1   # last step of the densified discovery after a DEGENERATE start (design §7)
        self._unknown_streak = 0   # discovery steps in a row that left an unknown candidate
        self._pending: list[_Carry] = []  # search candidates being continued (design 2026-10-11 §5.2)
        self._same_log: list[dict] = []   # this step's same-state judgments (design memo 2026-10-23 §2.2 2a)

    # ---- helpers ---------------------------------------------------------------------------------------
    def _active(self) -> list[_Member]:
        return [m for m in self.members if m.status is LineageStatus.ACTIVE]

    def _member(self, lineage_id) -> _Member | None:
        return next((m for m in self.members if m.id == lineage_id), None)

    def _protected(self) -> int | None:
        return self._followed if self.policy == "follow" else None

    def _new_engine(self, seed: Frame) -> Engine:
        engine = Engine(self.template, self.backend, fragments=self.fragments, thresholds=self.thresholds,
                        initial_state=seed)
        engine.step = self.step  # Result.step counts beam steps, also for a lineage born after step 0
        return engine

    def _admit(self, engine: Engine, result: Result, revive_id, events, kind=None, scf_result=None) -> None:
        verdict, victim = admission_slot(result.energy, {m.id: m.result.energy for m in self._active()},
                                         self.params.max_lineages, self._protected())
        if verdict == "full":
            events.append(("full", None))
            return
        if victim is not None:
            v = self._member(victim)
            v.status, v.lost_step, v.seed = LineageStatus.DORMANT, None, v.frame
            events.append(("evict", victim))
        frame = engine.lineage.last_frame
        if revive_id is None:  # a DORMANT lineage on this state keeps its id (spec §4.1)
            revive_id = next((m.id for m in self.members if m.status is LineageStatus.DORMANT
                              and m.seed is not None and self._same(m.seed, frame, "revive_match")), None)
        if revive_id is not None:
            m = self._member(revive_id)
            m.engine, m.status, m.strikes, m.lost_step = engine, LineageStatus.ACTIVE, 0, None
            events.append((kind or "revive", m.id))
        else:
            m = _Member(id=self._next_id, engine=engine)
            self._next_id += 1
            engine.lineage.birth_step = self.step
            self.members.append(m)
            events.append((kind or "spawn", m.id))
        engine.lineage.id = m.id
        m.result, m.forces, m.gap_last = dataclasses.replace(result, lineage_id=m.id), None, 0.0
        m.scf_result = scf_result

    def _reseat_forces(self, m: _Member, point_charges) -> np.ndarray:
        """Reuse a local converged solution, or validate a custom executor's re-seat and update its E/frame.

        Re-seating gets one full-budget retry if unconverged, then must satisfy the same state/energy
        criteria as admission. The default serial executor needs no re-seat SCF."""
        # Custom executors keep the task protocol: they may run remotely or override map() semantics. The process
        # executor keeps the SCF object on its worker: the gradient is computed there (design 2026-10-03 §9.2).
        kept = m.scf_result
        if kept is not None and (kept.handle is not None or type(self.executor) is SerialExecutor):
            forces = -(kept.handle.gradient() if kept.handle is not None else self.backend.gradient(kept.outcome))
            if forces.shape != (self.template.natm, 3) or not np.all(np.isfinite(forces)):
                raise ReseatError(f"lineage {m.id}: invalid reused gradient")
            m.result = dataclasses.replace(m.result, forces=forces)
            return forces
        res = self._ledger.map([m.engine.gradient_task(point_charges, self.params.grad_reseat_cycles)])[0]
        if not res.converged:
            retry = self._ledger.map([m.engine.gradient_task(point_charges, None)])[0]
            res = dataclasses.replace(retry, cycles=res.cycles + retry.cycles)
        if not res.converged:
            raise ReseatError(f"lineage {m.id}: gradient re-seat did not converge ({res.cycles} cycles)")
        forces = m.engine.forces_from(res)
        de = res.outcome.energy - m.result.energy
        if not (self._same(m.frame, res.frame, "reseat") and abs(de) < self.tol.admit_de
                and forces.shape == (self.template.natm, 3) and np.all(np.isfinite(forces))):
            raise ReseatError(f"lineage {m.id}: gradient re-seat left its state (dE = {de:.2e} Ha)")
        m.result = m.engine.finish_gradient(m.result, res)
        return forces

    def _same(self, a: Frame, b: Frame, where: str) -> bool:
        """Tolerances.same_state, and a record (sigma_min, E_b - E_a) of every pair it calls one state (design memo
        2026-10-23 §2.2 2a: near B a distinct state sits at sigma 0.84-0.90). Decisions are unchanged."""
        same = self.tol.same_state(a, b)
        if same:
            self._same_log.append(dict(where=where, sv=occupied_overlap(a, b).min_sv, de_mha=1e3 * (b.energy - a.energy)))
        return same

    def _advance_budget(self, m: _Member):
        b = advance_budget(m.id == self._selected, m.gap_last, self.params)
        return dataclasses.replace(b, with_gradient=False) if self.energy_only else b

    # ---- one step --------------------------------------------------------------------------------------
    _SCALARS = ("step", "_selected", "_followed", "_next_id", "_gap_prev", "_search_next", "_densify_until",
                "_unknown_streak")
    _NOTHING = object()

    def _snapshot(self):
        """Everything a step changes. The advance pushes frames into every lineage before discovery and forces; if a
        later stage raises, compute() puts it all back so the same instance can be called again (review 2026-10-23
        #3). Frames and results are immutable, so shallow copies of the containers are enough."""
        def engine(e):
            lin = e.lineage
            return (e, e.step, getattr(e, "_last_context", self._NOTHING), lin.id, lin.birth_step, list(lin.frames),
                    list(lin.fingerprints), list(lin.energies), list(lin.gaps))
        members = [(m, {f.name: getattr(m, f.name) for f in dataclasses.fields(m)}, engine(m.engine))
                   for m in self.members]
        return (members, {k: getattr(self, k) for k in self._SCALARS}, list(self._pending),
                dict(vars(self.scheduler)))

    def _restore(self, snap) -> None:
        members, scalars, pending, scheduler = snap
        for m, fields, (e, step, context, lid, birth, frames, fps, energies, gaps) in members:
            for k, v in fields.items():
                setattr(m, k, v)
            e.step, e.lineage.id, e.lineage.birth_step = step, lid, birth
            if context is self._NOTHING:
                e.__dict__.pop("_last_context", None)
            else:
                e._last_context = context
            e.lineage.frames.clear()
            e.lineage.frames.extend(frames)
            e.lineage.fingerprints[:], e.lineage.energies[:], e.lineage.gaps[:] = fps, energies, gaps
        self.members = [m for m, _, _ in members]
        for k, v in scalars.items():
            setattr(self, k, v)
        self._pending = pending
        self.scheduler.__dict__.clear()
        self.scheduler.__dict__.update(scheduler)

    def compute(self, coords_bohr, point_charges=None) -> BeamResult:
        snap = self._snapshot()
        try:
            return self._compute(coords_bohr, point_charges)
        except BaseException:
            self._restore(snap)  # a failed step leaves the beam as it was before the call
            raise
        finally:
            for m in self.members:
                m.scf_result = None  # also release PySCF grids/DF resources if discovery or gradients fail
            if hasattr(self.executor, "release"):
                self.executor.release()  # a process executor's workers drop this step's SCF objects

    def _compute(self, coords_bohr, point_charges=None) -> BeamResult:
        p, tol, step = self.params, self.tol, self.step
        coords = np.asarray(coords_bohr, dtype=float)
        if coords.shape != (self.template.natm, 3):
            raise ValueError(f"coords shape {coords.shape} != ({self.template.natm}, 3)")
        mol = self.template.set_geom_(coords, unit="Bohr", inplace=False)
        events: list[tuple[str, int | None]] = []
        self._ledger = ledger = _Ledger(self.executor)
        self._same_log = []
        trace = dict(advance=[], search=None, refresh=None, tasks=ledger.rows, same_state=self._same_log)
        search_frames = ()

        # 1. advance every ACTIVE lineage (spec §4.2-4.3)
        active = self._active()
        tasks = [m.engine.prepare(coords, point_charges, self._advance_budget(m)) for m in active]
        for m, task, res in zip(active, tasks, ledger.map(tasks)):
            m.result = m.engine.finish(task, res)
            m.forces = m.result.forces
            m.scf_result = res
            trace["advance"].append(dict(id=m.id, max_cycle=task.max_cycle, cycles=res.cycles,
                                         converged=res.converged, alarms=[a.name for a in m.result.alarms]))
        selected_lost = False
        for m in active:
            if is_lost(m.result, followed=m.id == self._protected()):
                m.status = LineageStatus.DORMANT
                m.lost_step = step if m.gap_last <= p.keep_window else None  # m.seed: last step's frame
                events.append(("lost", m.id))
                selected_lost |= m.id == self._selected
        active = self._active()
        sel_prev = self._member(self._selected)
        sel_alarms = (tuple(a.name for a in sel_prev.result.alarms)
                      if sel_prev is not None and sel_prev.status is LineageStatus.ACTIVE else ())

        # 2. discovery (spec §5, §4.3)
        cands: list[tuple[Frame, int | None]] = []
        refreshed, pending, new_pending = None, 0, ()
        reason = search_reason(step, len(active), selected_lost, sel_alarms, p.search_every) or self._search_next
        if reason is None and step <= self._densify_until:
            reason = "densify"
        if reason is not None:
            self._search_next = None
            fronts = sorted(active, key=lambda m: m.result.energy)
            low = fronts[0] if fronts else None
            # design §6: densified (§7), escalated and in-window searches are wide: every ACTIVE frame plus every
            # still-revivable lost-state seed; the others start from the lowest frame and this step's lost seeds
            wide = reason in ("densify", "escalate") or step <= self._densify_until
            if wide:
                bases = [m.frame for m in fronts] + [
                    m.seed for m in self.members if m.status is LineageStatus.DORMANT and m.seed is not None
                    and revive_eligible(m.lost_step, step, p.revive_steps)]
            else:
                bases = [m.frame for m in fronts[:1]] + \
                        [m.seed for m in self.members if m.lost_step == step and m.seed is not None]
            sr = full_search(ledger, mol, bases, self.metal_atoms, p.keep_window, tol,
                             point_charges=point_charges, retry_guesses=not active, merge_guesses=True,
                             anchor=low.result.energy if low is not None else None,
                             screen=(p.screen_cycles, p.screen_margin) if p.screen_margin is not None else None)
            events.append(("search", None))
            trace["search"] = dict(reason=reason, level1=sr.level1, level2=sr.level2, guesses=list(sr.guesses),
                                   n_states=len(sr.states), pending=sr.pending, screened=sr.screened,
                                   screened_mha=[[tag, 1e3 * de] for tag, de in sr.screened_de],
                                   n_bases=len(bases), wide=wide,
                                   merged=[[st, sv, 1e3 * de] for st, sv, de in sr.merged])
            if reason == "start":  # 2a: every converged start candidate in the window, before the dedupe
                search_frames = sr.window_frames
            if sr.pending:  # unconverged inside the window: unknown, not absent (spec 2026-10-05 §5.3)
                events.append(("pending", None))
                pending, new_pending = sr.pending, sr.pending_frames
            if not sr.states:
                events.append(("search_empty", None))
            cands += [(f, None) for f in sr.states]
            if p.densify_random and low is not None and step <= self._densify_until:
                # design memo 2026-10-23 1a: the lowest t2g orientation after a symmetric start cannot be built, and
                # about 1 in 9 random orientations reaches it (state.md §4.21, §4.23): sample it on the window steps
                rtasks = random_orientation_tasks(mol, low.frame, self.metal_atoms, p.densify_random,
                                                  np.random.default_rng(step), point_charges=point_charges)
                rres = ledger.map(rtasks)
                cands += [(r.frame, None) for r in rres if r.converged]
                trace["search"]["random"] = dict(n=len(rtasks), converged=sum(r.converged for r in rres))
        if active and self.scheduler.due(step):
            base = min(active, key=lambda m: m.result.energy).frame
            refreshed = refresh(ledger, mol, base, d_occupation_patterns(base, self.metal_atoms)[1:],
                                self.scheduler.stage2_cap(self._gap_prev, len(active)),
                                screen_cycles=p.screen_cycles, screen_window=p.screen_window,
                                same_state_sv=tol.same_state_sv, point_charges=point_charges)
            cands += [(f, None) for f in refreshed.siblings]
            trace["refresh"] = dict(log=list(refreshed.log), unknown=refreshed.unknown,
                                    merged=[[sv, 1e3 * de] for sv, de in refreshed.merged])
        revivable = [m for m in self.members if m.status is LineageStatus.DORMANT and m.seed is not None
                     and revive_eligible(m.lost_step, step, p.revive_steps)]
        rtasks = [SCFTask(mol=mol, dm0=m.engine.extrapolator.guess((m.seed,), mol), imom_ref=m.seed,
                          max_cycle=p.cap_near, point_charges=point_charges, tag="revive") for m in revivable]
        for m, r in zip(revivable, ledger.map(rtasks)):
            if r.converged:
                cands.append((r.frame, m.id))
        # Earlier searches' pending candidates (design 2026-10-11 §5.2): continue each for up to revive_steps steps
        # from its last frame (IMOM on it, cap_near, no fallback). Converged -> a candidate like any other; one that
        # is already an ACTIVE lineage is resolved without an SCF; still unconverged when it expires -> unknown.
        live = [c for c in self._pending if step - c.born <= p.revive_steps]
        expired = len(self._pending) - len(live)
        known = [c for c in live if any(self._same(m.frame, c.frame, "pending_known") for m in active)]
        carried = [c for c in live if c not in known]
        ctasks = [SCFTask(mol=mol, dm0=GrassmannGuess().guess((c.frame,), mol), imom_ref=c.frame, max_cycle=p.cap_near,
                          point_charges=point_charges, tag="pending") for c in carried]
        still: list[_Carry] = []
        for c, r in zip(carried, ledger.map(ctasks)):
            if r.converged:
                cands.append((r.frame, None))
            else:
                still.append(_Carry(r.frame, c.born))
        trace["pending"] = dict(carried=len(carried), known=len(known), converged=len(carried) - len(still),
                                expired=expired)

        # 3. admission, lowest first (spec §4.4-4.5, §5.4). Batches of up to max_lineages candidates; a failed
        #    admission does not use up a slot, so the candidates behind it get their chance in the next batch.
        emin = min([m.result.energy for m in active] + [f.energy for f, _ in cands], default=np.inf)
        queue: deque[tuple[Frame, int | None]] = deque()
        for f, rid in sorted(cands, key=lambda c: c[0].energy):
            if f.energy - emin > p.keep_window:
                break
            if any(self._same(m.frame, f, "cand_known") for m in active) or \
                    any(self._same(q, f, "cand_queue") for q, _ in queue):
                continue
            queue.append((f, rid))
        forced = None
        while queue:
            live = self._active()
            energies = {m.id: m.result.energy for m in live}
            batch: list[tuple[Frame, int | None]] = []
            while queue and len(batch) < p.max_lineages:
                f, rid = queue.popleft()
                if any(self._same(m.frame, f, "cand_known") for m in live):  # admitted from an earlier batch
                    continue
                if admission_slot(f.energy - tol.admit_de, energies, p.max_lineages, self._protected())[0] == "full":
                    events.append(("full", None))  # no slot even at the lowest energy admission may return
                    continue
                batch.append((f, rid))
            engines = [self._new_engine(f) for f, _ in batch]
            atasks = [dataclasses.replace(e.prepare(coords, point_charges,
                                                    Budget(max_cycle=p.cap, allow_fallback=False, with_gradient=False)),
                                          tag="admit") for e in engines]
            for (f, rid), e, task, res in zip(batch, engines, atasks, ledger.map(atasks)):
                r = e.finish(task, res)
                if res.converged and (forced is None or r.energy < forced[1].energy):
                    forced = (e, r, rid, res)
                stable = self._same(f, e.lineage.last_frame, "admit") and abs(r.energy - f.energy) < tol.admit_de
                if not (res.converged and stable):
                    events.append(("unstable", None))
                elif any(self._same(m.frame, e.lineage.last_frame, "duplicate") for m in self._active()):
                    events.append(("duplicate", None))  # re-converged onto a lineage, e.g. one admitted this batch
                else:
                    self._admit(e, r, rid, events, scf_result=res)
        if not self._active() and forced is not None:  # never end a step with an empty beam (spec §5.4)
            self._admit(forced[0], forced[1], forced[2], events, kind="forced", scf_result=forced[3])
        if not self._active():
            raise RuntimeError(f"step {step}: no converged electronic state")

        # 4. merge lineages on one state, keep the lower (spec §4.5)
        kept: list[_Member] = []
        for m in sorted(self._active(), key=lambda m: m.result.energy):
            if any(self._same(k.frame, m.frame, "merge") for k in kept):
                m.status = LineageStatus.PRUNED
                events.append(("merge", m.id))
            else:
                kept.append(m)
        active = kept

        # 5. select; forces only for the selected lineage (spec §4.6)
        lowest = min(active, key=lambda m: m.result.energy)
        if self.policy == "follow" and self._followed is not None:
            sel = self._member(self._followed)
            if sel is None or sel.status is not LineageStatus.ACTIVE:
                events.append(("follow_lost", self._followed))
                sel = lowest
                self._followed = sel.id
        else:
            sel = lowest
        if self._followed is None:
            self._followed = sel.id
        failures = []
        while not self.energy_only:  # energy_only: no forces, so no re-seat and nothing re-selects
            try:
                if sel.forces is None:
                    sel.forces = self._reseat_forces(sel, point_charges)
                if self.policy == "adiabatic_min":
                    # Re-seating can refine an energy within admit_de. Select again before returning E/F;
                    # each member gets forces at most once, so this loop ends after at most len(active) re-seats.
                    while sel is not (lowest := min(active, key=lambda m: m.result.energy)):
                        sel = lowest
                        if sel.forces is None:
                            sel.forces = self._reseat_forces(sel, point_charges)
                break
            except ReseatError as exc:  # spec 2026-10-05 §6.2: drop this lineage for the step, take the next lowest
                failures.append(str(exc))
                sel.status, sel.lost_step, sel.seed = LineageStatus.DORMANT, step, sel.frame
                events.append(("reseat_failed", sel.id))
                self._search_next = "reseat"
                active = [m for m in active if m is not sel]
                if not active:
                    raise RuntimeError(f"step {step}: every gradient re-seat failed: " + "; ".join(failures)) from exc
                sel = min(active, key=lambda m: m.result.energy)
                if self.policy == "follow":
                    events.append(("follow_lost", self._followed))
                    self._followed = sel.id
        switched = self._selected is not None and sel.id != self._selected
        if switched:
            events.append(("switch", sel.id))
        self._selected = sel.id
        energy_final = None
        if self.energy_only and hasattr(self.backend, "finalize") and sel.scf_result is not None:
            e_fin, conv_fin, cyc_fin = self.backend.finalize(sel.scf_result.outcome)
            trace["final"] = dict(id=sel.id, energy=e_fin, converged=conv_fin, cycles=cyc_fin)
            energy_final = e_fin if conv_fin else None

        # 6. prune with hysteresis (spec §4.5)
        emin = min(m.result.energy for m in active)  # a validated gradient re-seat may have refined its energy
        for m in active:
            if m.id == self._protected():
                m.strikes = 0
                continue
            m.strikes = next_strikes(m.strikes, m.result.energy, emin, p.prune_window)
            if m.strikes >= p.n_prune:
                m.status, m.lost_step, m.seed = LineageStatus.DORMANT, None, m.frame
                events.append(("prune", m.id))
        active = self._active()

        # 7. bookkeeping and schedule (spec §5.2)
        for m in active:
            m.gap_last, m.seed = m.result.energy - emin, m.frame
        energies = sorted(m.result.energy for m in active)
        gap = energies[1] - energies[0] if len(energies) > 1 else np.inf
        # this search's pending candidates are carried, not yet unknown: only those already an ACTIVE lineage or
        # already carried are dropped
        fresh: list[Frame] = []
        for f in new_pending:
            if not any(self._same(g, f, "pending_new") for g in [m.frame for m in active] + [c.frame for c in still] + fresh):
                fresh.append(f)
        self._pending = still + [_Carry(f, step) for f in fresh]
        trace["pending"]["alive"] = len(self._pending)
        # Unknown = a refresh candidate left unconverged, or a pending candidate that expired unresolved. A search's
        # own pending no longer counts: it is being continued, and counting it let every densified search extend
        # the window again (R1 2026-10-22, state.md §4.18: wide searches on every step 1-9).
        unknown = (refreshed.unknown if refreshed is not None else 0) + expired
        if reason is not None:
            self._unknown_streak = 1 if unknown else 0
        elif refreshed is not None:
            self._unknown_streak = self._unknown_streak + 1 if unknown else 0
        if p.escalate_unknown is not None and self._unknown_streak >= p.escalate_unknown:
            self._search_next, self._unknown_streak = "escalate", 0
            events.append(("escalate", None))
        status = status_of(energies, p.keep_window, tol.deg_band, events, tuple(a.name for a in sel.result.alarms))
        if reason == "start" and status == "DEGENERATE" and p.densify_steps > 0:
            self._densify_until = step + p.densify_steps
            trace["densify"] = dict(event="start", until=self._densify_until)
        elif step == self._densify_until:
            unresolved = bool(unknown) or any(kind in ("lost", "reseat_failed") for kind, _ in events)
            if unresolved and p.search_every is not None and (step + 1) % p.search_every != 0:
                self._densify_until += 1
                trace["densify"] = dict(event="extend", until=self._densify_until)
            else:  # resolved, or the next periodic search takes over (no periodic search: no unbounded extension)
                trace["densify"] = dict(event="end", until=step, unresolved=unresolved)
        beam_changed = any(kind in _CHANGES for kind, _ in events)
        open_unknown = pending + len(self._pending) + expired  # an unknown still pulls the next refresh (spec 10-05)
        if refreshed is not None:
            self.scheduler.record(step, gap, refreshed.unknown + open_unknown, status == "DEGENERATE", beam_changed)
        elif beam_changed or open_unknown:
            self.scheduler.poke(step)
        self._gap_prev = gap

        out = BeamResult(
            step=step,
            energy=sel.result.energy,
            forces=sel.forces,
            policy=self.policy,
            lineage_id=sel.id,
            switched=switched,
            status=status,
            manifold=tuple((m.id, m.result.energy - emin) for m in sorted(active, key=lambda m: m.result.energy)),
            lineage_energies={m.id: m.result.energy for m in active},
            lineage_status={m.id: m.status.value for m in self.members},
            events=tuple(events),
            frame=sel.frame,
            results={m.id: m.result for m in active},
            trace=trace,
            frames={m.id: m.frame for m in active},
            energy_final=energy_final,
            search_frames=search_frames,
        )
        self.step += 1
        return out
