"""Stateful QM force engine, single-lineage (FAST mode): spec §12, §19."""
from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum

import numpy as np

from sroqm.alarms import Alarm, AlarmThresholds, evaluate_alarms
from sroqm.backend import PySCFBackend
from sroqm.executor import SCFTask, TaskResult, run_task
from sroqm.extrapolate import Extrapolator, GrassmannGuess
from sroqm.fingerprint import compare, frontier_gaps
from sroqm.frame import Frame
from sroqm.lineage import Lineage
from sroqm.overlap import occupied_overlap

_CONTINUATIONS = ("imom", "aufbau")


class Status(Enum):
    FAST = "FAST"
    ALARM = "ALARM"  # Phase 3 hooks STATE_SEARCH here


@dataclass(frozen=True)
class Budget:
    """What one step of a lineage may spend (Phase-3b spec §4.2). The defaults are the plain engine's behavior."""
    max_cycle: int | None = None   # None: the backend's MethodSpec.max_cycle
    allow_fallback: bool = True    # redo an unconverged IMOM step with aufbau
    with_gradient: bool = True

    def __post_init__(self):
        if self.max_cycle is not None and self.max_cycle < 1:
            raise ValueError(f"max_cycle must be >= 1 or None, got {self.max_cycle}")


@dataclass(frozen=True, eq=False)
class Result:
    step: int
    energy: float
    forces: np.ndarray | None  # None when the step's Budget skipped the gradient
    status: Status
    lineage_id: int
    s2: float
    local_spins: dict[str, float]
    local_charges: dict[str, float]
    scf_iterations: int
    converged: bool
    overlap_det: float | None
    overlap_min_sv: float | None
    frontier_gap: float
    alarms: tuple[Alarm, ...]
    probe_energy: float | None = None  # independent scratch/aufbau SCF energy on probe steps


class Engine:
    def __init__(self, template, backend: PySCFBackend, extrapolator: Extrapolator | None = None,
                 continuation: str = "imom", fragments=None, thresholds: AlarmThresholds | None = None,
                 history_length: int = 6, initial_state=None, probe_every: int | None = None):
        """probe_every: every N steps also run an independent scratch/aufbau SCF and raise
        `lower_state_found` if it lands below the lineage (spec §10.5). Costs one extra SCF per probe."""
        if continuation not in _CONTINUATIONS:
            raise ValueError(f"continuation must be one of {_CONTINUATIONS}, got {continuation!r}")
        if template.spin != 0 and not backend.spec.unrestricted:
            raise ValueError(f"open-shell template (spin={template.spin}) needs uhf/uks, got {backend.spec.method!r}")
        if probe_every is not None and probe_every < 1:
            raise ValueError(f"probe_every must be >= 1 or None, got {probe_every}")
        self.template = template
        self.backend = backend
        self.extrapolator = extrapolator if extrapolator is not None else GrassmannGuess()
        self.continuation = continuation
        self.fragments = fragments
        self.probe_every = probe_every
        self.thresholds = thresholds if thresholds is not None else AlarmThresholds()
        self.lineage = Lineage(id=0, multiplicity=template.spin + 1, max_frames=history_length)
        self._seed = None
        if isinstance(initial_state, Frame):  # a state found at any geometry, e.g. by a beam search
            self._seed = initial_state
        elif initial_state is not None:  # (mo_coeff, mo_occ) at the template geometry
            mo_coeff, mo_occ = initial_state
            self._seed = Frame.from_orbitals(template, mo_coeff, mo_occ)
        self.step = 0

    def compute(self, coords_bohr, point_charges=None, budget: Budget | None = None) -> Result:
        task = self.prepare(coords_bohr, point_charges, budget)
        result = run_task(self.backend, task)
        probe_energy = None
        if self.probe_every is not None and self.step % self.probe_every == 0:
            probe = self.backend.run(task.mol, point_charges=point_charges, with_gradient=False)
            probe_energy = probe.energy if probe.converged else None
        return self.finish(task, result, probe_energy)

    def _history(self) -> tuple:
        return self.lineage.history or ((self._seed,) if self._seed is not None else ())

    def prepare(self, coords_bohr, point_charges=None, budget: Budget | None = None) -> SCFTask:
        """The SCF of the next step as a task; run it anywhere, then hand the result to finish()."""
        b = budget if budget is not None else Budget()
        coords = np.asarray(coords_bohr, dtype=float)
        if coords.shape != (self.template.natm, 3):
            raise ValueError(f"coords shape {coords.shape} != ({self.template.natm}, 3)")
        mol = self.template.set_geom_(coords, unit="Bohr", inplace=False)
        history = self._history()
        dm0 = self.extrapolator.guess(history, mol) if history else None
        ref = history[0] if history and self.continuation == "imom" else None
        return SCFTask(mol=mol, dm0=dm0, imom_ref=ref, max_cycle=b.max_cycle,
                       allow_fallback=b.allow_fallback and ref is not None, with_gradient=b.with_gradient,
                       point_charges=point_charges, tag="advance")

    def finish(self, task: SCFTask, result: TaskResult, probe_energy: float | None = None) -> Result:
        """Record `result` as this step. `task` must be this engine's last prepare(), with no other step in between."""
        history = self._history()
        out, mol = result.outcome, task.mol
        frame = result.frame
        fp = result.fingerprint(self.fragments)  # on the worker that keeps the SCF, if any
        prev_fp = self.lineage.last_fingerprint
        self._last_context = (history[0] if history else None, prev_fp, self.lineage.last_gaps)
        change = compare(prev_fp, fp) if prev_fp is not None else None
        overlap = occupied_overlap(history[0], frame) if history else None
        gaps = frontier_gaps(out.mo_energy, out.mo_occ)
        gap = min(gaps, key=abs)
        alarms = evaluate_alarms(converged=out.converged, cycles=result.cycles if history else None, gap=gap,
                                 change=change, overlap=overlap, thresholds=self.thresholds,
                                 gaps=gaps, prev_gaps=self.lineage.last_gaps, imom_fallback=result.fallback,
                                 probe_delta=out.energy - probe_energy if probe_energy is not None else None)
        self.lineage.push(frame, fp, gaps)
        if any(a.name == "overlap_drop" for a in alarms):
            self.lineage.cut_history()  # the old state's frames would poison the next extrapolation

        res = Result(
            step=self.step,
            energy=out.energy,
            forces=-out.gradient if out.gradient is not None else None,
            status=Status.ALARM if alarms else Status.FAST,
            lineage_id=self.lineage.id,
            s2=fp.s2,
            local_spins=fp.fragment_spins,
            local_charges=fp.fragment_charges,
            scf_iterations=result.cycles,
            converged=out.converged,
            overlap_det=overlap.det if overlap else None,
            overlap_min_sv=overlap.min_sv if overlap else None,
            frontier_gap=gap,
            alarms=alarms,
            probe_energy=probe_energy,
        )
        self.step += 1
        return res

    def finish_gradient(self, previous: Result, result: TaskResult) -> Result:
        """Replace the current step after a validated re-seat, without advancing or pushing a second frame."""
        if previous.step != self.step - 1 or previous.lineage_id != self.lineage.id:
            raise ValueError("gradient result must replace this lineage's current step")
        forces = self.forces_from(result)
        frame = result.frame
        fp = result.fingerprint(self.fragments)
        prev_frame, prev_fp, prev_gaps = self._last_context
        change = compare(prev_fp, fp) if prev_fp is not None else None
        overlap = occupied_overlap(prev_frame, frame) if prev_frame is not None else None
        gaps = frontier_gaps(result.outcome.mo_energy, result.outcome.mo_occ)
        gap = min(gaps, key=abs)
        cycles = previous.scf_iterations + result.cycles
        alarms = evaluate_alarms(
            converged=result.converged, cycles=cycles if prev_frame is not None else None, gap=gap,
            change=change, overlap=overlap, thresholds=self.thresholds, gaps=gaps, prev_gaps=prev_gaps,
            imom_fallback=any(a.name == "imom_fallback" for a in previous.alarms),
            probe_delta=frame.energy - previous.probe_energy if previous.probe_energy is not None else None,
        )
        self.lineage.frames[0] = frame
        self.lineage.fingerprints[-1] = fp
        self.lineage.energies[-1] = frame.energy
        self.lineage.gaps[-1] = gaps
        if any(a.name == "overlap_drop" for a in alarms):
            self.lineage.cut_history()
        return replace(previous, energy=frame.energy, forces=forces, s2=fp.s2,
                       local_spins=fp.fragment_spins, local_charges=fp.fragment_charges,
                       scf_iterations=cycles, frontier_gap=gap, alarms=alarms,
                       status=Status.ALARM if alarms else Status.FAST,
                       overlap_det=overlap.det if overlap else None,
                       overlap_min_sv=overlap.min_sv if overlap else None)

    def gradient_task(self, point_charges=None, max_cycle: int | None = 10) -> SCFTask:
        """Re-seat the last frame (IMOM on itself, from its own density) and compute its gradient (spec §4.6)."""
        last = self.lineage.last_frame
        if last is None:
            raise RuntimeError("no frame yet: compute a step first")
        return SCFTask(mol=last.mol, dm0=last.dm, imom_ref=last, max_cycle=max_cycle, with_gradient=True,
                       point_charges=point_charges, tag="reseat")

    @staticmethod
    def forces_from(result: TaskResult) -> np.ndarray:
        if not result.converged:
            raise RuntimeError("gradient re-seat did not converge")
        return -np.asarray(result.outcome.gradient)
