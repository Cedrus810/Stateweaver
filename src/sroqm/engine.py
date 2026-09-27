"""Stateful QM force engine, single-lineage (FAST mode): spec §12, §19."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from sroqm.alarms import Alarm, AlarmThresholds, evaluate_alarms
from sroqm.backend import PySCFBackend
from sroqm.extrapolate import Extrapolator, GrassmannGuess
from sroqm.fingerprint import compare, compute_fingerprint, frontier_gap
from sroqm.frame import Frame
from sroqm.imom import make_imom_hook
from sroqm.lineage import Lineage
from sroqm.overlap import occupied_overlap

_CONTINUATIONS = ("imom", "aufbau")


class Status(Enum):
    FAST = "FAST"
    ALARM = "ALARM"  # Phase 3 hooks STATE_SEARCH here


@dataclass(frozen=True, eq=False)
class Result:
    step: int
    energy: float
    forces: np.ndarray
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


class Engine:
    def __init__(self, template, backend: PySCFBackend, extrapolator: Extrapolator | None = None,
                 continuation: str = "imom", fragments=None, thresholds: AlarmThresholds | None = None,
                 history_length: int = 6, initial_state=None):
        if continuation not in _CONTINUATIONS:
            raise ValueError(f"continuation must be one of {_CONTINUATIONS}, got {continuation!r}")
        if template.spin != 0 and not backend.spec.unrestricted:
            raise ValueError(f"open-shell template (spin={template.spin}) needs uhf/uks, got {backend.spec.method!r}")
        self.template = template
        self.backend = backend
        self.extrapolator = extrapolator if extrapolator is not None else GrassmannGuess()
        self.continuation = continuation
        self.fragments = fragments
        self.thresholds = thresholds if thresholds is not None else AlarmThresholds()
        self.lineage = Lineage(id=0, multiplicity=template.spin + 1, max_frames=history_length)
        self._seed = None
        if initial_state is not None:
            mo_coeff, mo_occ = initial_state
            self._seed = Frame.from_orbitals(template, mo_coeff, mo_occ)
        self.step = 0

    def compute(self, coords_bohr, point_charges=None) -> Result:
        coords = np.asarray(coords_bohr, dtype=float)
        if coords.shape != (self.template.natm, 3):
            raise ValueError(f"coords shape {coords.shape} != ({self.template.natm}, 3)")
        mol = self.template.set_geom_(coords, unit="Bohr", inplace=False)

        history = self.lineage.history or ((self._seed,) if self._seed is not None else ())
        dm0 = self.extrapolator.guess(history, mol) if history else None
        hook = make_imom_hook(history[0], mol) if history and self.continuation == "imom" else None
        out = self.backend.run(mol, dm0=dm0, occ_hook=hook, point_charges=point_charges)

        frame = Frame.from_outcome(mol, out)
        fp = compute_fingerprint(out.mf, self.fragments)
        prev_fp = self.lineage.last_fingerprint
        change = compare(prev_fp, fp) if prev_fp is not None else None
        overlap = occupied_overlap(history[0], frame) if history else None
        gap = frontier_gap(out.mo_energy, out.mo_occ)
        alarms = evaluate_alarms(converged=out.converged, cycles=out.cycles, gap=gap, change=change,
                                 overlap=overlap, thresholds=self.thresholds)
        self.lineage.push(frame, fp)

        result = Result(
            step=self.step,
            energy=out.energy,
            forces=-out.gradient,
            status=Status.ALARM if alarms else Status.FAST,
            lineage_id=self.lineage.id,
            s2=fp.s2,
            local_spins=fp.fragment_spins,
            local_charges=fp.fragment_charges,
            scf_iterations=out.cycles,
            converged=out.converged,
            overlap_det=overlap.det if overlap else None,
            overlap_min_sv=overlap.min_sv if overlap else None,
            frontier_gap=gap,
            alarms=alarms,
        )
        self.step += 1
        return result
