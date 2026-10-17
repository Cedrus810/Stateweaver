"""One SCF as a self-contained task, and the executor that runs a batch of them (Phase-3b spec §8.1)."""
from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Protocol, Sequence

import numpy as np

from sroqm.backend import PySCFBackend, SCFOutcome
from sroqm.frame import Frame
from sroqm.imom import make_imom_hook


@dataclass(frozen=True, eq=False)
class SCFTask:
    mol: object
    dm0: np.ndarray | None = None
    imom_ref: Frame | None = None     # pin occupations to this frame (cross-geometry IMOM)
    max_cycle: int | None = None      # None: the backend's MethodSpec.max_cycle
    allow_fallback: bool = False      # redo an unconverged IMOM run with aufbau from the same dm0
    with_gradient: bool = False
    init_guess: str | None = None     # PySCF init-guess key, used only without dm0
    point_charges: object = None
    tag: str = ""                     # where the task came from, for the per-step ledger (e.g. "advance", "l2")

    def __post_init__(self):
        if self.max_cycle is not None and self.max_cycle < 1:
            raise ValueError(f"max_cycle must be >= 1 or None, got {self.max_cycle}")
        if self.allow_fallback and self.imom_ref is None:
            raise ValueError("allow_fallback needs an IMOM task (imom_ref)")


@dataclass(frozen=True, eq=False)
class TaskResult:
    mol: object
    outcome: SCFOutcome
    cycles: int       # including an aufbau retry
    fallback: bool    # IMOM did not converge and the aufbau retry did

    @property
    def converged(self) -> bool:
        return self.outcome.converged

    @cached_property
    def frame(self) -> Frame:
        return Frame.from_outcome(self.mol, self.outcome)


def run_task(backend: PySCFBackend, task: SCFTask) -> TaskResult:
    hook = make_imom_hook(task.imom_ref, task.mol) if task.imom_ref is not None else None
    kw = dict(point_charges=task.point_charges, with_gradient=task.with_gradient, max_cycle=task.max_cycle)
    out = backend.run(task.mol, dm0=task.dm0, occ_hook=hook, init_guess=task.init_guess, **kw)
    cycles, fallback = out.cycles, False
    if hook is not None and not out.converged and task.allow_fallback:
        retry = backend.run(task.mol, dm0=task.dm0, **kw)
        cycles += retry.cycles
        if retry.converged:
            out, fallback = retry, True
    return TaskResult(task.mol, out, cycles, fallback)


class Executor(Protocol):
    def map(self, tasks: Sequence[SCFTask]) -> list[TaskResult]: ...


class SerialExecutor:
    """Runs the tasks one after another in this process. A cross-host executor implements the same map()."""

    def __init__(self, backend: PySCFBackend):
        self.backend = backend

    def map(self, tasks: Sequence[SCFTask]) -> list[TaskResult]:
        return [run_task(self.backend, t) for t in tasks]
