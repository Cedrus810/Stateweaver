"""One SCF as a self-contained task, and the executors that run a batch of them (Phase-3b spec §8.1; process
executor: performance design 2026-10-03 §9)."""
from __future__ import annotations

import dataclasses
import os
import traceback
from collections import deque
from dataclasses import dataclass, field
from functools import cached_property
from typing import Protocol, Sequence

import numpy as np

from sroqm.backend import MethodSpec, PySCFBackend, SCFOutcome
from sroqm.fingerprint import Fingerprint, compute_fingerprint
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
    handle: object = field(default=None, repr=False)  # ProcessExecutor: the worker that keeps this SCF's mf

    @property
    def converged(self) -> bool:
        return self.outcome.converged

    @cached_property
    def frame(self) -> Frame:
        return Frame.from_outcome(self.mol, self.outcome)

    def fingerprint(self, fragments=None) -> Fingerprint:
        """Needs the SCF object: computed here, or by the worker that keeps it."""
        if self.outcome.mf is not None:
            return compute_fingerprint(self.outcome.mf, fragments)
        if self.handle is None:
            raise RuntimeError("no SCF object for the fingerprint: neither local nor on a worker")
        return self.handle.fingerprint(fragments)


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


# ---- single-node process executor (performance design 2026-10-03 §9) ------------------------------------------

def _rebuild_mole(cls, state):
    mol = cls.__new__(cls)
    mol.__dict__.update(state)
    return mol


def _reduce_mole(mol):
    """PySCF pickles a Mole through its JSON dumps(), which turns the empty (0, 8) `_ecpbas` of an all-electron
    molecule into shape (0,); the `atom` initial guess then fails (IndexError in atom_hf). Send the arrays as they
    are. Registered for multiprocessing only; plain pickle keeps PySCF's behavior."""
    return _rebuild_mole, (type(mol), dict(mol.__dict__))


def _register_mole_reducer() -> None:
    from multiprocessing.reduction import ForkingPickler
    from pyscf import gto
    ForkingPickler.register(gto.Mole, _reduce_mole)


_register_mole_reducer()  # in the driver (sends tasks) and in every worker (imports this module)

def _worker(conn, spec: MethodSpec, cpus) -> None:
    """Worker loop. Keeps each task's SCF object (by key) until the driver releases the step's results, so the
    selected lineage's gradient and fingerprints are computed here on the converged mf (design §9.2)."""
    if cpus is not None:
        os.sched_setaffinity(0, cpus)
    from pyscf import lib
    lib.num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))
    backend, kept = PySCFBackend(spec), {}
    while True:
        try:
            msg = conn.recv()
        except EOFError:
            return
        op = msg[0]
        if op == "stop":
            return
        try:
            if op == "run":
                _, key, task = msg
                r = run_task(backend, task)
                kept[key] = r.outcome.mf
                reply = (dataclasses.replace(r.outcome, mf=None), r.cycles, r.fallback)
            elif op == "grad":
                mf = kept[msg[1]]
                if not mf.converged:
                    raise RuntimeError("cannot compute a reused gradient from an unconverged SCF")
                reply = np.asarray(mf.nuc_grad_method().kernel())
            elif op == "fp":
                reply = compute_fingerprint(kept[msg[1]], msg[2])
            elif op == "release":
                kept.clear()
                reply = None
            else:
                raise ValueError(f"unknown op {op!r}")
            conn.send(("ok", reply))
        except Exception:
            conn.send(("err", traceback.format_exc()))


@dataclass(frozen=True, eq=False)
class _Handle:
    executor: "ProcessExecutor"
    worker: int
    key: int
    generation: int

    def gradient(self) -> np.ndarray:
        return self.executor._call(self, "grad")

    def fingerprint(self, fragments=None) -> Fingerprint:
        return self.executor._call(self, "fp", fragments)


def split_cpus(n_workers: int, threads: int, cpus=None) -> list[list[int]]:
    """Consecutive blocks of `threads` CPUs, one per worker, from `cpus` (default: this process's affinity, so a
    run under `numactl --physcpubind=0-39` gives 0-9 / 10-19 / 20-29 / 30-39: one socket per pair of workers)."""
    allowed = sorted(cpus if cpus is not None else os.sched_getaffinity(0))
    if n_workers * threads > len(allowed):
        raise ValueError(f"{n_workers} workers x {threads} threads > {len(allowed)} CPUs available")
    return [allowed[i * threads:(i + 1) * threads] for i in range(n_workers)]


class ProcessExecutor:
    """Runs a batch of tasks on long-lived worker processes on this node, in parallel.

    Each worker is spawned once, pinned to its own block of `threads` CPUs (`OMP_NUM_THREADS=threads`,
    `OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1`: state.md §3) and keeps its PySCF import. map() hands each task to the
    next idle worker and returns the results in task order (spec §8.1). A result carries a handle to the worker
    that keeps its SCF object: its gradient and fingerprint are computed there; release() drops every kept object
    and invalidates the handles (call it at the end of each step). A worker exception is raised in map() after the
    tasks already running have finished; a dead worker makes the executor unusable.
    """

    def __init__(self, spec: MethodSpec, n_workers: int, threads: int = 1, pin: bool = True):
        import multiprocessing as mp
        from multiprocessing.connection import wait
        if n_workers < 1 or threads < 1:
            raise ValueError("need n_workers >= 1 and threads >= 1")
        self.spec, self.n_workers, self.threads = spec, n_workers, threads
        self._wait = wait
        cpus = split_cpus(n_workers, threads) if pin else [None] * n_workers
        ctx = mp.get_context("spawn")
        env = dict(OMP_NUM_THREADS=str(threads), OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
        saved = {k: os.environ.get(k) for k in env}
        os.environ.update(env)  # a spawned child takes the parent's environment when it starts
        self._procs, self._conns = [], []
        try:
            for c in cpus:
                parent, child = ctx.Pipe()
                p = ctx.Process(target=_worker, args=(child, spec, c), daemon=True)
                p.start()
                child.close()
                self._procs.append(p)
                self._conns.append(parent)
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
        self._key, self._generation, self._dead = 0, 0, set()

    def _check(self) -> None:
        if self._dead:
            raise RuntimeError(f"worker(s) {sorted(self._dead)} died; the executor cannot continue")

    def map(self, tasks: Sequence[SCFTask]) -> list[TaskResult]:
        self._check()
        tasks = list(tasks)
        out: list[TaskResult | None] = [None] * len(tasks)
        queue, busy, idle, error = deque(range(len(tasks))), {}, list(range(self.n_workers))[::-1], None
        index = {id(c): w for w, c in enumerate(self._conns)}
        while busy or (queue and error is None):
            while queue and idle and error is None:
                w, i = idle.pop(), queue.popleft()
                self._key += 1
                self._conns[w].send(("run", self._key, tasks[i]))
                busy[w] = (i, self._key)
            for conn in self._wait([self._conns[w] for w in busy]):
                w = index[id(conn)]
                i, key = busy.pop(w)
                try:
                    status, payload = conn.recv()
                except EOFError:
                    self._dead.add(w)
                    error = error or f"worker {w} died while running task {i} ({tasks[i].tag})"
                    continue
                idle.append(w)
                if status == "err":
                    error = error or f"task {i} ({tasks[i].tag}) failed on worker {w}:\n{payload}"
                    continue
                outcome, cycles, fallback = payload
                out[i] = TaskResult(tasks[i].mol, outcome, cycles, fallback,
                                    handle=_Handle(self, w, key, self._generation))
        if error is not None:
            raise RuntimeError(error)
        return out

    def _call(self, handle: _Handle, op: str, *args):
        if handle.executor is not self or handle.generation != self._generation:
            raise RuntimeError("stale handle: its SCF object was released at the end of an earlier step")
        self._check()
        conn = self._conns[handle.worker]
        conn.send((op, handle.key, *args))
        status, payload = conn.recv()
        if status == "err":
            raise RuntimeError(f"{op} on worker {handle.worker} failed:\n{payload}")
        return payload

    def release(self) -> None:
        """Drop every kept SCF object; earlier handles become stale."""
        self._generation += 1
        for w, conn in enumerate(self._conns):
            if w in self._dead:
                continue
            conn.send(("release",))
            status, payload = conn.recv()
            if status == "err":
                raise RuntimeError(f"release on worker {w} failed:\n{payload}")

    def close(self) -> None:
        for w, (p, conn) in enumerate(zip(self._procs, self._conns)):
            if w not in self._dead and p.is_alive():
                try:
                    conn.send(("stop",))
                except (BrokenPipeError, OSError):
                    pass
            p.join(timeout=10)
            if p.is_alive():
                p.terminate()
            conn.close()
        self._procs, self._conns = [], []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
