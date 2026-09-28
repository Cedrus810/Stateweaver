"""Phase-1 continuation benchmark (spec §17.1).

replay: one reference trajectory (scratch guess, conv_tol 1e-10) is generated once;
        every strategy re-evaluates the same geometries -> cycles, time, |dE|, |dF|.
        dE/dF are taken only over steps where the strategy is on the same electronic state
        as the reference (occupied-space min_sv >= 0.9); other steps are counted in
        state_mismatch_steps, since a different state is not an extrapolation error.
nve:    every strategy runs its own NVE trajectory -> cycles, energy drift.

--jobs N runs strategies in N separate processes, each with --threads threads.
Thread variables are set before numpy/pyscf are imported; lib.num_threads alone
leaves BLAS oversubscribed.

Example:
  PYTHONPATH=src $PY benchmarks/bench_continuation.py --system fe3_hexaaqua --basis def2-svp \\
      --density-fit --steps 50 --conv-tol 1e-7 --threads 10 --jobs 4 --out benchmarks/out/fe3_replay.json
"""
from __future__ import annotations

import os
import sys

_THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")


def thread_env(argv) -> dict[str, str]:
    """Thread variables for --threads; they only take effect if set before numpy/pyscf are imported."""
    n = "8"
    for i, arg in enumerate(argv):
        if arg == "--threads" and i + 1 < len(argv):
            n = argv[i + 1]
        elif arg.startswith("--threads="):
            n = arg.split("=", 1)[1]
    return {var: n for var in _THREAD_VARS}


if __name__ == "__main__":
    os.environ.update(thread_env(sys.argv[1:]))

import argparse  # noqa: E402
import json  # noqa: E402
import multiprocessing  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from pyscf import lib  # noqa: E402

from sroqm.backend import MethodSpec, PySCFBackend  # noqa: E402
from sroqm.engine import Engine  # noqa: E402
from sroqm.extrapolate import make_extrapolator  # noqa: E402
from sroqm.frame import Frame  # noqa: E402
from sroqm.md import energy_drift, run_nve  # noqa: E402
from sroqm.overlap import occupied_overlap  # noqa: E402
from sroqm.testsystems import DEFAULT_METHOD, build  # noqa: E402

STRATEGIES = ("scratch", "read", "aspc", "grassmann")
SAME_STATE_SV = 0.9


@dataclass(frozen=True)
class RunConfig:
    system: str
    basis: str
    method: str
    conv_tol: float
    density_fit: bool
    threads: int
    dt: float
    steps: int
    temperature: float


def parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--system", default="water", choices=sorted(DEFAULT_METHOD))
    p.add_argument("--basis", default="def2-svp")
    p.add_argument("--method", default=None, help="override the system default, e.g. uks/tpssh")
    p.add_argument("--mode", default="replay", choices=("replay", "nve"))
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--dt", type=float, default=0.5, help="time step in fs")
    p.add_argument("--temperature", type=float, default=300.0)
    p.add_argument("--conv-tol", type=float, default=1e-7)
    p.add_argument("--density-fit", action="store_true")
    p.add_argument("--strategies", default=",".join(STRATEGIES))
    p.add_argument("--threads", type=int, default=8, help="threads per process")
    p.add_argument("--jobs", type=int, default=1, help="strategies run in this many processes at once")
    p.add_argument("--out", default=None, help="write JSON results to this path")
    args = p.parse_args(argv)
    args.strategies = args.strategies.split(",")
    bad = [s for s in args.strategies if s not in STRATEGIES]
    if bad:
        p.error(f"unknown strategies {bad}; choose from {list(STRATEGIES)}")
    if args.jobs < 1:
        p.error("--jobs must be >= 1")
    return args


def _engine(cfg: RunConfig, conv_tol: float, strategy: str) -> Engine:
    lib.num_threads(cfg.threads)
    spec = MethodSpec(cfg.method, conv_tol=conv_tol, density_fit=cfg.density_fit)
    return Engine(build(cfg.system, cfg.basis), PySCFBackend(spec), extrapolator=make_extrapolator(strategy))


class _Recorder:
    """Pass-through provider that keeps each step's converged frame."""

    def __init__(self, engine):
        self.engine = engine
        self.frames = []

    def compute(self, x):
        res = self.engine.compute(x)
        self.frames.append(self.engine.lineage.last_frame)
        return res


def reference_trajectory(cfg: RunConfig):
    rec = _Recorder(_engine(cfg, 1e-10, "scratch"))
    mol = rec.engine.template
    log = run_nve(rec, mol.atom_coords(), mol.atom_mass_list(isotope_avg=True), cfg.dt, cfg.steps, cfg.temperature)
    return log, [(f.mo_coeff, f.mo_occ) for f in rec.frames]


def replay_strategy(cfg: RunConfig, name: str, coords, energies, forces, ref_orbitals) -> dict:
    eng = _engine(cfg, cfg.conv_tol, name)
    cycles, de, df, mismatch = [], [], [], 0
    t0 = time.perf_counter()
    for x, e_ref, f_ref, (mo_coeff, mo_occ) in zip(coords, energies, forces, ref_orbitals):
        res = eng.compute(x)
        cycles.append(res.scf_iterations)
        frame = eng.lineage.last_frame
        if occupied_overlap(Frame.from_orbitals(frame.mol, mo_coeff, mo_occ), frame).min_sv < SAME_STATE_SV:
            mismatch += 1
            continue
        de.append(abs(res.energy - e_ref))
        df.append(float(np.abs(res.forces - f_ref).max()))
    return {"mean_cycles": float(np.mean(cycles[1:])), "wall_s": time.perf_counter() - t0,
            "max_abs_dE": max(de, default=float("nan")), "max_abs_dF": max(df, default=float("nan")),
            "state_mismatch_steps": mismatch}


def nve_strategy(cfg: RunConfig, name: str) -> dict:
    eng = _engine(cfg, cfg.conv_tol, name)
    mol = eng.template
    t0 = time.perf_counter()
    log = run_nve(eng, mol.atom_coords(), mol.atom_mass_list(isotope_avg=True), cfg.dt, cfg.steps, cfg.temperature)
    return {"mean_cycles": float(np.mean(log.scf_iterations[1:])), "wall_s": time.perf_counter() - t0,
            "drift_Ha_per_ps_per_atom": energy_drift(log) / mol.natm,
            "alarm_steps": sum(s != "FAST" for s in log.statuses)}


def _run_all(fn, jobs: int, per_strategy_args: dict) -> dict:
    if jobs == 1:
        return {name: fn(*a) for name, a in per_strategy_args.items()}
    # spawn, not fork: forking a process whose OpenMP/BLAS pools are already running can deadlock
    with ProcessPoolExecutor(max_workers=jobs, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = {name: pool.submit(fn, *a) for name, a in per_strategy_args.items()}
        return {name: f.result() for name, f in futures.items()}


def format_table(rows) -> str:
    cols = list(next(iter(rows.values())))
    lines = ["| strategy | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for name, row in rows.items():
        lines.append(f"| {name} | " + " | ".join(f"{row[c]:.4g}" for c in cols) + " |")
    return "\n".join(lines)


def main(argv=None) -> dict:
    args = parse_args(argv)
    cfg = RunConfig(args.system, args.basis, args.method or DEFAULT_METHOD[args.system], args.conv_tol,
                    args.density_fit, args.threads, args.dt, args.steps, args.temperature)
    if args.jobs > 1:
        os.environ.update(thread_env(["--threads", str(args.threads)]))  # inherited by spawned workers
    lib.num_threads(args.threads)
    if args.mode == "replay":
        ref, ref_orbitals = reference_trajectory(cfg)
        rows = _run_all(replay_strategy, args.jobs,
                        {name: (cfg, name, ref.coords, ref.potential, ref.forces, ref_orbitals)
                         for name in args.strategies})
    else:
        rows = _run_all(nve_strategy, args.jobs, {name: (cfg, name) for name in args.strategies})
    result = {"system": args.system, "method": cfg.method, "basis": args.basis, "mode": args.mode,
              "steps": args.steps, "conv_tol": args.conv_tol, "rows": rows}
    print(format_table(rows))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    main()
