"""Phase-1 continuation benchmark (spec §17.1).

replay: one reference trajectory (scratch guess, conv_tol 1e-10) is generated once;
        every strategy re-evaluates the same geometries -> cycles, time, |dE|, |dF|.
nve:    every strategy runs its own NVE trajectory -> cycles, energy drift.

Example:
  PYTHONPATH=src $PY benchmarks/bench_continuation.py --system fe_hexaaqua --basis def2-svp \\
      --density-fit --steps 50 --conv-tol 1e-7 --threads 32 --out benchmarks/out/fe_replay.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from pyscf import lib

from sroqm.backend import MethodSpec, PySCFBackend
from sroqm.engine import Engine
from sroqm.extrapolate import make_extrapolator
from sroqm.md import energy_drift, run_nve
from sroqm.testsystems import DEFAULT_METHOD, build

STRATEGIES = ("scratch", "read", "aspc", "grassmann")


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
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--out", default=None, help="write JSON results to this path")
    args = p.parse_args(argv)
    args.strategies = args.strategies.split(",")
    bad = [s for s in args.strategies if s not in STRATEGIES]
    if bad:
        p.error(f"unknown strategies {bad}; choose from {list(STRATEGIES)}")
    return args


def _engine(mol, args, conv_tol, strategy):
    spec = MethodSpec(args.method or DEFAULT_METHOD[args.system], conv_tol=conv_tol, density_fit=args.density_fit)
    return Engine(mol, PySCFBackend(spec), extrapolator=make_extrapolator(strategy))


def run_replay(mol, args):
    masses = mol.atom_mass_list(isotope_avg=True)
    ref = run_nve(_engine(mol, args, 1e-10, "scratch"), mol.atom_coords(), masses, args.dt, args.steps,
                  args.temperature)
    rows = {}
    for name in args.strategies:
        eng = _engine(mol, args, args.conv_tol, name)
        cycles, de, df = [], [], []
        t0 = time.perf_counter()
        for x, e_ref, f_ref in zip(ref.coords, ref.potential, ref.forces):
            r = eng.compute(x)
            cycles.append(r.scf_iterations)
            de.append(abs(r.energy - e_ref))
            df.append(float(np.abs(r.forces - f_ref).max()))
        rows[name] = {"mean_cycles": float(np.mean(cycles[1:])), "wall_s": time.perf_counter() - t0,
                      "max_abs_dE": max(de), "max_abs_dF": max(df)}
    return rows


def run_nve_mode(mol, args):
    masses = mol.atom_mass_list(isotope_avg=True)
    rows = {}
    for name in args.strategies:
        t0 = time.perf_counter()
        log = run_nve(_engine(mol, args, args.conv_tol, name), mol.atom_coords(), masses, args.dt, args.steps,
                      args.temperature)
        rows[name] = {"mean_cycles": float(np.mean(log.scf_iterations[1:])), "wall_s": time.perf_counter() - t0,
                      "drift_Ha_per_ps_per_atom": energy_drift(log) / mol.natm,
                      "alarm_steps": sum(s != "FAST" for s in log.statuses)}
    return rows


def format_table(rows) -> str:
    cols = list(next(iter(rows.values())))
    lines = ["| strategy | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for name, row in rows.items():
        lines.append(f"| {name} | " + " | ".join(f"{row[c]:.4g}" for c in cols) + " |")
    return "\n".join(lines)


def main(argv=None) -> dict:
    args = parse_args(argv)
    lib.num_threads(args.threads)
    mol = build(args.system, args.basis)
    rows = run_replay(mol, args) if args.mode == "replay" else run_nve_mode(mol, args)
    result = {"system": args.system, "method": args.method or DEFAULT_METHOD[args.system], "basis": args.basis,
              "mode": args.mode, "steps": args.steps, "conv_tol": args.conv_tol, "rows": rows}
    print(format_table(rows))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    main()
