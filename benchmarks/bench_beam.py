"""Phase-3b acceptance replay (spec §9.2): run BeamEngine along a trajectory and judge every step against ref3.

Each reference step is an .npz with the energies of every state found and the lowest state's orbitals
(on the 40-core node: ~/sroqm/out/ref3/<prefix>_<k:03d>.npz). A step passes if the selected lineage is the reference's
state (occupied sigma_min >= same_state_sv), else if it is within accept_de of the best known energy, else if
the beam flagged DEGENERATE and it is within deg_band (spec §3). An `energy` pass more than WARN_DE_MHA above the
best known is reported as a warning and fails the run.

Exit code: 0 every step passed without warning; 3 a miss or a warning (the run finished; see SUMMARY); anything else
(1 for an uncaught exception) means the replay itself failed (review 2026-10-23 #4: 1 used to mean both).

usage: python benchmarks/bench_beam.py --coords traj2_coords.npy --ref ~/sroqm/out/ref3 --prefix traj2 --out r.json

Energy-only replay with an r2scan_jax backend (r2scan_jax spec 2026-10-11 §3.6; needs r2scan_jax/src and
pyscf_wb97mv on PYTHONPATH): --backend r2scan|cosx-ref --a 0.25 --energy-only [--save-frames DIR]. Without --ref
the steps are not judged; --save-frames writes the selected state of every step (orbitals, occupations, energies)
so that two replays can be compared step by step.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import resource
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from sroqm.backend import MethodSpec, PySCFBackend
from sroqm.beam import BeamEngine
from sroqm.beamrules import BeamParams
from sroqm.executor import ProcessExecutor
from sroqm.frame import Frame
from sroqm.overlap import occupied_overlap
from sroqm.testsystems import build
from sroqm.tolerances import Tolerances


@dataclass(frozen=True)
class StepVerdict:
    passed: bool
    reason: str        # same-state | energy | degenerate-band | miss
    sigma_min: float
    de_mha: float      # selected minus best known (min of selected and reference), mHa


GATE_FAILED = 3     # exit code: the replay finished but missed or warned (1 is Python's code for an uncaught exception)
WARN_DE_MHA = 0.02  # same-state convergence noise at conv_tol=1e-7 is <= 0.016 mHa (state.md §4.8)


def is_warn(v: StepVerdict) -> bool:
    """An energy-only pass well above the noise: traj2 step 5 hid a lost state this way (spec 2026-10-05 §3.2)."""
    return v.reason == "energy" and v.de_mha > WARN_DE_MHA


def judge_step(selected: Frame, degenerate: bool, reference: Frame, tol: Tolerances) -> StepVerdict:
    sv = occupied_overlap(selected, reference).min_sv
    de = selected.energy - min(selected.energy, reference.energy)
    if sv >= tol.same_state_sv:
        reason = "same-state"
    elif de <= tol.accept_de:
        reason = "energy"
    elif degenerate and de <= tol.deg_band:
        reason = "degenerate-band"
    else:
        reason = "miss"
    return StepVerdict(reason != "miss", reason, float(sv), 1e3 * de)


def load_reference(path, template) -> Frame:
    z = np.load(path)
    mol = template.set_geom_(z["coords"], unit="Bohr", inplace=False)
    return Frame.from_orbitals(mol, z["mo_coeff"], z["mo_occ"], float(z["energies"][0]))


def save_frame(path, res) -> None:
    f = res.frame
    np.savez(path, coords=f.mol.atom_coords(), mo_coeff=f.mo_coeff, mo_occ=f.mo_occ, energy=res.energy,
             energy_final=np.nan if res.energy_final is None else res.energy_final, status=res.status,
             manifold=np.array([m[1] for m in res.manifold]))


def peak_rss_mb() -> float:
    """Peak RSS of this process so far, MiB (Linux ru_maxrss is KiB; spec §9 memory requirement, state.md §4.16)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def rss_mb() -> float:
    """Current resident set size of this process, MiB (/proc/self/status VmRSS, kB); NaN if unavailable."""
    try:
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return float(line.split()[1]) / 1024
    except OSError:
        pass
    return float("nan")


def replay(beam: BeamEngine, coords_list, ref_dir, prefix: str, tol: Tolerances, save_dir=None,
           rows_path=None) -> list[dict]:
    """rows_path: also append every row to this JSON-lines file as soon as its step finishes, so a run that is
    stopped or crashes keeps the trace of the steps it completed (R1 2026-10-22 lost all of it)."""
    rows = []
    if rows_path is not None:
        Path(rows_path).write_text("")

    def emit(row):
        rows.append(row)
        if rows_path is not None:
            with open(rows_path, "a") as fh:
                fh.write(json.dumps(row) + "\n")

    for k, x in enumerate(coords_list):
        t = time.perf_counter()
        res = beam.compute(x)
        seconds = time.perf_counter() - t
        if save_dir is not None:
            save_frame(Path(save_dir) / f"{prefix}_{k:03d}.npz", res)
            if res.search_frames:  # design memo 2026-10-23 §2.2 2a: the start search's converged candidates
                fs = res.search_frames
                np.savez(Path(save_dir) / f"{prefix}_{k:03d}_search.npz", coords=fs[0].mol.atom_coords(),
                         energies=np.array([f.energy for f in fs]), mo_coeff=np.array([f.mo_coeff for f in fs]),
                         mo_occ=np.array([f.mo_occ for f in fs]))
        if ref_dir is None:
            emit(dict(step=k, energy=res.energy, energy_final=res.energy_final, status=res.status,
                             events=[list(e) for e in res.events], manifold=[list(m) for m in res.manifold],
                             seconds=seconds, passed=True, warn=False, trace=res.trace,
                             peak_rss_mb=peak_rss_mb(), rss_mb=rss_mb()))
            fin = "" if res.energy_final is None else f" final={res.energy_final:.10f}"
            print(f"step {k:3d}: E={res.energy:.10f}{fin} {res.status:10s} {seconds:6.1f}s events={res.events}",
                  flush=True)
            continue
        ref = load_reference(Path(ref_dir) / f"{prefix}_{k:03d}.npz", beam.template)
        v = judge_step(res.frame, res.status == "DEGENERATE", ref, tol)
        emit(dict(step=k, energy=res.energy, status=res.status, events=[list(e) for e in res.events],
                         manifold=[list(m) for m in res.manifold], seconds=seconds, peak_rss_mb=peak_rss_mb(), rss_mb=rss_mb(),
                         passed=v.passed,
                         reason=v.reason, sigma_min=v.sigma_min, de_mha=v.de_mha, warn=is_warn(v), trace=res.trace,
                         e_ref=ref.energy, energy_final=res.energy_final,
                         lineages=[dict(id=i, de_ref_mha=1e3 * (e - ref.energy),
                                        sv_ref=occupied_overlap(res.frames[i], ref).min_sv)
                                   for i, e in res.lineage_energies.items()]))
        reason = v.reason + (" WARN" if is_warn(v) else "")
        print(f"step {k:3d}: {reason:15s} sigma_min={v.sigma_min:.3f} dE={v.de_mha:+.4f} mHa {res.status:10s} "
              f"{seconds:6.1f}s events={res.events}", flush=True)
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--coords", required=True)
    ap.add_argument("--ref", default=None, help="ref3 directory; omit to replay without judging")
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--system", default="fe2_hexaaqua")
    ap.add_argument("--basis", default="def2-svp")
    ap.add_argument("--method", default="uks/b3lyp")
    ap.add_argument("--conv-tol", type=float, default=Tolerances().conv_tol)
    ap.add_argument("--metal-atoms", default="0", help="comma-separated atom indices; empty for none")
    ap.add_argument("--backend", choices=("pyscf", "r2scan", "cosx-ref"), default="pyscf",
                    help="pyscf: PySCFBackend(--method); r2scan / cosx-ref: r2scan_jax R2scanBackend / its stock "
                         "FP64 COSX reference at --a (energy only)")
    ap.add_argument("--a", type=float, default=None, help="exact-exchange fraction for --backend r2scan|cosx-ref")
    ap.add_argument("--energy-only", action="store_true", help="no gradients (BeamEngine energy_only)")
    ap.add_argument("--save-frames", default=None, help="directory for the selected state of every step")
    ap.add_argument("--densify-steps", type=int, default=0,
                    help="full search on each of the n steps after a DEGENERATE start (BeamParams.densify_steps)")
    ap.add_argument("--escalate-unknown", type=int, default=None,
                    help="n discovery steps in a row with unknowns -> full search (BeamParams.escalate_unknown)")
    ap.add_argument("--screen-margin-mha", type=float, default=None,
                    help="screen search candidates at keep_window + this many mHa (BeamParams.screen_margin)")
    ap.add_argument("--densify-random", type=int, default=0,
                    help="random t2g-orientation SCFs on each densify-window step (BeamParams.densify_random)")
    ap.add_argument("--workers", type=int, default=0,
                    help="run the SCFs on this many worker processes (ProcessExecutor); 0: serial in this process")
    ap.add_argument("--worker-threads", type=int, default=1,
                    help="OpenMP threads per worker; workers are pinned to consecutive blocks of this process's CPUs")
    args = ap.parse_args(argv)

    tol = Tolerances(conv_tol=args.conv_tol)
    mol = build(args.system, args.basis)
    if args.backend == "pyscf":
        backend = PySCFBackend(MethodSpec(args.method, conv_tol=tol.conv_tol, density_fit=args.system != "water"))
    else:
        if args.a is None or not args.energy_only:
            ap.error(f"--backend {args.backend} needs --a and --energy-only")
        if args.backend == "r2scan":
            from r2scan_jax.backend import R2scanBackend, R2scanSpec
            backend = R2scanBackend(R2scanSpec(a=args.a, conv_tol=tol.conv_tol))
        else:
            from r2scan_jax.reference import ReferenceBackend, ReferenceSpec
            backend = ReferenceBackend(ReferenceSpec(a=args.a, conv_tol=tol.conv_tol))
    if args.save_frames is not None:
        Path(args.save_frames).mkdir(parents=True, exist_ok=True)
    metals = [int(i) for i in args.metal_atoms.split(",") if i.strip()]
    coords = np.load(args.coords)[: args.steps]
    params = BeamParams(densify_steps=args.densify_steps, escalate_unknown=args.escalate_unknown,
                        densify_random=args.densify_random,
                        screen_margin=None if args.screen_margin_mha is None else 1e-3 * args.screen_margin_mha)
    executor = None
    if args.workers:
        if args.backend != "pyscf":
            ap.error("--workers needs --backend pyscf")
        executor = ProcessExecutor(backend.spec, args.workers, args.worker_threads)
    try:
        rows = replay(BeamEngine(mol, backend, metal_atoms=metals, tol=tol, energy_only=args.energy_only,
                                 params=params, executor=executor),
                      coords, args.ref, args.prefix, tol, save_dir=args.save_frames,
                      rows_path=Path(args.out).with_suffix(".rows.jsonl"))
    finally:
        if executor is not None:
            executor.close()
    t = np.array([r["seconds"] for r in rows])
    summary = dict(steps=len(rows), miss=sum(not r["passed"] for r in rows),
                   missed=[r["step"] for r in rows if not r["passed"]],
                   warn=sum(r["warn"] for r in rows), warned=[r["step"] for r in rows if r["warn"]],
                   mean_s=float(t.mean()), median_s=float(np.median(t)), max_s=float(t.max()))
    run = dict(workers=args.workers, worker_threads=args.worker_threads if args.workers else None)
    Path(args.out).write_text(json.dumps(dict(params=dataclasses.asdict(params), run=run, summary=summary, rows=rows),
                                         indent=1))
    print(f"SUMMARY {summary}", flush=True)
    return GATE_FAILED if summary["miss"] or summary["warn"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
