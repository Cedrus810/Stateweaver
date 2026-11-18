"""Coordination replay along a fixed path (Stateweaver design docs/coordination-electronic-lineage-research-design.zh-CN.md
§7, §9 V1, §10.1; WP3 entry): the beam runs along the path as in bench_beam (the observer changes nothing in it), and
every ACTIVE lineage's bonding descriptors are recorded; coordination events are then derived offline from the saved
observations, so detector variants and thresholds are compared on the same electronic frames.

  run     beam + observer along COORDS; writes OUT_DIR/obs.jsonl (one line per step: selected lineage, status, beam
          events, and per ACTIVE lineage its energy, converged / continuity-doubtful flags and its pairs'
          BondObservation's), OUT_DIR/frames/<prefix>_<k>_beam.npz (bench_beam.save_branches), OUT_DIR/run.json
          (params, pairs) and the source snapshot OUT_DIR/run.src/ (bench_beam.snapshot_source). Not resumable.
  events  replays obs.jsonl through coordination.CoordinationMonitor with the given thresholds; writes the events
          as JSON lines and the monitor's final state.

usage: python benchmarks/bench_coordination.py run --system fe2_hexaaqua --coords C.npy --out-dir D [--prefix scan]
           [--steps N] [beam options as bench_beam] [--workers 4 --worker-threads 10]
       python benchmarks/bench_coordination.py events --obs D/obs.jsonl --thresholds '{"Fe-O": [r_on, r_off, m_on, m_off, n]}'
           --out D/events.jsonl
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path

import numpy as np

from bench_beam import save_branches, snapshot_source
from sroqm.backend import MethodSpec, PySCFBackend
from sroqm.beam import BeamEngine
from sroqm.beamrules import BeamParams
from sroqm.bonding import BondObservation, PairSpec, observe_beam
from sroqm.coordination import UNCERTAIN_ALARMS, BondThresholds, CoordinationMonitor
from sroqm.executor import ProcessExecutor
from sroqm.testsystems import build
from sroqm.tolerances import Tolerances


def metal_donor_pairs(mol, metal: int = 0, donors=("O", "N")) -> list[PairSpec]:
    """Every metal-donor pair of the fixed QM atom list, ids <Metal>-<Donor><atom index>."""
    sym = [mol.atom_symbol(i) for i in range(mol.natm)]
    return [PairSpec(f"{sym[metal]}-{sym[i]}{i}", (metal, i), f"{sym[metal]}-{sym[i]}")
            for i in range(mol.natm) if i != metal and sym[i] in donors]


def step_record(k, res, pairs) -> dict:
    obs = observe_beam(res, pairs, k)
    return dict(step=k, selected=res.lineage_id, status=res.status, events=[list(e) for e in res.events],
                lineages=[dict(id=lid, energy=res.lineage_energies[lid], converged=bool(res.results[lid].converged),
                               uncertain=any(a.name in UNCERTAIN_ALARMS for a in res.results[lid].alarms),
                               observations=[dataclasses.asdict(o) for o in obs[lid]])
                          for lid in sorted(obs, key=lambda i: res.lineage_energies[i])])


def run(args) -> int:
    out = Path(args.out_dir)
    (out / "frames").mkdir(parents=True, exist_ok=True)
    source = snapshot_source(out / "run.json")
    tol = Tolerances()
    mol = build(args.system, args.basis)
    pairs = metal_donor_pairs(mol, args.metal)
    params = BeamParams(densify_steps=args.densify_steps, escalate_unknown=args.escalate_unknown,
                        densify_random=args.densify_random, shadow_sv=args.shadow_sv,
                        shadow_per_frame=not args.shadow_all, shadow_skip_densify=args.shadow_skip_densify,
                        screen_margin=None if args.screen_margin_mha is None else 1e-3 * args.screen_margin_mha)
    spec = MethodSpec(args.method, conv_tol=tol.conv_tol, density_fit=True)
    (out / "run.json").write_text(json.dumps(dict(params=dataclasses.asdict(params), pairs=[dataclasses.asdict(p)
                                                                                         for p in pairs],
                                                  coords=args.coords, system=args.system, source=source), indent=1))
    coords = np.load(args.coords)[: args.steps]
    obs_path = out / "obs.jsonl"
    obs_path.write_text("")
    executor = ProcessExecutor(spec, args.workers, args.worker_threads) if args.workers else None
    try:
        beam = BeamEngine(mol, PySCFBackend(spec), metal_atoms=[args.metal], tol=tol, params=params, executor=executor)
        for k, x in enumerate(coords):
            t = time.perf_counter()
            res = beam.compute(x)
            rec = step_record(k, res, pairs)
            rec["seconds"] = time.perf_counter() - t
            save_branches(out / "frames" / f"{args.prefix}_{k:03d}_beam.npz", res)
            with open(obs_path, "a") as fh:
                fh.write(json.dumps(rec) + "\n")
            sel = next(l for l in rec["lineages"] if l["id"] == res.lineage_id)["observations"]
            first = f"{sel[0]['pair_id']} r={sel[0]['distance_angstrom']:.3f} M={sel[0]['mayer_order']:.4f} " if sel else ""
            print(f"step {k:3d}: {res.status:10s} lineages={len(rec['lineages'])} {first}{rec['seconds']:6.1f}s "
                  f"events={res.events}", flush=True)
    finally:
        if executor is not None:
            executor.close()
    print("RUN_DONE", flush=True)
    return 0


def obs_from(d) -> BondObservation:
    return BondObservation(**{**d, "charges": tuple(d["charges"]), "spins": tuple(d["spins"]),
                              "quality": tuple(d["quality"])})


def events(args) -> int:
    header = json.loads((Path(args.obs).parent / "run.json").read_text())
    pairs = [PairSpec(p["pair_id"], tuple(p["atoms"]), p["donor_class"]) for p in header["pairs"]]
    thresholds = {k: BondThresholds(*v) for k, v in json.loads(args.thresholds).items()}
    mon = CoordinationMonitor(pairs, thresholds)
    out = []
    for line in open(args.obs):
        r = json.loads(line)
        obs = {l["id"]: [obs_from(o) for o in l["observations"]] for l in r["lineages"]}
        out += mon.step(r["step"], obs, r["selected"], [tuple(e) for e in r["events"]],
                        {l["id"] for l in r["lineages"] if not l["converged"]},
                        {l["id"] for l in r["lineages"] if l["uncertain"]})
    Path(args.out).write_text("".join(json.dumps(e) + "\n" for e in out))
    Path(args.out).with_suffix(".state.json").write_text(json.dumps(dict(thresholds=json.loads(args.thresholds),
                                                                         state=mon.to_state())))
    counts = {}
    for e in out:
        counts[e["event_type"]] = counts.get(e["event_type"], 0) + 1
    print("EVENTS", json.dumps(counts), flush=True)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--system", required=True)
    r.add_argument("--coords", required=True)
    r.add_argument("--out-dir", required=True)
    r.add_argument("--prefix", default="scan")
    r.add_argument("--basis", default="def2-svp")
    r.add_argument("--method", default="uks/b3lyp")
    r.add_argument("--metal", type=int, default=0)
    r.add_argument("--steps", type=int, default=None)
    r.add_argument("--densify-steps", type=int, default=0)
    r.add_argument("--escalate-unknown", type=int, default=None)
    r.add_argument("--screen-margin-mha", type=float, default=None)
    r.add_argument("--densify-random", type=int, default=0)
    r.add_argument("--shadow-sv", type=float, default=None)
    r.add_argument("--shadow-all", action="store_true")
    r.add_argument("--shadow-skip-densify", action="store_true")
    r.add_argument("--workers", type=int, default=0)
    r.add_argument("--worker-threads", type=int, default=1)
    e = sub.add_parser("events")
    e.add_argument("--obs", required=True)
    e.add_argument("--thresholds", required=True,
                   help='JSON {donor_class: [r_on, r_off, m_on, m_off, n_confirm]} (Angstrom, Mayer)')
    e.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    return run(args) if args.cmd == "run" else events(args)


if __name__ == "__main__":  # spawned workers re-import this file
    raise SystemExit(main())
