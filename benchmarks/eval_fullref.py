"""Score a beam replay against a full-state reference (paper-1 protocol docs/2026-11-06-paper1-validation-protocol.zh-CN.md
§6): per step, the selected lineage against the reference's lowest solution with the energy difference kept signed,
recall of the reference solutions within a window by the ACTIVE lineages, and the SCF cost from the replay's ledger.

Inputs: the reference directory of ref_full2_20261111.py (<ref_prefix>_<k:03d>.npz: every solution within 3 mHa, tight
energies and orbitals) and the replay's --save-frames directory (<prefix>_<k:03d>_beam.npz from bench_beam.save_branches:
every ACTIVE lineage) plus its <out>.rows.jsonl for the ledger.

Per step:
  lowest      sigma_min(selected, reference lowest); de_mha = E_selected - E_ref_min, signed (bench_beam.judge_step clips
              it at 0); miss / warn as bench_beam (same state, or within accept_de, or DEGENERATE within deg_band; warn
              above WARN_DE_MHA);
  below_ref   the lowest ACTIVE lineage lies more than WARN_DE_MHA below the reference lowest: the reference missed a
              solution (protocol §3: send the geometry back to the reference search);
  recall(W)   reference solutions within W of the reference lowest that an ACTIVE lineage matches (sigma_min >=
              same_state_sv and |dE| <= accept_de), counted two ways: any (a lineage may match several solutions) and
              one-to-one (an optimal assignment, one lineage per solution).

usage: python benchmarks/eval_fullref.py --system fe2_hexaaqua --ref-dir R --ref-prefix l1_s21 --frames-dir F
       --prefix l1_s21 [--rows run.rows.jsonl] [--windows-mha 2 1] --out score.json
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from bench_beam import WARN_DE_MHA
from sroqm.frame import Frame
from sroqm.overlap import occupied_overlap
from sroqm.testsystems import build
from sroqm.tolerances import Tolerances


def load_frames(path, template, energies_key="energies"):
    z = np.load(path)
    mol = template.set_geom_(z["coords"], unit="Bohr", inplace=False)
    frames = [Frame.from_orbitals(mol, c, o, e) for c, o, e in zip(z["mo_coeff"], z["mo_occ"], z[energies_key])]
    return frames, z


def score_step(branches, selected, degenerate, refs, tol: Tolerances, windows_mha=(2.0, 1.0)) -> dict:
    """branches: ACTIVE lineage frames; selected: the selected one (one of branches); refs: reference solutions,
    lowest first. Energies in Hartree; the result's in mHa."""
    e0 = refs[0].energy
    sv = occupied_overlap(selected, refs[0]).min_sv
    de = 1e3 * (selected.energy - e0)
    if sv >= tol.same_state_sv:
        reason = "same-state"
    elif de <= 1e3 * tol.accept_de:
        reason = "energy"
    elif degenerate and de <= 1e3 * tol.deg_band:
        reason = "degenerate"
    else:
        reason = "miss"
    below = 1e3 * (min(f.energy for f in branches) - e0)
    out = dict(sigma_min=sv, de_mha=de, reason=reason, passed=reason != "miss",
               warn=reason in ("energy", "same-state") and de > WARN_DE_MHA,
               lowest_branch_de_mha=below, below_ref=below < -WARN_DE_MHA, n_branches=len(branches), n_ref=len(refs))
    match = np.array([[occupied_overlap(b, r).min_sv >= tol.same_state_sv and abs(b.energy - r.energy) <= tol.accept_de
                       for r in refs] for b in branches], dtype=bool).reshape(len(branches), len(refs))
    for w in windows_mha:
        cols = [j for j, r in enumerate(refs) if 1e3 * (r.energy - e0) <= w]
        sub = match[:, cols]
        any_ = int(sub.any(axis=0).sum())
        rows, cs = linear_sum_assignment(-sub.astype(float)) if sub.size else ((), ())
        one = int(sum(sub[r, c] for r, c in zip(rows, cs)))
        out[f"recall_{w:g}mha"] = dict(n_ref=len(cols), any=any_, one_to_one=one)
    return out


def ledger_cost(rows_path) -> dict:
    """Total SCF tasks / cycles by source tag (screen suffixes :s1 / :s2 folded in) and step-time percentiles."""
    by, secs = {}, []
    for line in open(rows_path):
        r = json.loads(line)
        secs.append(r["seconds"])
        for t in r["trace"].get("tasks", ()):
            tag = re.sub(r":s[12]$", "", t["tag"])
            n, c = by.get(tag, (0, 0))
            by[tag] = (n + 1, c + t["cycles"])
    s = np.array(secs)
    return dict(steps=len(secs), tasks=sum(n for n, _ in by.values()), cycles=sum(c for _, c in by.values()),
                by_tag={k: dict(tasks=n, cycles=c) for k, (n, c) in sorted(by.items())},
                seconds_total=float(s.sum()), seconds_p50=float(np.percentile(s, 50)),
                seconds_p95=float(np.percentile(s, 95)))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    for a in ("--system", "--ref-dir", "--ref-prefix", "--frames-dir", "--prefix", "--out"):
        ap.add_argument(a, required=True)
    ap.add_argument("--rows", default=None)
    ap.add_argument("--windows-mha", type=float, nargs="+", default=[2.0, 1.0])
    args = ap.parse_args(argv)
    tol, template = Tolerances(), build(args.system, "def2-svp")
    steps = []
    for bp in sorted(Path(args.frames_dir).glob(f"{args.prefix}_[0-9][0-9][0-9]_beam.npz")):
        k = int(bp.name[len(args.prefix) + 1:len(args.prefix) + 4])
        rp = Path(args.ref_dir) / f"{args.ref_prefix}_{k:03d}.npz"
        if not rp.exists():
            steps.append(dict(step=k, ref="missing"))
            continue
        branches, zb = load_frames(bp, template)
        refs, zr = load_frames(rp, template)
        if not np.allclose(zb["coords"], zr["coords"]):
            raise ValueError(f"step {k}: replay and reference geometries differ")
        sel = branches[list(zb["ids"]).index(int(zb["selected"]))]
        steps.append(dict(step=k, status=str(zb["status"]),
                          **score_step(branches, sel, str(zb["status"]) == "DEGENERATE", refs, tol, args.windows_mha)))
    scored = [s for s in steps if "reason" in s]
    summary = dict(steps=len(steps), scored=len(scored), ref_missing=len(steps) - len(scored),
                   miss=[s["step"] for s in scored if not s["passed"]], warn=[s["step"] for s in scored if s["warn"]],
                   below_ref=[s["step"] for s in scored if s["below_ref"]],
                   de_mha_min=min((s["de_mha"] for s in scored), default=None),
                   de_mha_max=max((s["de_mha"] for s in scored), default=None))
    for w in args.windows_mha:
        key = f"recall_{w:g}mha"
        n = sum(s[key]["n_ref"] for s in scored)
        summary[key] = dict(n_ref=n, any=sum(s[key]["any"] for s in scored),
                            one_to_one=sum(s[key]["one_to_one"] for s in scored))
    out = dict(summary=summary, cost=ledger_cost(args.rows) if args.rows else None, steps=steps)
    Path(args.out).write_text(json.dumps(out, indent=1))
    print("SUMMARY", json.dumps(summary), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
