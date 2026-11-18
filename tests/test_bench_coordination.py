import json

import numpy as np

from bench_coordination import main, metal_donor_pairs
from sroqm.testsystems import build


def test_metal_donor_pairs_list_every_o_and_n_of_the_metal():
    pairs = metal_donor_pairs(build("fe2_pentaaqua_ammine", "sto-3g"))
    assert [p.pair_id for p in pairs] == ["Fe-N1", "Fe-O5", "Fe-O8", "Fe-O11", "Fe-O14", "Fe-O17"]
    assert {p.donor_class for p in pairs} == {"Fe-N", "Fe-O"} and all(p.atoms[0] == 0 for p in pairs)


def test_run_records_every_lineage_and_events_replay_it(tmp_path, water):
    # water: the "metal" is atom 0 (O), the donors none -> no pairs; use H as a donor through the events replay below
    np.save(tmp_path / "c.npy", np.array([water.atom_coords()] * 2))
    code = main(["run", "--system", "water", "--coords", str(tmp_path / "c.npy"), "--out-dir", str(tmp_path / "o"),
                 "--basis", "6-31g", "--method", "uks/b3lyp"])
    assert code == 0
    recs = [json.loads(l) for l in (tmp_path / "o" / "obs.jsonl").read_text().splitlines()]
    assert [r["step"] for r in recs] == [0, 1] and all(r["lineages"] and r["lineages"][0]["converged"] for r in recs)
    assert (tmp_path / "o" / "frames" / "scan_001_beam.npz").exists() and (tmp_path / "o" / "run.src").is_dir()
    # a synthetic obs.jsonl for the events replay: one Fe-O pair pulled out over 6 steps
    (tmp_path / "e").mkdir()
    (tmp_path / "e" / "run.json").write_text(json.dumps(dict(pairs=[dict(pair_id="Fe-O1", atoms=[0, 1],
                                                                          donor_class="Fe-O")])))
    rows = []
    for k, (r, m) in enumerate([(2.1, 0.33), (2.1, 0.33), (2.6, 0.15), (3.1, 0.04), (3.3, 0.02), (3.5, 0.01)]):
        o = dict(pair_id="Fe-O1", lineage_id=0, step=k, distance_angstrom=r, mayer_order=m, charges=[1.0, -0.3],
                 spins=[3.8, 0.0], population_scheme="mulliken", quality=[])
        rows.append(dict(step=k, selected=0, status="FAST", events=[],
                         lineages=[dict(id=0, energy=-1.0, converged=True, uncertain=False, observations=[o])]))
    (tmp_path / "e" / "obs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    code = main(["events", "--obs", str(tmp_path / "e" / "obs.jsonl"), "--thresholds",
                 json.dumps({"Fe-O": [2.4, 3.0, 0.2, 0.05, 2]}), "--out", str(tmp_path / "e" / "ev.jsonl")])
    assert code == 0
    ev = [json.loads(l) for l in (tmp_path / "e" / "ev.jsonl").read_text().splitlines()]
    assert [e["event_type"] for e in ev] == ["INITIALIZED", "DISSOCIATED"] and ev[1]["onset_step"] == 2
    assert json.loads((tmp_path / "e" / "ev.state.json").read_text())["state"]["monitors"][0]["stable"] == "UNBOUND"
