from types import SimpleNamespace

import numpy as np
import pytest

from bench_beam import save_branches
from eval_fullref import ledger_cost, score_step
from sroqm.backend import MethodSpec, PySCFBackend
from sroqm.frame import Frame
from sroqm.tolerances import Tolerances

TOL = Tolerances()
WIDE = (1e6,)  # mHa: every reference solution of water counts (the excitation is ~0.3 Ha up)


@pytest.fixture(scope="module")
def uks_ground(water):
    return Frame.from_outcome(water, PySCFBackend(MethodSpec("uks/b3lyp", conv_tol=1e-9)).run(water, with_gradient=False))


def shifted(f, de_ha):
    return Frame.from_orbitals(f.mol, f.mo_coeff, f.mo_occ, f.energy + de_ha)


def test_the_selected_state_is_judged_against_the_reference_lowest_with_a_signed_difference(uks_ground, excited_water):
    s = score_step([uks_ground], uks_ground, False, [uks_ground, excited_water], TOL, WIDE)
    assert s["reason"] == "same-state" and s["passed"] and s["de_mha"] == pytest.approx(0.0, abs=1e-9)
    lower = shifted(uks_ground, -1e-4)  # the replay found a lower solution: kept negative, flagged
    s = score_step([lower], lower, False, [uks_ground, excited_water], TOL, WIDE)
    assert s["de_mha"] == pytest.approx(-0.1) and s["below_ref"]
    s = score_step([excited_water], excited_water, False, [uks_ground, excited_water], TOL, WIDE)
    assert s["reason"] == "miss" and not s["passed"]


def test_recall_counts_reference_solutions_matched_by_any_branch_and_one_to_one(uks_ground, excited_water):
    s = score_step([uks_ground], uks_ground, False, [uks_ground, excited_water], TOL, WIDE)
    assert s[f"recall_{WIDE[0]:g}mha"] == dict(n_ref=2, any=1, one_to_one=1)
    s = score_step([uks_ground, excited_water], uks_ground, False, [uks_ground, excited_water], TOL, WIDE)
    assert s[f"recall_{WIDE[0]:g}mha"] == dict(n_ref=2, any=2, one_to_one=2)
    # two reference copies of one state (dE within accept_de) and one branch: any 2, one-to-one 1
    twin = shifted(uks_ground, 1e-5)
    s = score_step([uks_ground], uks_ground, False, [uks_ground, twin], TOL, (1.0,))
    assert s["recall_1mha"] == dict(n_ref=2, any=2, one_to_one=1)
    # the window counts only reference solutions within W of the lowest
    s = score_step([uks_ground], uks_ground, False, [uks_ground, excited_water], TOL, (2.0,))
    assert s["recall_2mha"] == dict(n_ref=1, any=1, one_to_one=1)


def test_save_branches_writes_every_active_lineage_lowest_first(tmp_path, uks_ground, excited_water):
    res = SimpleNamespace(frames={4: excited_water, 7: uks_ground}, lineage_energies={4: excited_water.energy,
                          7: uks_ground.energy, 9: 0.0}, lineage_id=7, status="MULTISTATE", frame=uks_ground)
    save_branches(tmp_path / "b.npz", res)
    z = np.load(tmp_path / "b.npz")
    assert list(z["ids"]) == [7, 4] and int(z["selected"]) == 7 and str(z["status"]) == "MULTISTATE"
    assert z["mo_coeff"].shape[0] == 2 and np.allclose(z["energies"], [uks_ground.energy, excited_water.energy])


def test_ledger_cost_folds_screen_segments_into_their_tag(tmp_path):
    rows = [dict(seconds=10.0, trace=dict(tasks=[dict(tag="l1:scratch:s1", cycles=4), dict(tag="l1:scratch:s2", cycles=20),
                                                  dict(tag="shadow", cycles=7)])),
            dict(seconds=30.0, trace=dict(tasks=[dict(tag="advance", cycles=5)]))]
    p = tmp_path / "r.rows.jsonl"
    p.write_text("".join(__import__("json").dumps(r) + "\n" for r in rows))
    c = ledger_cost(p)
    assert c["by_tag"]["l1:scratch"] == dict(tasks=2, cycles=24) and c["cycles"] == 36 and c["tasks"] == 4
    assert c["seconds_total"] == 40.0
