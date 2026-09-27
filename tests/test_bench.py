import pytest

import bench_continuation


def test_replay_mode_smoke():
    out = bench_continuation.main(["--system", "water", "--basis", "sto-3g", "--steps", "3",
                                   "--strategies", "scratch,grassmann", "--threads", "2"])
    assert out["mode"] == "replay"
    assert set(out["rows"]) == {"scratch", "grassmann"}
    for row in out["rows"].values():
        assert row["mean_cycles"] > 0
        assert row["max_abs_dE"] < 1e-5


def test_nve_mode_smoke():
    out = bench_continuation.main(["--system", "water", "--basis", "sto-3g", "--steps", "3", "--mode", "nve",
                                   "--strategies", "read", "--threads", "2"])
    assert "drift_Ha_per_ps_per_atom" in out["rows"]["read"]


def test_unknown_strategy_is_rejected():
    with pytest.raises(SystemExit):
        bench_continuation.main(["--strategies", "magic"])
