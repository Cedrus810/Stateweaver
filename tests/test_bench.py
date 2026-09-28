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


def test_thread_env_is_taken_from_argv():
    env = bench_continuation.thread_env(["--system", "water", "--threads", "12"])
    assert env == {"OMP_NUM_THREADS": "12", "OPENBLAS_NUM_THREADS": "12", "MKL_NUM_THREADS": "12"}
    assert bench_continuation.thread_env(["--threads=4"])["OMP_NUM_THREADS"] == "4"
    assert bench_continuation.thread_env([])["OMP_NUM_THREADS"] == "8"


def test_replay_counts_steps_on_a_different_state_than_the_reference():
    out = bench_continuation.main(["--system", "water", "--basis", "sto-3g", "--steps", "3",
                                   "--strategies", "read", "--threads", "2"])
    assert out["rows"]["read"]["state_mismatch_steps"] == 0


def test_parallel_jobs_reproduce_the_serial_rows():
    argv = ["--system", "water", "--basis", "sto-3g", "--steps", "3", "--strategies", "read,grassmann",
            "--threads", "2"]
    serial = bench_continuation.main(argv + ["--jobs", "1"])["rows"]
    parallel = bench_continuation.main(argv + ["--jobs", "2"])["rows"]
    assert set(parallel) == {"read", "grassmann"}
    for name in serial:
        assert parallel[name]["mean_cycles"] == pytest.approx(serial[name]["mean_cycles"], abs=0.5)
        assert parallel[name]["max_abs_dE"] == pytest.approx(serial[name]["max_abs_dE"], abs=1e-8)
