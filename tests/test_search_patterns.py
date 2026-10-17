"""Compare capped, lazy enumeration with a small exhaustive oracle, including tie order."""
import itertools
from types import SimpleNamespace

import numpy as np
import pytest

from sroqm.search import d_occupation_patterns


def synthetic(monkeypatch, occupations):
    occ = np.asarray(occupations, dtype=float)
    n = occ.shape[-1]
    base = SimpleNamespace(mol=object(), mo_occ=occ, ovlp=np.eye(n), unrestricted=occ.ndim == 2,
                           mo_coeff=np.stack([np.eye(n)] * 2) if occ.ndim == 2 else np.eye(n))
    monkeypatch.setattr("sroqm.search._metal_d_rows", lambda *args: list(range(n)))
    monkeypatch.setattr("sroqm.search.lo.orth_ao", lambda *args: np.eye(n))
    return base


def exhaustive(occupations):
    occ = np.asarray(occupations, dtype=float)
    channels = occ if occ.ndim == 2 else occ[None]
    choices = []
    for channel in channels:
        filled = set(np.flatnonzero(channel))
        choices.append([(combo, len(filled - set(combo)))
                        for combo in itertools.combinations(range(len(channel)), len(filled))])
    ordered = sorted(itertools.product(*choices), key=lambda cs: sum(m for _, m in cs))
    patterns = []
    for cs in ordered:
        pattern = np.zeros_like(channels)
        for s, (combo, _) in enumerate(cs):
            pattern[s, list(combo)] = 1.0 if occ.ndim == 2 else 2.0
        patterns.append(pattern if occ.ndim == 2 else pattern[0])
    return patterns


@pytest.mark.parametrize("unrestricted", [False, True])
def test_lazy_patterns_keep_exhaustive_order_and_cutoff(monkeypatch, unrestricted):
    rng = np.random.default_rng(42)
    for _ in range(12):
        occ = (rng.random((2, 6) if unrestricted else 6) > .5).astype(float)
        if not unrestricted:
            occ *= 2
        base = synthetic(monkeypatch, occ)
        expected = exhaustive(occ)
        for cap in (1, 3, 24, len(expected) + 1):
            actual = d_occupation_patterns(base, [0], window=6, max_candidates=cap)
            assert len(actual) == min(cap, len(expected))
            for got, want in zip(actual, expected):
                np.testing.assert_array_equal(got, want)


def test_large_space_only_generates_a_bounded_prefix(monkeypatch):
    import sroqm.search as search

    occ = np.zeros((2, 32))
    occ[:, 16:] = 1  # the base is last in lexicographic order: scanning combinations is also too expensive
    base = synthetic(monkeypatch, occ)
    original = search._occupation_combinations
    yielded = 0

    def counted(*args):
        nonlocal yielded
        for combo in original(*args):
            yielded += 1
            yield combo

    monkeypatch.setattr(search, "_occupation_combinations", counted)
    patterns = d_occupation_patterns(base, [0], window=32, max_candidates=24)
    assert len(patterns) == 24 and yielded < 100
    np.testing.assert_array_equal(patterns[0], occ)
    for pattern in patterns:
        np.testing.assert_array_equal(pattern.sum(axis=1), occ.sum(axis=1))


def test_no_metal_and_zero_cap_skip_orthogonalization(monkeypatch):
    base = synthetic(monkeypatch, [[1, 0, 1], [0, 1, 0]])
    monkeypatch.setattr("sroqm.search._metal_d_rows", lambda *args: [])

    def unexpected(*args):
        pytest.fail("no orthogonalization is needed without candidate d orbitals")

    monkeypatch.setattr("sroqm.search.lo.orth_ao", unexpected)
    (pattern,) = d_occupation_patterns(base, [])
    np.testing.assert_array_equal(pattern, base.mo_occ)
    assert d_occupation_patterns(base, [], max_candidates=0) == []
    with pytest.raises(ValueError, match="max_candidates"):
        d_occupation_patterns(base, [], max_candidates=-1)
