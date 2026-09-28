import dataclasses
import itertools

import numpy as np
import pytest

from sroqm.backend import MethodSpec, PySCFBackend
from sroqm.frame import Frame
from sroqm.overlap import occupied_overlap
from sroqm.search import d_occupation_patterns, search_states


@pytest.fixture(scope="module")
def fe2_backend():
    return PySCFBackend(MethodSpec("uks/b3lyp", conv_tol=1e-7, density_fit=True))


@pytest.fixture(scope="module")
def fe2_base(fe2_distorted, fe2_backend):
    return Frame.from_outcome(fe2_distorted, fe2_backend.run(fe2_distorted, with_gradient=False))


@pytest.fixture(scope="module")
def fe2_states(fe2_distorted, fe2_backend):
    return search_states(fe2_backend, fe2_distorted, None, metal_atoms=[0])


def test_d_patterns_put_the_minority_electron_in_each_d_orbital(fe2_base):
    patterns = d_occupation_patterns(fe2_base, metal_atoms=[0])
    assert len({tuple(p[1]) for p in patterns}) == 5  # high-spin d6: one beta electron, five d orbitals
    for p in patterns:
        assert p[0].sum() == fe2_base.mo_occ[0].sum()
        assert p[1].sum() == fe2_base.mo_occ[1].sum()


def test_d_patterns_without_a_metal_are_just_the_base(fe2_base):
    (only,) = d_occupation_patterns(fe2_base, metal_atoms=[])
    assert np.array_equal(only, fe2_base.mo_occ)


def test_search_finds_the_three_t2g_states_sorted_and_distinct(fe2_states):
    energies = [f.energy for f in fe2_states]
    assert energies == sorted(energies)
    assert sum(e - energies[0] < 5e-3 for e in energies) == 3  # 5T2g: three t2g occupations
    for a, b in itertools.combinations(fe2_states, 2):
        assert occupied_overlap(a, b).min_sv < 0.9


def test_search_from_a_high_state_still_reaches_the_lowest(fe2_distorted, fe2_backend, fe2_states):
    low = [f for f in fe2_states if f.energy - fe2_states[0].energy < 5e-3]
    again = search_states(fe2_backend, fe2_distorted, low[-1], metal_atoms=[0])
    assert again[0].energy == pytest.approx(fe2_states[0].energy, abs=1e-6)


def test_search_without_a_metal_returns_the_aufbau_state(water):
    (state,) = search_states(PySCFBackend(MethodSpec("rks/b3lyp")), water, None, metal_atoms=[])
    assert np.isfinite(state.energy)


class ScratchNeverConverges(PySCFBackend):
    """Real SCF, but the scratch (no-guess) run reports failure, as on a hard metal geometry."""

    def run(self, mol, dm0=None, occ_hook=None, **kw):
        out = super().run(mol, dm0=dm0, occ_hook=occ_hook, **kw)
        return dataclasses.replace(out, converged=False) if dm0 is None else out


def test_search_enumerates_from_an_unconverged_scratch_solution(water):
    (state,) = search_states(ScratchNeverConverges(MethodSpec("rks/b3lyp")), water, None, metal_atoms=[])
    assert np.isfinite(state.energy)


def test_max_candidates_caps_the_enumeration(fe2_base):
    assert len(d_occupation_patterns(fe2_base, metal_atoms=[0], max_candidates=2)) == 2
