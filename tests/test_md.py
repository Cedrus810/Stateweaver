from types import SimpleNamespace

import numpy as np
import pytest

from sroqm.backend import MethodSpec, PySCFBackend
from sroqm.engine import Engine
from sroqm.md import AMU_TO_AU, KB_AU, energy_drift, maxwell_boltzmann, run_nve


class Harmonic:
    """Isotropic well E = k|r|^2/2 with Engine-like result fields."""

    def __init__(self, k=0.5):
        self.k = k

    def compute(self, coords):
        return SimpleNamespace(energy=0.5 * self.k * float(np.sum(coords ** 2)), forces=-self.k * coords,
                               scf_iterations=0, status=SimpleNamespace(value="FAST"))


def test_velocity_verlet_conserves_energy_in_a_harmonic_well():
    x0 = np.array([[0.1, 0.0, 0.0], [0.0, -0.1, 0.0]])
    log = run_nve(Harmonic(k=0.1), x0, np.array([1.0, 1.0]), dt_fs=0.5, nsteps=400)
    assert len(log.potential) == 401
    assert log.kinetic[0] == 0.0
    assert np.ptp(log.total) < 0.02 * log.total[0]                  # checked: 0.6 %
    assert abs(energy_drift(log)) * 0.2 < 1e-3 * log.total[0]       # drift over 0.2 ps; checked: 6e-5


def test_maxwell_boltzmann_temperature_and_zero_momentum():
    masses = np.full(3000, 12.0) * AMU_TO_AU
    v = maxwell_boltzmann(masses, 300.0, np.random.default_rng(0))
    ke = 0.5 * np.sum(masses[:, None] * v ** 2)
    assert ke / (1.5 * len(masses) * KB_AU * 300.0) == pytest.approx(1.0, rel=0.05)
    assert np.allclose((masses[:, None] * v).sum(axis=0), 0.0, atol=1e-8)


def test_run_nve_with_the_real_engine(water):
    eng = Engine(water, PySCFBackend(MethodSpec("rks/b3lyp", conv_tol=1e-9)))
    log = run_nve(eng, water.atom_coords(), water.atom_mass_list(isotope_avg=True), dt_fs=0.5, nsteps=4,
                  temperature_K=300.0)
    assert len(log.coords) == 5
    assert all(s == "FAST" for s in log.statuses)
    assert np.ptp(log.total) < 1e-3
