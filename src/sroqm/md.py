"""Minimal velocity-Verlet NVE driver for Phase-1 continuation benchmarks (atomic units inside)."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

AMU_TO_AU = 1822.888486209
FS_TO_AU = 41.341373335
KB_AU = 3.166811563e-6  # Hartree / K


@dataclass
class MDLog:
    time_fs: list = field(default_factory=list)
    coords: list = field(default_factory=list)
    potential: list = field(default_factory=list)
    kinetic: list = field(default_factory=list)
    forces: list = field(default_factory=list)
    scf_iterations: list = field(default_factory=list)
    statuses: list = field(default_factory=list)

    @property
    def total(self) -> np.ndarray:
        return np.asarray(self.potential) + np.asarray(self.kinetic)


def maxwell_boltzmann(masses_au, temperature_K, rng) -> np.ndarray:
    m = np.asarray(masses_au, dtype=float)
    v = rng.normal(size=(len(m), 3)) * np.sqrt(KB_AU * temperature_K / m)[:, None]
    v -= (m[:, None] * v).sum(axis=0) / m.sum()
    return v


def _kinetic(m, v) -> float:
    return 0.5 * float(np.sum(m[:, None] * v ** 2))


def _record(log, t_fs, x, res, m, v):
    log.time_fs.append(t_fs)
    log.coords.append(x.copy())
    log.potential.append(res.energy)
    log.kinetic.append(_kinetic(m, v))
    log.forces.append(np.asarray(res.forces).copy())
    log.scf_iterations.append(res.scf_iterations)
    log.statuses.append(res.status.value)


def run_nve(provider, coords0_bohr, masses_amu, dt_fs, nsteps, temperature_K=0.0, seed=0) -> MDLog:
    m = np.asarray(masses_amu, dtype=float) * AMU_TO_AU
    dt = dt_fs * FS_TO_AU
    x = np.asarray(coords0_bohr, dtype=float).copy()
    v = maxwell_boltzmann(m, temperature_K, np.random.default_rng(seed)) if temperature_K > 0 else np.zeros_like(x)
    log = MDLog()
    res = provider.compute(x)
    a = np.asarray(res.forces) / m[:, None]
    _record(log, 0.0, x, res, m, v)
    for n in range(1, nsteps + 1):
        x = x + v * dt + 0.5 * a * dt ** 2
        res = provider.compute(x)
        a_new = np.asarray(res.forces) / m[:, None]
        v = v + 0.5 * (a + a_new) * dt
        a = a_new
        _record(log, n * dt_fs, x, res, m, v)
    return log


def energy_drift(log: MDLog) -> float:
    t_ps = np.asarray(log.time_fs) / 1000.0
    return float(np.polyfit(t_ps, log.total, 1)[0])
