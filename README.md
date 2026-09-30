# sroqm — Stateful Reduced-Order QM engine for QM/MM

English | [简体中文](README.zh-CN.md) | [日本語](README.ja.md)

`sroqm` is a stateful electronic-structure engine built on [PySCF](https://github.com/pyscf/pyscf)
for QM/MM molecular dynamics. A plain SCF restarts from a guess and lands wherever the guess takes
it. `sroqm` is built for trajectories that must stay on *one chosen electronic state* — a metal
spin state, a broken-symmetry solution — and that need to know when they are about to slip off it.

## What it does

**Continuation.** Instead of re-converging from scratch every step, each SCF is seeded from the
previous step: the previous orbitals (`read`), an ASPC-style density extrapolation (`aspc`), or a
Grassmann geodesic extrapolation of the orbital frame (`grassmann`). On def2-SVP/B3LYP this cuts
mean SCF cycles per step from 7.0 to 3.7 (water dimer) and from 12.3 to 4.6 ([Fe(H₂O)₆]³⁺), with
energies matching scratch to ≤ 2×10⁻⁷ Hartree (see [Benchmarks](#benchmarks)).

**State tracking (single lineage).** A *lineage* carries one electronic state across geometries:
IMOM overlap continuation between consecutive orbitals, rotation-invariant fingerprints and
frontier gaps as state descriptors, and graded alarms — `gap_sign_flip`, `overlap_drop`,
`imom_fallback`, `lower_state_found`, … — when the continuation may have hopped to a different
state or is stuck away from its own minimum. `Engine(..., probe_every=N)` additionally runs an
independent scratch SCF every N steps to detect lower states the lineage cannot see.

**Multi-state beam.** For near-degenerate manifolds a single lineage is not enough: high-spin
[Fe(H₂O)₆]²⁺ (⁵T₂g) has three t₂g occupations within 0.5 mHa of each other, and the SCF lands in
one of them essentially at random. `search_states` enumerates the distinct low-lying SCF solutions
at one geometry (scratch/aufbau plus every metal-d occupation, converged with MOM); `BeamEngine`
keeps up to `max_lineages` competing lineages alive — spawning, reviving, merging and pruning
them — and returns energy and forces of the lowest one.

### Status

Implements Phase 1 (continuation baseline), Phase 2 (single-lineage stateful engine) and Phase 3a
(state search + multi-lineage beam) of the
[design spec](<Stateful Reduced-Order QM-MM：核空间与电子空间的双层自动化框架.md>) (Chinese).
Phase 3b — beam hygiene rules and discovery scheduling on near-degenerate manifolds (per-lineage
SCF budgets, a candidate admission test, lost/dormant semantics, refresh scheduling, a two-layer
full search, explicit DEGENERATE output) — is designed but not yet implemented. Not started:
XL-BOMD, MM-side gradients / OpenMM coupling, ORCA backend. Search candidates from neighbouring
multiplicities (M±2) and broken-symmetry solutions are planned after Phase 3b.

## Installation

Python ≥ 3.12, PySCF ≥ 2.14, NumPy ≥ 2.0.

```bash
pip install -e .
```

Set `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS` and `MKL_NUM_THREADS` *before* launching Python
(or let `benchmarks/bench_continuation.py --threads` do it for you); setting them after import
over-subscribes BLAS (measured 2–2.7× slower).

Run the tests with:

```bash
python -m pytest -q
```

## Quick start

### Single lineage

```python
from sroqm.backend import MethodSpec, PySCFBackend
from sroqm.engine import Engine
from sroqm.testsystems import build

mol = build("fe2_hexaaqua", "def2-svp")      # or any pyscf gto.Mole
coords = mol.atom_coords()                   # the geometry for this step, in Bohr

engine = Engine(
    mol,
    PySCFBackend(MethodSpec("uks/b3lyp", density_fit=True)),
    fragments={"Fe": [0], "ligands": list(range(1, mol.natm))},
)
res = engine.compute(coords)   # energy, forces (Hartree/Bohr), s2, local_spins, alarms, status
print(res.energy, res.alarms)
```

Call `engine.compute(coords)` once per MD step; the engine continues the lineage from its own
history (default continuation: `imom`). For open-shell systems use a `uhf`/`uks` method spec.

### Multi-lineage beam

```python
from sroqm.beam import BeamEngine

beam = BeamEngine(
    mol,
    PySCFBackend(MethodSpec("uks/b3lyp", density_fit=True)),
    metal_atoms=[0],
    search_every=5,   # also searches on gap_sign_flip / overlap_drop / imom_fallback
)
res = beam.compute(coords)            # E/F of the lowest lineage (policy="adiabatic_min")
res.lineage_energies                  # every lineage kept alive, by id
res.switched, res.events              # cusp flag on switch; ("spawn"|"revive"|"merge"|"dormant", id)
```

On an 11-step [Fe(H₂O)₆]²⁺ NVE trajectory, the beam with `search_every=5` stays on the lowest
state to within +0.004 mHa once the first alarm-triggered search fires, while a single lineage
drifts up to +0.4 mHa and only *reports* `lower_state_found` without switching.

Known limitations on near-degenerate manifolds (what Phase 3b addresses): only `search_every=1`
holds the lowest state from the very first step — sparser schedules miss the first steps after
symmetry breaking; the keep window (default 8 mHa) far exceeds the t₂g splitting, so the beam can
fill up with near-degenerate variants; and a single search layer can miss the lowest state
entirely. On these manifolds keep `conv_tol` at 1e-7 or tighter — at 1e-6 the SCF declares
convergence halfway along the soft directions and cannot separate 0.1 mHa states.

### One-geometry state search

```python
from sroqm.search import search_states

states = search_states(backend, mol, base=None, metal_atoms=[0])   # list[Frame], lowest first
```

## Benchmarks

`benchmarks/bench_continuation.py` compares continuation strategies (`scratch`, `read`, `aspc`,
`grassmann`) on replayed or fresh NVE trajectories, with multi-process `--jobs`. Energy/force
errors are counted only on steps where the strategy is on the same electronic state as the
reference; other steps are counted separately as `state_mismatch_steps`.

Test systems (`sroqm.testsystems`): `water`, `water_dimer`, `fe3_hexaaqua` (high-spin d⁵ — the
Phase-1 metal benchmark) and `fe2_hexaaqua` (high-spin d⁶, ⁵T₂g — three near-degenerate t₂g
occupations, a state-tracking stress test).

```bash
PYTHONPATH=src python benchmarks/bench_continuation.py --system water_dimer --steps 50
PYTHONPATH=src python benchmarks/bench_continuation.py --system fe3_hexaaqua --basis def2-svp \
    --density-fit --mode nve --steps 100 --threads 10 --jobs 4 --out benchmarks/out/fe3_nve.json
```

Measured mean SCF cycles per step (def2-SVP/B3LYP, conv tol 1×10⁻⁷):

| system | scratch | read | aspc | grassmann |
|---|---|---|---|---|
| water dimer (closed shell, 20 steps) | 7.0 | 5.0 | 3.95 | **3.65** |
| [Fe(H₂O)₆]³⁺ sextet (d⁵, 10 steps) | 12.3 | 6.4 | 5.1 | **4.6** |

Maximum energy deviation vs. scratch: ≤ 2×10⁻⁸ Ha (water dimer), ≤ 2×10⁻⁷ Ha (Fe³⁺). On small
systems the wall-clock win is smaller than the cycle count suggests (gradients, integration grids
and DF rebuild are fixed per-step costs), and one Fe²⁺ step costs ~9 s even with a single SCF —
hence the beam, which avoids re-converging the wrong state.

## Code layout

| module | role |
|---|---|
| `backend.py` | PySCF wrapper: SCF from a given guess, gradients, MM point charges |
| `grassmann.py`, `frame.py`, `extrapolate.py` | Löwdin + Grassmann log/exp; orbital frames; the four initial-guess extrapolators |
| `overlap.py`, `imom.py` | cross-geometry occupied-space overlap (det and σ_min); IMOM continuation |
| `fingerprint.py`, `alarms.py` | rotation-invariant fingerprints, per-channel frontier gaps; Tier 0–2 alarms |
| `lineage.py`, `engine.py` | single-lineage engine (IMOM fallback, history truncation, periodic probing) |
| `search.py` | one-geometry state search: scratch/aufbau + metal-d occupation enumeration (MOM) |
| `beam.py` | multi-lineage beam: spawn / revive / merge / dormant / capacity-replacing prune |
| `md.py`, `testsystems.py` | velocity-Verlet NVE driver; water, water dimer, Fe³⁺/Fe²⁺ hexaaqua |

## Design documents

- [Design spec (Chinese)](<Stateful Reduced-Order QM-MM：核空间与电子空间的双层自动化框架.md>) —
  full framework design; §9–10 cover the beam and the alarm tiers.

## Roadmap

- Phase 3b: lineage hygiene rules (per-lineage SCF budgets, a candidate admission test,
  lost/dormant semantics) and discovery scheduling (refresh intervals, two-layer full search),
  with DEGENERATE output and tolerances matched to the acceptance criteria; single node by default.
- Search-candidate extensions: neighbouring multiplicities (M±2), broken-symmetry fragments,
  stability following.
- Cross-machine task-level dispatch (prototyped, ~47–68 s/step at 40 cores).
- XL-BOMD.
- MM-side gradients and OpenMM coupling.
- ORCA backend.
