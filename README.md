# sroqm — Stateful Reduced-Order QM engine

Design: `Stateful Reduced-Order QM-MM：核空间与电子空间的双层自动化框架.md` (v2).
Implemented so far: Phase 1 (continuation baseline) and Phase 2 (single-lineage tracking).

## Environment

```bash
PY=/home/ruigengji/miniforge3/envs/pyscf-env/bin/python   # PySCF 2.14, no install needed
$PY -m pytest -q                                          # full test suite
```

## Usage

```python
from pyscf import gto
from sroqm.backend import MethodSpec, PySCFBackend
from sroqm.engine import Engine

mol = gto.M(atom="...", basis="def2-svp", charge=2, spin=4)
engine = Engine(mol, PySCFBackend(MethodSpec("uks/b3lyp", density_fit=True)),
                fragments={"Fe": [0], "ligands": list(range(1, mol.natm))})
res = engine.compute(coords_bohr)          # energy, forces (Hartree/Bohr), s2, local_spins, alarms, status
```

## Benchmarks

```bash
PYTHONPATH=src $PY benchmarks/bench_continuation.py --system water_dimer --steps 50
PYTHONPATH=src $PY benchmarks/bench_continuation.py --system fe_hexaaqua --basis def2-svp \
    --density-fit --mode nve --steps 100 --threads 32 --out benchmarks/out/fe_nve.json
```

## Not yet implemented (see spec §20)

Phase 3 state search / branching / pruning; surface policies beyond single lineage;
XL-BOMD; MM-side gradients and OpenMM coupling (Phase 5, via `~/openmm-pyscf`); ORCA backend.
