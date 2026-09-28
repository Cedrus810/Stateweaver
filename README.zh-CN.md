# sroqm — 有状态的约化阶 QM/MM 引擎

[English](README.md) | 简体中文

`sroqm` 是构建在 [PySCF](https://github.com/pyscf/pyscf) 之上的有状态电子结构引擎，面向 QM/MM
分子动力学。普通 SCF 每步从初猜重新收敛，落在哪个解上算哪个；`sroqm` 面向的则是必须**停留在某条
指定电子态上**的轨迹——金属自旋态、破对称解等——并且要在快要滑到别的态上时发出警告。

## 它做什么

**Continuation（续算）。** 不再每步从零收敛，而是用上一步的信息做初猜：上一步的轨道（`read`）、
ASPC 式密度外推（`aspc`）、或轨道帧的 Grassmann 测地线外推（`grassmann`）。在 def2-SVP/B3LYP 上，
每步平均 SCF 循环数从 7.0 降到 3.7（水二聚体）、从 12.3 降到 4.6（[Fe(H₂O)₆]³⁺），能量与从头算
之差 ≤ 2×10⁻⁷ Hartree（见[基准测试](#基准测试)）。

**态跟踪（单 lineage）。** 一条 *lineage* 就是一个被携带跨越几何空间的电子态：相邻两步轨道之间做
IMOM 重叠续算，用旋转不变的 fingerprint 和 frontier gap 作态描述符，跟踪可能出问题时给出分级
alarm——`gap_sign_flip`、`overlap_drop`、`imom_fallback`、`lower_state_found` 等——分别对应
"可能跳到了别的态""重叠骤降""IMOM 不收敛已回退 aufbau""别处存在更低的态"。
`Engine(..., probe_every=N)` 还会每 N 步额外跑一次独立的从头 SCF，探测 lineage 自己看不到的更低态。

**多态 beam。** 近简并流形上单条 lineage 不够用：高自旋 [Fe(H₂O)₆]²⁺（⁵T₂g）的三个 t₂g 占据彼此
只差 0.5 mHa 以内，SCF 基本上随机落进其中一个。`search_states` 在单一几何上枚举不同的低 lying
SCF 解（scratch/aufbau 加上每种金属 d 占据，用 MOM 收敛）；`BeamEngine` 同时保活至多
`max_lineages` 条竞争 lineage——spawn、revive、merge、prune——并返回最低态的能量和力。

### 进度

已实现设计规范（[中文](<Stateful Reduced-Order QM-MM：核空间与电子空间的双层自动化框架.md>)）中的
Phase 1（continuation 基线）、Phase 2（单 lineage 有状态引擎）和 Phase 3a（态搜索 + 多 lineage
beam）。尚未实现：Phase 3b（搜索中加入相邻多重度和破对称候选、按 d 流形限制 lineage 数、对称性
破缺后的搜索调度）、XL-BOMD、MM 端梯度 / OpenMM 耦合、ORCA 后端。

## 安装

Python ≥ 3.12，PySCF ≥ 2.14，NumPy ≥ 2.0。

```bash
pip install -e .
```

`OMP_NUM_THREADS`、`OPENBLAS_NUM_THREADS`、`MKL_NUM_THREADS` 必须在启动 Python **之前**设置
（或交给 `benchmarks/bench_continuation.py --threads` 处理）；启动后再设会让 BLAS 超额订阅
（实测慢 2–2.7 倍）。

运行测试：

```bash
python -m pytest -q
```

## 快速上手

### 单 lineage

```python
from sroqm.backend import MethodSpec, PySCFBackend
from sroqm.engine import Engine
from sroqm.testsystems import build

mol = build("fe2_hexaaqua", "def2-svp")      # 或任意 pyscf gto.Mole
coords = mol.atom_coords()                   # 本步几何，单位 Bohr

engine = Engine(
    mol,
    PySCFBackend(MethodSpec("uks/b3lyp", density_fit=True)),
    fragments={"Fe": [0], "ligands": list(range(1, mol.natm))},
)
res = engine.compute(coords)   # 能量、力（Hartree/Bohr）、s2、local_spins、alarm、status
print(res.energy, res.alarms)
```

每个 MD 步调用一次 `engine.compute(coords)`，引擎从自己的历史里续算 lineage（默认
continuation：`imom`）。开壳层体系要用 `uhf`/`uks` 方法。

### 多 lineage beam

```python
from sroqm.beam import BeamEngine

beam = BeamEngine(
    mol,
    PySCFBackend(MethodSpec("uks/b3lyp", density_fit=True)),
    metal_atoms=[0],
    search_every=5,   # gap_sign_flip / overlap_drop / imom_fallback 时也会触发搜索
)
res = beam.compute(coords)            # 最低 lineage 的 E/F（policy="adiabatic_min"）
res.lineage_energies                  # 保活的所有 lineage（按 id）
res.switched, res.events              # 换根标志（有 cusp）；("spawn"|"revive"|"merge"|"dormant", id)
```

在 11 步 [Fe(H₂O)₆]²⁺ NVE 轨迹上：单 lineage 漂移最高 +0.4 mHa 且只会报告 `lower_state_found`
而不切换；beam 设 `search_every=5`，第一次 alarm 触发搜索之后一直停在最低态（+0.004 mHa 以内）。

### 单几何态搜索

```python
from sroqm.search import search_states

states = search_states(backend, mol, base=None, metal_atoms=[0])   # list[Frame]，能量从低到高
```

## 基准测试

`benchmarks/bench_continuation.py` 在重放或全新 NVE 轨迹上比较各 continuation 策略（`scratch`、
`read`、`aspc`、`grassmann`），支持多进程 `--jobs`。能量/力误差只在策略与参考轨迹处于同一电子态的
步上统计，其余步单独计入 `state_mismatch_steps`——换了态不算外推误差。

测试体系（`sroqm.testsystems`）：`water`、`water_dimer`、`fe3_hexaaqua`（高自旋 d⁵，Phase 1 的
金属基准）和 `fe2_hexaaqua`（高自旋 d⁶，⁵T₂g——三个近简并 t₂g 占据，态跟踪的压力测试）。

```bash
PYTHONPATH=src python benchmarks/bench_continuation.py --system water_dimer --steps 50
PYTHONPATH=src python benchmarks/bench_continuation.py --system fe3_hexaaqua --basis def2-svp \
    --density-fit --mode nve --steps 100 --threads 10 --jobs 4 --out benchmarks/out/fe3_nve.json
```

实测每步平均 SCF 循环数（def2-SVP/B3LYP，收敛阈值 1×10⁻⁷）：

| 体系 | scratch | read | aspc | grassmann |
|---|---|---|---|---|
| 水二聚体（闭壳层，20 步） | 7.0 | 5.0 | 3.95 | **3.65** |
| [Fe(H₂O)₆]³⁺ 六重态（d⁵，10 步） | 12.3 | 6.4 | 5.1 | **4.6** |

与从头算的最大能量偏差：水二聚体 ≤ 2×10⁻⁸ Ha，Fe³⁺ ≤ 2×10⁻⁷ Ha。小体系上墙钟时间的收益小于循环
数的收益（梯度、积分格点和 DF 重建是每步固定开销）；[Fe(H₂O)₆]²⁺ 一步哪怕只做 1 次 SCF 也要约
9 秒——所以需要 beam 来避免反复收敛到错误的态上。

## 代码结构

| 模块 | 作用 |
|---|---|
| `backend.py` | PySCF 封装：给定初猜跑 SCF、梯度、MM 点电荷 |
| `grassmann.py`、`frame.py`、`extrapolate.py` | Löwdin + Grassmann log/exp；电子帧；四种初猜外推 |
| `overlap.py`、`imom.py` | 跨几何占据空间重叠（det 与 σ_min）；IMOM 续算 |
| `fingerprint.py`、`alarms.py` | 旋转不变 fingerprint、逐通道 frontier gap；Tier 0–2 alarm |
| `lineage.py`、`engine.py` | 单 lineage 引擎（IMOM 回退、历史截断、定期探测） |
| `search.py` | 单几何态搜索：scratch/aufbau + 金属 d 占据枚举（MOM） |
| `beam.py` | 多 lineage beam：spawn / revive / merge / dormant / 满员替换 |
| `md.py`、`testsystems.py` | velocity-Verlet NVE 驱动；水、水二聚体、Fe³⁺/Fe²⁺ 六水合 |

## 设计文档

- [设计规范（中文）](<Stateful Reduced-Order QM-MM：核空间与电子空间的双层自动化框架.md>)——
  完整框架设计；§9–10 是 beam 与 alarm 分级。

## 路线图

- Phase 3b：搜索中加入相邻多重度（M±2）与破对称候选；按 d 流形限制 lineage 数，避免近简并变体
  占满 beam；对称性破缺后的搜索调度。
- XL-BOMD。
- MM 端梯度与 OpenMM 耦合。
- ORCA 后端。
