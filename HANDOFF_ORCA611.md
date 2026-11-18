# HANDOFF：ORCA 6.1.1 工作文档要点 × sroqm ORCA 后端开工包

> 来源：ORCA 6.1.1 对接工作文档（kasuga 整理），全部材料入口：
> `/home/kasuga/orca_ext_interface/ORCA_6_1_1_COMPLETE_TEARDOWN.md`。
> 对应 sroqm README "Not started: … ORCA backend" —— 这份是开工包。
> 证据分级：✅ 实测验证 / 已核实 = 工作文档确认的直接证据 / inferred 推断。
> 注意：早前口头说过 "ORCA 无 MOM"，**那是检索口径失误，工作文档已纠正复核**
> ——ORCA 有完整 MOM 与手动占据键族（§1）。以本文档为准。

## 1. search_states / beam 的原生等价物（%scf 键族，已核实）

```
DOMOM / PMOM            最大重叠占据（MOM）开关及变体
SETALPHA / SETBETA      手动指定 α/β 占据 ←→ search_states 的 d-占据枚举播种
REMOVEALPHA / REMOVEBETA      从占据移除轨道
IONIZEALPHA / IONIZEBETA      制备空穴态
ALPHACONF / BETACONF    组态指定
FINALMS                 收敛到目标 M_S
DELTASCF                ΔSCF 激发态
GUESSMIX / GUESSROTATION  初猜混合/旋转
FLIPSPIN                翻自旋原子（仅 UHF；仅 QM 原子）←→ BS 谱系制备
```

含义：beam 谱系在 ORCA 后端可用**原生 DOMOM 锁占据**（不必只靠外部 IMOM）；
`search_states` 的"枚举金属 d 占据 + MOM 收敛"用 SETALPHA/SETBETA 播种 +
DOMOM 收敛原生复现。邻近多重态（M±2）候选用 FlipSpin+UHF 制备破对称解。

## 2. 续算 / 外推的 ORCA 侧接口

- 每步播种：`%moinp` + GBW。openmm-ORCA 已踩过的坑直接适用：**moinp 文件
  不得与作业同基名**（ORCA 会先覆盖 gbw 再读它），stage 成 `guess.gbw`。
- 上游先例：松弛面扫描的 `SCANGUESS` 键（扫描点间复用初猜）——外推续算
  在上游是有共识的模式，不是野路子。

## 3. 多谱系布局 = 上游 PES2 模式（已核实）

ORCA 双势能面机器（CIOpt/MECP）的同步布局，可直接推广到 N 谱系：

```
<base>_PES2.gbw / .hostnames / .nodes / .property.txt    ← 每面独立全套目录
ciopten.tmp                                              ← 每步单 double 能量交换
GBW 段接口同步几何（ReadGeometry/WriteGeometry）
```

即：**每谱系一个目录 + 一个 GBW + 每步一个能量文件**——天然并行、崩溃可恢复、
和 ORCA 自己的习惯一致。上游 MD（CMDInterfaceOrca）为独立子作业编排
（`> %s.scf.last.log`），同级参考。

## 4. Beam 卫生的新探针与下游（ideas）

1. **分支面夹角探针**：上游双面优化每步跟踪 "Angle between branching plane
   vectors"（(Δg, NACME d) 向量对，已核实）。近简并谱系间该夹角骤变 =
   避免交叉临近 → 应触发 refresh/合并判定（Phase-3b admission/refresh 的新证据源）。
2. **M±2 候选的下游**：不同多重态交叉 NACME 恒为零，正是 MECP-OPT 领地：
   sroqm 找候选 → ORCA `MECP-OPT` 定交叉点 → `SURFCROSSNUMFREQ` 表征。
   限制：MECP 与 RHF/RKS 组合会被警告（开壳层方法为主）。
3. **DEGENERATE 表征**：BeamResult.status=DEGENERATE 时，用 SurfCross 式
   双面 Hessian 合成区分"真缝合面简并 vs 孤立近能"，影响保留策略。
4. **导数耦合校验**：`TNACMEIntegral::CalcPrimitiveIntegrals` 与
   `TSharkInterface::Calc_Gradient_NACME`（均已核实为可用接口）可作为 IMOM
   重叠近似 d≈⟨φ|∂φ⟩ 的解析对照（同多重度谱系）。

## 5. 数据交换与运行事实

- 能量/力：EXT/engrad 接口（✅ 端到端验证）：`! ORCAEXT` + `EXTOPTEXE`，
  六行 extinp + engrad 回报（原子数/能量Eh/3N 梯度 Eh·Bohr⁻¹）。可编译
  C++ 模板：`~/orca_ext_interface/otool_external_template/`。
- 返回码不可信（已知）；判据 `.out` 的 `ORCA TERMINATED NORMALLY`。
- **ORCA 6.1.1 不提供电子级 PBC**（已核实）——周期场景留给 OpenMM 侧。
- 性能：该 build 使用 OpenBLAS 0.3.29 DYNAMIC_ARCH（EPYC 回落 SkylakeX），
  QM 节点先试 `OPENBLAS_CORETYPE=ZEN*`；AVX2 专用无 AVX-512。

## 6. 需要深挖时

工作文档可持续扩充：需要更细的行为说明（FLIPSPIN 收敛行为、DOMOM 细节、
MECP 梯度混合装配等），直接向 kasuga 提需求，补充条目按同一证据分级交付。
相关专题文档：`FLIPSPIN_SCAN_TEARDOWN.md`（键族）、`DUAL_PES_TEARDOWN.md`
（双面机器 + 6 ideas）、`GBW_FORMAT.md`、`INTERFACE.md`（engrad）。
