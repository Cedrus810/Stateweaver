# Stateful Reduced-Order QM/MM  
## 核空间导航与电子空间导航的双层自动化框架

> v2 修订说明（相对 v1，原稿保存为 `*.v1.md`）：
>
> 1. 重新定位创新点：核心贡献是**多 lineage 电子态管理 + 显式电子歧义输出**；低秩外推降级为采用已有方法（ASPC / XL-BOMD / Grassmann extrapolation）的工程组件。
> 2. 修正 §6 中对 \(C_{\rm occ}\) 直接做 POD/SVD 的问题（规范自由度、移动基组、Fock build 成本）。
> 3. 新增 surface policy（多态时 E,F 来自哪个面）与 continuation 规则（MOM vs aufbau）。
> 4. 针对 heme：多态是常态而非例外；阈值需与泛函误差挂钩。
> 5. MR_REQUIRED 明确为 flag + 选定帧 single-point 验证，而非 force engine。
> 6. alarm 分级；fingerprint 要求轨道旋转不变。
> 7. 新增评测设计（state recall）、benchmark 体系与相关工作。
>
> v2.2（2026-09-28）：§10.6 Phase 3a（STATE_SEARCH 的 d 占据枚举、多 lineage beam）的实现与 [Fe(H₂O)₆]²⁺ 实测、遗留问题。
>
> v2.1（2026-09-28，Phase 1–2 验收后）：§7.4 单 lineage IMOM 的静默漂移与 `gap_sign_flip` / `imom_fallback`；§8 alarm 表更新与"事件驱动看不到别处更低态"的局限；§10.5 定期态探测；§17 同态统计的 replay 与实测数据、Phase 1/2–3 测试体系。

---

## 1. 目标

目标不是开发新的通用 MLP，也不是要求机器学习势直接理解过渡金属的：

- 氧化态变化；
- 高/低自旋；
- broken-symmetry；
- 电荷转移；
- 多参考电子结构；
- spin crossing。

真正目标是把传统 QM/MM 中大量依赖人工经验的电子结构操作，封装成一个尽可能接近 MM force engine 的自动化 QM 层。

理想接口为：

\[
(\mathbf R_t,\mathrm{MM\ environment},Q,\mathcal S_{t-1},\pi)
\rightarrow
(E_t,\mathbf F_t,\mathcal S_t,\mathrm{metadata})
\]

其中：

- \(\mathbf R_t\)：当前核坐标；
- \(Q\)：QM 区总电荷；
- \(\mathcal S_{t-1}\)：上一帧电子状态（可包含多条 lineage）；
- \(\pi\)：surface policy（多态时 \(E,\mathbf F\) 取自哪个面，见 §11）；
- \(E_t\)：当前能量；
- \(\mathbf F_t\)：当前力；
- \(\mathcal S_t\)：更新后的电子状态；
- `metadata`：态间能隙、spin character、SCF 稳定性、可信度等。

核心思想：

> **TS/path 方法负责导航核空间；stateful QM engine 负责导航电子空间并管理电子态历史。**

---

## 2. 定位与贡献

### 2.1 本项目**不**声称的新颖性

以下内容已有成熟方法，本项目直接采用并作为 baseline：

| 问题 | 已有方法 |
|---|---|
| 利用历史帧减少每步 SCF 次数 | ASPC（Kolafa 2004）、Fock matrix dynamics（Pulay & Fogarasi 2004） |
| 不完全 SCF 下的能量守恒 / 时间可逆 | XL-BOMD（Niklasson 2008 及后续） |
| 密度矩阵在流形上的外推 | Grassmann extrapolation（Polack, Dusson, Stamm, Lipparini 2021） |
| 激发/非基态 SCF 解的追踪 | MOM / IMOM（Gilbert–Besley–Gill 2008；Barca–Gilbert–Gill 2018） |
| 寻找多个 SCF 解 | SCF metadynamics（Thom & Head-Gordon 2008）；Vaucher & Reiher 2017 |
| 两态反应性、MECP | Shaik et al.（TSR，P450）；Harvey（MECP） |

### 2.2 本项目的核心贡献

\[
\boxed{
\text{一个把电子态历史、多 lineage 分叉/剪枝、电子歧义显式化}
\text{ 封装进 QM/MM force-provider 接口的自动化引擎}
}
\]

具体包括：

1. **Electronic lineage**：把每条电子态分支作为有身份、有历史、有 fingerprint 的对象持续追踪；
2. **事件驱动的分叉 + 持续多态并行**：正常区域 \(K=1\)，heme 等体系允许 \(K>1\) 长期并存；
3. **显式 surface policy**：多态时明确 \(E,\mathbf F\) 的来源与不连续性；
4. **可信度与歧义作为一等输出**：`UNRESOLVED` / `MR_REQUIRED` 是合法返回值；
5. **可量化的评测协议**：state recall（§17）。

> 命名备注：标题中的 "Reduced-Order" 容易让审稿人把注意力放在最不新颖的部分（低秩外推）。后续可考虑改为强调 stateful / multi-lineage 的名称。

---

# 3. 总体结构

```text
Optional upstream proposal
MLPATH / manual hypothesis / known enzyme chemistry
                    │
                    ▼
        ┌───────────────────────┐
        │  Nuclear-space layer  │
        │                       │
        │ MD / string / NEB     │
        │ dimer / P-RFO / MECP  │
        │ constrained dynamics  │
        └───────────┬───────────┘
                    │  requests E,F  (+ surface policy π)
                    ▼
       ┌─────────────────────────────┐
       │  Stateful QM force engine   │
       │                             │
       │  continuation (外推 + SCF)   │  ← 采用 ASPC / XL-BOMD / Grassmann
       │  lineage manager            │  ← 核心贡献
       │  alarm → state search       │
       │  branching / pruning        │
       │  surface policy             │
       │  MR flag & verification     │
       └───────────┬─────────────────┘
                   │
                   ▼
    E / F / lineages / state metadata / status
```

这里不假设 MLPATH 能可靠完成 reaction-space navigation。

它最多是：

\[
\boxed{\text{candidate generator}}
\]

而不是整个方法成立的必要条件。

---

# 4. Layer I：Nuclear-Space Navigation

## 4.1 这一层解决什么问题

这一层只回答：

\[
\boxed{\text{下一个核坐标 } \mathbf R_{t+1} \text{ 应该在哪里？}}
\]

它不负责决定电子结构。

可使用传统或现代方法：

- MD；
- constrained MD；
- NEB / CI-NEB；
- string method；
- dimer；
- eigenvector following；
- P-RFO；
- MECP geometry optimization；
- umbrella / metadynamics；
- externally generated reaction corridor。

对于已知反应物/产物，还可以使用：

\[
R_0,R_1,\ldots,R_N
\]

这样一条粗糙的 nuclear corridor。

## 4.2 与传统 TS 方法的关系

TS/path 方法优化的是核坐标：

\[
\mathbf R_k\rightarrow\mathbf R_{k+1}
\]

例如 TS 搜索最终希望满足：

\[
\nabla_R E(\mathbf R)=0
\]

并具有一个负 Hessian 本征值。

本项目不试图替代这一层，而是让这些方法调用一个更鲁棒、状态可追踪的 QM force provider：

```text
P-RFO
   │
   ├── geometry R1 → AutoQM → E,F,lineages
   ├── geometry R2 → AutoQM → E,F,lineages
   ├── geometry R3 → AutoQM → E,F,lineages
   └── ...
```

同样的 AutoQM 可以被 MD、NEB、MECP、string 等共同使用。

## 4.3 不同上层调用方式的收益差异

| 上层 | 相邻调用间 \(\Delta R\) | continuation 收益 | 对收敛精度要求 |
|---|---|---|---|
| MD | 很小（~0.01 Å） | 最大 | 中（需控制漂移） |
| NEB / string（迭代间同一 image） | 小 | 较大 | 高 |
| P-RFO / dimer | 中等 | 中（主流程序已默认读上一步 guess） | 高 |
| MECP | 中等 | 中；但**两态同时追踪**价值最大 | 高 |

结论：speedup 主要体现在 MD；对优化器而言，主要价值在于**态身份稳定**（不在两步之间悄悄换根），而不是省 SCF 次数。

---

# 5. Layer II：Electronic-Space Navigation

传统电子结构计算通常近似表现为：

```text
geometry R
   ↓
guess orbitals
   ↓
SCF
   ↓
converged wavefunction
   ↓
E,F
```

而连续 QM/MM trajectory/path 中：

\[
R_{t+1}\approx R_t
\quad\Rightarrow\quad
P_{t+1}\approx P_t
\]

其中 \(P\) 为一粒子密度矩阵（占据子空间投影子）。

因此把电子问题改写成一个**有记忆的动态系统**。但要注意：

- 在**单态**意义上，"有记忆的 SCF" 已被 ASPC / XL-BOMD 等充分研究（§2.1）；
- 本项目的新意在于**多态**意义上的记忆：记住有哪些电子态、它们各自从哪里来、是否仍然存活。

---

# 6. Electronic Frame 与 Lineage

## 6.1 Electronic frame

单条电子态在某一几何下的快照：

\[
\mathcal F_t=
\{
P_t^{\alpha},P_t^{\beta},
C_{\mathrm{occ},t}^{\alpha},C_{\mathrm{occ},t}^{\beta},
n_t,
M,
\boldsymbol\rho_s,
\text{SCF history}
\}
\]

可包括：

- α/β 密度矩阵；
- occupied orbital subspace；
- occupations（MOM 参考轨道）；
- total multiplicity \(M\)；
- local spin populations \(\boldsymbol\rho_s\)；
- DIIS vectors；
- previous Fock/Kohn–Sham matrices；
- 外推器内部状态（ASPC 历史 / XL-BOMD 辅助密度）。

## 6.2 Electronic lineage

一条 lineage 是同一电子态沿核路径的连续历史：

```text
Lineage {
    id
    parent_id, birth_step          # 从哪条 lineage、在哪一步分叉出来
    multiplicity, BS_pattern       # 例如 M=2, Fe↑ / porphyrin↓
    fingerprint_history            # 见 §6.3
    energy_history
    frames (最近 m 帧)              # 供外推使用
    status: ACTIVE | DORMANT | PRUNED
}
```

QM engine 实际计算的是：

\[
(\mathbf R_t,\{\mathcal L^{(k)}_{t-1}\}_{k=1}^{K})
\rightarrow
(\{\mathcal L^{(k)}_{t}\}_{k=1}^{K'},E_t,\mathbf F_t)
\]

## 6.3 Fingerprint：必须对轨道旋转不变

态身份判定不能依赖 MO 系数或轨道序号（近简并时会任意旋转/换序）。使用旋转不变量：

- \(\langle S^2\rangle\)（以及 Yamaguchi 投影前后的差）；
- Fe、O、S（axial Cys）、卟啉环、底物上的局域自旋布居（Mulliken / Hirshfeld / IAO）；
- 局域电荷；
- 基于局域轨道的氧化态分析（IBO / effective oxidation state）；
- 与上一帧同 lineage 的 determinant overlap（§8.1）。

例如 P450 Compound I 中，doublet 与 quartet 的区别是 Fe=O 三重态单元与卟啉 \(a_{2u}\) 自由基（或 thiyl 自由基）的自旋耦合方式，fingerprint 需要能区分卟啉自由基与 thiyl 自由基。

---

# 7. Continuation：外推 + 少量 SCF（采用已有方法）

## 7.1 为什么不能直接对 \(C_{\rm occ}\) 做 POD/SVD

v1 中提出：

\[
U_t=\operatorname{POD/SVD}(C_{t-m},\ldots,C_t)
\]

该做法存在三个问题：

1. **规范自由度**：\(C_{\rm occ}\rightarrow C_{\rm occ}W\)（\(W\) 为任意幺正矩阵）不改变物理状态；符号翻转、近简并轨道换序同理。不同帧的 \(C\) 直接拼接做 SVD 是 ill-posed 的，必须先对齐（Procrustes）或改为在 Grassmann 流形上操作。
2. **移动基组**：AO 基函数随原子移动，\(C_t\) 与 \(C_{t+1}\) 属于不同基组，\(S_t\neq S_{t+1}\)。需在 Löwdin 正交化表示 \(\tilde C=S^{1/2}C\) 下处理。
3. **子空间求解省不下主要成本**：求解
   \[
   U_t^\dagger F U_t\,c=\epsilon\,U_t^\dagger S U_t\,c
   \]
   前仍需构建完整 \(F[P]\)，且 \(F\) 依赖于 \(P\)（仍是自洽问题）。省下的只是对角化，而 heme QM 区（~2000 AO）的瓶颈是 Fock build。此外 \(r\ge N_{\rm occ}\)（~300）是硬约束，压缩比有限。

因此 reduced-basis 的正确定位是：

\[
\boxed{\text{高质量初猜（外推器），而不是廉价求解器}}
\]

## 7.2 采用的外推方案

**Grassmann extrapolation**（推荐作为默认，对每条 lineage、每个自旋分量独立执行）：

\[
\tilde C_i = S_i^{1/2}C_{{\rm occ},i},
\qquad
\Gamma_i=\operatorname{Log}_{\tilde C_{\rm ref}}(\tilde C_i)
\]

在切空间对 \(\{\Gamma_{t-m},\ldots,\Gamma_t\}\) 做线性/最小二乘外推得到 \(\Gamma_{t+1}\)，再映回：

\[
\tilde C_{t+1}^{\rm guess}=\operatorname{Exp}_{\tilde C_{\rm ref}}(\Gamma_{t+1}),
\qquad
C_{t+1}^{\rm guess}=S_{t+1}^{-1/2}\tilde C_{t+1}^{\rm guess}
\]

"低秩结构"在这里体现为：外推系数可在切空间历史上用 POD/最小二乘确定——这是 v1 思路在数学上正确的版本。

**MD 下的能量守恒**：不完全收敛的 SCF + 非时间可逆外推会导致系统性能量漂移。MD 场景推荐：

- ASPC（时间可逆、稳定，CP2K 默认）；或
- XL-BOMD（辅助密度随核一起 Verlet 传播，力来自 shadow energy，可做到每步 1–2 次 SCF 且长期守恒）。

## 7.3 正常流程

```text
lineage 最近 m 帧
        ↓
Grassmann / ASPC / XL-BOMD 外推
        ↓
1–5 次 SCF correction（MOM 约束，见 §7.4）
        ↓
E,F + fingerprint
```

## 7.4 Continuation 规则：MOM vs aufbau

heme 的 Fe 3d 轨道近简并，每步 SCF 可按两种规则填充：

| 规则 | 行为 | 对应的面 |
|---|---|---|
| aufbau | 每步按轨道能量重新填充 | adiabatic；近简并处会悄悄换根 |
| MOM / IMOM | 保持与上一帧占据空间最大重叠 | diabatic-like；沿同一 lineage 连续 |

默认对每条 lineage 使用 **IMOM continuation**（保证身份连续），由 lineage manager 在更高层比较各 lineage 的能量决定 adiabatic 排序。这样换根变成一个**被检测、被记录的事件**，而不是 SCF 内部的静默行为。

**v2.1 验收修正（实测）**：只有一条 lineage 时，"更高层比较"不存在，IMOM 会**静默地**沿 diabatic 态走到激发态上：乙烯扭转 0°→180°，IMOM 在 180° 落在 (π*)² 上，比基态高 0.461 Ha，且每步 overlap 都很高、|gap| 从未进入 small-gap 窗口。因此：

- 新增 Tier-0 alarm `gap_sign_flip`：任一自旋通道 frontier gap 相对上一步变号即报警（与步长无关）；
- IMOM 钉在一个已不是 SCF 极小的态上时会不收敛（[Fe(H₂O)₆]²⁺ 上 10 步中有 4 步到 100 cycles）：此时用同一初猜改 aufbau 重做该步，并报 `imom_fallback`。

---

# 8. Electronic-State Alarm（分级）

## 8.1 各级指标

**Tier 0 —— 每步，几乎零成本**

- SCF 迭代次数异常升高（只在用了历史初猜的步上判断；从头算的第 0 步不算）；
- 各自旋通道 frontier gap **变号**（`gap_sign_flip`，见 §7.4）；
- IMOM 不收敛、改用 aufbau 重做（`imom_fallback`）；
- 实际能量与外推预测能量之差；
- \(\langle S^2\rangle\) 变化；
- 局域自旋布居变化（Fe / O / S / 卟啉 / 底物）；
- 各自旋通道 frontier gap；
- \(\|P_{t+1}-P_t\|\)（在正交化表示下）。

**Tier 1 —— 每步或每 n 步，低成本**

- 跨几何 determinant overlap：
  \[
  O=\left|\det\!\left(C_{{\rm occ},t}^{\dagger}\,S(\mathbf R_t,\mathbf R_{t+1})\,C_{{\rm occ},t+1}\right)\right|
  \]
  其中 \(S(\mathbf R_t,\mathbf R_{t+1})\) 为跨几何 AO overlap（PySCF：`gto.intor_cross`），α/β 分别计算后相乘；
- **alarm 使用占据重叠矩阵的最小奇异值** \(\sigma_{\min}\)（最大主角的余弦），而不是 \(O\) 本身：\(O=\prod_i\sigma_i\) 随电子数指数衰减（heme ~300 电子时正常一步也可能 \(O<0.8\)），\(\sigma_{\min}\) 是尺寸强度量，单个轨道换根时 \(\to0\)。\(O\) 仍作为 metadata 返回；
- 活跃 lineage 之间的能隙变化。

注：v1 中的 \(\|P_{t+1}-P_t\|\) 由 \(\sigma_{\min}\) 取代（跨几何比较密度需要同样的 cross overlap，且信息冗余）；"实际能量 vs 外推预测能量"指标推迟到有 MD 积分器之后再加。

**Tier 2 —— 触发后执行，或按计划定期执行，中等成本**

- SCF 稳定性分析（internal / RKS→UKS / 实→复）：需要对轨道 Hessian 做 Davidson 求解，代价接近若干次 Fock build；
- 局域轨道氧化态分析；
- **定期态探测**（`probe_every`，§10.5）：与 lineage 无关的独立 SCF，低于 lineage 超过 `probe_tol` 即报 `lower_state_found`。

**Tier 3 —— 触发后执行，高成本**

- STATE_SEARCH（§10）。

## 8.2 触发逻辑

```text
Tier 0/1 任一指标超阈值
        │
        ▼
   Tier 2 确认
        │
   ├── 稳定 & fingerprint 连续 → 误报，回到 FAST
   └── 不稳定 或 fingerprint 跳变 → Tier 3
```

**局限（v2.1 实测）**：这条链是**事件驱动**的，只能发现"本 lineage 发生了变化"，发现不了"别处出现了更低的态"。continuation（IMOM 或 aufbau）停留在自己的 SCF 局部极小里，轨迹局部完全平滑：[Fe(H₂O)₆]²⁺ 上 continuation 最多比从头算找到的另一个 t₂g 占据高 1.75 mHa（≈1.1 kcal/mol），overlap 连续、gap 不变号、稳定性分析也不会报（它本身是稳定极小）。这类情况只能靠 §10.5 的定期探测。

---

# 9. 单电子帧的失效：Heme / Transition Metal

对于普通闭壳层区域，\(P(R)\) 通常是平滑的。

但 heme 可能出现两条（或多条）竞争电子分支 \(P_A(R)\)、\(P_B(R)\)，例如：

- doublet / quartet（Compound I）；
- FeII / FeIII-like；
- superoxo / peroxo；
- different broken-symmetry states；
- 卟啉自由基 vs thiyl 自由基 vs 底物自由基。

不能将两者平均成：

\[
\bar P=\tfrac12(P_A+P_B)
\]

否则会得到没有明确物理意义的"平均电子态"。

## 9.1 对 heme 而言，多态是常态而非例外

两个事实：

1. DFT 对 heme 自旋态能量的泛函依赖本身就在 **5–15 kcal/mol** 量级（B3LYP / B3LYP* / TPSSh / 不同 HF exchange 比例之间）；
2. P450 Compound I 的 doublet / quartet 在整条 H 抽提路径上几乎简并（two-state reactivity）。

因此：

- 若 \(\Delta E_{\rm keep}\) 设得比泛函误差小，会错误地剪掉物理上不可排除的态；
- 若设得合理（~3–5 kcal/mol 或更大），heme 体系很多时候会长期处于 \(K\ge2\)。

设计上把

\[
\boxed{\text{K 条 lineage 长期并行}}
\]

作为 heme 的**常规运行模式**，成本约为 \(K\) 倍。这不是缺点：它直接为 TSR / MECP 分析提供输入。

---

# 10. Multi-Lineage Tracking：分叉、剪枝与 State Search

## 10.1 Beam

\[
\mathcal B_t=\{\mathcal L_t^{(1)},\ldots,\mathcal L_t^{(K)}\},
\qquad K\approx1\sim4
\]

而不是穷举 spin × oxidation × BS × occupation。

- 普通区域：\(K=1\)；
- alarm 触发：\(K=1\rightarrow2\rightarrow3\)；
- heme 活性中心：允许 \(K\ge2\) 长期存在。

## 10.2 STATE_SEARCH：候选态生成

触发后自动尝试（每项均产生一个 SCF 初猜，收敛后去重）：

- 当前 lineage 的 continuation；
- 相邻多重度 \(M\pm2\)（即 \(S\pm1\)）；
- 沿稳定性分析最负本征向量方向扰动后重新收敛（stability following）；
- fragment-based BS guess（翻转 Fe / 卟啉 / 底物 / O₂ 片段上的自旋）；
- MOM / IMOM 占据激发（在 Fe 3d 与卟啉 π 之间交换占据）；
- charge-localized guess；
- 必要时 SCF metadynamics（在已找到的解附近加 bias 以寻找新解）。

**去重**：用 §6.3 的 fingerprint 与 determinant overlap 聚类；相同 fingerprint 且 overlap > 阈值视为同一态。

## 10.3 剪枝（带滞回）

\[
E_i-E_{\min}>\Delta E_{\rm prune}
\quad\text{连续 } n_{\rm prune}\text{ 步}
\;\Rightarrow\;
\text{ACTIVE}\rightarrow\text{DORMANT}
\]

- 设置连续步数要求，避免在热涨落下反复生灭；
- DORMANT lineage 保留最后一帧，若之后能隙回落可**直接复活**，无需重新 search；
- 长期 DORMANT 后才转为 PRUNED。

\[
|E_i-E_j|<\Delta E_{\rm keep}
\;\Rightarrow\;
\text{同时保留}
\]

## 10.4 能量比较的注意事项

- BS 态与纯自旋态比较前需做自旋投影（Yamaguchi）；
- 可选的 **functional sensitivity** 诊断：对 DORMANT 边缘的态，在少量帧上用不同 HF exchange 比例重算态间能隙；若排序翻转，则标记低可信度。

关键原则：

> **Near-degeneracy 不要求算法强行选出唯一 ground state。**

## 10.5 定期态探测（v2.1 新增）

§8.2 的事件链看不到"lineage 之外出现更低的态"。因此除了 alarm 触发的 search，还需要**按计划**的探测：

- 每 `probe_every` 步，在同一几何上做一次与 lineage 无关的 SCF（不用历史、不用 IMOM、不算梯度）；
- 若 \(E_{\rm lineage}-E_{\rm probe}>\) `probe_tol`（默认 0.1 mHa）→ Tier-2 alarm `lower_state_found`，交给 Phase 3 分叉；
- 成本：每次探测多一次完整 SCF；金属中心 / 活性位点建议 `probe_every` 取 5–20。

**单一初猜不够**：Phase 2 的最小实现只用 minao 一个初猜。在轨道简并的金属中心（⁵T₂g 的三个 t₂g 占据）上，minao 会任意落进其中一个解，甚至比 lineage 还高，探测就看不到更低的那个。实测还发现：同一初猜只因线程数不同，就会落到相差 < 0.03 mHa 的不同解上，即这类 SCF 能面极其平坦。Phase 3 的探测必须**系统枚举**金属 d 壳层的占据（每个 t₂g/e_g 占据模式一个 MOM 初猜，加 §10.2 的 BS / 相邻多重度），并以最低者为参照。

## 10.6 Phase 3a 实现与实测（v2.2）

**STATE_SEARCH（`search_states`）**：候选 = 一次独立 scratch/aufbau SCF + 对基态轨道逐自旋通道枚举金属 d 电子在"d 主导 MO"（meta-Löwdin d 布居 > 0.5，HOMO ±15 内）之间的全部占据方式，每个用 MOM 从基态轨道收敛；scratch 不收敛时其轨道仍作为枚举基；以占据空间 \(\sigma_{\min}\ge0.9\) 去重，按能量排序。原型（[Fe(H₂O)₆]²⁺ MD 第 3 步）：5 个候选共 ~80 s，找回了单一初猜漏掉的最低 t₂g 解（β 电子在另一个 t₂g：0 / +0.369 / +0.488 mHa，e_g：+37–38 mHa）。

**Beam（`BeamEngine`）**：每条 lineage 是一个单 lineage `Engine`；第 0 步搜索并为 keep window 内每个态建 lineage；之后每步推进全部活跃 lineage，在 `gap_sign_flip` / `overlap_drop` / `imom_fallback` 或按 `search_every` 时重新搜索；新态 spawn、匹配到的休眠 lineage revive；beam 满时低于最高 lineage 的新态替换它；变成同一态的 lineage 合并；超出 prune window `n_prune` 步转休眠。surface policy：`adiabatic_min`（`switched` 标记 cusp）/ `follow`；每步结果都带全部活跃 lineage 的 E、F。

**实测（[Fe(H₂O)₆]²⁺，第一轮参考几何，def2-SVP/B3LYP）**：

- 第 0 步在严格八面体上建出 3 条**简并**的 t₂g lineage，它们只是简并空间里的任意取向；几何一扭曲，最低态是随扭曲方向走的特定组合，3 条 IMOM lineage 都不是它（第 1–3 步 +0.24–0.49 mHa）。只有重新搜索才能找到它。
- `search_every=5`：第 1–3 步漏掉，第 4 步 alarm 触发搜索后全程停在最低态（+0.001–0.004 mHa）；11 步 ~26 min（单 lineage ~8–12 min）。
- `search_every=1`：第 1 步起全程停在最低态（+0.000–0.004 mHa）；11 步 ~53 min（~290 s/步），beam 在 t₂g 变体间反复 spawn / merge / revive。该次运行还暴露出 revive 不检查容量（活跃 lineage 达到 5 > `max_lineages`=4），已修复并加回归测试。
- 不合并时，简并空间里"任意取向"的 lineage 是鞍点，IMOM 守不住，每条最多 100 + 100（fallback）cycles：一步 604 cycles / 656 s。加入合并后同一步 214 cycles。

**尚未解决（Phase 3b）**：

1. 对称点出发的 lineage 取向任意：对称性破缺初期需要更频繁的搜索（或在对称点只建一条、之后按需分叉）；
2. keep window（8 mHa）远大于 t₂g 劈裂（≤ 1.5 mHa），beam 会被同一流形里的近简并变体占满并反复 spawn/merge/revive；需要"同一 d 流形只保留最低 + 必要的变体"之类的上限，或按 fingerprint 聚类；
3. 成本：每次搜索 ≈ 5–6 次 SCF；beam 每步 = K 次 SCF+梯度。

---

# 11. Surface Policy：多态时 E,F 从哪里来

当 \(K>1\) 时，上层必须明确使用哪个面。这不是实现细节，而是物理选择，因此作为接口参数 \(\pi\)：

| policy | 含义 | 适用 | 代价 / 风险 |
|---|---|---|---|
| `follow_lineage` | 始终返回指定 lineage 的 \(E,\mathbf F\)（diabatic，固定自旋） | 单自旋态 MD、单态 NEB | 可能跑在激发态面上；需报告与最低态的能隙 |
| `adiabatic_min` | 每步返回能量最低 lineage | 快速探索 | 切换时 PES 出现 cusp，\(E,\mathbf F\) 不连续，NVE 能量不守恒；每次切换必须记录 |
| `all_states` | 返回所有 lineage 的 \(E^{(k)},\mathbf F^{(k)}\) | MECP、TSR 分析、外部 surface hopping（需 SOC） | 成本 \(\times K\) |

对优化器：

- 单态 TS 搜索：`follow_lineage`；
- MECP：`all_states`，取两条 lineage；
- two-state reactivity 分析：`all_states` + 沿路径记录各 lineage 的能量曲线。

---

# 12. QM Engine 状态机

## FAST

所有活跃 lineage 电子态连续。

```text
每条 lineage：外推 → 1–5 次 SCF (IMOM) → E,F,fingerprint
Tier 0/1 alarm 检查
按 surface policy 返回
```

## STATE_SEARCH

Tier 2 确认异常。执行 §10.2，聚类去重后保留少数候选，进入 FAST 或 MULTISTATE。

## MULTISTATE

两个或多个 lineage 低能竞争。返回：

```text
lineages: A, B, ...
ΔE (投影前/后)
fingerprints
crossing risk（能隙趋势外推）
surface-policy 下的 E,F
```

上层决定：continuation / MECP / ET treatment / surface hopping / mark 后停止。

## MR_REQUIRED

检测到明显 strong correlation 迹象：

- SCF 解高度不稳定，多个解几乎简并；
- 态间排序对泛函异常敏感（§10.4）；
- 稳定性分析在多个方向同时不稳定。

注意：UKS 自然轨道占据数偏离整数**多数时候反映 BS 自旋污染，而不是真正的多参考特征**，只能作为弱信号，不能单独触发。

**MR 的现实定位**：heme 的活性空间（Fe 3d + 3d′ 双壳层 + 卟啉 π + O₂ π*）至少在 (16,15) 量级。在 QM/MM MD 或 TS 搜索中逐步运行 CASSCF/NEVPT2 梯度不可行。因此：

\[
\boxed{\text{MR\_REQUIRED} = \text{标记} + \text{选定帧上的 CASSCF/NEVPT2/DMRG single-point 验证}}
\]

而不是切换 force engine。

---

# 13. Full-Auto 不等于 Full-Answer

系统必须允许：

\[
\boxed{\texttt{UNRESOLVED}}
\]

例如：

```text
status: ELECTRONICALLY_AMBIGUOUS

lineages:
  A (M=2, Fe=O triplet ⊗ a2u radical, AF):  +0.0 kcal/mol
  B (M=4, Fe=O triplet ⊗ a2u radical, F):   +0.8 kcal/mol
  C (M=2, substrate radical):               +1.7 kcal/mol

functional_sensitivity: A/B ordering flips at 15% HF exchange
single_reference_confidence: medium
recommended_action: keep A,B as parallel lineages; MR single-point on selected frames
```

这比自动选：

```text
quartet
```

但实际上选错电子根安全得多。

注意：若没有泛函误差条，"+0.8 kcal/mol" 这类数字本身几乎没有区分意义，因此 `functional_sensitivity` 应尽量作为输出的一部分。

---

# 14. 与 QM/MM 的结合

```text
protein / solvent
       │
       └── MM

heme + substrate + reactive residues
       │
       └── Stateful QM
```

QM 区保持相对固定：

- heme；
- metal；
- axial ligand；
- substrate reactive atoms；
- O₂ / oxo；
- 必要 proton relay；
- 少量关键 residue。

这样避免同时解决 adaptive QM region + electronic-state tracking 两个困难问题。

补充：

- 静电嵌入下，即使 QM 几何不变，MM 环境运动也会改变 \(P\)；外推器与 alarm 应把 MM 点电荷变化视为输入的一部分；
- \(Q\) 固定意味着只处理 QM 区**内部**的电荷转移（Fe ↔ 卟啉 ↔ 底物）；来自 QM 区外（如 reductase、远程氧化还原中心）的电子转移不在本框架范围内。

---

# 15. 成本模型

传统连续计算：

\[
N_{\rm step}\times N_{\rm SCF}\times C_{\rm Fock}
\]

本框架：

\[
N_{\rm step}\times\sum_{k=1}^{K_t}N^{(k)}_{\rm correct}\times C_{\rm Fock}
\;+\;
N_{\rm step}\times C_{\rm alarm}
\;+\;
N_{\rm event}\times C_{\rm search}
\;+\;
N_{\rm verify}\times C_{\rm MR}
\]

- \(N_{\rm correct}\ll N_{\rm SCF}\)：这部分收益来自已有外推方法，是 baseline 能力；
- \(N_{\rm event}\ll N_{\rm step}\)：对普通有机/闭壳层区域成立；**对 heme 活性中心不一定成立**，此时成本主项是 \(K\times\)；
- 真正的价值不只是 speedup，而是：

\[
\boxed{\text{在相近成本下，不漏掉低能电子态、不静默换根}}
\]

---

# 16. 真正的性能瓶颈

即使 SCF iteration 从 \(20\rightarrow3\)，仍然需要：

- Fock/Kohn–Sham build；
- Coulomb；
- exact exchange；
- XC grid；
- gradient；
- QM/MM electrostatic embedding。

所以 continuation 不会把 DFT 直接变成 MACE 速度。它优化的是：

\[
\boxed{\text{重复求解成本}}
\]

不是：

\[
\boxed{\text{单次 Hamiltonian evaluation 成本}}
\]

若继续优化性能：

- density fitting / RIJCOSX；
- incremental Fock build（\(\Delta P\) 较小时只算增量）；
- localized orbital / sparse exchange；
- GPU integral kernels（如 GPU4PySCF）；
- MTS QM/MM；
- 多 lineage 共享积分 / 批量 Fock build（\(K\) 条 lineage 在同一几何下可共享 J/K 引擎的积分筛选）。

---

# 17. 评测设计

## 17.1 Continuation（Phase 1）

- 每步 SCF 次数；
- NVE 能量漂移（per ps per atom）；
- 力误差：相对 tight convergence 的 RMS force error；
- wall-clock；
- baseline：`guess=read`（上一步轨道）、ASPC、XL-BOMD。

**v2.1 修正**：replay 的参考轨迹（从头算）本身会在近简并态之间跳，所以 \(|\Delta E|\) 只在"与参考同一态"（占据空间 \(\sigma_{\min}\ge0.9\)）的步上统计，其余步单独计为 `state_mismatch_steps`——不同态不是外推误差。

实测（kasuga02，def2-SVP / B3LYP，每步平均 SCF 次数）：

| 体系 | scratch | read | aspc | grassmann | \(\max|\Delta E|\) |
|---|---|---|---|---|---|
| 水二聚体（闭壳层，20 步，1e-7） | 7 | 5 | 3.95 | **3.65** | ≤ 2e-8 |
| [Fe(H₂O)₆]³⁺ 六重态（⁶A₁g，10 步，1e-7） | 12.3 | 6.4 | 5.1 | **4.6** | ≤ 2e-7 |

总耗时的下降远小于 SCF 次数的下降（水二聚体只少 15%）：小体系上每步的梯度 + 格点 / DF 重建是固定成本，与 §16 一致。

## 17.2 State recall（核心指标）

在轨迹 / 路径上采样 \(N_{\rm sample}\) 帧，对每帧做**穷举式参考搜索**：

\[
\text{spin}\in\{M, M\pm2\}\times\text{BS patterns}\times\text{MOM 占据激发}\times\text{多个初猜}
\]

得到能量窗口 \(W\)（如 10 kcal/mol）内的参考态集合 \(\mathcal R\)。

定义：

\[
\text{Recall}=\frac{|\mathcal R\cap\mathcal B_{\rm engine}|}{|\mathcal R|},
\qquad
\text{Ground-state miss rate}=P(\text{最低参考态}\notin\mathcal B_{\rm engine})
\]

以及：

- 伪分叉率（spawn 后很快被剪枝的比例）；
- 静默换根次数（lineage fingerprint 跳变但未触发 alarm）；
- 相对穷举搜索的成本比。

这比单纯报告 speedup 有说服力得多。

## 17.3 Benchmark 体系

| 阶段 | 体系 | 考察点 |
|---|---|---|
| Phase 1 | 水二聚体；**[Fe(H₂O)₆]³⁺ 六重态**（d⁵，⁶A₁g，无轨道简并） | continuation 的 SCF 次数、精度、漂移 |
| Phase 2–3 | **[Fe(H₂O)₆]²⁺ 五重态**（d⁶，⁵T₂g，三个近简并 t₂g 占据） | 态追踪压力测试：continuation 停在高 ≤1.75 mHa 的解上而无 overlap alarm；定期探测 / 占据枚举 |
| 气相小模型 | Fe(P)、Fe(P)(Im)、Fe(P)(SH⁻)=O（Cpd I 模型） | 自旋态 lineage、BS continuation |
| 气相小模型 | Fe(P)(Im)–O₂ | open-shell singlet / triplet / quintet 竞争，MR 争议 |
| QM/MM | **P450cam Cpd I + camphor H 抽提** | 经典 TSR，大量 QM/MM 文献可对照 |
| QM/MM | **Mb–O₂** | Fe–O₂ 键合电子结构，测试 MR_REQUIRED |

---

# 18. 与 MLPATH 的关系

当前不把以下假设作为核心：

\[
\boxed{\text{MLPATH can reliably navigate arbitrary reaction space}}
\]

MLPATH 只保留为 optional upstream component：

```text
MLPATH
manual mechanism
experimental hypothesis
known substrate/product
external reaction generator
          │
          ▼
   nuclear-space layer
```

即使 MLPATH 最终只能给出"这里大概有个坑"，后两层方法仍然成立。

---

# 19. 最小可行版本

第一阶段不碰完整 heme catalytic cycle。先实现一个 stateful QM calculator：

```python
engine = autoqm.Engine(
    method="UKS/B3LYP-D3/def2-SVP",
    extrapolator="grassmann",        # or "aspc", "xlbomd"
    continuation="imom",
    surface_policy="follow_lineage", # or "adiabatic_min", "all_states"
)

result = engine.compute(
    geometry=R,
    embedding=mm,
    charge=Q,
)
```

返回：

```python
result.energy
result.forces
result.surface_policy
result.surface_switched          # adiabatic_min 下本步是否换面

result.lineages                  # 每条: id, parent, M, fingerprint, E, F(可选), status
result.primary_lineage
result.state_gaps                # 投影前/后

result.s2
result.local_spins

result.scf_iterations            # 每条 lineage
result.alarms                    # 触发的 tier 与指标
result.events                    # spawn / dormant / revive / prune

result.status                    # FAST / MULTISTATE / UNRESOLVED / MR_REQUIRED
result.confidence
```

优先解决：

1. 接入现成外推（Grassmann / ASPC），作为 baseline；
2. IMOM continuation；
3. 旋转不变 fingerprint 与跨几何 overlap；
4. lineage 数据结构与事件日志；
5. alarm 分级；
6. 相邻自旋 / BS 分叉 + 带滞回的剪枝；
7. surface policy；
8. explicit unresolved state；
9. state-recall 评测脚手架。

---

# 20. 推荐开发顺序

### Phase 1 — Continuation baseline（短周期，~1–2 周）

直接采用 Grassmann extrapolation / ASPC（PySCF 实现），闭壳层与固定自旋体系上验证：

- SCF 次数 \(20\rightarrow1\sim5\)；
- NVE 能量漂移；
- force continuity；
- 对比 `guess=read`。

**不作为创新点，只作为基础设施。**

### Phase 2 — Stateful open-shell tracking

- UHF/UKS；
- IMOM continuation；
- 旋转不变 fingerprint；
- 跨几何 determinant overlap；
- lineage 数据结构 + 静默换根检测。

### Phase 3 — Adaptive branching

- Tier 0–2 alarm；
- STATE_SEARCH：\(M\rightarrow M,M\pm2\)、fragment BS、stability following；
- 聚类去重；
- 带滞回剪枝与 DORMANT 复活。

### Phase 4 — Heme-like multi-lineage tracking

体系：Fe-porphyrin、Fe–O₂、Fe=O、简化 axial ligand 模型。

重点不是势垒，而是：

\[
\boxed{\text{能否稳定保持多个电子态 lineage，且 state recall 足够高}}
\]

同时建立 state-recall 参考数据集（§17.2）。

### Phase 5 — QM/MM force engine

\[
\text{heme model}
\rightarrow
\text{heme + point charges}
\rightarrow
\text{heme protein QM/MM}
\]

### Phase 6 — Nuclear search coupling

最后接：MD、TS search、string、NEB、MECP（`all_states`）、以及可能的 MLPATH。

目标案例：P450cam Cpd I + camphor 的 two-state H 抽提。

---

# 21. 核心概念总结

### 1.

\[
\boxed{\text{Nuclear algorithms choose where to evaluate.}}
\]

### 2.

\[
\boxed{\text{Stateful QM decides which electronic states exist there, and tracks them.}}
\]

### 3.

\[
\boxed{\text{Electronic ambiguity is represented, not forcibly eliminated.}}
\]

最终目标不是创造一个"永不失败的 QM"，而是创造一个：

\[
\boxed{
\textbf{像 MM calculator 一样被调用，}
\\
\textbf{但在内部自动管理电子态 lineage、分叉/剪枝和歧义输出的 QM engine。}
}
\]

---

# 参考文献（待补全卷页号）

- J. Kolafa, "Time-reversible always stable predictor–corrector method for molecular dynamics of polarizable molecules", *J. Comput. Chem.* 25, 335 (2004).
- P. Pulay, G. Fogarasi, "Fock matrix dynamics", *Chem. Phys. Lett.* 386, 272 (2004).
- A. M. N. Niklasson, "Extended Born–Oppenheimer molecular dynamics", *Phys. Rev. Lett.* 100, 123004 (2008).
- É. Polack, G. Dusson, B. Stamm, F. Lipparini, "Grassmann extrapolation of density matrices for Born–Oppenheimer molecular dynamics", *J. Chem. Theory Comput.* (2021).
- A. T. B. Gilbert, N. A. Besley, P. M. W. Gill, "Self-consistent field calculations of excited states using the maximum overlap method (MOM)", *J. Phys. Chem. A* (2008).
- G. M. J. Barca, A. T. B. Gilbert, P. M. W. Gill, "Simple models for difficult electronic excitations" (IMOM), *J. Chem. Theory Comput.* (2018).
- A. J. W. Thom, M. Head-Gordon, "Locating multiple self-consistent field solutions: an approach inspired by metadynamics", *Phys. Rev. Lett.* 101, 193001 (2008).
- A. C. Vaucher, M. Reiher, "Steering orbital optimization out of local minima and saddle points toward lower energy", *J. Chem. Theory Comput.* (2017).
- S. Shaik, D. Kumar, S. P. de Visser, A. Altun, W. Thiel, "Theoretical perspective on the structure and mechanism of cytochrome P450 enzymes", *Chem. Rev.* 105, 2279 (2005).
- J. N. Harvey 等，MECP 方法与自旋禁阻反应相关工作。
- C. Duan, F. Liu, A. Nandy, H. J. Kulik 等，"Learning from failure: predicting electronic structure calculation outcomes with machine learning models", *J. Chem. Theory Comput.* (2019)；以及该组关于多参考特征检测的后续工作。
- 自旋态能量泛函依赖：Reiher（B3LYP*）、Swart、Radoń 等的 benchmark 工作。
