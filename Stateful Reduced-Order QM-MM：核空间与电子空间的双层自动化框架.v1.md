# Stateful Reduced-Order QM/MM  
## 核空间导航与电子空间导航的双层自动化框架

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
(\mathbf R_t,\mathrm{MM\ environment},Q,\mathcal S_{t-1})
\rightarrow
(E_t,\mathbf F_t,\mathcal S_t,\mathrm{metadata})
\]

其中：

- \(\mathbf R_t\)：当前核坐标；
- \(Q\)：QM 区总电荷；
- \(\mathcal S_{t-1}\)：上一帧电子状态；
- \(E_t\)：当前能量；
- \(\mathbf F_t\)：当前力；
- \(\mathcal S_t\)：更新后的电子状态；
- `metadata`：态间能隙、spin character、SCF 稳定性、可信度等。

核心思想：

> **TS/path 方法负责导航核空间；stateful QM engine 负责导航和压缩电子空间。**

---

# 2. 总体结构

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
                    │  requests E,F
                    ▼
       ┌─────────────────────────┐
       │ Stateful QM force engine│
       │                         │
       │ electronic continuation │
       │ reduced subspace        │
       │ state branching         │
       │ SCF recovery            │
       │ MR escalation           │
       └───────────┬─────────────┘
                   │
                   ▼
        E / F / state metadata
```

这里不假设 MLPATH 能可靠完成 reaction-space navigation。

它最多是：

\[
\boxed{\text{candidate generator}}
\]

而不是整个方法成立的必要条件。

---

# 3. Layer I：Nuclear-Space Navigation

## 3.1 这一层解决什么问题

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

---

## 3.2 与传统 TS 方法的关系

TS/path 方法优化的是核坐标：

\[
\mathbf R_k\rightarrow\mathbf R_{k+1}
\]

例如 TS 搜索最终希望满足：

\[
\nabla_R E(\mathbf R)=0
\]

并具有一个负 Hessian 本征值。

本项目不试图替代这一层。

相反，目标是让这些方法调用一个更鲁棒、更便宜的 QM force provider。

即：

```text
P-RFO
   │
   ├── geometry R1 → AutoQM → E,F
   ├── geometry R2 → AutoQM → E,F
   ├── geometry R3 → AutoQM → E,F
   └── ...
```

同样的 AutoQM 可以被 MD、NEB、MECP、string 等共同使用。

---

# 4. Layer II：Electronic-Space Navigation

这是主要方法学创新所在。

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

每次核坐标变化都重复大量相同工作。

而连续 QM/MM trajectory/path 中：

\[
R_{t+1}\approx R_t
\]

对应的电子结构通常也高度相关：

\[
P_{t+1}\approx P_t
\]

其中 \(P\) 为一粒子密度矩阵或 occupied-subspace projector。

因此应把电子问题改写成一个**有记忆的动态系统**。

---

# 5. Electronic Frame

定义电子帧：

\[
\mathcal F_t=
\{
P_t,
C_{\mathrm{occ},t},
n_t,
S_t,
\rho_s,
\text{SCF history}
\}
\]

可包括：

- density matrix；
- occupied orbital subspace；
- occupations；
- total multiplicity；
- local spin populations；
- orbital ordering；
- DIIS vectors；
- previous Fock/Kohn–Sham matrices；
- electronic-state identity；
- competing low-energy roots。

因此 QM engine 实际计算的是：

\[
(\mathbf R_t,\mathcal F_{t-1})
\rightarrow
(\mathcal F_t,E_t,F_t)
\]

而不是：

\[
\mathbf R_t\rightarrow\text{SCF from scratch}.
\]

---

# 6. Low-Rank / Reduced-Subspace Representation

占据空间：

\[
C_{\rm occ}
\in
\mathbb R^{N_{\rm AO}\times N_{\rm occ}}
\]

对应：

\[
P=C_{\rm occ}C_{\rm occ}^{\dagger}.
\]

因此电子状态天然具有低秩结构。

收集最近若干帧：

\[
\{C_{t-m},\ldots,C_t\}
\]

建立压缩子空间：

\[
U_t=
\operatorname{POD/SVD}
\left(
C_{t-m},\ldots,C_t
\right).
\]

其中：

\[
U_t\in\mathbb R^{N_{\rm AO}\times r},
\qquad
r\ll N_{\rm AO}.
\]

下一帧首先在该子空间中求解：

\[
U_t^\dagger F U_t c
=
\epsilon
U_t^\dagger S U_t c.
\]

得到预测电子结构后，再通过少量完整 SCF correction 收敛。

正常情况下：

```text
previous electronic frames
          ↓
reduced-basis prediction
          ↓
2–5 SCF corrections
          ↓
E,F
```

目标不是完全取消 QM，而是避免：

\[
10\sim30
\]

次重复 SCF iteration。

---

# 7. 为什么这不等于 TS 算法

二者压缩的是两个完全不同的问题。

## Nuclear-space algorithm

解决：

\[
\boxed{\mathbf R_{t+1}\text{ 在哪里？}}
\]

例如：

- dimer；
- P-RFO；
- NEB；
- MD。

## Electronic-space engine

解决：

\[
\boxed{
\text{给定 }\mathbf R_{t+1},
\text{怎样最快、最稳定得到正确的电子状态和 }E,F？
}
\]

因此两者可以直接组合：

\[
\boxed{
\text{nuclear-space search}
+
\text{electronic-space compression}
}
\]

而不是互相替代。

---

# 8. 单电子帧的失效：Heme / Transition Metal

对于普通闭壳层区域：

\[
P(R)
\]

通常是平滑的。

但 heme 可能出现：

\[
P_A(R)
\]

和：

\[
P_B(R)
\]

两条竞争电子分支，例如：

- doublet / quartet；
- FeII / FeIII-like；
- superoxo / peroxo；
- different broken-symmetry states。

因此不能将两者平均成：

\[
\bar P=
\frac12(P_A+P_B).
\]

否则会得到没有明确物理意义的“平均电子态”。

因此需要：

\[
\boxed{\text{multi-frame electronic tracking}}
\]

---

# 9. Multi-Frame / Beam Electronic Tracking

维护少量电子态：

\[
\mathcal B_t=
\{
\mathcal F_t^{(1)},
\mathcal F_t^{(2)},
\ldots,
\mathcal F_t^{(K)}
\}
\]

其中：

\[
K\approx1\sim4
\]

而不是穷举所有：

\[
\text{spin}\times
\text{oxidation}\times
\text{BS}\times
\text{occupation}.
\]

正常区域：

\[
K=1.
\]

只有电子态出现异常才自动扩展：

\[
K=1\rightarrow2\rightarrow3.
\]

---

# 10. Electronic-State Alarm

触发 branching 的指标可以包括：

### Wavefunction continuity

\[
O=
|\langle\Psi_t|\Psi_{t+1}\rangle|
\]

突然降低。

### Density change

\[
\|P_{t+1}-P_t\|
\]

突然变大。

### SCF stability

当前 solution 出现 unstable orbital rotation。

### Spin change

\[
\Delta\langle S^2\rangle
\]

或者 local spin population 突变。

### Orbital gap

frontier/relevant orbital gap 接近零。

### Occupation instability

natural occupations 明显偏离：

\[
0,\;2
\]

或 UHF 对应整数 occupation。

只有出现这些信号时才启动额外电子态搜索。

---

# 11. Automatic Branching

例如当前为 quartet-like state：

```text
quartet frame
      │
      │ normal continuation
      ▼
quartet frame
      │
      ├── electronic alarm
      │
      ├── continue quartet
      │
      └── spawn doublet / BS candidate
```

随后比较：

\[
E_1,E_2
\]

并计算 continuity/fingerprint。

如果一个状态明显升高：

\[
E_i-E_{\min}>\Delta E_{\rm prune}
\]

即可删除。

如果两个状态始终接近：

\[
|E_1-E_2|<\Delta E_{\rm keep}
\]

则同时保留。

关键原则：

> **Near-degeneracy 不要求算法强行选出唯一 ground state。**

---

# 12. QM Engine 状态机

建议至少定义四级运行模式。

## FAST

电子状态连续。

```text
reduced-space extrapolation
→ few SCF corrections
→ return E,F
```

---

## STATE_SEARCH

发现 electronic alarm。

自动尝试：

- previous-state continuation；
- neighboring multiplicity；
- selected spin flip；
- broken-symmetry guess；
- MOM/IMOM；
- charge-localized guess。

聚类和去重后保留少数候选。

---

## MULTISTATE

两个或多个状态低能竞争。

返回：

```text
state A
state B
ΔE
state fingerprints
crossing risk
```

上层可以决定：

- continuation；
- MECP；
- ET treatment；
- surface hopping；
- 或 mark 后停止。

---

## MR_REQUIRED

检测到明显 strong correlation：

- SCF solutions 高度不稳定；
- 多个 determinants 几乎简并；
- natural occupations 明显 fractional；
- spin ordering 对方法异常敏感。

此时停止强迫 single-reference DFT 给出唯一答案。

升级：

\[
\text{CASSCF/NEVPT2/DMRG/...}
\]

或者直接返回：

```text
status = MR_REQUIRED
```

---

# 13. Full-Auto 不等于 Full-Answer

系统必须允许：

\[
\boxed{\texttt{UNRESOLVED}}
\]

例如：

```text
status: ELECTRONICALLY_AMBIGUOUS

states:
  A: +0.0 kcal/mol
  B: +0.8 kcal/mol
  C: +1.7 kcal/mol

single_reference_confidence: low
recommended_action: multireference
```

这比自动选：

```text
quartet
```

但实际上选错电子根安全得多。

---

# 14. 与 QM/MM 的结合

整体体系：

```text
protein / solvent
       │
       └── MM

heme + substrate + reactive residues
       │
       └── Stateful QM
```

QM 区最好保持相对固定：

- heme；
- metal；
- axial ligand；
- substrate reactive atoms；
- O2 / oxo；
- 必要 proton relay；
- 少量关键 residue。

这样避免同时解决：

\[
\text{adaptive QM region}
+
\text{electronic-state tracking}
\]

两个困难问题。

---

# 15. 为什么可能加速

传统连续计算：

\[
N_{\rm step}
\times
N_{\rm SCF}
\times
C_{\rm SCF}.
\]

目标变成：

\[
N_{\rm step}
\times
N_{\rm correct}
\times
C_{\rm SCF}
+
N_{\rm event}
\times
C_{\rm search}
\]

其中：

\[
N_{\rm correct}\ll N_{\rm SCF}
\]

并且：

\[
N_{\rm event}\ll N_{\rm step}.
\]

也就是说：

> 大多数帧只做廉价 continuation；  
> 少数真正发生电子变化的帧才付昂贵 state-search 成本。

---

# 16. 真正的性能瓶颈

即使 SCF iteration 从：

\[
20\rightarrow3
\]

仍然需要：

- Fock/Kohn–Sham build；
- Coulomb；
- exact exchange；
- XC grid；
- gradient；
- QM/MM electrostatic embedding。

所以 reduced electronic frame 不会把 DFT 直接变成 MACE 速度。

它优化的是：

\[
\boxed{\text{重复求解成本}}
\]

不是：

\[
\boxed{\text{单次 Hamiltonian evaluation 成本}}.
\]

之后如果还要继续优化性能，需要进一步研究：

- reduced-rank exchange；
- localized orbital / sparse exchange；
- incremental Fock build；
- density fitting；
- GPU integral kernels；
- local correlation；
- MTS QM/MM。

---

# 17. 与 MLPATH 的关系

当前不把以下假设作为核心：

\[
\boxed{
\text{MLPATH can reliably navigate arbitrary reaction space}
}
\]

因此 MLPATH 只保留为 optional upstream component：

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

即使 MLPATH 最终只能：

> “这里大概有个坑。”

整个后两层方法仍然成立。

---

# 18. 最小可行版本

第一阶段不要碰完整 heme catalytic cycle。

先实现一个 stateful QM calculator：

```python
result = autoqm.compute(
    geometry=R,
    embedding=mm,
    charge=Q,
    context=context,
)
```

返回：

```python
result.energy
result.forces

result.primary_state
result.competing_states

result.state_gap
result.s2
result.local_spins

result.scf_iterations
result.subspace_residual

result.status
result.confidence
```

优先解决：

1. 前一帧 MO/density continuation；
2. reduced electronic basis；
3. SCF iteration reduction；
4. state identity tracking；
5. neighboring-spin branching；
6. automatic pruning；
7. explicit unresolved state。

---

# 19. 推荐开发顺序

### Phase 1 — Single-state reduced QM

目标：

\[
20\ {\rm SCF}
\rightarrow
3\sim5\ {\rm SCF}.
\]

先用闭壳层或固定-spin 系统验证：

- energy conservation；
- force continuity；
- MD stability；
- speedup。

---

### Phase 2 — Stateful open-shell tracking

加入：

- UHF/UKS；
- local spin fingerprint；
- orbital/density overlap；
- BS continuation。

---

### Phase 3 — Adaptive spin branching

程序自己：

\[
S
\rightarrow
S,\;S\pm1
\]

生成少数候选态，并自动 prune。

---

### Phase 4 — Heme-like multi-frame tracking

测试：

- Fe-porphyrin；
- Fe–O2；
- Fe=O；
- simplified axial ligand models。

重点不是反应势垒，而是：

\[
\boxed{\text{能否稳定保持多个电子态 lineage}}
\]

---

### Phase 5 — QM/MM force engine

放进蛋白 environment。

测试：

\[
\text{heme model}
\rightarrow
\text{heme + point charges}
\rightarrow
\text{heme protein QM/MM}.
\]

---

### Phase 6 — Nuclear search coupling

最后才接：

- MD；
- TS search；
- string；
- NEB；
- MECP；
- 以及可能的 MLPATH。

---

# 20. 核心概念总结

整个方法可以浓缩成三句话：

### 1.

\[
\boxed{
\text{Nuclear algorithms choose where to evaluate.}
}
\]

### 2.

\[
\boxed{
\text{Stateful QM decides how to evaluate there efficiently.}
}
\]

### 3.

\[
\boxed{
\text{Electronic ambiguity is represented, not forcibly eliminated.}
}
\]

最终目标不是创造一个“永不失败的 QM”。

而是创造一个：

\[
\boxed{
\textbf{像 MM calculator 一样被调用，}
\\
\textbf{但在内部自动管理电子态历史、低秩子空间和多态分叉的 QM engine。}
}
\]

这应该是整个项目后两层最清晰的定义。