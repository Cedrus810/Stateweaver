# StatefulReducedOrder 搜索性能设计

日期：2026-10-11（Asia/Tokyo）。状态：设计；同日已做 §4 阶段 A（账本）并在 Fe 体系实测一轮、§5.1 离线影子筛选校准、DF 落盘修复（§4、§8），§7 的 `beam_changed` 事件集合单独修正（见审查文档"处理记录"）；阶段 B 筛选本身、C、加密发现、并行均未实现。下一步 Phase 3b 工作以本文为准，**但优先级以 `docs/2026-10-13-roadmap.zh-CN.md` 为准（用户 2026-10-13 决定）：覆盖（§6–7）先于成本（§5）；§8 的 DF / 格点共享移出 Phase 3b（修复后只占 1–3%）；基线为 2026-10-11b 复跑。**

## 1. 最新基线与优先级

本设计依据当前工作区源码、`state.md` §4.15 / §5 第 13 条，以及 `benchmarks/out/accept_20261006_v2/` 的六份 JSON 和 full.log。承接本日审查；当前目标是减少搜索中的无效收敛，并验证对称破缺阶段更频繁的发现能否消除 sym1 漏态。

已核对：

| 运行 | 已完成范围 | miss / warn | 含搜索步骤占总耗时 |
|---|---|---|---|
| r0 | traj2 0–20 | 0 / 0 | 80.4% |
| r1 | traj2 0–20 | 0 / 0 | 83.0% |
| r2 | traj2 0–20 | 0 / 0 | 67.6% |
| r3 | traj2 0–20 | 0 / 0 | 78.1% |
| full | traj2 0–19，日志到此为止 | 已完成步全通过 | 无完整 JSON，不计算全程占比 |
| sym0 | 对称起点 0–10 | 0 / 0 | 88.1% |
| sym1 | 对称起点 0–10 | 8 / 1 | 81.9% |

`seconds` 计时覆盖整个 `beam.compute()`。上述比例包含搜索步内的推进、refresh、复活、准入与梯度，不能当作搜索函数的独立计时。sym0 第 10 步的 11233 s 也具有这个计时边界。

traj2 第 0 步 B 态发现、第 5/12 步交叉已有五次运行的支持；第 6 步简并取向不同、能量相同不应误报。当前版本第 21–59 步仍未验收，对称起点仍失败，不能称 Phase 3b 已通过。

`state.md` §5 第 13 条的旧诊断统计：55% 周期花在高出最低 8 mHa 以上的候选、约 25% 回到已知态；固定开销拟合约 15%。这是另一批旧诊断的证据，不能归给此次 v2 的每个任务。它支持“先改搜索筛选/来源，再做公共资源缓存”的优先级；单独缓存的收益不能填平数小时搜索。

## 2. sym1 对设计的具体约束

`sym1.json` 中：第 0/1 步为 DEGENERATE；第 1 步 WARN，随后第 2–9 步漏。第 2–9 步没有 beam 事件，但每步都有 refresh：第 1–3 步找到的兄弟态都在选中态之上（+0.064 / +0.132 / +0.202 mHa），第 4–10 步每步 `refresh.unknown=1`（一个候选 `nc`）。所以问题不是 refresh 不够密，而是 refresh 从当前最低态的有限枚举没有找回正确取向，unknown 也没有触发更强发现。

修改后必须验证两件独立的事：搜索更便宜；对称破缺阶段的搜索确实找到正确分支。前者不能证明后者。不能用 ref3 的 sigma 或误差作为运行时触发器，因为生产运行没有参考答案。

## 3. 固定的不变量

- 维持 `conv_tol=1e-7`、`same_state_sv=0.9`、`admit_de=1e-5 Ha`、`deg_band=1e-4 Ha`、`accept_de=5e-5 Ha`。
- 初版保留两层搜索、占据上限 24、稳定的提交/结果顺序、准入/复活 ID 和同态检查。
- 零先验或 beam 清空时保留 minao/atom/huckel 三种 scratch；未收敛 scratch 的轨道仍可作枚举种子。
- 已知态同态判据用于输出与准入；不能自动升级成“两个初猜的后续发现完全等价”。
- 不按近简并轨道能分块合并候选，避免删掉 B；已提议的严格几何对称 orbit 简化保持为低优先级独立实验，本设计不扩大其范围。
- 未收敛、未尝试、经验筛掉的候选分别记录，不能一概表述为“没有其他态”。
- 阶段 A（账本）与影子筛选校准串行，保证逐任务计时可复现。校准完成后**紧接着**做任务级并行（排在 DF / 格点共享之前：v6 原型 4 个 worker 的搜索步 160–431 s，正式实现串行的含搜索步 40–187 分钟）。并行需要 worker outcome handle、梯度路由和内存策略，并须在真实回放中走通 re-seat 退路（`state.md` §5 第 12 条），单独实施、单独验收。

## 4. 实施阶段 A：建立每个候选的账本

优先修改 `executor.py`、`search.py`、`schedule.py` 与 beam trace 的数据边界。每个物理 SCF 尝试至少记录：

```text
step, task_id, source(s), level, base/seed identity, occupation index,
geometry/method/MM identity, initial-guess kind, IMOM-reference identity,
cycle budget, actual cycles, converged, fallback,
initial/short/final energy, screening decision/reason,
build/SCF/frame/fingerprint/gradient wall time,
returned state identity, duplicate-result identity
```

同一个物理任务被多个来源复用时，只计一次执行成本，保留来源列表。fallback/retry 各有 attempt ID。峰值内存、缓存命中和搜索步骤整体计时分开记录。

**实现状态（2026-10-11，第一版，单元测试通过，未在 Fe 体系真实回放中跑过）**：
- `SCFOutcome` 新增 `energies`（每个 SCF 周期后的 e_tot，经 PySCF `callback`）与 `seconds`（`build` / `df` / `grids` / `scf` / `gradient` 墙钟）。DF 积分与格点在 kernel 前按 kernel 自身的方式显式构建以便单独计时（`with_df.build()`、`initialize_grids`；`small_rho_cutoff > 0` 时格点留给 kernel），能量与原路径一致（`tests/test_backend.py::test_prebuilt_df_and_grids_leave_the_energy_unchanged`）。
- `SCFTask.tag` 标来源：`advance`、`reseat`、`admit`、`revive`、`refresh:s1` / `refresh:s2`、`scratch:<guess>`、`l1:base`、`l1:scratch`、`l2`。
- `BeamEngine` 每步用 `_Ledger` 包住 executor，每个物理 SCF 一行写进 `BeamResult.trace["tasks"]`：`tag, max_cycle, imom, cycles, converged, fallback, energy, energies, seconds`；`bench_beam.py` 的 json 原样保存。
- **尚未记录**：几何 / IMOM 参照 / base 身份、占据序号、fallback 的第一次尝试（只记返回的那次，`imom_fallback` 在真实数据上未触发过）、结果去重身份、frame / fingerprint 计时、峰值内存。按首轮真实数据再决定补哪些。

这能区分：任务数量多、每个任务周期多、单周期变慢、资源重建/内存压力四种原因。当前 summary 与 level1/level2 状态数不足以解释 sym0 的 11233 s，不能据此直接指定根因或提速倍数。

**首轮实测（2026-10-11，`state.md` §4.16）**：四种原因里主因是**内存压力导致单周期变慢**。保留的 `SCFOutcome` 持有 mf 及其 DF 张量（116 MB），一次搜索留到约 30 个后 RSS 超过 PySCF `max_memory`（4000 MB），之后的 SCF 把 DF 积分放到磁盘，单周期 0.7 → 7–8 s；四个 run 里这部分占 19–118 分钟。已修（`run()` 末尾 `with_df.reset()`），修复后复跑验证：无任务再落盘，四个 run 90 / 100 / 157 / 117 → 72 / 75 / 67 / 54 min。去掉这一项后，按账本估计的步时间里 SCF 迭代本身仍占 97–99%，DF / 格点重建只占 1–2%：§8 的共享收益比旧拟合（约 15%）更小。

## 5. 实施阶段 B：搜索采用两阶段筛选

### 5.1 短段用于排序与候选筛选，严格收敛用于状态确认

每个占据候选先跑短段，初步沿用 refresh 的 4 周期作为对照起点，随后根据账本校准；窗口内或不能可靠排除的候选继续到现有严格阈值。不降低最终能量/同态的确认精度。

初猜轨道能差只作为优先级，不能直接证明最终解在窗口外。短段 SCF 能量也不是“后续能量下降幅度”的严格下界，不能因短段高能就宣称排除成立。

以可靠收敛态的最低能量作为 E_anchor，经验筛选条件可设计为：

```text
E_short - E_anchor > keep_window + relaxation_margin
```

`relaxation_margin` 必须由已记录的完整候选的短段→最终能量下降校准；考虑层级、几何、方法、金属候选种类和 NC 失败样本。有限样本最大值不是数学保证。没有收敛 anchor 或校准不足时不启用硬淘汰。

先运行影子筛选：给出“本来会淘汰”的标记，仍完成原任务；查找被淘汰候选最终进入窗口、成为最低态或提供第二层必要种子的反例。没有每周期能量的旧日志无法完成此校准，需要在用户授权新计算后取得。

**离线影子筛选结果（2026-10-11 / 10-13，账本 8 个 run、16 个搜索步、555 个搜索候选，`state.md` §4.16；窗口按 `full_search` 实际的 `keep_window` = 2 mHa，首写误用 8 mHa 已更正）**：窗口内收敛候选 E_4 − anchor 最大 2.59 mHa，高 e_g 一团 ≥ 22.1 mHa；margin 0 有 14 个反例，margin 1–5 mHa 反例 0、省 49–50% 搜索周期；收敛候选 E_4 − E_final 最大 1.46 mHa。建议 k=4、`relaxation_margin` = 5 mHa（阈值 anchor + 7 mHa，余量约 3 × 1.46 mHa）。被筛掉的都在窗口外，本来也不作第二层种子。anchor 用本步事后最低收敛能量（实时只会更高、筛得更少）。样本只有 Fe²⁺(H₂O)₆ B3LYP/def2-SVP 两条轨迹；上线第一轮带影子标记复核，换体系 / 方法要重新校准。

### 5.2 第二阶段与 pending

第二阶段用短段密度热启动，同时显式记录 IMOM 参照选择；先保持现有 refresh 的参照重设语义，不把继续 DIIS 混进缓存优化。

NC 返回候选记录：DM、轨道/占据、参照所属几何、最近能量、实际周期、来源、剩余工作预算。跨步使用时构造当前几何任务，保留正确的跨几何参照；旧能量不能进入新几何的排序或准入。

pending 只做有上限的恢复：初版沿用现有 revive 时间尺度作为试验起点，不无界携带所有候选。过期或预算耗尽记录 unresolved；跨步续接改变原搜索语义，必须单独验证。起点候选不能依赖“下一步再算”来满足起点 B 态覆盖。

## 6. 实施阶段 C：分开起点搜索、带先验搜索和兜底搜索

| 模式 | 种子 | scratch 策略 |
|---|---|---|
| 初始化 / beam 清空 | 零先验及合法丢态种子 | 保留三种初猜 |
| 常规带先验发现 | 本步 ACTIVE 帧及合法恢复种子 | base-first；不一开始重跑三个 scratch |
| 对称破缺阶段 | 当前各 ACTIVE 代表、仍合法的起点/丢态种子 | 密集 base-first；保留独立 scratch 兜底 |
| 恢复 / 覆盖不足 | 已有帧、pending、丢态种子 | 升级完整多初猜搜索 |

当前 full_search 的 bases 只有最低 ACTIVE 与本步丢态 lineage 的种子（`beam.py` 中 `bases = [low.frame] + [m.seed … lost_step == step]`），其他 ACTIVE 不在其中；更改为多 ACTIVE 来源会增加任务，但也可能减少对独立 scratch 的依赖，必须和筛选/严格任务合并一起衡量，不能预设净收益。

“有 base 就永远禁用 scratch”不能作为安全定理：sym1 说明稳定且有多个 ACTIVE 时也可能缺最低分支。base-first 是待验的搜索策略，仍要保留丢态、持续 unknown、发现停滞和周期覆盖检查的 scratch 兜底。

初版保留现有第二层；先不引入无限闭包。若后续实验按“只从新态扩展”构造 worklist，必须承认 sigma>=0.9 的同态分组可能掩盖不同轨道初猜的搜索能力。只有严格相同输入可无条件跳过；被分组而省掉的扩展必须有来源记录和可审计的回退，工作上限耗尽不能报闭包完成。

## 7. 对称破缺时的加密发现

触发器必须使用运行时可见信息：起点几何的对称诊断、起点 DEGENERATE、随后几何变化、unknown、谱隙变化和 lineage 丢失。几何对称性与电子态简并分别记录，不能把仅有 DEGENERATE 的所有轨迹直接视为相同几何问题。

为验证因果，第一组明确试验采用：从对称简并起点出发后，前 4 个推进步（1–4）每步进行加强发现；4 是设计对照参数，不是已验证最佳长度。延长条件为仍有 pending/持续 unknown/丢态；退出加密模式的原因与覆盖是否未解决都记录。持续 unknown 的升级条件先作为独立规则，避免强行宣称它必能找回漏态。

此加强发现应覆盖多 ACTIVE 来源并具有 scratch 兜底，不能仅把现有 refresh 再跑一遍。不同阈值/窗口长度属于明确的实验变量，不能反复加临时启发式直至某一次轨迹通过。

同时复核 `beam_changed` 的事件集合：`full`/`duplicate` 不代表 ACTIVE 变化；保留真实变化和独立风险信号。修正该调度问题不能替代上述加密发现，sym1 即使没有拒绝事件也可能长期漏态。**（2026-10-11 已修：`beam.py` `_CHANGES` 白名单，见审查文档末尾"处理记录"；单元测试通过，未在真实回放中验证。）**

## 8. 公共资源缓存与生命周期

每步 GeometryContext 共享 overlap/Löwdin/meta-Löwdin、几何对 overlap；独立 mf 共享兼容的 DF tensor 和格点数据，保留独立 DM、hook、DIIS 与结果。hcore/后端资源缓存包含 MM 等依赖；密度筛选后的格点不能无条件共享。

搜索侧抽取 Frame 和账本后尽早释放不再需要的 mf，保留未收敛 scratch 的枚举用途和当前 selected 的梯度对象。**（2026-10-11 先做了最小一步：每个 outcome 的 DF 张量在 `run()` 末尾释放，见 §4 首轮实测；mf 本身仍保留，每个约 11 MB。）**推进的梯度延迟到最终选态；此项另行检查自定义执行器 re-seat 路径。

缓存仍有价值，但旧诊断中的固定成本约 15% 是拟合分量的估计，包含多种准备工作；不能声称共享 DF/grid 一定省 15%，更不能据此承诺搜索从数小时降到几分钟。

## 9. 验收与比较

先校准影子筛选，再比较三种完整配置，避免把加密搜索收益归给缓存：

1. 原搜索范围/频率 + 性能改动：确认原覆盖没有因筛选和来源简化退化。
2. 原搜索算法 + 对称起点加密发现：单独判断加密是否接住 sym1 类型失败，记录增加的成本。
3. 性能改动 + 加密发现：判断组合是否同时降低成本并守住覆盖。

以上是未来执行方案，不是本轮启动任务的授权。若预算不足以做完整对照，记录缺失项目与证据限制，不以局部成功替代全程验收。

正确性要求：

- 起点 B 态、traj2 第 5/12 步交叉维持原身份与能量判据；第 6 步简并取向不能误报。
- 对称起点至少沿用既有两次重复/不同核配置要求，sym1 类型的第 2–9 步漏态与第 1 步 WARN 必须消除；不放宽 acceptance 阈值。
- 完成当前版本 traj2 0–59 的全程回放，补齐第 20 步及 21–59；原 r0–r3 的 0–20 重复要求保持。
- 原验收目录未保存完整每步 seed/候选快照，仅同线程数/核配置不能保证精确复现 sym1；新的实验保存输入与状态快照，记录这一限制。

性能要求：同时报告 total/搜索函数独立计时、搜索步计时、task/周期数、P50/P95/max、筛选反例、pending/未尝试数量、峰值内存；固定方法、精度和资源配置。早期加密会增加搜索次数，最终比较必须计入这些次数。

判据是全部要求的覆盖通过且总成本降低；本设计没有现成证据支持特定提速倍数，也没有预先承诺更频繁搜索一定解决 sym1。

## 10. 代码改动边界

- `executor.py`：候选/attempt 账本、计时与生命周期协议。
- `search.py`：两阶段任务、输入等价合并、search mode、pending payload；第二层初版保留。
- `schedule.py` / `beamrules.py`：加密发现、unknown 升级、明确调度原因。
- `beam.py`：来源计划、真实变化事件、各模式调用与 trace；保持准入/选态不变量。
- `backend.py` / `frame.py` / `fingerprint.py` 等：公共几何/后端资源共享。
- `bench_beam.py`：组件计时、coverage/pending 与候选账本输出；参考答案只用于验收，不回流调度。

本轮只生成设计文档，没有改变上述源码、原状态文档或历史验收结果。文档保存于项目 `docs/` 目录。
