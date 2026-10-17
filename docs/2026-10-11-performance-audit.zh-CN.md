# StatefulReducedOrder 重复计算审查

日期：2026-10-11，Asia/Tokyo。

补充：本轮随后核对了 `state.md` §4.15 / §5 第 13 条及 `benchmarks/out/accept_20261006_v2/`。最新完成的六份 JSON 显示，含完整搜索的步骤占总耗时 67.6–88.1%；这不是搜索函数独立计时。由此更新优先级：先处理候选筛选与搜索来源，再做公共资源缓存。下文原“实施与验证顺序”是核对这批证据前的建议，以 [最新搜索性能设计](docs/2026-10-11-search-performance-design.zh-CN.md) 为准。没有启动新计算。

结论：当前实现有明确的同几何重复准备、调度过度触发和提前计算无用梯度的问题。搜索还会从多个来源提交候选，执行完 SCF 才进行电子态去重；这是潜在的大开销，但不能把“结果同态”直接等同于“任务重复”。~~优先做每步公共资源复用、修正刷新触发条件、延迟梯度，再研究严格等价任务合并。~~ **（已被取代）** 优先级以搜索性能设计与 `state.md` §5 第 13 条为准：先改搜索（含完整搜索的步骤占总时间 68–88%），公共资源缓存其次。各条处理结果见文末"处理记录"。

## 证据范围

- 只读审查当前工作区源码、项目历史记录、已安装 PySCF 源码；没有运行 Python、SCF、测试、profiling 或远端任务。
- Git HEAD：`71fd937ef66ffe8eaa6a642d066e80de7571eaa0`。工作区有既有未提交改动，以下结论针对工作区，不能归给 HEAD。
- `energy_only` 和三种初猜合并已出现在当前源码；2026-10-04 的性能数字不能代表这个版本。
- 后端内部结论针对 `PySCFBackend`，未审查其他项目中的 r2scan/COSX 后端。
- 报告保存于项目 `docs/` 目录；没有更改目录权限或项目源码。

源码指纹（SHA256，审查时）。2026-10-11 处理记录中的修改之后，`beam.py` 已变为 `82abc743…`，`backend.py` 已变为 `8af623fe…`，其余三个不变；下文行号以审查时为准：

```text
beam.py     6d88da417cac3242e55dc14d7357f3d9f700219ac1317425568e6a6c30b8e873
backend.py  c8baa29c55a96e03baa6b6314fabce84b88cb5e9c5813e75981b6d3fd8cd3609
search.py   a08faf917b15a2ed044c5c00fafff40a0cde74be71ee1617e73046ec30784866
schedule.py 91d5466cedcc36fec148a5a3da00af9a57675be7a6c1392d2260399b6e63d0fd
engine.py   45049f74e4c9d751834335e8f5fd6ccf0a737e1eda665046562cc24de6945105
```

## 1. 每个候选重新构造后端资源：结构性重复，潜在收益大

位置：[backend.py:63](src/sroqm/backend.py:63)、[backend.py:91](src/sroqm/backend.py:91)。

每次 `run()` 都新建 RHF/UHF/RKS/UKS；DF、DFT grids 都挂在新的对象上。即使几何、基组、方法完全相同，也没有跨候选公共资源层。

已安装 PySCF 源码证据：

- `scf/hf.py:125` 附近：每次 kernel 获取 overlap、hcore，构建初始 veff，并准备正交化/DIIS。
- `df/df_jk.py:77` 附近：未传 `with_df` 时新建 DF 对象。
- `df/df.py:102` 附近：新 DF 的 `_cderi=None`；`loop()` 在无 tensor 时触发 `build()`。
- `dft/rks.py:499`：新 grids 没有坐标时触发 `grids.build()`。

这意味着“把候选 SCF 的循环数降到 4”并没有免掉格点、DF 与初始矩阵准备。候选多时固定开销被反复支付。当前源码足以证明没有共享路径，不能据此宣称具体耗时占比。

建议：同一步建立共享 overlap/hcore、DF 积分与几何格点数据；每个候选仍有独立的密度、MO、占据 hook 和 DIIS。先实现矩阵共享，再单独验证 DF/grid 共享。不要复用一个可变 `mf` 去承载多个独立结果，否则之前保留的 `SCFOutcome.mf` 会被后续任务污染。

DF 缓存需包含 AO/辅助基、几何和算子等依赖；hcore 额外包含 MM 点电荷。grids 需包含格点配置；若启用密度筛选，筛选后的 grids 还依赖初始密度，不能无条件共享。PySCF 此安装源码的默认 `small_rho_cutoff` 为 0，但实际运行配置可覆盖默认值。

## 2. 同几何的 overlap、Löwdin 和 meta-Löwdin 被反复计算：明确可复用

位置：

- [engine.py:103](src/sroqm/engine.py:103)：每条 lineage 的 `prepare()` 都复制同一几何的 Mole。
- [frame.py:46](src/sroqm/frame.py:46)：每个新的结果 Frame 再取 overlap 并对其特征分解。
- [extrapolate.py:91](src/sroqm/extrapolate.py:91)：每条历史长度至少为 2 的 Grassmann 外推再次算本步 overlap 和 Löwdin。
- [fingerprint.py:43](src/sroqm/fingerprint.py:43)：电荷和自旋人口分析分别调用 PySCF；UHF/UKS 下，两次都重新算 meta-Löwdin 和变换后的 alpha/beta 密度。
- [search.py:79](src/sroqm/search.py:79)：每次 d 占据枚举再算一遍 meta-Löwdin。
- [overlap.py:27](src/sroqm/overlap.py:27)：每次同态比较都重新积分跨几何 overlap，连同几何比较也走此路径。
- [imom.py:17](src/sroqm/imom.py:17)：每个 task 的 hook 重新积分 overlap；不同占据使用同一几何对时仍不共享。

仅正常推进 K 条 lineage、各有至少 2 帧历史，就会在项目层调用约 `2K` 次本步 Löwdin 分解（外推 K 次 + 结果 Frame K 次）；UHF/UKS fingerprint 再调用 `2K` 次 meta-Löwdin。此计数不含搜索、准入、后端内部准备。

已有 `TaskResult.frame` 缓存只防止同一个结果反复建 Frame；`Frame.with_occupations()` 只共享同一个基帧的矩阵。它们没有让不同结果、不同 lineage 共用几何数据。

建议：每步有限的 `GeometryContext`，一次提供 Mole、S、S_half、S_invhalf、meta-Löwdin 变换、AO 索引；跨几何 S 按几何对缓存；同一 Frame 对的比较结果按实际帧身份缓存。fingerprint 的电荷/自旋共用一次变换后的密度。不能仅用可变 Mole 的对象 ID 判定几何相同，跨几何比较必须保留真实交叉积分。

## 3. 搜索先花完 SCF 成本，再去重：需区分任务等价与结果同态

位置：[search.py:179](src/sroqm/search.py:179)。

当前 `merge_guesses=True` 路径：

1. 跑 minao、atom、huckel 三个 scratch，加上各 base 的全部占据枚举。
2. 对每个 scratch 的轨道再枚举，包括 own occupation；scratch 是否收敛都不阻止此步骤。
3. 所有结果返回后才 `_dedupe()`。
4. 对窗口内各独立态再枚举第二层。

`tests/test_search.py:161` 的既有测试明确预期：水分子没有金属，也从三个 scratch 各再提交一个 IMOM 任务，最后只剩一个态。此处是既有测试源码的断言，不是本轮运行结果。

第一层并不提前比较 scratch 间的重复输入。即使两个 scratch 最终同态，也可能轨道、密度和 IMOM 参照不同；在软方向上这些差别影响后续发现，不能按 `sigma_min >= 0.9` 合并整个枚举源。

任务数公式（不含推进、refresh、复活、准入和 fallback）：

```text
N_search = 3 + sum(P_base) + sum(P_scratch) + sum(P_level1_seed - 1)
```

`P` 是该帧实际返回的占据数，默认最多 24；第二层 seed 数不受 `max_lineages` 限制。举例：1 个 base、每个源 24 模式、第二层 3 个 seed，则 `3 + 4*24 + 3*23 = 168` 个 SCF task。这是静态算术示例，未测得真实体系每次都提交 168 个。

建议：先记录严格输入等价的命中率；任务键包含几何/方法/MM、dm0、IMOM 参照、初猜、预算、fallback、梯度选项。能量接近、占据数组相同或结果同态都不足以证明输入等价。默认保留三种初猜与两层搜索：`state.md` §4.14 记录多初猜改善起点覆盖，这部分有正确性用途。

## 4. 同一步完整搜索、refresh、复活独立执行：来源重叠但预算不同

位置：[beam.py:221](src/sroqm/beam.py:221)、[beam.py:237](src/sroqm/beam.py:237)、[beam.py:245](src/sroqm/beam.py:245)。

完整搜索运行后，只要 `active` 非空且 scheduler 到期，同一步继续 refresh；搜索没有替 refresh 提供已覆盖任务的记录。丢态 seed 可同时进入 full_search 枚举和 revive 路径。三路结果到准入队列时才再次比较。

不能简单取消本步 refresh：它的筛选 4 周期、第二阶段 30/70 周期与 search 的默认预算不同，输出 unknown 与调度含义也不同。严格按当前输入键合并时，这些任务往往并不等价。

建议：统一来源账本，先复用公共资源和严格等价任务；若以后复用满预算搜索去覆盖短预算 refresh，需另外定义筛选、unknown、候选顺序、revive ID 如何回填，并验证发现覆盖。

## 5. 拒绝事件也强迫下一步 refresh：明确的调度反馈问题

**状态：2026-10-11 已修**（`_CHANGES` 白名单；`unstable` 按 spec §5.2 不算 beam 变化，不另设强制刷新规则）。见文末处理记录。

位置：[beam.py:40](src/sroqm/beam.py:40)、[beam.py:370](src/sroqm/beam.py:370)、[schedule.py:70](src/sroqm/schedule.py:70)。

`_QUIET` 只有 `search` 和 `search_empty`；任何其他事件都被当作 `beam_changed=True`。因此：

- 候选因无位置被 `full` 拒绝，ACTIVE 集合完全没变，也触发下一步 refresh。
- 重收敛后撞上 ACTIVE 的 `duplicate` 被拒绝，也触发下一步 refresh。
- `unstable` 作为准入失败同样进入此布尔值；它可以具有科学风险意义，但不等同于 membership 变化。

当反复发现相同、满员或不稳定候选时，可形成：refresh → SCF/准入 → 拒绝 → 强制下一步 refresh → 再次重复。当前控制流证明这条路径存在；是否占某条真实轨迹多数步骤，需事件统计，尚未测量。

Phase 3b 设计 §5.2（文件约 112 行）列举强制刷新的 beam 事件为 spawn/revive/lost/merge/prune/evict/switch；当前代码对拒绝事件的处理比这一列举更宽。

建议：按实际变化事件显式计算 `beam_changed`，把 unknown、degenerate、reseat failure、follow loss 等风险信号分别保留。`full`、`duplicate` 不应自动表示 beam 变化。`unstable` 是否需要独立强制刷新规则，要显式决定，不能顺手全部放进 quiet。

## 6. 先给上一步选中态算梯度，之后才决定本步选中态

**状态：未改。** 实测上界 < 0.5% 总时间，列入搜索性能设计 §8；"不收敛 IMOM 也算梯度"一项不修（理由见处理记录）。

位置：[beamrules.py:49](src/sroqm/beamrules.py:49)、[beam.py:176](src/sroqm/beam.py:176)、[backend.py:99](src/sroqm/backend.py:99)。

非 energy_only 模式下，上一步 selected 的推进预算为 `Budget()`，其中 `with_gradient=True`。这时尚未 discovery、准入、merge 和本步选态。

若本步换选、旧 selected 丢失或被撤掉，刚算的梯度不会用于返回；新的 selected 再算一份。backend 也没有先判断 `converged` 再决定是否计算请求的梯度，因此不收敛 IMOM 的推进与 fallback 可能付出无用梯度成本。

默认 `SerialExecutor` 已具备“从保留 outcome 直接算梯度”的路径，调整为所有推进先 energy-only、本步选态后再求梯度有明确实现基础。此调整针对非 energy_only beam；当前 energy_only 路径本身已禁止梯度。

需要保留自定义执行器的 re-seat 同态/能量检查、重选和错误处理。远端或进程执行器更适合支持 outcome handle；不能把 `type(executor) is SerialExecutor` 随意改成 `isinstance` 来绕开任务协议。

## 7. refresh 第二阶段保留密度，却重建求解器，而且改变 IMOM 参照

位置：[schedule.py:34](src/sroqm/schedule.py:34)、[executor.py:49](src/sroqm/executor.py:49)。

第一阶段 4 周期未收敛、通过能量筛选的候选，第二阶段用 `r.outcome.dm` 做初始密度，用第一阶段 `r.frame` 作新的 IMOM 参照，再从新 `mf` 启动。DF、grids、DIIS 等没有续接。

所以这不是同一个求解过程直接增加剩余周期。新密度是有效热启动，但求解器准备仍重复；IMOM 参照也已经换过，不能直接等价于“在旧 mf 上继续跑”。

建议：第一步只复用与状态无关的几何资源；保持当前两阶段参照和预算语义。真正续接 DIIS/求解器作为后续算法改动，需要单独比较 unknown 数、电子态覆盖和周期数，不能宣传为纯缓存改动。

## 8. 批量搜索同时保留许多 mf：重复准备还放大峰值内存

位置：[backend.py:109](src/sroqm/backend.py:109)、[executor.py:73](src/sroqm/executor.py:73)、[search.py:183](src/sroqm/search.py:183)。

SerialExecutor 返回整个结果 list，每个 outcome 持有 mf；full_search 保留 scratches/found/found2，refresh 保留 stage1。多个候选的 DF/grid 对象可能同时存活，串行也不意味着只占一个 SCF 的内存。

搜索最终只返回 Frames，不保留 mf；准入阶段会独立重收敛。可在按需抽取 Frame/标量后释放搜索侧 mf，或分块处理，但必须保留第二层依赖、未收敛 scratch 的枚举用途、pending 统计和稳定候选顺序。准入/推进可能成为 selected 的 mf 仍需保留到本步选态结束。

## 9. 次级问题

- IMOM 每轮计算 `C_ref.T @ S_cross @ C`，固定的左半边 `C_ref.T @ S_cross` 可在 hook 创建时预计算。收益需按矩阵尺寸测量；这不能代替 DF/grid 共享。
- PySCF 默认会创建临时 chkfile，并在 SCF 周期内 dump checkpoint；backend 没有禁用它。若运行配置未 mute，这是候选级重复磁盘 I/O。项目不靠这些内部临时文件恢复结果，值得核对后显式禁用；不能在没有检查实际 TMPDIR/mute 配置时断言它正在写 NFS。**（2026-10-11 已修：核对默认为 `/tmp/tmp…`，`backend.py` 设 `mf.chkfile = None`。）**
- `BeamEngine.members` 与 lineage 标量日志没有长度上限，准入匹配会遍历历史 DORMANT seed 并算跨几何 overlap。长轨迹可能出现内存与匹配成本增长；不能直接删除休眠 seed，因为它承载复活 ID 和历史身份。

## 实施与验证顺序（已被取代）

**以下顺序是核对验收数据前的建议，已由 `2026-10-11-search-performance-design.zh-CN.md` 与 `state.md` §5 第 13 条的优先级取代**：候选账本 → 影子筛选校准 → 两段筛选与按模式拆分搜索 → 任务级并行 → DF / 格点共享。第 2 条中的刷新触发已修。下文保留原样作记录。

1. 每步 GeometryContext + 合并 fingerprint 的公共变换；先把明确相同的矩阵计算从“每候选一次”变成“每几何一次”。
2. 用真实状态变化与独立风险信号驱动 refresh，消除 `full`/`duplicate` 的无效调度反馈；推进延迟梯度到本步选态以后。
3. 分离公共 DF/grid 数据与独立 SCF 状态，控制搜索侧 mf 生命周期、checkpoint I/O；逐项实施和核对。
4. 加来源账本，衡量严格任务等价的命中率，再合并；不同预算的覆盖、两阶段 solver 续接另行设计。
5. 之后再做候选并行；否则并行会复制更多 DF/grid 和内存资源，掩盖重复计算。

后续允许执行验证时，应记录每步各来源 task 数、SCF 总周期、预算/收敛、prepare/SCF/gradient/fingerprint/compare 的计时、公共缓存命中、梯度计算后未使用次数、refresh 触发原因、峰值内存。现有 trace 记录了推进预算和部分搜索/refresh 信息，但缺少搜索/准入/复活的完整 task/周期账本和组件计时。

验证同时检查能量、力、同态身份、候选顺序、lineage ID、unknown 与事件语义。轻量回归与完整真实轨迹性能/覆盖分别报告；历史 224 个测试与小水体系约 8.5% 提速不能作为当前版本或 Fe 长轨迹的性能证明。

## 处理记录（2026-10-11）

逐条对照源码与 `benchmarks/out/accept_20261006_v2/` 的六份 JSON 核对后：

| 条目 | 核对 | 处理 |
|---|---|---|
| §5 拒绝事件强迫下一步 refresh | 属实：`_QUIET` 只有 `search`/`search_empty`，比 spec 2026-09-30 §5.2 宽。按 JSON，上一步只有拒绝类事件（且无 DEGENERATE、无 unknown）而触发的 refresh 每次运行 ≤ 2–3 次（每次约 100–160 s），约占总时间 1–3%；修它主要是为了与 spec 一致。 | **已修**。`beam.py` 改为白名单 `_CHANGES`（spawn / revive / lost / merge / prune / evict / switch，加 forced / reseat_failed / follow_lost）；`full` / `duplicate` / `unstable` 不再算 beam 变化；`pending` 仍单独 poke。回归测试 `tests/test_beam.py::test_a_rejected_candidate_does_not_pull_the_next_refresh`（修前失败：下一次 refresh 被从第 6 步拉到第 4 步）、`::test_an_unknown_search_result_still_pulls_the_next_refresh`。 |
| §9 PySCF 临时 chkfile | 属实：`dft.RKS(...).chkfile` 默认为 `/tmp/tmp…`，每个 SCF 每周期 dump；项目不读它。 | **已修**。`backend.py` `build()` 设 `mf.chkfile = None`。回归测试 `tests/test_backend.py::test_scf_objects_write_no_checkpoint_file`（修前失败）。 |
| §6 后半：不收敛 IMOM 也算梯度 | 属实，但只在"IMOM 不收敛 → aufbau fallback 成功"时浪费一次梯度；`imom_fallback` 在真实数据上从未触发（`state.md` §5 第 5 条）。要保持"两次都不收敛"时的现有语义还需额外代码。 | **不修**：收益为零。 |
| §6 前半：梯度早于选态 | 属实。按 JSON，换选或丢态的步每次运行 ≤ 8 次，梯度约 5.5 s（10 核），合计 < 0.5%。改动涉及非 `SerialExecutor` 的 re-seat 路径。 | 不在本轮；列入搜索性能设计 §8。 |
| §1、§2、§3、§4、§7、§8 | 属实（§2 收益未测）。 | 按 `2026-10-11-search-performance-design.zh-CN.md` 推进；搜索（含完整搜索的步骤占总时间 68–88%，步计时口径）优先，公共资源缓存其次。 |

测试（40 核节点，pyscf-env，每个文件一个进程、4 线程）：`test_beam.py` 42 passed，`test_backend.py` 15 passed，`test_executor.py` 8 passed，`test_beam_integration.py` 5 passed / 1 skipped，`test_search.py` 8 passed / 7 skipped（慢速 Fe 测试按 conftest 默认跳过）。两处修改都未在真实回放中验证。

**补记（2026-10-11 晚，候选账本首轮实测，`state.md` §4.16）**：审查各条之外的最大一项是**内存压力导致 DF 积分落盘**——保留的 `SCFOutcome` 各持有 mf 的 DF 张量（116 MB），一次搜索约 30 个后 RSS 超过 PySCF `max_memory`，后续 SCF 单周期 0.7 → 7–8 s，四个 run 里占 19–118 分钟。已修（`backend.py` `run()` 末尾 `with_df.reset()`，回归测试 `tests/test_backend.py::test_kept_outcomes_do_not_hold_df_integrals` 修前失败、修后通过；修复后复跑无任务再落盘，四个 run 90 / 100 / 157 / 117 → 72 / 75 / 67 / 54 min）。去掉它后 DF / 格点重建只占步时间 1–2%（§1 的收益比原估计更小）。
