验收基准：docs/2026-10-11-search-performance-design.zh-CN.md、docs/superpowers/specs/2026-09-30-phase3b-beam-hygiene-design.md（已全文读完；验收方法另按 docs/superpowers/specs/2026-10-05-phase3b-crossing-robustness-design.md §3，同样已全文读完）

# Phase 3b 验收报告

日期：2026-10-29。对照对象：工作区当前源码（`src/sroqm/`、`benchmarks/bench_beam.py`）与实测（`state.md` §4.18–4.27）。用户原话（2026-10-29）："可以"，指同意写本报告。

**结论**：两道覆盖门都通过——G1（对称起点 11 步 × 4）与 G2（traj2 60 步 × 4）在当前代码上 0 miss / 0 warn（`state.md` §4.24、§4.27）。Phase 3b spec 的规则全部实现，验收判据比 spec §3 更严（warn 门、同态也看能量）。截至2026-10-29，筛选开/关对照、离线候选审计与成本报告均已补齐；搜索性能设计里的"按模式分搜索"（§6）、原设计三配置对照（§9）仍未完成，筛选保留候选的逐输入等价性尚未证明。下文逐条列出。

## 1. 设计要点

**Phase 3b spec（2026-09-30）**

| # | 要点 | 章节 |
|---|---|---|
| H1 | 兄弟态连续跟踪（beam），状态 ACTIVE / DORMANT / PRUNED，复活沿用原 id | §1、§4.1 |
| H2 | 推进预算分三档：选中 100 + 回退 + 梯度；近最低 `cap_near` 70；其余 `cap` 30 | §4.2 |
| H3 | 丢态（NC / overlap_drop / imom_fallback；follow 例外），窗口内保留上一步帧作复活种子，`revive_steps` 3 | §4.3 |
| H4 | 准入：从候选自身密度 IMOM、`cap`、不回退；收敛 + 同态 + \|ΔE\| < `admit_de` | §4.4 |
| H5 | 容量 3、满员替换最高者（adiabatic_min 不保护）、合并保留低者、剪枝 4 mHa × 2 步 | §4.5 |
| H6 | 选最低；补算梯度在已收敛解上，不从密度重跑 | §4.6 |
| H7 | 刷新：4 周期筛 5 mHa、第二段 `cap` / `cap_near`、NC 计未知；刷新间隔规则 | §5.1–5.2 |
| H8 | 完整搜索：触发条件、两层、base = 最低 ACTIVE + 本步丢态种子 | §5.3 |
| H9 | 空搜索 `search_empty`、空 beam 换初猜、`forced` | §5.4 |
| H10 | 输出：status 四档、manifold、events、frame | §6 |
| H11 | `Tolerances` 唯一来源，五个默认值 | §7 |
| H12 | `executor.map` 批次接口、串行实现；运行约定（绑核、OPENBLAS=1、环境不来自 NFS） | §8 |
| H13 | 验收：同态 σ ≥ 0.9 / 能量 0.05 mHa / 简并带 0.1 mHa；对称起点 11 步、traj2 前 20 步零遗漏 | §3、§9.2 |
| 明确不做 | M±2 / BS / stability following；跨机分发；heme 等；搜索结果可复现 | §2 |

**搜索性能设计（2026-10-11）**

| # | 要点 | 章节 |
|---|---|---|
| S1 | 不变量：容差不变、两层、占据上限 24、零先验三种 scratch、不按轨道能分块合并、未收敛 / 未尝试 / 筛掉分别记录 | §3 |
| S2 | 候选账本（每个物理 SCF 一行） | §4 |
| S3 | 两段筛选：k 周期后 E_short − E_anchor > keep_window + margin 才丢；先影子校准；上线首轮带影子复核 | §5.1 |
| S4 | pending：NC 候选带到后面续算，有上限，过期记 unresolved；须单独验证 | §5.2 |
| S5 | 按模式分搜索：常规带先验 base-first、不一开始重跑三个 scratch；对称破缺阶段多 ACTIVE + scratch 兜底；恢复时升级 | §6 |
| S6 | 对称破缺加密发现（第一组 4 步）、延长与退出原因记录、持续 unknown 升级为独立规则；`beam_changed` 白名单 | §7 |
| S7 | 公共资源缓存（已按路线移出 Phase 3b） | §8；路线 §2 |
| S8 | 三配置对照、正确性要求（B、第 5 / 12 / 6 步、对称起点第 1 步 WARN 与第 2–9 步漏态消除、traj2 0–59 全程、保存快照）、性能报告 | §9 |
| 交叉 spec | 重复运行；warn（能量判据 > 0.02 mHa）= 0；逐 lineage 记录；re-seat 失败退到次低 | §3、§6 |

## 2. 逐条对照

| # | 判定 | 位置 | 说明 |
|---|---|---|---|
| H1 | 符合 | `src/sroqm/beam.py:309`–`:640`（一步的 1–7 段）、`src/sroqm/lineage.py` | 复活沿用 id 在 `_admit` |
| H2 | 符合 | `src/sroqm/beamrules.py:61` `advance_budget` | |
| H3 | 符合 | `src/sroqm/beamrules.py:68` `is_lost`、`src/sroqm/beam.py:322`、`:390` | |
| H4 | 符合 | `src/sroqm/beam.py:487`、`:493` | 影子开时，已分开的影子不按 known / duplicate 跳过（设计 2026-10-26） |
| H5 | 符合（影子开时合并规则有补充） | `src/sroqm/beamrules.py:92`、`:103`；`src/sroqm/beam.py:512` | 影子开时，σ ≥ 0.9 但 σ < `shadow_sv` 且 \|ΔE\| > `accept_de` 的两条不合并（`_separated`，`:233`）；关时与 spec 相同 |
| H6 | 符合 | `src/sroqm/beam.py:197` `_reseat_forces` | 进程执行器时梯度回到保留 `mf` 的 worker（`:205`） |
| H7 | 符合 | `src/sroqm/schedule.py:20`、`:57` | |
| H8 | 符合，有扩展 | `src/sroqm/beamrules.py:76`；`src/sroqm/beam.py:345`–`:354` | 另加 reseat / densify / escalate 触发与加宽 base（S5、S6） |
| H9 | 符合 | `src/sroqm/beam.py:371`、`:354`（`retry_guesses=not active`）、`:507`–`:510` | 三种初猜现在每次都合并跑（交叉 spec D2 = S2），空 beam 的退路因此总是生效 |
| H10 | 符合 | `src/sroqm/beamrules.py:111`；`BeamResult` | |
| H11 | 符合 | `src/sroqm/tolerances.py:9`–`:13`；`src/sroqm/beam.py:129` | 五个默认值未改 |
| H12 | 符合，有一处偏离 | `src/sroqm/executor.py:82`、`:177` | 另实现了 `ProcessExecutor`（性能优化设计 §9）。**偏离**：R2、R2′ 在 PBS 上用的是 NFS 上的 `pyscf-env`（§8.3 要求环境不来自 NFS；只影响速度，版本与 183 一致） |
| H13 | 符合，且更严 | `benchmarks/bench_beam.py:61` `judge_step`、`:54` `is_warn`、`:51` | 加了 warn 门（交叉 spec §3.2，用户 2026-10-23 "D-C 保持原标准"）与同态也看能量（2d，用户 2026-10-25 "2d做了。。。"）；轨迹从 traj2 前 20 步扩到 60 步 × 4、对称起点 × 4 |
| S1 | 部分符合 | `src/sroqm/search.py:85`、`:270`–`:271`；`src/sroqm/tolerances.py` | 容差、两层、上限 24、三种 scratch、不按轨道能分块：符合。pending、筛掉的候选分别记录；"未尝试"没有单独记（当前没有工作上限，不会出现） |
| S2 | 部分符合 | `src/sroqm/beam.py` `_Ledger`；`src/sroqm/backend.py` | 设计 §4 列出的未记录项（几何 / 参照 / base 身份、占据序号、去重身份等）仍未补；去重身份由 2a 的 σ / ΔE 记录部分补上 |
| S3 | 符合（结果层面），逐候选未证明 | `src/sroqm/search.py:240` `_screened`；`BeamParams.screen_margin` | 按校准值（k = 4、margin 5 mHa）实现。**2026-10-29 补**：筛选关闭的 traj2 × 4 也 4 / 4（`state.md` §4.28）；离线套用筛选规则 1300 个候选 0 反例、省 57% 搜索周期。第二段换 IMOM 参照对保留候选的影响没有逐候选证明（搜索不可复现） |
| S4 | 符合，验证不足 | `src/sroqm/beam.py:84` `_Carry`、`:390`–`:412` | 续算、`revive_steps` 上限、过期计未知都在；**没有单独的验证实验**，只在整条回放里出现（§4.20：续算收敛若干次、过期触发升级） |
| S5 | **未实现** | — | 除加宽 base 外，"常规带先验 base-first、不先跑三个 scratch"没有做：每次完整搜索都合并跑三种初猜 |
| S6 | 符合，有两处偏离 | `src/sroqm/beam.py:630`–`:640`；`:627`；`:45` | 加密 4 步、延长 / 结束记录、升级为独立规则、白名单：符合。**偏离 1**：延长条件不再计本步搜索自己的 pending（用户 2026-10-22 决定修性能，§4.18–4.20）。**偏离 2**：触发只看"起点 DEGENERATE"，没有用几何对称诊断。另加了随机取向采样（1a，用户 2026-10-23 决定） |
| S7 | 未实现（按决定移出） | — | 路线 §2：DF / 格点共享移出 Phase 3b |
| S8 | 正确性部分符合；成本报告已补，原设计对照未齐 | — | 正确性：B、第 5 / 12 步、第 6 步不误报、对称起点第 1 步与第 2–9 步、traj2 0–59 全程：全部满足（§4.24、§4.27）；快照：`--save-frames` 存选中帧与起点候选（不是每步全部 seed）。三配置对照（§9.1–9.3）没有按设计做；2026-10-29成本报告已补每步P50/P95、周期与任务来源分解，但搜索函数独立计时、worker峰值内存与同硬件的完整受控对照尚未齐备 |
| 交叉 spec | 符合 | `src/sroqm/beam.py:559`（`reseat_failed`）；`bench_beam.py` | 重复运行、warn = 0、逐 lineage 与逐任务记录、re-seat 退路 |
| 明确不做 | 符合 | — | M±2 / BS / stability following、跨机分发、heme 都没有做；搜索结果确实不可复现（E7，§4.25） |

## 3. 测试、门与阈值（最后看）

- **门**：G1 R1″（§4.24）与复核（§4.27）各 4 / 4；G2 R2″（§4.27）4 / 4；全部 0 miss / 0 warn，`exit_codes` 全 pass。
- **G2 通过的限定**：这一轮 4 个 traj2 run 第 0 步都没有出现 B′，所以 G2 通过不单独证明影子救回了 B；影子起作用的证据是 E8 的确定性对照（§4.26：关影子复现第 5 步 WARN，开影子第 4 步晋升 B、第 5 步选中）。
- **2026-11-01 补：G2 加跑 × 4（yayoi14，`state.md` §4.29）3 / 4，t1 第 5 步 WARN**。B′ 自然出现 3 次：t2、t3 影子救回 B（t2 走的是"能量已分开"晋升）；t1 第 0 步找到了 B，但 B′ 的唯一影子槽位给了另一个近重复（+0.826 mHa，σ 0.905），B 没登记。分析与改法见 `docs/2026-11-01-shadow-slot-analysis.zh-CN.md`。按 warn = 0，**当前代码在这一批上 G2 不通过**；上面"G2 4 / 4"只对 R2″ 那一批成立。**同日 S1（`--shadow-all`）：E9 确定性对照救回 B（§4.30），G2 × 4 4 / 4（yayoi13，§4.31；这一批没出现会分开的 B′）。**
- **单元测试**（kasuga02，最近一轮）：test_beam 66、test_search 19 + 7 skip、test_beamrules 32、test_bench_beam 6、test_schedule 12、test_executor 13、test_beam_integration 5 + 1 skip。spec §9.1 列出的规则测试没有逐项对照清点；pytest 通过不当作正确性证据。
- **阈值**：`Tolerances` 五个值未改；新增的 `WARN_DE_MHA` 0.02 mHa（依据：同态收敛噪声 ≤ 0.016 mHa，§4.8）、`screen_margin` 5 mHa（离线校准，§4.16）、`densify_random` 40（E5 命中率 0.11，§4.23，未经用户确认）、`shadow_sv` 0.999 / `shadow_steps` 5（§4.24 的 σ–ΔE 分布与 t2 的分开时间，未经用户确认）都只在 Fe²⁺(H₂O)₆ / B3LYP / def2-SVP 上校准。

## 4. 已知限制（交给下一阶段前要知道的）

1. 对称起点第 1 步靠随机取向采样，按 E5 的命中率估计单步仍有约 1–8% 的漏态概率（4 / 4 + 4 / 4 只说明降低了）。
2. σ ≥ 0.9 的同态判据会把相差 0.04–0.24 mHa 的驻点并成一个态（§4.24）；靠影子跟踪补救，影子的参数只在一个体系、一例 B / B′ 上验证过。**2026-11-01：每个保留帧只留一个影子的规则已在真实运行里漏掉过 B（§4.29 t1）**。
3. 起点搜索不可逐位复现（E7）；验收只能靠重复运行。
4. 筛选对结果的影响未证明（审查第 5 条）。
5. 相邻多重度、broken-symmetry 候选没有（spec §2 明确不做）；扩到 heme 时覆盖要重新取证（路线 §1.4）。
6. 1a 的旋转块只对高自旋八面体 d⁶ 校准；换体系要重测。

## 5. 收口还差的事（收口计划 Task 6、7）

- ~~筛选开 / 关对照与反例审计~~：2026-10-29 已做（`state.md` §4.28）。
- ~~成本报告~~：2026-10-29 已写，`docs/2026-10-29-phase3b-cost-report.zh-CN.md`（worker 内存未记录、跨硬件墙钟不可比，见其 §1）。
- ~~同步路线文档、三语 README、设计稿状态行~~：2026-10-29论文准备度梳理时已同步；README补记PBS上NFS环境例外，源码和实验配置未改。下一阶段论文证据与建议工作包见 `docs/2026-10-29-paper-readiness.zh-CN.md`。
