# P0 证据清单：Phase 3b 现有结论 → 文件

日期：2026-11-06。依据：论文准备度 `docs/2026-10-29-paper-readiness.zh-CN.md` §5–6 的 P0（证据冻结）第一项"归档已有结果与脚本、源码/输入 hash、参数与环境清单"；用户原话（2026-11-06）："继续往下做啊？"。本文只整理已有文件，不跑计算、不改代码；P0 的另一项"首篇冻结协议"未写（需用户确认范围）。

## 1. 结论 → 证据

每批运行的参数都在其 json 的 `params` 字段（`bench_beam.py` 写入）；退出码在 `exit_codes`（0 pass / 3 有 miss 或 warn / 其他崩溃）。"NFS" = `/home/ruigengji/StatefulReducedOrder/benchmarks/out/`；"183" = `korakuen@192.168.0.183:~/sroqm/out/`（本地慢速 HDD）。"归档" = NFS `archive_183/`：183 上单份的 6 批已于 2026-11-06 拷入（见第 3 节第 1 条）。

| 结论 | 运行 | 日期 / 节点 | 代码特征 | 结果（`exit_codes`） | 位置 | `state.md` |
|---|---|---|---|---|---|---|
| 参考态 ref3（traj2 61 步 + 对称起点 11 步，最低态轨道 + 全部态能量） | `proto_ref3.py` | 10-01 起 / 183 | — | — | 183 `ref3/`（72 npz）；NFS `inputs_r2/ref3/` 为副本，md5 逐个一致 | §4.9 |
| G1 4 / 4（无影子） | R1″ | 10-23 / 183 | 四项性能修复 + 1a | s0–s3 全 pass | 183 `r1pp_20261023/`；归档 | §4.24 |
| G2 3 / 4（无影子，B′ 失败复现） | R2′ | 10-25 / PBS yayoi14 | + 2a / 2d | t2 exit 3，其余 pass | NFS `r2p_pbs_20261025/` | §4.25 |
| 起点搜索不可复现 | E7 | 10-25 / 183 | 同 R2′ | — | 183 `e7_20261025/`；归档 | §4.25 |
| 影子救回 B（确定性对照） | E8 | 10-28 / 183 | 影子（每帧一个） | off exit 3 / on exit 0 | 183 `e8_20261028/`；归档 | §4.26 |
| G2 4 / 4、G1 4 / 4（影子每帧一个） | R2″ | 10-28 / 183 | 影子（每帧一个） | t0–t3、s0–s3 全 pass | 183 `r2pp_20261028/`；归档 | §4.27 |
| 筛选关闭 G2 4 / 4、离线审计 0 反例 | 筛选关闭 | 10-29 / PBS yayoi11 / 12 | 同 R2″，无筛选 | t0–t3 全 pass | NFS `screen_off_pbs_20261029/`；审计 `audit_screen_off_20261029.py` | §4.28 |
| （部分，不计入） | G2 加跑 × 5 | 10-29 / PBS yayoi45 | 同 R2″ | 空（中途被停，第 4–7 步） | NFS `g2x5_yayoi45_20261029/` | §4.29 |
| G2 3 / 4（单槽位漏 B） | G2 × 4 现行规则 | 11-01 / PBS yayoi14 | 同 R2″ | t1 exit 3，其余 pass | NFS `g2x4_yayoi14_20261101/`；分析 `analyze_g2x4_step0_20261101.py` / `.out` | §4.29 |
| S1 救回 B（确定性对照） | E9 | 11-01 / PBS yayoi28 | S1 开关 | t1_one exit 3；t1_all、r2p_all pass | NFS `e9_pbs_yayoi28_20261101/` | §4.30 |
| G2 4 / 4（S1） | G2 S1 | 11-01 / PBS yayoi13 | S1 | t0–t3 全 pass | NFS `g2s1_pbs_20261101/` | §4.31 |
| G1 4 / 4（S1） | G1 S1 | 11-01 / PBS yayoi11 | S1 | s0–s3 全 pass | NFS `g1s1_pbs_20261101/` | §4.32 |
| G1 4 / 4（S1 + D1），影子 15.7% → 3.2% | G1 S1 + D1 | 11-02 / 183 | S1 + D1 | s0–s3 全 pass | 183 `g1d1_20261102/`；归档；分析 `analyze_shadow_cost_20261102.py` | §4.33 |
| 1a 的 N = 40（命中率 p̂ = 0.11） | E5 | 10-23 / 183 | — | — | 183 `e5_20261023/`；归档 | §4.23 |
| 筛选 margin 5 mHa（离线校准） | 账本 | 10-11 / 183 | 账本 | — | NFS `ledger_20261011/`、`ledger_20261011b/`（183 上也有） | §4.16 |

## 2. 输入与源码

- **输入**：`experiments/2026-11-06_p0_evidence/inputs_r2_md5.txt`（NFS 副本 74 个文件：`traj2_coords.npy`、`fe_ref_coords.npy`、`ref3/*.npz` 72 个）与 `inputs_183_md5.txt`（183 原件）按文件名排序后逐行一致。
- **源码（当前工作区）**：`experiments/2026-11-06_p0_evidence/source_md5.txt`（`src/sroqm/*.py` 19 个 + `benchmarks/bench_beam.py`）。
- **环境**：181 venv `/home/kasuga/venvs/pyscf`：Python 3.14.9、PySCF 2.14.0、numpy 2.5.3、scipy 1.18.1。183 `/home/korakuen/micromamba/envs/pyscf-env` 与 PBS 用的 NFS `/home/ruigengji/miniforge3/envs/pyscf-env`：Python 3.12.14、PySCF 2.14.0、numpy 2.5.3、scipy 1.18.1（第 4 节）。

## 3. 缺口

1. ~~183 上有 6 批证据只有一份~~（已备份）（ref3 原件除外，已有 NFS 副本）：R1″、E5、E7、E8、R2″、G1 S1 + D1。183 是慢速 HDD，没有备份。数据量（2026-11-06，逐目录不递归 `ls` 求和，含 `frames_*`）：R1″ 64 MB、E5 < 1 MB、E7 11 MB、E8 < 1 MB、R2″ 216 MB、G1 S1 + D1 60 MB，合计约 350 MB。**2026-11-06 已备份**：在 183 上 `cp -a` 到 NFS `benchmarks/out/archive_183/<原目录名>/`，逐文件 md5 比对 6 批全部一致（479 个文件），列表存 `archive_183/MD5SUMS`；183 原件保留。
2. **旧运行没有代码快照**：每批只记了日期与参数，没有记当时源码的版本标识。新加的开关都默认关闭，"用当前代码加同样参数能重现同样算法"是推断，没有逐批核对。git 提交由用户执行，提交草稿见 `.commit_drafts/`。
3. **起点搜索不可逐位复现**（E7）：即使代码和输入相同，端到端重跑也只能比较统计结果；逐步对照要用存下的候选帧注入（E8、E9 的做法）。
4. **`judge_step` 只对照最低参考态**，ref3 只存最低态轨道（论文准备度 §4.1）：现有证据只支持"选中态 = 已知最低态"，不支持窗口内逐态 recall。

## 4. 环境核对（2026-11-06）

| 环境 | 用在 | Python | PySCF | numpy | scipy |
|---|---|---|---|---|---|
| 181 `/home/kasuga/venvs/pyscf` | 181 上的分析、E9 的 181 版（未用） | 3.14.9 | 2.14.0 | 2.5.3 | 1.18.1 |
| 183 `/home/korakuen/micromamba/envs/pyscf-env` | 183 上的全部运行 | 3.12.14 | 2.14.0 | 2.5.3 | 1.18.1 |
| NFS `/home/ruigengji/miniforge3/envs/pyscf-env` | PBS 上的全部运行 | 3.12.14 | 2.14.0 | 2.5.3 | 1.18.1 |

三者 PySCF / numpy / scipy 版本相同；181 的 Python 版本不同（只用于分析，没有跑过门测试）。核对的是现在的版本，运行当时的版本没有记录。

## 未确认（Claude 推断）

- 第 3 节第 2 条："当前代码 + 同参数 = 当时算法"未逐批核对。
- 第 1 节"代码特征"一栏按日期与 `state.md` 记录归纳，不是从代码快照读出的。
