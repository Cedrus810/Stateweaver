# sroqm — QM/MM のための状態保持型電子構造エンジン

[English](README.md) | [简体中文](README.zh-CN.md) | 日本語

`sroqm` は [PySCF](https://github.com/pyscf/pyscf) の上に作った電子構造計算エンジンで、QM/MM
分子動力学（MD）向けです。普通の SCF は毎ステップ初期値から収束させるため、どの電子状態に落ちるかは
制御できません。`sroqm` は「指定した電子状態——金属のスピン状態や破対称（broken-symmetry）解など——を
トラジェクトリの間じゅう守り続け、別の状態へ滑りそうになったら警告する」ためのエンジンです。

## できること

**継続計算（continuation）。** 毎ステップゼロから収束させず、前のステップの情報を初期値にします：
前のステップの軌道（`read`）、ASPC 式の密度外挿（`aspc`）、Grassmann 測地線による軌道外挿
（`grassmann`）。def2-SVP/B3LYP では、1 ステップあたりの平均 SCF サイクル数が 7.0 → 3.7（水二量体）、
12.3 → 4.6（[Fe(H₂O)₆]³⁺）に減り、エネルギーはゼロから収束させた結果と 2×10⁻⁷ Hartree 以内で
一致します。

**状態追跡（単一 lineage）。** *lineage* は「1 つの電子状態」を構造の変化に沿って運ぶ単位です。
隣接ステップの軌道の間で IMOM の重なり継続をし、回転不変の fingerprint とフロンティアギャップを
状態の目印にします。うまくいかなそうなときは、段階別のアラームを出します——
`gap_sign_flip`（軌道交差で別の根に移った疑い）、`overlap_drop`（重なりの急落）、
`imom_fallback`（IMOM が収束せず aufbau に退避）、`lower_state_found`（もっと低い状態が別にある）。

**多状態 beam。** 近縮退状態では 1 本の lineage では足りません。高スピン [Fe(H₂O)₆]²⁺（⁵T₂g）は
3 つの t₂g 占有が 0.5 mHa 以内に並んでいて、SCF はほぼランダムにその一つに落ちます。
`search_states` は 1 つの構造での低エネルギー SCF 解を列挙し（スクラッチ/aufbau ＋金属 d 占有の
全パターンを MOM で収束）、`BeamEngine` は最大 `max_lineages` 本の lineage を同時に生かして、
最低状態のエネルギーと力を返します。

### 進捗

Phase 1（継続計算のベースライン）、Phase 2（単一 lineage エンジン）、Phase 3a（状態探索＋
多 lineage beam）まで実装済みです。Phase 3b（近縮退状態での beam の衛生ルールと発見
スケジューリング）は設計済み・未実装。XL-BOMD、MM 側の勾配 / OpenMM 連携、ORCA バックエンドは
未着手です。

## インストール

Python ≥ 3.12、PySCF ≥ 2.14、NumPy ≥ 2.0。

```bash
pip install -e .
```

`OMP_NUM_THREADS`・`OPENBLAS_NUM_THREADS`・`MKL_NUM_THREADS` は Python を起動する**前**に
設定してください。起動後に設定すると BLAS が必要以上のスレッドを起こし、実測で 2〜2.7 倍
遅くなります。

テストの実行：

```bash
python -m pytest -q
```

## 使い方

### 単一 lineage

```python
from sroqm.backend import MethodSpec, PySCFBackend
from sroqm.engine import Engine
from sroqm.testsystems import build

mol = build("fe2_hexaaqua", "def2-svp")      # 任意の pyscf gto.Mole でもよい
coords = mol.atom_coords()                   # このステップの構造（Bohr）

engine = Engine(
    mol,
    PySCFBackend(MethodSpec("uks/b3lyp", density_fit=True)),
    fragments={"Fe": [0], "ligands": list(range(1, mol.natm))},
)
res = engine.compute(coords)   # エネルギー・力（Hartree/Bohr）・s2・アラーム
print(res.energy, res.alarms)
```

MD の各ステップで `engine.compute(coords)` を 1 回呼びます。エンジンは自分の履歴から状態を
継続します（既定は `imom`）。開殻系は `uhf`/`uks` を使ってください。

### 多 lineage beam

```python
from sroqm.beam import BeamEngine

beam = BeamEngine(
    mol,
    PySCFBackend(MethodSpec("uks/b3lyp", density_fit=True)),
    metal_atoms=[0],
    search_every=5,   # gap_sign_flip / overlap_drop / imom_fallback でも探索が走る
)
res = beam.compute(coords)            # 最低 lineage の E/F（policy="adiabatic_min"）
res.lineage_energies                  # 生きている lineage のエネルギー
res.switched, res.events              # 状態切替のフラグ・イベント
```

### 1 構造での状態探索

```python
from sroqm.search import search_states

states = search_states(backend, mol, base=None, metal_atoms=[0])   # list[Frame]、エネルギー昇順
```

## ベンチマーク

`benchmarks/bench_continuation.py` が継続戦略（`scratch`、`read`、`aspc`、`grassmann`）を
NVE トラジェクトリ上で比較します。実測した 1 ステップあたりの平均 SCF サイクル数
（def2-SVP/B3LYP、収束しきい値 1×10⁻⁷）：

| 系 | scratch | read | aspc | grassmann |
|---|---|---|---|---|
| 水二量体（閉殻、20 ステップ） | 7.0 | 5.0 | 3.95 | **3.65** |
| [Fe(H₂O)₆]³⁺ 六重項（d⁵、10 ステップ） | 12.3 | 6.4 | 5.1 | **4.6** |

エネルギーのずれは最大でも 2×10⁻⁸ Ha（水二量体）、2×10⁻⁷ Ha（Fe³⁺）です。テスト体系は
`water`、`water_dimer`、`fe3_hexaaqua`（高スピン d⁵）、`fe2_hexaaqua`（高スピン d⁶、⁵T₂g——
3 つの近縮退 t₂g 占有を持つ、状態追跡のストレステスト）です。

なお近縮退状態では、対称性が破れ始めた直後の数ステップを取りこぼすことがあります
（`search_every=1` なら回避可能）。この改良が Phase 3b です。

## 設計資料

- [設計仕様（中国語）](<Stateful Reduced-Order QM-MM：核空间与电子空间的双层自动化框架.md>)——
  フレームワーク全体の設計。§9–10 が beam とアラームの段階づけです。

## ロードマップ

- Phase 3b：beam の衛生ルール（lineage ごとの SCF 予算、候補の准入テスト、喪失/休止の扱い）と
  発見スケジューリング（更新間隔、二層の完全探索）、DEGENERATE 出力。
- 探索候補の拡張：隣接多重度（M±2）、破対称フラグメント、stability following。
- 計算機間のタスク分散（プロトタイプで検証済み）。
- XL-BOMD。
- MM 側の勾配と OpenMM 連携。
- ORCA バックエンド。
