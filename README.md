# Entropy-driven Memory Reconstruction (Minimal Demo)

## English

This project is a small executable prototype of an entropy-driven reasoning core. It is not a language model and not an agent framework. Its purpose is to decide **when** the system is allowed to enter expression.

The core idea is: reasoning is treated as driving an internal state into a region that is safe/stable to express, instead of "always produce the most likely text".

### Quick start

Requirements: Python 3.11+ and `numpy`.

```bash
pip install numpy
python -m entropy_demo.cli --query "should state contain an answer string" --k 2 --epsilon 0.35 --tmax 6
```

What you should see:

- S1 retrieval log (`M0`)
- Entropy breakdown at `t=0`
- If the gate fails (`E > epsilon`), S2 steps `t=1..` with `rewrite=...` and strictly decreasing `E`
- Final report: `M0` (retrieved) + `DeltaM` (added by S2) + final entropy

### Model shape (three layers; intentionally separated)

- Layer 0 (representation): embedding + cosine similarity only (`entropy_demo/embedding.py`)
- Layer 1 (cognitive core): state `(q, M)`, S1 retrieval, entropy gate, S2 rewrite loop (`entropy_demo/engine.py`, `entropy_demo/retrieval.py`, `entropy_demo/entropy.py`, `entropy_demo/rewrite.py`)
- Layer 2 (expression): projection `G(q, M)` that prints selected memories (`entropy_demo/generate.py`)

Important constraints implemented in code:

- State excludes answer text: state is `(q, M)` only (see `entropy_demo/types.py`).
- No-DROP: S2 can only add new memory atoms.
- Hard entropy descent: every accepted rewrite must strictly reduce `E(q, M)`.

### Seed memory format (`data/memory_seed.json`)

Each memory atom contains the paper fields (plus a stable `id` used for signatures and reconciliation pairs):

- `id`: stable identifier (string)
- `q_i`: applicability context (string)
- `v_i`: natural language projection (string)
- `z_i`: embedding (computed at load time from `v_i`, not stored in JSON)
- `c_i`: cost (float)
- `s_i`: directional weight in `[-1, 1]` (not truth, not confidence)
- `eta_i`: metadata (dict), used for conflict and bookkeeping

Conflict detection in this demo supports two ways:

- `v_i` pattern: `FACT:key=value` / `NOT:key=value` (parsed at load time), or
- explicit metadata: `eta_i.claim_key` (string) + `eta_i.polarity` (`+1` / `-1`)

Conflict mediation:

- a `CONSTRAINT:` atom can carry `eta_i.reconcile_pairs=[(idA,idB), ...]`, and then that pair is treated as reconciled (no conflict) without deleting anything.

### Tests

```bash
python -m unittest discover -s tests
```

---

## 中文

这是一个最小可运行的「熵驱动推理核心」原型工程：它不是语言模型，也不是 Agent/Planner 框架，核心目标是决定系统**何时**允许进入表达（生成）阶段。

核心观点是：推理不等价于「随时生成概率最高的文本」，而是把内部状态推进到一个足够稳定、风险可控、可以表达的区域；表达只是最后一步的投影。

### 快速运行

依赖：Python 3.11+ 与 `numpy`。

```bash
pip install numpy
python -m entropy_demo.cli --query "should state contain an answer string" --k 2 --epsilon 0.35 --tmax 6
```

你应该能看到：

- S1 检索日志（`M0`）
- `t=0` 的熵分解
- 若未过门（`E > epsilon`），则会出现 `t=1..` 的 S2 步骤，带 `rewrite=...`，并且 `E` 严格下降
- 最终汇总：`M0`（检索到的）+ `DeltaM`（S2 新增的）+ 最终熵值

### 模型结构（三层，不混在一起）

- 第 0 层（表征层）：embedding 与余弦相似度（只提供几何相似，不判断可表达性）`entropy_demo/embedding.py`
- 第 1 层（认知核心）：状态 `(q, M)`、S1 检索、熵门控、S2 重写循环 `entropy_demo/engine.py` 等
- 第 2 层（表达层）：投影 `G(q, M)`，只负责把结构投影成文本 `entropy_demo/generate.py`

代码里实现了这些关键约束：

- 状态不含答案文本：状态只有 `(q, M)`（见 `entropy_demo/types.py`）
- No-DROP：S2 只能新增记忆原子
- 严格下降：每一步接受的重写必须让 `E(q, M)` 严格变小

### 种子记忆格式（`data/memory_seed.json`）

每条记忆原子包含论文字段（额外加了 `id`，用于签名与冲突调解配对）：

- `id`：稳定 ID（字符串）
- `q_i`：适用上下文（字符串）
- `v_i`：自然语言投影（字符串）
- `z_i`：embedding（加载时由 `v_i` 计算；JSON 不存）
- `c_i`：成本（float）
- `s_i`：方向权重 `[-1,1]`（不等同于真值/置信度）
- `eta_i`：元数据（dict）

冲突检测支持两种方式：

- 从 `v_i` 解析 `FACT:key=value` / `NOT:key=value`，或
- 用元数据 `eta_i.claim_key` + `eta_i.polarity`（`+1/-1`）直接定义冲突结构

冲突调解：

- `CONSTRAINT:` 原子可以携带 `eta_i.reconcile_pairs=[(idA,idB), ...]`，则对应 pair 被视为已调解（不删除任何原子）。

### 测试

```bash
python -m unittest discover -s tests
```
