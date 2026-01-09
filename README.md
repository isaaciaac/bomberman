# Entropy-driven Memory Reconstruction (Minimal Cognitive Core)

## English

This repo is a minimal executable implementation of an entropy-driven cognitive core. It is **not** a language model, and it does not treat "generate the most likely text" as reasoning. It makes an explicit decision about **when** the system is allowed to enter expression.

### Architecture (three layers)

- Layer 0 (representation): tokenize -> embedding -> cosine similarity only (`entropy_demo/embedding.py`)
- Layer 1 (cognitive core): state `(q, M)`, S1 retrieval, entropy gate, S2 REWRITE loop (`entropy_demo/engine.py`, `entropy_demo/retrieval.py`, `entropy_demo/entropy.py`, `entropy_demo/rewrite.py`, `entropy_demo/types.py`)
- Layer 2 (projection): a trivial `G(q, M)` that prints selected memories (`entropy_demo/generate.py`)

Key constraints implemented:

- State excludes answer text (only `(q, M)`).
- No-DROP: S2 only adds new memory atoms (monotone extension).
- Hard entropy descent: every accepted REWRITE strictly decreases `E(q, M)`.

### Run (single query)

Requirements: Python 3.11+ and `numpy`.

```bash
python -m pip install -r requirements.txt
python -m entropy_demo.cli --query "should state contain an answer string" --k 3 --epsilon 0.35 --tmax 6
```

Notes:

- S2 behavior is visible in logs (`t=1..` lines with `rewrite=...`).
- Stability history is stored in `.run/history.json`. Delete that file to reset `p_succ` back to the prior `p0`.
- For a machine-readable trace, add `--trace-json .run/trace.json`.

### Sample transcript (fresh history)

```text
S1: retrieved |M0|=3 (k=3) ids=['m003', 'm004', 'm002']
t=0 |M|=3 E=0.3689 (E_cov=0.3377 E_conf=0.3333 E_stab=0.5000 p_succ=0.50)
t=1 |M|=4 E=0.2418 (E_cov=0.1835 E_conf=0.1667 E_stab=0.5000) rewrite=bridge added=['rw_bridge_1']
...
success=True reason=E(q,M*) <= epsilon
```

### Configuration (centralized tunables)

All tunable parameters are centralized in `entropy_demo/config.py` (`ModelConfig`). You can override defaults via a JSON file:

```bash
python -m entropy_demo.cli --config config.json --query "..." 
```

Convenience overrides still exist: `--k`, `--epsilon`, `--tmax`.

Representation-layer note: the hashing BoW embedding can optionally filter tokens via config (`embedding.token_min_len`, `embedding.use_english_stopwords`) for controlled experiments.

### External verifier (engineering gate)

Optionally, an external deterministic verifier `V(q, M)` can be enabled as an additional acceptance condition for expression. When enabled, the run is considered successful only if:

- `E(q, M) <= epsilon`, and
- the verifier passes (e.g., no unreconciled strong conflicts remain).

This is configured via `verifier` in the JSON config.

Practical note: if you enable the verifier, consider setting `verifier.min_evidence_sim` (e.g. 0.2-0.3) so a synthetic bridge cannot pass without any non-generated supporting memories.

### Batch verification

Run multiple queries and get a JSON summary (default is isolated: no history/write-back):

```bash
python -m entropy_demo.batch --queries data/queries_demo.txt --mode isolated
```

To allow persistence (history/write-back/capacity) across the batch, use `--mode persistent`.

### Objective task suite (`data/tasks.json`)

This repo includes a small, deterministic task suite that checks *structural* expectations (not answer quality), e.g.:

- whether the run succeeds or refuses,
- whether S2 was invoked,
- whether a conflict mediator (`constraint`) was added when needed,
- whether entropy strictly decreases for accepted S2 steps.

Run it in isolated mode (no cross-task history/write-back):

```bash
python -m entropy_demo.eval --tasks data/tasks.json --mode isolated --out .run/eval_report.json
```

The printed summary includes `core_pass_rate`. The full per-task trace is stored in `.run/eval_report.json`.

Scoring note: the evaluator scores only objective fields (express/refuse decision, and cited memory IDs), not the natural-language `answer`.

### Pitfalls benchmark (LLM-error-leaning, 100 tasks)

To evaluate “common LLM failure modes” with a strictly scorable protocol (answer vs refuse + citations), run the included 100-task benchmark:

- tasks: `data/tasks_pitfalls_100.json`
- benchmark memory: `data/memory_pitfalls.json`
- recommended config: `configs/pitfalls_benchmark.json`

Run:

```bash
python -m entropy_demo.eval --tasks data/tasks_pitfalls_100.json --config configs/pitfalls_benchmark.json --mode isolated --out .run/eval_report_pitfalls_core.json
```

This benchmark is designed to stress:
- missing-fact temptation (should refuse when required claim keys are absent),
- strong conflict (should not express until conflict is mediated),
- and citation discipline (only cite retrieved ids).

### Transformer baseline comparison (offline)

You can score an external Transformer model on the same task suite, without adding any ML dependencies to this repo.

1) Generate a prompt pack (one prompt per task) and an outputs template:

```bash
python -m entropy_demo.baseline_pack --tasks data/tasks.json --mode isolated
```

This writes:
- `.run/baseline_prompt_pack.jsonl` (inputs/prompts you feed to the Transformer)
- `.run/baseline_outputs_template.jsonl` (a JSONL skeleton you can fill)

2) Run your Transformer using each `prompt` and save one JSON object per line to a file, e.g. `.run/baseline_outputs.jsonl`.

Required fields per line:

```json
{"task_id":"t001...","refuse":false,"used_ids":["m003"],"answer":"...","reason":""}
```

Notes:
- If `refuse=false`, `used_ids` must be non-empty and must reference only S1-provided ids for that task.
- The evaluator only scores objective fields (decision + cited ids), not the natural-language `answer`.

3) Score the baseline against the same expectations:

```bash
python -m entropy_demo.eval --tasks data/tasks.json --mode isolated --baseline-jsonl .run/baseline_outputs.jsonl
```

### Reference baseline (DeepSeek-V3.2)

In our reference runs, we used DeepSeek-V3.2 as an external Transformer baseline under the same scorable protocol (decide vs refuse + citations from S1). This repo keeps the baseline interface generic and does not ship any API credentials.

### Offline baseline (no Transformer required)

If you don’t have a Transformer available, there is an offline deterministic baseline that only uses S1 retrieval (no S2):

```bash
python -m entropy_demo.baseline_heuristic --tasks data/tasks.json --mode isolated --out .run/baseline_heuristic_outputs.jsonl --require-no-conflicts --require-claim-keys --use-en-stopwords --clip-cosine
python -m entropy_demo.eval --tasks data/tasks.json --mode isolated --baseline-jsonl .run/baseline_heuristic_outputs.jsonl
```

### Reference results (example run)

- Unit tests: `python -m unittest discover -s tests` -> `14` tests OK
- Core task suite: `python -m entropy_demo.eval --tasks data/tasks.json --mode isolated` -> `core_pass_rate=1.0` (`8/8`)
- Pitfalls benchmark: `python -m entropy_demo.eval --tasks data/tasks_pitfalls_100.json --config configs/pitfalls_benchmark.json --mode isolated` -> `core_pass_rate=1.0` (`100/100`)
- DeepSeek-V3.2 reference baseline on pitfalls (same prompts): `baseline_pass_rate=0.88`, `baseline_safety_pass_rate=0.78` (reference run)

### End-to-end engineering trace

If you want a single input to produce a compact, auditable JSON trace (S1 memories, S2 steps, added atoms, and any write-back/capacity effects), run:

```bash
python -m entropy_demo.demo --query "should state contain an answer string" --runs 2 --out .run/demo_trace.json
```

The default is `--mode persistent`, so it uses `.run/history.json` and (if enabled in config) the write-back store under `.run/`.

### Virtual-environment training (rewrite templates)

This learns a simple bucket -> rewrite ordering template file used by `rewrite.mode="template_then_search"`:

```bash
python -m entropy_demo.train --queries data/queries_demo.txt --episodes 200 --out .run/rewrite_templates.json
```

### Seed memory format (`data/memory_seed.json`)

Each atom stores the paper fields (plus an `id` and optional `eta_i` metadata):

- `id`: stable identifier (string; required for conflict mediation/signatures)
- `q_i`: applicability context (string)
- `v_i`: natural language projection (string)
- `z_i`: embedding (computed at load time from `v_i`; not stored in JSON)
- `c_i`: cost / resource weight (float)
- `s_i`: directional valence in `[-1, 1]` (not truth, not confidence)
- `eta_i`: metadata (dict)

Conflict detection (`chi`) supports:

- parsing `v_i`: `FACT:key=value` / `NOT:key=value`, or
- explicit metadata: `eta_i.claim_key` + `eta_i.polarity` (`+1` / `-1`)

Conflict mediation:

- a `CONSTRAINT:` atom can carry `eta_i.reconcile_pairs=[[idA,idB], ...]`; those pairs are treated as reconciled (chi=0) without deletion.

### Explicit simplifications (kept visible on purpose)

This repo is a verification-oriented prototype, not a production system. The following are deliberate simplifications:

- Embedding is hashing bag-of-words (no Transformer).
- `chi` is a small rule-based heuristic; no formal logic.
- S2 explores a tiny candidate set (bridge/constraint/abstraction); it is local descent, not global search.
- `cluster(q)` is a token-bucket hash; `sig(M)` is an ID signature (configurable inclusion rules).
- Memory-capacity control uses folding + suppression overlays; the capacity check can use `total` cost or an `effective` cost approximation (`capacity.cost_mode`).
- The generator is a trivial projection; removing it does not break the cognitive core.

### Tests

```bash
python -m unittest discover -s tests
```

### License

Apache-2.0 (see `LICENSE`).

---

## 中文

这个仓库实现了一个最小可运行的「熵驱动认知核心」：它不是语言模型，也不是 Agent/Planner 框架；它的目标是把“是否允许进入表达阶段”做成一个显式、可计算、可调的机制。

### 结构（三层）

- 第 0 层（表征层）：分词 -> embedding -> 余弦相似度（`entropy_demo/embedding.py`）
- 第 1 层（认知核心）：状态 `(q, M)`、S1 检索、熵门控、S2 REWRITE 循环（`entropy_demo/engine.py`, `entropy_demo/retrieval.py`, `entropy_demo/entropy.py`, `entropy_demo/rewrite.py`, `entropy_demo/types.py`）
- 第 2 层（表达投影）：一个最小 `G(q, M)`，只把结构投影成文本（`entropy_demo/generate.py`）

代码里保证了这些约束：

- 状态不含答案文本（只有 `(q, M)`）。
- No-DROP：S2 只能新增记忆原子（单调扩展）。
- 严格下降：每次接受的 REWRITE 必须让 `E(q, M)` 严格变小。

### 运行（单条 query）

依赖：Python 3.11+ 与 `numpy`。

```bash
python -m pip install -r requirements.txt
python -m entropy_demo.cli --query "should state contain an answer string" --k 3 --epsilon 0.35 --tmax 6
```

说明：

- S2 是否启动可以从日志里直接看到（`t=1..` 且带 `rewrite=...`）。
- 稳定性历史会写入 `.run/history.json`。删除它即可把 `p_succ` 重置为先验 `p0`。
- 如果需要机器可读的 trace，加 `--trace-json .run/trace.json`。

### 参数（集中管理）

所有可调参数集中在 `entropy_demo/config.py`（`ModelConfig`）。可以用 JSON 覆盖默认值：

```bash
python -m entropy_demo.cli --config config.json --query "..."
```

也可以用便捷覆盖：`--k` / `--epsilon` / `--tmax`。

表征层说明：hashing BoW embedding 支持通过配置做 token 过滤（`embedding.token_min_len`、`embedding.use_english_stopwords`），方便做可控实验。

### 外部验证器（工程门）

可选启用一个外部、确定性的验证器 `V(q, M)`，作为表达阶段的额外通过条件。启用后，只有同时满足：

- `E(q, M) <= epsilon`，并且
- 验证器通过（例如：不存在未调解的强冲突）

才会判定成功并进入表达。

对应配置在 JSON 里的 `verifier` 字段。

实用建议：启用 verifier 后，可以设置 `verifier.min_evidence_sim`（例如 0.2-0.3），避免仅靠合成 bridge 把熵做低就过门。

### 批量验证

对多条 query 跑一遍并输出 JSON 汇总（默认隔离模式：不写 history/write-back）：

```bash
python -m entropy_demo.batch --queries data/queries_demo.txt --mode isolated
```

需要跨 query 保留历史/写回/容量机制时，用 `--mode persistent`。

### 客观任务集（`data/tasks.json`）

仓库自带一个小型、确定性的任务集，用来验证“结构行为”而不是文本质量，例如：

- 这一条 query 应该成功还是拒答；
- 是否触发了 S2；
- 冲突场景下是否新增了 `constraint`（调解原子）；
- 所有被接受的 S2 步必须满足熵严格下降。

建议用隔离模式跑（避免跨任务的 history/write-back 影响）：

```bash
python -m entropy_demo.eval --tasks data/tasks.json --mode isolated --out .run/eval_report.json
```

命令会打印 `core_pass_rate`；每个任务的 trace 会写到 `.run/eval_report.json`。

打分说明：evaluator 只对“客观字段”（表达/拒答决策 + 引用的记忆 id）打分，不评价 `answer` 文本好不好。

### Pitfalls benchmark（偏向 LLM 易错点，100 题）

为了用“可打分”的方式评估常见 LLM 错误（该拒不拒 / 冲突未调解就回答 / 引用不规范），仓库提供了一个 100 题的基准：

- tasks：`data/tasks_pitfalls_100.json`
- 记忆库：`data/memory_pitfalls.json`
- 推荐配置：`configs/pitfalls_benchmark.json`

运行：

```bash
python -m entropy_demo.eval --tasks data/tasks_pitfalls_100.json --config configs/pitfalls_benchmark.json --mode isolated --out .run/eval_report_pitfalls_core.json
```

这个 benchmark 重点覆盖：
- 缺关键事实的诱导题（应拒答），
- 强冲突题（应先调解再表达），
- 引用约束（只能引用 S1 检索到的 id）。

### Transformer 基线比对（离线）

你可以把同一套任务喂给外部 Transformer（本仓库不引入任何重型依赖），再用同一个 evaluator 做对比。

1) 生成 prompt 包（每个 task 一个 prompt）以及基线输出模板：

```bash
python -m entropy_demo.baseline_pack --tasks data/tasks.json --mode isolated
```

会生成：
- `.run/baseline_prompt_pack.jsonl`（给 Transformer 的输入/prompt）
- `.run/baseline_outputs_template.jsonl`（你可以照这个格式写输出）

2) 用 Transformer 跑每条 `prompt`，把每条输出保存成一行 JSON（例如 `.run/baseline_outputs.jsonl`）。

每行至少需要这些字段：

```json
{"task_id":"t001...","refuse":false,"used_ids":["m003"],"answer":"...","reason":""}
```

注意：
- 如果 `refuse=false`，`used_ids` 不能为空，并且只能引用该 task 的 S1 提供的 id。
- evaluator 只对“客观字段”（决策 + 引用的 id）打分，不评价 `answer` 文本好不好。

3) 运行对比打分：

```bash
python -m entropy_demo.eval --tasks data/tasks.json --mode isolated --baseline-jsonl .run/baseline_outputs.jsonl
```

### 参考基线（DeepSeek-V3.2）

在我们的参考跑数里，我们用 DeepSeek-V3.2 作为外部 Transformer 基线，在同一套“可打分协议”（答/拒 + S1 引用）下做对比。本仓库仅保留通用的 baseline 接口，不包含任何 API 凭据。

### 离线基线（不需要 Transformer）

如果你手头没有 Transformer，可以用一个本地的确定性基线：它只做 S1 检索（不做 S2），用阈值规则决定“答/拒”，并输出引用的 `used_ids`：

```bash
python -m entropy_demo.baseline_heuristic --tasks data/tasks.json --mode isolated --out .run/baseline_heuristic_outputs.jsonl --require-no-conflicts --require-claim-keys --use-en-stopwords --clip-cosine
python -m entropy_demo.eval --tasks data/tasks.json --mode isolated --baseline-jsonl .run/baseline_heuristic_outputs.jsonl
```

### 参考结果（示例跑数）

- 单元测试：`python -m unittest discover -s tests` -> `14` tests OK
- 核心任务集：`python -m entropy_demo.eval --tasks data/tasks.json --mode isolated` -> `core_pass_rate=1.0`（`8/8`）
- Pitfalls 100 题：`python -m entropy_demo.eval --tasks data/tasks_pitfalls_100.json --config configs/pitfalls_benchmark.json --mode isolated` -> `core_pass_rate=1.0`（`100/100`）
- DeepSeek-V3.2 参考基线（同 prompt）：`baseline_pass_rate=0.88`，`baseline_safety_pass_rate=0.78`（参考跑数）

### 端到端工程 trace

如果你希望“一个输入信号进来 -> S1/S2/写回/投影”的全过程输出成一个不追求可读性、但可审计/可做实验的 JSON：

```bash
python -m entropy_demo.demo --query "should state contain an answer string" --runs 2 --out .run/demo_trace.json
```

默认是 `--mode persistent`，因此会使用 `.run/history.json`，并且在配置启用 write-back 时写入 `.run/memory_store.json`。

### 虚拟环境训练（rewrite 模板）

训练得到一个简单的 bucket -> rewrite 顺序模板，供 `rewrite.mode="template_then_search"` 使用：

```bash
python -m entropy_demo.train --queries data/queries_demo.txt --episodes 200 --out .run/rewrite_templates.json
```

### 种子记忆格式（`data/memory_seed.json`）

每条记忆原子包含论文字段（另外加了 `id` 和可选 `eta_i`）：

- `id`：稳定 ID（冲突调解/签名需要）
- `q_i`：适用上下文
- `v_i`：自然语言投影
- `z_i`：embedding（加载时由 `v_i` 计算；JSON 不存）
- `c_i`：成本/资源权重
- `s_i`：方向权重 `[-1, 1]`（不是真值/置信度）
- `eta_i`：元数据

冲突检测（`chi`）支持：

- 从 `v_i` 解析 `FACT:key=value` / `NOT:key=value`，或
- 使用元数据 `eta_i.claim_key` + `eta_i.polarity`（`+1/-1`）

冲突调解：

- `CONSTRAINT:` 原子可携带 `eta_i.reconcile_pairs=[[idA,idB], ...]`，对应 pair 被视为已调解（chi=0），不删除任何原子。

### 明确的简化点（为了可验证而保留）

这是面向验证的原型，不是生产系统。下面这些简化是刻意保留、并且在代码中显式可见的：

- embedding 用 hashing bag-of-words（没有 Transformer）。
- `chi` 是规则启发式，不做形式逻辑。
- S2 的候选集很小（bridge/constraint/abstraction），属于局部下降，不保证全局最优。
- `cluster(q)` 用 token bucket hash；`sig(M)` 用 ID 签名（可配置包含规则）。
- 容量控制用 folding + suppression overlay；容量检查可选 `total` 成本或 `effective` 近似（`capacity.cost_mode`）。
- 表达层只是投影；移除它不影响认知核心的运行与可验证性。

### 测试

```bash
python -m unittest discover -s tests
```

### 许可

Apache-2.0（见 `LICENSE`）。
