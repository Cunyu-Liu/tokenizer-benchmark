# TokBench-RNA Goal Document V3

## Authority

- **Document**: `NCRNA_TOKENIZATION_SEGMENTATION_BENCHMARK_GOAL_V3.md`
- **Version**: V3
- **Status**: ACTIVE
- **Supersedes**: V2 (SHA-256 `ad2e9467a30c63081e5e265389b32b88d7087783ea139e6efd109e1fea75276b`)
- **SHA-256**: computed at materialization (recorded in the authority manifest)
- **Revision rule**: No silent gate modification. Any change to science, acceptance criteria, thresholds, or phase gating requires an amendment log entry and either owner approval or a new Goal version. Bug fixes that do not alter gates/SHA may be committed as amendments with evidence.

## Amendment Log (V2 -> V3)

| Date | Section | Change | Reason / Evidence |
|---|---|---|---|
| 2026-08-19 | Whole document | V3 replaces V2 as the single authoritative contract | Owner confirmed the reviewer major-revision plan `提示词/PLAN_审稿修订版_20260819.md` (915 lines) is the only effective science & engineering execution contract. It closes eight review issues and adds dynamic variable-length controls plus BiomBenchmark-inspired design. |
| 2026-08-19 | §1.2 / Phase 0 | Broadened collision matrix | Added H-Net, DNAChunker, MergeDNA, PatchDNA (biorxiv v1), BiomBenchmark and latest genomic tokenizer benchmark as direct/adjacent work; G4_BROAD_NOVELTY=FAIL, narrow contribution held CONDITIONAL_PASS_PENDING_UPDATED_COLLISION_MATRIX. |
| 2026-08-19 | §1.1/1.5/2.3/3.7 | Generation positioning | `true-suffix continuation` promoted to required main-text secondary; unconditional generation demoted to exploratory panel. |
| 2026-08-19 | §1.4/3.5/5.3 | Project scope | CORE_PAPER_SCOPE fixed at 33 runs (Track R 30 + B1 3) and independently publishable; Track D (6) and 350M (12) are conditionally-started enhancement; full 51 is no longer a core-paper prerequisite. |
| 2026-08-19 | §1.5/3.9 | Decision Map simplification | Main text uses a 4-step abstention-capable decision map (WINNER/TRADE_OFF_SET/NO_RESOLVED_WINNER/INSUFFICIENT_SUPPORT); full pairwise/regret selector demoted to supplementary. |
| 2026-08-19 | §3.1/3.7/5.2 | Biological evidence | Added length×GC×RNA-type matched views, CM×nearest-train identity cross-tabulation, non-rRNA/tRNA sensitivity, rare-family slice, and RNAGym mutation-likelihood diagnostic. |
| 2026-08-19 | §3.6 | BPN semantics | Made explicit this is a complete canonical codec system; adds uniform/order-k Markov/CTW/PPM calibration baselines and storage/build-cost reporting. |
| 2026-08-19 | §2.2/3.2/3.9 | Causal/universality wording | Uses controlled patch-rule effect / randomized boundary-rule intervention / complete-system effect; conclusions restricted to the evaluated Flat and BLT backbones. |
| 2026-08-19 | §1.4/3.1/Phase 1 | Data release version | release-22 training anchor unchanged; release 23–27 shift (release-27 composition audit added, ~10M new sequences incl. circRNA/mirtron). |
| 2026-08-19 | §1.6/3.2.2 | Variable-length controls | Registered Track L (learned causal dynamic chunking, PILOT_ONLY_NOT_IN_CORE_51) and BIO-DIAG (conservation/structure-guided patching, oracle/diagnostic); neither blocks the core. |
| 2026-08-19 | §1.7/3.5/3.8 | BiomBenchmark-inspired design | Scenario matrix, MethodProfileCard, simple baselines, low-resource/efficiency stress, unified RNAARAdapter/result parser, time-to-quality. |
| 2026-08-30 | §1.6/3.2.2/4 | Track L2 same-backbone dynamic-input pilot | 新增独立增强实验 L2（同一条 Flat 骨干，静态 NUC vs 变长 patch 输入；Stage A pilot seed101 + 可选 Stage B 3-seeds×4臂），用作架构无关性敏感性证据，不进 R_FLAT/R_BLT 主比较，不改变 33-run 门禁；执行遵守单 GPU 约束与 final access=0。批准依据：docs/decisions/amendment_L2_same_backbone_pilot_DRAFT_20260830.md（owner 批准 2026-08-30）；登记：docs/data/track_l2_same_backbone_pilot_registration_20260830.json |
| 2026-08-30 | §3.8 / Phase 8 | BLT framing + external reference registry | BLT treated as a reproducible instance of the blt_like patching-latent-transformer family; PatchDNA / H-Net (dnaHNet) / MergeDNA / DNAChunker registered as Phase-8 reference-only external systems; H-Net venue corrected to dnaHNet ICML 2026 Spotlight. Evidence: docs/decisions/20260830_blt_framing_and_external_registry.md; claims/collision_matrix_v3_20260830.json; claims/external_registry_v1_20260830.json. Owner-approved 2026-08-30. No gates or acceptance thresholds changed. |
| 2026-08-31 | §3.2 / §3.9.2 | P1–P3 mean patch length 与 LENGTH_STRATA 分箱裁决 | owner 裁决：(1) P1–P3 统一按 mean patch length=6 执行（代码 target_patch_len 8→6，旧 8 口径 P 系 run 标记 FAIL_CLOSED_WITH_EVIDENCE/superseded，按 6 重跑）；(2) Gate 2「逐长度层 patch-count 相对误差 ≤2%」的预注册长度层以代码 LENGTH_STRATA=(16–127, 128–511, 512–2047, 2048–4096) 四箱为准；Track R/B1 训练数据上限 4096，合同决策轴 4097–16384 层在 Gate 2 中恒空，不再参与该层门。证据：2026-08-31 服务器 checkpoint calib 实测（8 口径 mean≈8.60 vs 6 口径 mean≈6.59）、09 交接模板 §6。 |
| 2026-09-02 | §3.2 / §3.5 / §3.6 / §3.2.1 | Open-patch causal forward、nt 口径 exposure 与 E_S 重述 | owner 批准（2026-09-02 审计后）：(1) P 臂训练泄漏修复——closed-patch fold 的训练 forward 在 patch 内泄漏未来 nt，替换为 open-patch running-mean 前向（训练=评估口径、逐位因果、patch 内可逆无信息损失）；字面“仅 patch 末位计 loss”方案因会使 5/6 评估位置面对训练中从未出现的输入（P1 最受损）而在实现期否决；(2) exposure 按 nt 计（token_nt_counts 加权 + nt 加权 CE）；F2/F3/F6/F7 已 DONE 的 6 cell 标记 superseded 重跑，F1/F4/F5 保留；(3) E_S 重述——实证 B1(patch=1) 与 Flat 逐位同构（测试固化），B1−F1 仅报告为 lr 敏感性（3e-4 vs 6e-4），2×2 对角降级为同一 trunk 的输入参数化分解，全文措辞改为“同一 trunk 的 patched-input 变体”；(4) Track L2 与 P 系前向逐位相同，不建议按现登记启动（owner 待重定义）；(5) codec BLT 路径改单前向批量计分（O(T²)→O(T)，≤4096 nt；更长序列保留逐前缀滚动窗口）。证据：docs/decisions/20260902_amendment_open_patch_forward_nt_exposure.md；tests/test_open_patch_and_exposure.py（7 tests）；全套 251 passed / 0 failed。 |
| 2026-09-02 (follow-up) | §1.6/3.2.2（Track L2） | Track L2 取消；单卡约束维持；执行顺序确认 | owner 决定：(1) Track L2 pilot 在任何 GPU run 之前取消——开 patch 前向下 PatchInputFlatCausalLM 与 P 系前向逐位相同（测试固化），现登记方案为 P1–P3 的低预算重复，注册文件标记 CANCELLED_OWNER_2026-09-02 / CANCELLED_BEFORE_ANY_RUN，零算力消耗，代码保留作等价性证据；未来若重启需真正不同的参数化（如可学习 patch encoder）+ 全新前瞻 amendment；(2) 2026-08-28 单卡训练约束维持不变（owner 接受约 6 个月收口周期）；(3) 执行顺序：B1 s17 不受扰动自然收束 → 调度器按优先级重跑 6 个 superseded F cell 与剩余 F1/F4/F5 seeds（E_R 为完整主交付），随后 P1–P3 以开 patch 前向重跑。证据：docs/decisions/20260902_amendment_open_patch_forward_nt_exposure.md（Follow-up 节）；docs/data/track_l2_same_backbone_pilot_registration_20260830.json（cancellation 字段）。 |

---

# TokBench-RNA：同源感知、计算可核算的 ncRNA 自回归建模与条件生成分词/动态分段 Benchmark 科学预注册合同及执行计划

> **当前 authority 状态：`ACTIVE`。** 本文件（Goal V3）是 TokBench-RNA 的唯一有效科学与执行合同，由负责人验收并取代 Goal V2。取代前后的任何冲突均以本较新、明确验收的版本为准，禁止静默修改。
>
> **审稿修订状态：`REVIEWER_MAJOR_REVISION_V2_20260819`。** 本次修订关闭八项审稿问题：窄新颖性、生成定位、项目范围、选择图谱复杂度、生物学证据、BPN 语义、因果/普适性措辞、RNAcentral release 27；并新增动态变长分段对照策略及对 BiomBenchmark 的可复用设计。所有新增实验均受“核心论文不扩张、增强实验条件启动、final test 一次性解封”约束。

## 1. 总体结论与已锁定决策

### 1.1 项目定位与论文中心

项目不再以“提出一种新的 RNA 分词方法并冲击生成 SOTA”为中心，而定位为：

> 构建一个序列相似性与家族隔离、数据一致、骨干受控、训练核苷酸暴露量受控、计算与端到端部署成本可核算的 ncRNA 自回归建模与条件生成 benchmark；系统比较静态 tokenization 与动态 segmentation/patching，判断不同方案在实际 canonical codec 码长、真实后缀续写、有效上下文、计算效率、记忆风险、家族可识别性和 RNA proxy fidelity 上的真实权衡。

论文暂定名修订为：

> **TokBench-RNA: A Homology-Aware and Compute-Accounted Benchmark of Tokenization and Dynamic Segmentation for Autoregressive ncRNA Modeling and Conditional Generation**

论文唯一中心问题不是“哪个 tokenizer 平均第一”，而是：

> **在同源隔离的数据和明确的公平性估计量下，RNA 自回归模型的 tokenizer/segmentation 排名是否随任务、序列长度和预算定义而改变；观察到的质量收益是否伴随更高的记忆风险、计算代价或 RNA 分布保真损失？**

项目成功不要求 entropy patching 获胜，也不要求提出新架构。以下结果均可形成有效 benchmark 发现：任务依赖的排名反转、稳定的性能–成本 Pareto、entropy 与 matched-random 无实质差异、所有方法在预注册区间内实际等价，或无法稳定选出赢家但能形成明确的 failure/abstention map。前提是区间足够精确、数据/evaluator gate 全部闭合，而不是把“未显著”写成“等价”。

### 1.2 更新后的新颖性边界与 collision gate

已确认的直接或邻近工作至少包括：

- [BEACON](https://proceedings.neurips.cc/paper_files/paper/2024/hash/a8ea503d91320fcfe12cba61c8a6d285-Abstract-Datasets_and_Benchmarks_Track.html)：RNA encoder 中比较 single nucleotide、BPE、overlapping/non-overlapping k-mer，并覆盖 13 个下游任务。
- [GARNET](https://www.nature.com/articles/s41467-024-54812-y)：RNA 自回归模型中比较单碱基、重叠二联体和三联体，并报告生成与实验结果。
- BiRNA-BERT、RNAret 等：已覆盖 RNA 中 NUC/BPE 或不同 k-mer 的任务依赖比较。
- [PatchDNA](https://www.biorxiv.org/content/10.1101/2025.11.28.691095v1)：DNA 中提出 fixed、entropy、conservation-guided patching 与 re-patching；“首次核酸 entropy patching/生物引导 patching”均不可主张。
- [H-Net](https://iclr.cc/virtual/2026/poster/10008794)：端到端学习 content/context-dependent dynamic chunking，并在 DNA 模态报告结果。
- [DNAChunker](https://holymollyhao.github.io/dnachunker/) 与 [MergeDNA](https://ojs.aaai.org/index.php/AAAI/article/view/37032)：已在 DNA 中研究 learnable router、动态分块或 differentiable token merging。
- [Zero-shot benchmarking of RNA language models](https://academic.oup.com/bib/article/27/2/bbag098/8509095)：统一评估多种开放 RNA 模型并建立结构、家族和突变适应度子集。
- [Benchmarking pre-trained genomic language models for RNA sequence-related predictive applications](https://www.nature.com/articles/s41467-025-66899-y)：在统一 split 与训练流程下比较 11 个 gLM、四类 RNA 任务、数据规模/不平衡/上下文/效率，并给出场景化模型选择指南。
- GenerRNA：提供 RNAcentral release 22、BPE-1024 和 350M decoder 公开参照，必须区分更新权重和历史权重。

因此冻结：

- `G4_BROAD_NOVELTY = FAIL`。
- 禁止宣称“首次 RNA tokenizer benchmark”“首次 RNA 动态分词”“首次核酸 entropy patching”“RNA 模型尚无统一 benchmark”。
- `G4_NARROW_BENCHMARK = CONDITIONAL_PASS_PENDING_UPDATED_COLLISION_MATRIX`。
- 可保留的窄贡献是：**在广谱、多家族 ncRNA 自回归建模中，以共同数据、同源/家族隔离、统一核苷酸 exposure、共同 FLOP 检查点、实际可解码 codec、真实后缀续写和记忆/RNA proxy 约束，分离 Flat 内静态表示效应、BLT 内受控 patch-rule 效应，以及跨骨干完整系统的部署 Pareto。**
- 投稿前的 collision matrix 必须逐项覆盖：任务、训练目标、数据版本、同源隔离、family/clan OOD、骨干、参数规模、raw context、representation slots、tokenizer/patcher、是否输入自适应、预算、指标语义、生成、记忆、生物 proxy、代码/权重和发表状态。
- collision matrix 至少纳入 BEACON、GARNET、BiRNA-BERT、RNAret、PatchDNA、H-Net、DNAChunker、MergeDNA、RNA zero-shot benchmark、BiomBenchmark、GenerRNA 及最新 genomic tokenizer benchmark。
- 投稿前重新执行一次系统检索。若新工作完整覆盖上述交集，项目转为 evaluator/data resource、受控复核或负结果论文，不通过改名维持“首创”。

### 1.3 旧 BLT 项目的证据状态

Goal 文档必须完整记录但不得继承旧结论：

- 旧训练数据 lineage 不闭合，当前可见 cleaned-train artifact 中已经确认至少存在原始 test 记录。
- 当前可见候选代码中，显式 n-gram 查表、生成和 PPL 路径不一致；显式 n-gram 训练可能退化为默认 ID，生成使用全零占位，PPL 又关闭该通道。
- hash byte-groups、explicit lookup n-gram 和 entropy patch 曾被混合，旧结果不能归因于任何单独机制。
- 旧 H100/H800/no-ngram/ngram 结果无法完整闭合到唯一 `code → data → config → checkpoint → decoder → output → evaluator`。
- 当前本地 `outputs/` 中未发现上一轮最终报告和账本。按用户选择，记为 `artifact_lineage_status=MISSING`，后续新产物标记 `REGENERATED_NEW_LINEAGE`，不得冒充恢复旧字节。
- 旧 checkpoint、生成文件和 PPT 数值只能放入“历史记录/排错证据”附录，不能进入新 benchmark 主表、模型选择或先验胜负判断。

### 1.4 项目范围：核心论文、增强实验与探索性模块

#### A. 核心论文层（`CORE_PAPER_SCOPE`，必须完成）

- 数据锚点：RNAcentral release 22；来源、license、accession、hash 与 canonical entity lineage 闭合。
- Database-release-shift：release 23–27 相对 release 22 的新增 accession，经 exact/80-80 cross-search 后进入 sensitivity；不称生物时间泛化。
- Track R：100M 十臂主矩阵，10 臂 × 3 formal seeds = 30 runs。
- Bridge B1：BLT patch-size=1，1 臂 × 3 seeds = 3 runs；B2 继续复用 F7。
- 核心正式训练数：`33 runs`。这是最小可发表 benchmark 的训练闭环，不要求 Track D 或 350M 才能成立。
- 必须交付：actual canonical codec BPN、true-suffix continuation、共同 FLOP 轨迹、端到端推理成本、记忆风险、family/clan/length 分层、RNA proxy controls、开放 manifest/evaluator/adapter。

#### B. 增强论文层（`ENHANCEMENT_SCOPE`，条件启动）

- Track D：100M Flat-NUC/Flat-BPE，2 臂 × 3 seeds = 6 runs；固定 representation slots 与共同累计 FLOPs，回答部署/上下文容量的完整系统差异。
- 350M：NUC、BPE、fixed patch、entropy patch，4 臂 × 3 seeds = 12 runs；只复验规模趋势。
- 若 Track D 与 350M 均通过预注册启动门，完整既定程序为 `33 + 6 + 12 = 51 runs`。
- 任一增强模块未启动或资源受阻，只删除相应主张，不阻断 `CORE_PAPER_SCOPE`。

#### C. 探索性模块（`EXPLORATORY_SCOPE`，不得阻塞论文）

- learned dynamic chunking 系统参考；
- conservation/covariation/structure-guided patching 的有标注子集诊断；
- 无条件生成的大规模 decoder 网格；
- 复杂 selector graph、逐 cell winner 和完整 regret/coverage machinery；
- 可运行外部模型与生态比较。

固定训练 seed 为 `17、29、43`；独立调参 seed 为 `101`。调参 run 不得续训为正式 run。

### 1.5 主证据结构：两个质量面板、一个系统 Pareto、一个会弃权的决策图

论文主文不再以复杂 choice-atlas 算法作为成败条件，而固定为四项交付：

1. **Main Evidence A — Canonical modeling quality**：冻结 model–tokenizer–patcher–codec 系统的实际可解码 `canonical_code_length_BPN`。
2. **Main Evidence B — Conditional generative quality**：相同 raw prefix 下真实后缀的 continuation code length；生成样本的 edit distance/accuracy 只称 reference-continuation fidelity。
3. **Main Figure — Performance–cost–harm Pareto**：质量对累计 FLOPs、端到端延迟、显存、有效上下文、记忆风险和 RNA proxy harm。
4. **Decision Map — task × length × compute**：输出推荐、trade-off、无可判定赢家或支持不足，并明确弃权。

决策图的条件轴仅为：

| 主轴 | 条件 |
|---|---|
| 任务 | canonical codec、true-suffix continuation；无条件生成单列为 exploratory panel |
| 长度 | `16–127 / 128–511 / 512–4096 / 4097–16384`，并区分真正 length OOD 与 held-out long context |
| 计算 | valid-nt exposure、共同累计 FLOPs、端到端推理延迟、显存和 effective raw context |

生物分辨率、validity、family/clan recoverability、预测结构/分布 proxy 与 memorization 是约束/结果族，不再作为第四个轴。

主文只展示：分层效应量与区间、Pareto frontier、明确的弃权原因和 failure map。完整 pairwise dominance graph、regret/coverage/harm/abstention 算法如实现，放入补充材料和 `decision_map.yaml`，不得反向决定核心论文是否成立。

实用门槛不再仅以“已预注册”为依据。`1% BPN、15% latency、2 pp validity/family、1 pp memorization、5% distribution distance` 均必须在 final unseal 前由 validation 重复噪声、测量精度、部署意义或文献给出书面 justification；缺乏依据的门槛只能作为 sensitivity，不能产生强 winner claim。

### 1.6 变长分段方法：除 entropy patch 以外的对照策略

首先区分两类“变长”：

- **静态变长 tokenization**：BPE 与 Unigram 对不同序列产生不同 token 长度，但词表和切分规则在训练前冻结，不根据当前位置的模型状态自适应。
- **动态变长 segmentation**：边界随输入内容、prefix 状态、学习式 router 或生物先验变化。

当前正式 BLT 核心保留三臂：

| ID | 规则 | 作用 | 证据边界 |
|---|---|---|---|
| P1 | fixed-6 patch | 层级架构固定长度基线 | 非变长；用于隔离 patch 架构本身 |
| P2 | supported-strata length-matched random variable patch | 与 P3 匹配 patch budget 的随机边界反事实 | 只在 positivity/balance gate 通过的 supported strata 解释边界位置增量 |
| P3 | prefix-causal entropy patch | 基于 prefix 条件不确定性的动态变长规则 | 只能称冻结 BLT 中的 controlled adaptive-rule effect |

新增两类但不强制扩张核心矩阵：

1. **L1 learned causal dynamic chunking（开发门控的独立系统参考）**  
   参考 H-Net、DNAChunker、MergeDNA/GBST 的 learnable router 或 differentiable merging。因为它会改变参数路径、优化和边界学习机制，默认放入独立 `Track L`，不与 P1–P3 混作纯 patch-rule 因果比较。先仅使用 train/validation 和 `tuning_seed=101` 做实现/稳定性 pilot；只有 prefix causality、round-trip、参数/FLOP 记账、平均 patch budget、三次重复训练可行性全部通过，并在 final-test access=0 时由负责人签署 amendment，才允许增加三个 formal seeds。否则登记为 `REFERENCE_ONLY` 或未来工作。

2. **BIO-DIAG biological-prior patching（有标注子集诊断）**  
   参考 PatchDNA 的 conservation-guided 思路，在 Rfam 可比对家族中探索 covariance-model information content、alignment conservation 或结构置信度引导的边界。由于 broad RNAcentral 缺乏统一逐位注释，且全序列结构/家族信息可能违反自回归 prefix causality或引入标签泄漏，本方法只作为有标注子集的 oracle/diagnostic upper bound；不得进入 broad primary ranking，除非未来建立 train-only、family-held-out、prefix-causal 的可部署规则并通过独立 amendment。

因此，本项目不仅比较 entropy：正式核心已有 matched-random 反事实；同时注册 learned dynamic chunking 与 biological-prior patching 两条扩展路线，但不以无限增加训练臂换取“看起来完整”。

### 1.7 从 BiomBenchmark 借鉴并落地的设计

从优秀 RNA gLM benchmark 借鉴以下可复用原则：

- **场景矩阵而不是平均榜单**：按数据支持量、序列长度、任务类型、计算预算和类别/家族不平衡解释模型选择。
- **模型/方法 Profile Card**：对每个 arm 记录训练数据、目标、骨干、参数、词表/patcher、最大上下文、训练 exposure、FLOPs、端到端延迟、license 和可运行状态。
- **简单基线校准复杂模型**：除神经模型外加入 uniform、order-k Markov、CTW/PPM 类无损压缩基线；避免只在昂贵模型之间内部排名。
- **低资源与困难场景压力测试**：用早期共同 FLOP checkpoints、rare-family/family-balanced slice、长序列、低复杂度和 release-shift 代替事后挑选有利子集。
- **性能与时间共同报告**：提供 time-to-quality、吞吐、显存、在线 tokenizer/patcher 开销和每个有效生成的成本。
- **可扩展统一接口**：`RNAARAdapter + dataset registry + model/tokenizer profile + result parser + frozen analysis notebook`，让社区新增方法而不重写 evaluator。
- **决策指南可解释**：最终给出按任务/长度/预算选择或弃权的流程图，而不是只给一个总榜。

明确不照搬的部分：BiomBenchmark 比较的是不同公开 gLM 的生态表现；这些模型同时改变预训练数据、骨干、规模和 tokenizer。它可作为“如何做场景化推荐和开放框架”的模板，但不能替代本项目的同骨干 tokenizer/patch-rule 受控估计。

## 2. 科学问题、估计量与分析层级

### 2.1 唯一核心问题

> 在同源隔离的数据、统一自回归目标和明确的 raw-context/exposure/FLOP/部署预算下，ncRNA 模型的静态表示粒度与动态分段规则如何改变实际 canonical codec 码长和真实后缀续写；这些质量变化是否伴随有效上下文、计算成本、记忆风险、family/clan 泛化或 RNA proxy fidelity 的系统性代价？

任何结论必须限定到实际运行的 Flat Transformer 与 BLT backbones。项目不把“最优 tokenizer”视为先验存在，也不把跨骨干差异升级为 tokenizer 的普适因果效应。

### 2.2 估计量与证据边界

| 对象 | 实验来源 | 回答的问题 | 明确不回答 |
|---|---|---|---|
| `E_R_STATIC_REPRESENTATION` | Track R，F1–F7 | 同 Flat 骨干、同 raw context、同 valid-nt exposure 下静态表示/tokenization 的完整效应 | 不回答 BLT 是否更优；不包含固定 representation slots 带来的额外 raw context |
| `E_P_CONTROLLED_PATCH_RULE` | Track R，P1–P3 | 同 BLT 骨干和 mean patch length=6 下 fixed、matched-random、entropy adaptive rule 的受控差异 | balance/positivity 失败时不解释 fine boundary placement；不推广到所有动态分段方法 |
| `E_S_SYSTEM_DECOMPOSITION` | F1/F7/B1/P1 固定 2×2 | 粒度与层级系统同时改变时，完整系统响应如何分解 | 不升级为纯 tokenizer、纯架构或单模块因果效应 |
| `E_D_DEPLOYMENT` | Track D，D1/D2 | 固定 4096 representation slots 与累计 FLOPs 下，更长 raw context 的部署系统收益与代价 | 不把 tokenizer、上下文和执行路径拆成纯边界效应 |
| `E_L_LEARNED_DYNAMIC_SYSTEM` | 可选 Track L | learned router/dynamic merging 的完整系统表现 | 不与 P1–P3 混成同一受控 patch-rule estimand；未 formalize 时仅 reference-only |
| `DECISION_MAP` | validation 冻结、final test 一次确认 | 分层效应/Pareto 是否能在 task × length × compute 上稳定给出推荐、trade-off 或弃权 | 单个 cell 不升级为 confirmatory；完整 selector 不是项目成败前提 |

全文统一使用：`controlled patch-rule effect`、`randomized boundary-rule intervention`、`complete-system effect`。仅“prefix-causal”表示自回归不看未来，不等于因果推断。禁止无修饰的“causal tokenization effect”。

### 2.3 Endpoint 与论文层级

| 层级 | 冻结内容 | 论文用途 |
|---|---|---|
| Headline confirmatory | fixed final-budget checkpoint；primary cluster-held-out final test；`canonical_code_length_BPN`；canonical entities 等权；seed 17/29/43 paired effects | Main Table 1/2；跨 tokenizer 唯一 confirmatory headline |
| Required main-text secondary | true-suffix continuation code length；固定 raw-prefix manifest；相同 target raw length；逐 seed paired effects | Main Figure/Table 的第二个质量面板，保证论文确实评价条件生成，而非只评价压缩 |
| Major secondary | common-FLOP curves；端到端 latency；Track D；family/length 分层；decision-map overall regret/coverage/harm/abstention（如实现） | 解释适用范围、部署权衡和决策可迁移性 |
| Diagnostic/exploratory | 无条件生成；boundary mechanism；结构/CM/RNAGym proxy；BIO-DIAG；Track L pilot；external reference-only；database-release shift | failure map、机制假设和限制；不得事后升级为 primary |

三个 training seeds 是独立训练重复；测试 cluster、family、prefix 和 generation output 均不是额外模型重复。项目采用效应量与区间估计，不将三个 seeds 包装成高功效显著性检验。

### 2.4 主文档阅读顺序

科学合同按以下顺序执行：核心问题与 estimand（本节）→ 数据与 split（3.1）→ 核心 Track R/Bridge 与可选 Track L/Track D/350M（3.2–3.5）→ codec、续写/生成、生物控制和统计决策图（3.6–3.9）→ scope gate 与 final-test 生命周期（Phase 0–8）。GPU、Git、PID、目录和恢复规则位于附录，不参与 scientific claim 的定义。

## 3. 科学 Benchmark 合同

### 3.1 数据集构建

主训练数据：

- 官方或可验证归档的 RNAcentral release 22，作为与 GenerRNA 可比且在项目开始前冻结的训练锚点。
- 如果官方 release 22 原始快照无法合法、完整地恢复，可使用 GenerRNA 发布的 release-22-derived 数据作为单独标记的重建版本，但不得声称它等于官方原始快照。
- 如果两者都不能闭合 accession、license 和 artifact identity，`gate_status=FAIL_CLOSED_WITH_EVIDENCE`，禁止以未知本地旧数据替代。

Database-release-shift：

- sensitivity 更新为 RNAcentral release 23–27 中相对 release 22 新增的 accession；不再停在 23–26。
- release 27 新增约千万级序列，并引入大量 circRNA、mirtron 等来源，因此必须同时报告 `release_id × source_database × RNA_type × length` 构成；任何性能变化不得仅解释为“时间泛化”。
- 每个 release-shift entity 必须再次移除对 release-22 train 的 canonical exact 和 80% identity/80% bidirectional coverage overlap。
- release 27 的 Parquet/Hugging Face 导出可以用于数据获取便利性，但本项目仍须冻结原始 accession、sequence hash、release manifest 和 canonicalization 结果，不能把可变在线数据集当作 authority。

Primary 数据规则：

- 大小写统一；
- `T → U`；
- primary alphabet 为 `A/C/G/U`；
- 含其他 IUPAC 字符的记录保留在 QC 账本，但不进入 primary 训练；另建 ambiguity stress subset；
- Track R、B1 和 350M 的主训练长度为 `16–4096 nt`；
- Track D 为回答固定 representation slots 下的上下文容量问题，使用独立但前瞻冻结的 `D_long_context_train_view`：仅包含已经分配到 train clusters 的 `16–16384 nt` canonical entities；D1/D2 使用完全相同的 entities 与冻结顺序，禁止引入 validation/final clusters；
- Track D 另建立 `D_long_context_validation_view`：只包含 validation clusters 中满足相同 `16–16384 nt` 规则的 canonical entities，用于 Track D HPO、上下文容量检查和 validation-only diagnostics；它与 train view、所有 final test 严格互斥；
- `4097–16384 nt` final subset 对 Track R/B1/350M 标为 `local-window length_OOD`；对 Track D 只能标为 `held_out_long_context`；
- RNA 有方向性，不把 reverse complement 静默视为同一序列；单独报告 reverse-complement 近邻；
- canonical exact duplicate 合并为一个训练实体，但完整保留 accession 和 metadata 映射；
- train、validation、test 之间 canonical exact overlap 必须为零。

同源隔离：

- primary cluster 使用 MMseqs2 `80% identity / 80% query-and-target coverage`；
- `90%/90%` 作为敏感性分析；
- 同一 primary cluster 不能跨 split；
- 工具版本、参数、cluster membership 和全部 hash 冻结；
- split 完成后直接执行 train→validation/test cross-search，验证不存在满足 primary 80/80 条件的跨 split sequence pair；只比较 cluster ID 不足以关闭 leakage gate；
- 对短 RNA 预注册高敏感 alignment/精确枚举复核，并以已知 Rfam family/clan 测试聚类召回；若 80/80 对短 RNA 的检出不足，相关样本使用更严格的 family-aware 隔离并在 datasheet 报告。

Split：

- 在 family/clan held-out 分配完成后，剩余同源簇按稳定 hash seed `20260808` 分为 `98% train / 1% validation / 1% cluster-held-out test`；
- 分层变量至少包括长度区间、RNA type、来源数据库和 Rfam 标注状态；
- eligible Rfam family 定义为清洗后至少 100 条序列且至少 10 个同源簇；
- eligible family 中 10% 分配给 family-validation、10% 分配给 family-test；
- 有 clan 的 family 另外构建完整 clan-held-out sensitivity split；
- family/clan test 对应的同源簇全部从训练集移除；
- primary cluster-held-out test、family test、clan test 和 database-release-shift test 均为 final sealed test；Phase 0–7 只能访问 train、validation 和预注册的未封存 diagnostics，Phase 8 是唯一授权 unseal。

训练以去重后的 canonical sequence entity 为等权采样单位；accession multiplicity 只保留为 metadata/QC，不得称为自然丰度。

评估必须同时报告：

- canonical-entity-weighted micro average；
- family-balanced macro average；
- RNA type/source/length 分层结果；
- 去除 rRNA/tRNA 后的 sensitivity；
- rare-family 与 high-frequency-family 分开结果；
- length × GC × RNA type 匹配后的结构/分布 proxy 比较。

不得只用一个总体平均值掩盖 rRNA、tRNA 或高频 family 的主导效应。

### 3.2 Track R：100M 十臂核心受控矩阵

每臂使用三个训练 seed，共 30 个科学运行：

| ID | 骨干 | 表示或分段 | 核心用途 |
|---|---|---|---|
| F1 | Flat causal Transformer | 单核苷酸 NUC | 静态主基线 |
| F2 | 同一 Flat backbone | BPE，vocab 1024 | GenerRNA 对齐的静态子词基线 |
| F3 | 同一 Flat backbone | Unigram，vocab 1024 | 排除收益仅来自 BPE merge 算法 |
| F4 | 同一 Flat backbone | overlapping 3-mer，stride 1 | 对齐 GARNET 风格局部表示 |
| F5 | 同一 Flat backbone | overlapping 6-mer，stride 1 | 对齐 BEACON 常见 6-mer |
| F6 | 同一 Flat backbone | non-overlapping 3-mer，stride 3 | 区分 overlap 与块状 token |
| F7 | 同一 Flat backbone | non-overlapping 6-mer，stride 6 | k 和 stride 敏感性 |
| P1 | 同一 BLT backbone | fixed-6 patch | 平均 patch length=6 的实用固定基线 |
| P2 | 同一 BLT backbone | supported-strata hybrid matched-random variable patch | 在平衡门通过时提供长度/预算匹配的随机边界反事实 |
| P3 | 同一 BLT backbone | prefix-causal entropy patch，阈值校准到 mean patch length=6 | 估计冻结 entropy adaptive rule 相对 fixed/random 的受控系统效应 |

约束：

- F1–F7 使用相同层数、宽度、注意力实现、位置编码、context_nt 和训练序列顺序；
- 不通过调整 backbone depth/width 匹配参数；
- 对较大词表使用 factorized/tied embeddings，使 total trainable params 落入共同容差；
- BPE/Unigram 只在 Track R 的 `16–4096 nt` primary train canonical entities 上训练一次，冻结唯一 tokenizer artifact、trainer config 和独立 trainer seed；不得用 `D_long_context_train_view` 更新 merge/vocabulary。该 BPE artifact 原样供 F2 三个 seeds、Track D D2 和 350M C2 复用，Unigram artifact 供 F3 三个 seeds 复用；D2 是词表的长序列外推应用，不是重新训练的 tokenizer；
- P1–P3 使用完全相同 BLT 参数和模块；
- BLT explicit lookup n-gram、hash byte-groups 以及任何遗留 dummy n-gram 全部关闭，且测试确认对应参数数量为零；
- P1 固定为 patch length `6`；P1 只匹配目标平均长度，不声称匹配 P3 的完整 patch-length distribution；
- P3 的 threshold 只在 train split 校准，使 train-only 平均 patch length 为 `6`，并冻结 estimator、threshold、最大 patch 长度和校准报告；
- P2 使用独立于模型训练 seed 的 `patch_randomization_seed=20260815`；位置 `i` 的随机数只由 `sequence_or_generation_id + i + patch_randomization_seed` 决定，三个 model seed 共用同一冻结规则，并满足 prefix causality；
- P2 在 train-only 数据上拟合 `q(boundary | causal prefix-length stratum, causal-entropy stratum)` 以匹配 P3 的 patch budget；运行时只能使用当前位置、已观察 prefix 和 causal entropy，不得读取最终序列长度或未来碱基。校准报告同时按预注册完整序列长度层汇总 patch count；不得使用 validation/test 序列重新校准；
- P2/P3 总 patch count 相对误差不得超过 `0.5%`，各预注册 sequence-length stratum 不得超过 `2%`，实测 FLOPs 差异不得超过 `5%`；
- `q` 必须以 train-only P3 的 causal boundary labels 拟合；在预注册的 `causal prefix-length × causal-entropy` 联合层内，P2/P3 boundary-rate 绝对差不超过 `2` 个百分点。只有 `0.05≤q≤0.95` 的 supported strata 执行随机边界干预；在 `q<0.05` 或 `q>0.95` 的 strata，P2 必须逐位精确 replay P3 的 deterministic causal boundary，不能用近似概率随机化。supported strata 必须覆盖至少 `80%` 的训练 positions 和 P3 boundaries；
- 另外审计每序列 patch count、patch-length distribution、maximum-patch-cap 触发率，以及 length、GC、low-complexity、source、family strata 的 patch-budget balance；连续分布的预注册 standardized difference 不超过 `0.1`，每个有至少 100 clusters 的离散层 patch-count 相对差不超过 `2%`；
- final test 只能用冻结规则执行平衡审计，不能重新拟合 `q`、改变 strata 或修正边界；Phase 8 必须用冻结 `q` 重新报告 supported positions 与 P3 boundaries 的覆盖率。任一覆盖率低于 `80%` 时，自动撤销 fine boundary-placement claim，只保留 total adaptive-rule system comparison，且不得用 final test 结果回头重拟合；
- 若 non-supported strata 不能精确 replay P3，或任一平衡/positivity/FLOP 门失败，P2/P3 仍可作为完整系统比较，但只允许称为 `total adaptive-rule effect`；全部通过时，也只允许称“在报告的 supported-strata coverage 下，相对这一冻结 matched-random control 的 fine boundary-placement rule effect”，不得推广为所有 random-patching family、所有边界或单个边界的局部因果效应；
- P3 entropy predictor 只在 train split 训练一次，使用独立于 model seeds 的 patcher trainer seed；冻结同一 predictor/threshold/最大 patch 长度/boundary implementation供 P3 三个 seeds 与 350M C4 使用，并记录参数量、训练 FLOPs、checkpoint identity 和在线/离线成本；
- static track 内只能解释该 Flat 实现中的静态 representation/tokenization effect；
- patch track 内使用 `controlled patch-rule effect`；只有 P2/P3 balance/positivity gate 通过时，才可进一步描述 supported strata 中的 randomized boundary-rule intervention；
- F1 与 P3 等跨骨干比较只能标为 complete architecture/system comparison，不能称为纯 tokenizer effect；
- BPE/Unigram 是静态变长 tokenization；P2/P3 是输入相关的动态变长 segmentation，二者不得混称。

### 3.2.1 Flat 与 BLT 的三层比较、主结果和桥接模型

本 benchmark 必须把结论分成三层，禁止将三层混成一个总排行榜：

**第一层：Flat 赛道内部的静态 tokenizer 比较**

- 比较对象：F1–F7，即 NUC、BPE、Unigram、overlapping/non-overlapping k-mer；
- 保持不变：Flat causal Transformer 骨干、训练数据、序列顺序、`context_nt`、有效核苷酸 exposure、优化协议和 evaluator；
- 允许结论：在 Flat 骨干中，某种静态 tokenizer 在指定任务、长度和预算下更优或更具性价比；
- 禁止结论：仅凭 Flat 赛道结果推断 BLT 的层级 patching 是否更优。

**第二层：BLT 赛道内部的受控 patch-rule 比较**

- 比较对象：P1–P3，即 fixed、patch-length-matched random、causal entropy patch；
- 保持不变：BLT 骨干、参数、平均 patch 数/长度分布口径、训练数据、有效核苷酸 exposure、优化协议和 evaluator；
- 允许结论：在该冻结 BLT 实现、报告的 supported-strata coverage 和平衡门下，entropy adaptive rule 相对 fixed/matched-random 是否存在稳定增量；
- 禁止结论：把 BLT 相对 Flat 的全部差异归因于 entropy 或“动态分词”。

**第三层：Flat 与 BLT 的跨赛道系统比较**

- 比较单位是完整系统，不是单一 tokenizer；
- 两条赛道使用同一数据/split、相同有效核苷酸 exposure、相同 raw-nucleotide context、共同累计 FLOP 检查点、相同测试序列和冻结的解码协议；
- 同时报告参数量、训练 FLOPs、推理 FLOPs、吞吐、延迟、峰值显存、BPN/续写/生成、训练记忆、family recoverability 和结构代理；
- 允许结论：在某个任务、长度和计算预算下，哪一个完整系统处于性能–效率–生物代价 Pareto 前沿；
- 禁止结论：从任意 Flat-vs-BLT 差值直接声称“某 tokenizer 导致提升”。

主结果固定为三类输出：

1. **Main Table 1 — Flat 静态 tokenizer 内部表**：F1–F7 的 headline、逐 seed paired effect、CI、计算和生物代价；
2. **Main Table 2 — BLT controlled patch-rule table**：P1–P3 在共同 mean patch length=6、且 P2/P3 通过分层 patch-budget/FLOP 平衡门后的 actual codec、true-suffix continuation、机制与代价；
3. **Main Figure 1 — System Pareto**：预先固定三个 panel：训练累计 FLOPs–`canonical_code_length_BPN`、端到端推理延迟–true-suffix continuation code length、推理成本–memorization/RNA-proxy constraints。无条件生成另列 exploratory panel；跨赛道图只用于完整系统选择，不用于 tokenizer 因果归因。

为缩小 Flat 与 BLT 之间的归因缺口，100M 阶段固定一个不增加未知数据流的 2×2 system decomposition。B1 是新增正式桥接臂；B2 只是 F7 的桥接别名，不新增训练：

| 表示粒度 | Flat Transformer | BLT | 可计算差值 | 允许解释 |
|---|---|---|---|---|
| 单碱基 | F1：Flat NUC | B1：BLT patch-size=1 | `B1 − F1` | 在单碱基表示下的 hierarchy/system effect |
| 固定 6-nt 分组 | B2=F7：Flat non-overlap 6-mer | P1：BLT fixed-6 | `P1 − F7` | 在固定 6-nt 表示下的 hierarchy/system effect |

B1 固定为 100M、seed `17/29/43`、`2.0B cumulative_valid_target_nt`、`4096 raw nt`；其参数、训练核苷酸、checkpoint 和 common-FLOP 记录规则与 Track R 一致。F1、F7、B1、P1 使用相同数据顺序和 evaluator。

2×2 判读固定为：

- `F7 − F1`：Flat 系统内从单碱基到固定 6-mer 的完整表示效应；
- `P1 − B1`：BLT 系统内从 patch-size=1 到 fixed-6 的完整 patch-compression effect；
- `(P1 − B1) − (F7 − F1)`：差中之差只用于描述“表示粒度改变在两类骨干中的系统响应是否不同”；
- `B1 − F1` 与 `P1 − F7`：在相应表示粒度下的层级系统差异。

该 2×2 只能提供系统分解证据。因为 Flat 与 BLT 的 local/global 计算、参数路径和预测实现仍不同，它不能升级为纯 tokenizer、纯架构或单一模块的因果效应；若参数、raw context、exposure 或共同 FLOP 记录未闭合，对应差值降级为描述性 Pareto 证据。

### 3.2.2 动态变长分段扩展：Track L 与 BIO-DIAG

#### Track L：learned causal dynamic chunking（条件性系统参考）

候选实现参考 H-Net/DNAChunker/MergeDNA/GBST，但必须针对自回归 RNA 场景重新审计 prefix causality 与无损恢复。默认状态为：

```text
L1_LEARNED_DYNAMIC_CHUNKING = PILOT_ONLY_NOT_IN_CORE_51
```

Phase 3 只允许使用 train/validation 与 `tuning_seed=101` 完成一个开发 pilot。升格为三个 formal seeds 前必须同时满足：

- 边界/合并决策只依赖当前及历史 prefix；suffix perturbation 不改变已观察 prefix 的 chunk boundaries 或 next-step logits；
- source RNA 可通过冻结 encoder/decoder 无损恢复；
- mean patch length、patch count 和 maximum-cap 规则可冻结，且与 P1–P3 的预算关系可审计；
- boundary/router 参数、训练 FLOPs、在线推理 FLOPs 和延迟单列；总参数无法落入 2% 容差时，只能作为完整系统参考；
- 三次独立训练在资源上可行；
- `final_test_access_count=0`；
- 负责人签署前瞻 amendment，明确新增 run 数和允许主张。

即使正式运行，L1 也进入独立 `LEARNED_DYNAMIC_SYSTEM` comparison family，不进入 P1–P3 的 controlled patch-rule contrasts。

#### BIO-DIAG：conservation/covariation/structure-guided patching

- 仅在 Rfam family/clan 信息和多序列比对可闭合的子集运行；
- 默认允许 covariance-model information content、alignment conservation、pairing confidence 等作为 oracle/diagnostic boundary signal；
- 必须报告 annotation coverage、family leakage 风险、是否需要完整未来序列、额外计算成本和不可部署条件；
- 结果只用于判断“生物先验是否可能改变边界分配”，不得进入 broad primary ranking；
- 若未来要升格为正式方法，必须建立 train-only、family-held-out、prefix-causal 的边界信号并重新预注册。

该扩展回答“除了 entropy 还有什么”，但不以牺牲核心论文可完成性为代价。

### 3.3 Track D：100M 部署/上下文容量增强赛道

Track D 属于 `ENHANCEMENT_SCOPE`，只在核心 100M validation gate 与资源 gate 通过后启动。它包含两个 Flat Transformer 系统，各使用三个正式训练 seed：

| ID | 表示 | 固定条件 | 主要回答 |
|---|---|---|---|
| D1 | NUC | 100M、4096 representation slots、共同累计 FLOP 预算 | 单碱基系统在固定槽位与计算预算下的部署表现 |
| D2 | BPE-1024 | 100M、4096 representation slots、共同累计 FLOP 预算 | BPE 在固定槽位下扩大 raw-nt context 后的完整系统收益与代价 |

Track D 的合同：

- D1/D2 使用相同 Flat backbone family、同一个 `D_long_context_train_view`、split、硬件 cohort、优化器候选和正式 seed `17/29/43`；`D_stream_seed=20260816` 只在 train entities 上生成一次冻结的 stratum-interleaved permutation：先按 length bin、source 和主要 RNA type 分层，各层内 seeded shuffle，再按冻结 round-robin 顺序交织；同一 homology cluster 的 entities 在有其他 cluster 可选时不得连续出现。D1/D2 共用该 entity order，不得按质量结果重排；
- 两臂共享冻结窗口策略。每个窗口固定最多 `2048 context slots + 2048 target-bearing slots`，不足部分只做 padding；首个 target interval 从 BOS 后第一个可评分 token 开始，之后本臂的下一个 target interval 必须紧接上一个已评分 raw interval 的末端；
- 两臂各自按本臂冻结 tokenizer 的 canonical token path 推进 raw-nt cursor，因此训练中不要求 D1/D2 在同一步预测完全相同的 raw positions。正式训练采用单次无放回 schedule：已评分 target 只能在后续窗口作为 loss-masked 左上下文重新载入，不能再次进入 loss 或 exposure；每个 eligible target nucleotide 的 primary repetition count 必须为 `1`，禁止第二个 pass。每个 checkpoint 保存 unique-target/entity/cluster coverage 与 target-repetition histogram，任何 `count>1` 都使 primary Track D exposure gate 失败；
- Track D 的 validation/final scorer 使用共同 raw-target-position manifest：validation manifest 由 Phase 3 在 `D_long_context_validation_view` 上按冻结的“逐 canonical entity、逐 raw nucleotide 评分一次”算法生成；final 只冻结同一纯确定性算法，到 Phase 8 unseal 后才对 final split 实例化。D1/D2 对完全相同的目标碱基各评分一次，只允许其可见左上下文因 tokenizer 和 4096-slot 容量不同；Phase 8 前不得读取或预生成可见的 final manifest；
- `4096 representation slots` 是模型输入槽位，不是 `4096 raw nt`；D2 的实际 raw-nt context 由冻结 tokenizer 在每个窗口中的编码长度决定，必须逐窗口、长度层和训练阶段报告。跨独立 RNA 的 packing 只计吞吐，不得计为单序列 context；
- 先定义 `D_NOMINAL_FLOP_BUDGET` 为“D1 在 frozen batch/model/config 下处理 `2.0B valid target nt` 所需的模型训练 FLOPs”。Phase 3 用冻结 tokenizer、entity order 和窗口策略完整模拟两臂的单次无放回 schedule，得到各臂在保留 `5%` 未消费 unique-target 容量时可达到的 `F_cap_arm`；最终 `D_FLOP_BUDGET=floor_common_checkpoint(min(D_NOMINAL_FLOP_BUDGET,F_cap_D1,F_cap_D2))`。该容量门只读取 train view 与 FLOP counter，不读取任何 validation 质量；formal run 禁止因耗尽数据临时开启第二个 pass；
- formal scheduler、warmup 和 checkpoints 按 `D_FLOP_BUDGET` 的 `25%/50%/75%/100%` 推进；必须报告最终预算相对 nominal budget 的比例、两臂预算所需/可用 unique target nt 和 5% capacity reserve；
- 两臂最终累计 FLOPs 必须达到同一预算；允许的最后一步离散 overshoot 不超过共同预算的 `0.5%`，并分别报告 within-sequence visible raw context、valid-target-nt exposure、每 batch raw-nt throughput、每 batch 独立 sequence 数、wall time、吞吐、显存和各训练阶段分布；这些量不得合并成一个“raw context”；
- `effective_raw_context_nt` 按被评分 raw target nucleotide 定义：在产生包含该 nucleotide 的 canonical token 条件分布时，模型因果上实际可见的、位于同一 RNA 内且去重的前序 raw nucleotides 数；不计 padding、跨 RNA packing、当前 multi-nucleotide token 内部碱基、当前或未来 targets。同一 multi-nucleotide target token 内的各碱基共享 token 产生前的 context 值；
- Phase 3 context-separation gate 在至少 `100` 个 train-only homology clusters 的同一组 frozen raw-target probes 上，以 raw target nucleotide 等权形成 `effective_raw_context_nt` 分布后取 median；D2 median 必须比 D1 高至少 `15%`，并报告 target-level、sequence-level 分布及 effective context `>4096 nt` 的 target 比例。若未通过，Track D 仍可报告固定槽位效率，但删除“更长单序列上下文收益”和对应 Decision Map 推荐；
- 两臂另报实际接触的 unique entity/cluster 数、length/source/RNA type/family、GC 与 low-complexity 构成及 valid target nt；若任一预注册连续暴露变量的绝对 SMD `>0.1` 或离散构成差超过 `2` 个百分点，则保留 `E_D_DEPLOYMENT`，但上下文解释降级为“该冻结数据流下的完整系统差异”，不得单独归因于更长上下文；
- Track D 不复用 Track R 的中间 checkpoint 冒充 compute-matched training；D1/D2 是单独从头训练的六个科学 run，调参计算另列；
- D2−D1 同时改变 tokenizer、可见 raw context 与系统执行路径，只允许称 `deployment/system effect`；它与 Track R 的 F2−F1 共同用于区分“matched raw context 的表示效应”和“固定槽位下更长上下文的系统收益”，不能被解释为边界的纯生物意义。

### 3.4 350M 条件性规模复验

350M 属于 `ENHANCEMENT_SCOPE`，不是默认扩张。只有核心数据/evaluator/100M validation、资源和论文价值 gate 通过后才固定启动四臂：

- C1：NUC flat；
- C2：BPE-1024 flat；
- C3：fixed-patch BLT；
- C4：entropy-patch BLT。

每臂固定 seed `17、29、43`，共 12 个科学运行。

规模复验只允许改变模型规模和训练预算：C2 必须复用 F2/D2 的同一个冻结 BPE-1024 artifact；C3 继续使用 fixed-6；C4 必须复用 P3 的同一个 RNA entropy estimator、threshold、最大 patch 长度和 boundary implementation。不得为 350M 重新训练 tokenizer/patcher或重新校准 threshold，否则结果只能视为新表示配置，不能称规模趋势复验。

350M 不包含 random patch，因此：

- 350M 可以复验 NUC vs BPE 和 entropy vs fixed 的规模趋势；
- “entropy 边界位置相对随机边界的因果价值”仍由 100M P2/P3 证据承担；
- 不得依据 100M test 或生成结果事后替换四个 350M 臂；
- 350M 启动条件是 100M 数据、执行和 evaluator gate 全部通过，而不是要求 entropy 在 100M 获胜。

### 3.5 训练预算、公平性与 scope gate

- `CORE_PAPER_SCOPE` 固定为 33 个 formal runs（Track R 30 + B1 3）。
- `FULL_ENHANCEMENT_SCOPE` 在 Track D 与 350M 均启动时为 51 个 formal runs。
- Track L 若升格，必须以 amendment 增加 run 数，不能悄悄计入原 51。
- 在任何 final test 解封前，由 validation、资源和 prior-art gate 冻结本次投稿 scope；未启动增强模块不允许在 unseal 后补跑并复用同一 test 作为 confirmatory。

- 100M 目标参数量：`98M–102M`；
- 350M 目标参数量：`343M–357M`；
- 同 track matched pair 的 backbone non-embedding 参数完全一致；
- total trainable params 差异不超过 2%，词表/embedding 参数单独列出；
- 100M 主训练预算：每个 run `2.0B cumulative_valid_target_nt`；
- B1 使用与 Track R 相同的 `2.0B cumulative_valid_target_nt`；Track D 不以固定 valid-nt 作为停止点，而以预冻结的共同 `D_FLOP_BUDGET` 作为停止点；
- 350M 主训练预算：每个 run `7.0B cumulative_valid_target_nt`；
- Track R、B1 和 350M 的 primary context：`4096 canonical nucleotides`；Track D 另按 `4096 representation slots`；
- Track R、B1 和 350M 的 scheduler、warmup、checkpoint 和停止点以有效 target nucleotide 计数，而不是 optimizer step 或 tokenizer token 数；Track D 按累计 FLOP 比例推进；
- 每个受控 comparison family/track 内的 matched arms 使用同一冻结原始 sequence order；不同数据视图之间不伪称逐样本相同，Track D 另遵守 3.3 的 long-context entity order 与 target schedule；
- padding、BOS、separator 和 overlap 重复窗口不增加 nucleotide exposure；
- 主科学 track 为 data-matched；
- Track R/Bridge/350M 的 exposure-matched 训练轨迹另外按各配对臂共同可达的累计 FLOP 分位点报告 compute-performance 曲线；FLOP 计数项、检查点网格和插值规则必须在 formal run 前冻结，不把该曲线冒充单独 compute-matched training；
- Track D 才是独立 compute-matched training：两臂共享固定 `D_FLOP_BUDGET`，scheduler 按 FLOP 进度推进；
- `D_FLOP_BUDGET` 与 common-FLOP curves 使用模型训练 forward/backward FLOPs；BPE/Unigram/entropy estimator 的一次性构建成本另列，并同时给“含构建成本”的 amortization sensitivity；在线 entropy/local encoder 若属于模型图，其 FLOPs 必须计入训练与推理模型 FLOPs。BPE/Unigram encode、canonicalization、packing 与 detokenization 等非模型在线操作不强塞进模型 FLOPs，但必须进入 3.9 冻结的端到端部署延迟，任何 tokenizer/patcher 都不得被当作免费；
- 同一硬件 cohort 才能比较 wall-clock、吞吐和显存；
- H100/H800/不同 A100 cohort 不能直接混排速度；
- 参照 BiomBenchmark 的场景化效率报告，所有正式 arm 另报 `time-to-quality`：到达冻结 BPN/continuation 阈值或共同 FLOP checkpoint 所需的 wall time、能耗（若可测）、吞吐和峰值显存；
- 模型权重、tokenizer/patcher artifact、在线缓存和必要 side information 的存储大小单列。

超参数规则：

- 每臂允许完全相同预算的 train/validation-only 调优；
- 固定候选为基础学习率的 `0.5×、1×、2×`，其余 optimizer 配置保持相同；
- 100M 基础学习率初值 `3e-4`，350M 初值 `2e-4`；
- AdamW `β=(0.9,0.95)`、weight decay `0.1`、bf16；
- 每个候选使用独立 `tuning_seed=101`；Track R/Bridge/350M 候选最多 `100M valid target nt`；Track D 候选使用“D1 处理 `100M valid target nt` 所需模型 FLOPs”的同一确定性 pilot 上限；
- 按 validation 指标选择后冻结；
- tuning checkpoint 不得成为正式 checkpoint，也不得续训；正式 seed `17/29/43` 必须以冻结配置从头训练；
- primary 使用固定最终预算 checkpoint；best-validation checkpoint 只作为 secondary sensitivity，二者不得混在 headline 或 Decision Map/可选 selector 中；
- 不允许给失败臂额外调参预算；
- 不允许使用 primary/family/clan final test、database-release-shift sensitivity 或生成主表选择超参数。

### 3.6 统一跨 tokenizer 码长口径与校准基线

不得把不同含义的 token perplexity 直接横向排列。`canonical_code_length_BPN` 测量的是冻结的 model–tokenizer–patcher–codec **完整系统**在 deterministic canonical representation 下的实际码长；它不是对所有可能分段路径边缘化后的 RNA string likelihood，也不自动包含模型、词表、manifest 或一次性构建成本。

建立三个语义显式分开的指标：

1. `canonical_code_length_BPN`（唯一跨 tokenizer headline）

   使用冻结的 deterministic canonical encoder 将每条 RNA 映射到唯一 canonical token/patch path，再使用同一个冻结的整数 arithmetic/range coder、概率量化规则和模型条件概率产生真实可解码 bitstream。每个 `model seed × split` 按冻结序列顺序写成一个连续 coder stream；模型上下文在序列边界重置，序列长度由 split manifest 作为共同 side information 提供且不计入码长，EOS 另报。数据集总 coded bits 除以 canonical nucleotide 总数。该指标是“冻结 canonical codec 下的实际码长”，不是对所有可解码 tokenization 路径边缘化后的字符串 likelihood。

2. `canonical_code_nll_BPN`（码长一致性诊断）

   同一 canonical path 的理想累计 `−log2 p(token|prefix)` 除以 canonical nucleotide 数。它与 quantized-CDF NLL、真实 coder bit count 的一致性按下述固定公式验收，但不替代 headline 实际码长。

3. `next_base_BPN`（兼容模型子集 secondary）

   \[
   -\frac{\sum_i\log_2 p(x_i\mid x_{<i})}{N_{\mathrm{canonical\ nt}}}
   \]

   只用于能够给出精确逐碱基条件分布的模型。每个真实 nucleotide 必须恰好计分一次；不得与仅有 canonical-path probability 的 BPE/Unigram 混成全模型总榜。

具体规则：

- headline 的完整名称为 “actual canonical codec length conditional on the frozen split manifest and model/tokenizer/patcher artifacts”；不得称为包含数据 manifest、模型或词表开销的无条件文件压缩率；
- integer CDF 固定总频数为 `2^24`：每个 vocabulary symbol 先分配 minimum frequency 1，其余频数按模型概率的 largest-remainder rule 分配，余数并列按 token ID 决定；使用同一个冻结的 64-bit range-coder implementation、初始化、renormalization 与 final flush；
- common side information 只包括 split manifest 中的序列顺序/长度、冻结模型/tokenizer/patcher identity 和 coder specification；P2 另外固定 `patch_randomization_seed` 与 sequence ordinal，P3 另外固定 entropy estimator/threshold/state。所有 side information 必须列出，不能隐藏为人工步骤；
- 同时累计基于 quantized integer CDF 的 `quantized_cdf_nll_bits_sum`。每个连续 stream 必须满足 `|coded_bits−quantized_cdf_nll_bits_sum|≤64 bits`；`canonical_code_nll_BPN` 与 quantized-CDF NLL 的绝对差必须不超过 `1e-4 bits/nt`，否则 codec evaluator `FAIL_CLOSED`；
- 每个 deterministic arm 必须执行独立 decoder：仅凭 bitstream、冻结模型和已列 side information 恢复 canonical RNA；validation fixture 与 Phase 8 最终 stream 均须逐字节等于源序列。只验证 bit count 接近 NLL、而不能解码，不得称 actual codec；
- 固定 BOS policy；BOS、padding 和 packing separator 不计分，EOS 单独报告 `EOS_NLL`，均不混入 headline 分子/分母；
- dataset `canonical_code_length_BPN` 使用总 coded bits/总 nucleotide；NLL 类 BPN 使用总 NLL/总 nucleotide；均不对 sequence BPN 做无权平均；
- BPE/Unigram 如果没有实现精确 prefix marginalization，只进入 `canonical_code_length_BPN` 与 `canonical_code_nll_BPN`，不能把结果标为 `next_base_BPN`；
- overlapping k-mer 的 canonical path 使用 full vocabulary coding；首个 k-mer 负责前 `k` 个碱基，之后每个 stride-1 token 只归属一个新碱基，确保 target attribution 不重复；
- non-overlapping k-mer 必须为序列尾部 `1…k−1` 个碱基定义冻结、无损的 canonical tail-token 规则；禁止丢弃、padding 后计分或重复计分尾部；
- overlap generation 只允许四个与已有后缀一致的合法转移，并同时报告被模型分配给非法转移的概率质量；
- 对四个合法后继重新归一化得到的逐碱基诊断单独标为 `constrained_next_base_BPN`；full-vocabulary canonical code length仍是 headline，不用 constrained score 替代；
- 所有 deterministic arms 必须通过上述 bitstream decode 与 `64 bits / 1e-4 bits-per-nt` 两级一致性门；任一 fixture 失败时整个 headline evaluator `FAIL_CLOSED`；
- token PPL 仅在相同 tokenizer 内作训练诊断，不进入跨 tokenizer 排名。


校准基线必须进入主结果或紧邻主结果的表格：

- uniform A/C/G/U：理论参考 `2 bits/nt`，并单列 EOS；
- order-0/1/2/3 Markov 无损模型：只在 train split 拟合，validation 冻结阶数/平滑；
- CTW 或 PPM 类传统上下文压缩器：冻结实现、窗口和字母表；
- 可选轻量 nucleotide autoregressive baseline，只在参数/FLOP/训练数据可闭合时加入。

这些基线用于解释“1% BPN 差异的实际规模”，不参与 tokenizer 因果归因。另报模型权重、词表/patcher artifact 与一次性构建成本；headline 仍是条件于冻结 side information 的数据流码长。

Track R/B1/350M 的 length-OOD primary 评分固定为 rolling `4096 raw nt` context：对位置 `i`，所有模型只观察相同的最近 `min(4096, i−1)` 个真实 raw nucleotides，每个目标碱基恰好计分一次；窗口重叠不得重复进入分母。该结果称“长序列上的局部窗口 OOD”，不得据此声称模型学习了超过 4096 nt 的长程依赖。Track D 可运行同一 raw-context-controlled sensitivity，但因其 long-context train view，只能标 `held_out_long_context`；其 4096-slot 部署上下文另表报告，不混入 length-OOD claim。

### 3.7 True-suffix continuation、无条件生成与生物控制

冻结的 required main-text continuation endpoint：

- 从 sealed test 的真实序列构建 `10% / 25% / 50%` prefix；
- prefix 以 raw nucleotide 定义，而不是 token 数；共同切点固定为 `6 × floor((ratio × sequence_length)/6)`，仅保留至少有 1 nt suffix 的样本，并逐样本报告实际 prefix 长度和实际比例；
- tokenizer 对观测 prefix 独立编码，禁止 BPE token 跨过观测与隐藏后缀边界；
- non-overlapping k-mer、NUC、BPE 和 BLT 必须使用完全相同的共同 raw-prefix 切点，任何模型不得静默多看或少看碱基；
- 给所有模型相同 target raw length；
- 对超过窗口的 prefix，所有模型只观察相同最近 `4096 raw nt`；该条件是 local-window continuation，不称长程 continuation；
- 若最后一个 token 超出目标长度，保留原始完整输出，同时另存固定长度评估视图，并记录 truncation；
- continuation 的主指标是真实 suffix 的 `canonical_code_length_BPN`；它是 required main-text secondary，与 headline codec 共同构成论文的两个质量面板。suffix edit distance、nucleotide accuracy、k-mer recovery、Rfam family/clan recoverability、CM bit score、结构代理偏差和训练最近邻均为 diagnostic，不把单一真实 suffix 称为唯一合理生成答案。

Decoder 协议：

- validation-only grid：temperature `{0.7,0.9,1.1}` × top-p `{0.90,0.95,1.00}`，top-k 固定为 0，另加 greedy；
- validation grid 的 query/prompt manifest 固定由 `validation_query_seed=1201` 生成；Phase 8 final query/prompt manifest 的生成算法与 `final_query_seed=1202` 在 unseal 前冻结。每个 decoder cell 使用完全相同的 query manifest；
- 主表同时报告三个 grid 内预注册共同点：headline `balanced=(0.9,0.95)`，sensitivity `conservative=(0.7,0.95)` 与 `exploratory=(1.1,1.0)`；
- 可以为每个模型在 validation 上选择一个 Pareto 点，但必须给相同搜索预算，并在 test 前冻结；
- final test 只执行冻结设置；
- 每个 `training seed × decoder condition` 固定执行恰好 `10,000 attempted generations`；generation seed 列表固定为 `[1101,1102,1103,1104,1105]`，每个 seed 恰好 `2,000 attempts`。若 model-selected Pareto decoder 与任一预注册共同 decoder tuple 完全相同，必须复用同一个 attempt bundle，不得重复采样后择优；禁止为凑足 valid 输出而追加主分析样本；
- 所有 raw、decoded、valid、invalid、early-EOS、truncated 和 overshoot 输出及其分母全部保留；validity 以 10,000 attempts 为分母；
- valid-only 的 family/结构/分布指标只作为条件分析，必须同时报告 valid 数、selection rate、attempted 总数和每个 accepted output 的成本；如额外补采样，只能进入明确标记的 secondary precision analysis；
- 训练 seed 是独立模型重复；generation seed 只量化同一模型内的 Monte Carlo uncertainty，10,000 个输出不能冒充 10,000 个独立模型重复。

生成指标：

- exact uniqueness；
- exact train/val/test memorization；
- 100%、90%、80% identity novelty，同时报告 alignment coverage；
- cluster-level uniqueness；
- nearest-training-neighbor identity/coverage；
- Rfam 15.1 `cmscan --cut_ga` family/clan hit、coverage、bit score；
- family/clan coverage；
- 长度、GC、dinucleotide、k-mer 分布；
- ViennaRNA MFE、ensemble diversity、paired fraction；
- 有效率、非法字符率、EOS 完整率、truncation/overshoot；
- memorization–validity Pareto；
- 95% homology-cluster bootstrap CI。

MFE、CM hit、预测 pairing 和 embedding score只能称为 computational proxy，不能写成真实功能、湿实验验证或天然 RNA 证明。所有 structure/distribution/family 结果还必须执行以下控制：

- 在 `length × GC × RNA type` 匹配或分层后重新比较；
- 联合报告 `CM hit/bit score × nearest-train identity/coverage`，避免把近记忆误写成 family fidelity；
- 报告 family-balanced macro、去除 rRNA/tRNA 后的 sensitivity，以及 rare-family slice；
- 低复杂度、homopolymer、dinucleotide bias 和 source database 分层；
- RNAGym mutation likelihood/fitness 作为实验数据锚定的 diagnostic；
- 任何 proxy 改善若伴随 memorization harm，不能进入无条件推荐。

### 3.8 外部 benchmark、横向模型与可扩展框架

内部结果固定为 `Main Table 1 Flat`、`Main Table 2 BLT`、`Main Figure 1 System Pareto` 和简化 Decision Map；外部结果分表，禁止混成单一 SOTA 排名。

#### External Table 1 — 公共自回归/生成模型参考

按 best-effort 优先冻结和运行：

- GenerRNA `model_updated.pt`；
- GenerRNA 历史 `model.pt`；
- GARNET 公开 GPT checkpoint；
- EVA 145M 与 437M，在权重、许可和代码可运行时作为外部 single-nucleotide 参考；
- learned dynamic systems（H-Net/DNAChunker/MergeDNA）只有在公开权重、任务语义和无损自回归 adapter 可闭合时才列入；否则只进入 prior-art/profile table。

公共模型的数据、架构、预算不匹配时只称 `ecological/reference-only comparison`，不用于 tokenizer 或 patch-rule 归因。GenerRNA 使用 release-22-derived 语料；在 release-22 primary test 上默认标记 `PRETRAIN_OVERLAP_EXPECTED_REFERENCE_ONLY`，更公平的外部参考优先使用经过 sequence-similarity 过滤的 release-27-shift subset。

#### External Table 2 — family/structure-conditioned 参考

- RfamGen；
- RNAgg；
- 仅在相同 Rfam family 子集上比较；
- 不与 broad unconditional generation 混排。

#### Secondary diagnostic resources

- GARNET 16S/23S 与 231-family continuation；
- RfamSample family recoverability；
- ArchiveII-Nr 结构分层；
- RNAGym mutation likelihood/fitness；
- TS-Hard 仅在能够定义兼容、冻结的结构 probe 时使用；
- BEACON 与 BiomBenchmark 用作 prior-art、任务/场景设计和可选表示诊断，不把其全部下游任务扩入核心项目。

#### BiomBenchmark 风格的可扩展工程交付

每个内部/外部模型建立统一 `MethodProfileCard`：

- model ID/revision、paper/preprint status；
- code commit、checkpoint SHA-256、license；
- pretraining data、目标、可能 overlap；
- architecture、parameter count、context limit；
- tokenizer/patcher 类型、词表/平均 patch 长度、是否输入自适应；
- training exposure/FLOPs、推理 latency/显存；
- decoder、evaluator adapter、comparability status。

代码结构至少包含：`dataset registry / model & tokenizer registry / RNAARAdapter / evaluator / result parser / frozen analysis notebook`。社区新增方法必须通过 adapter fixtures，而不是复制一套任务脚本。所有结果表由结构化 `ScoreSums` 和 registry 自动生成，禁止手工从日志挑数。

任何外部模型不可获得或不可运行，都不得阻断内部核心论文；只删除或降低 external-comparison/SOTA claim。

### 3.9 统计分析、Decision Map 与论文主张

#### 3.9.1 统计单位与核心 contrasts

统计采用估计优先，不把三个 training seeds 包装成高功效显著性检验。

| Estimand | 正式 contrasts | 共同条件 | 允许解释 |
|---|---|---|---|
| `E_R_STATIC_REPRESENTATION` | F2–F7 分别减 F1 | 同 Flat backbone、4096 raw nt、2B valid target nt、相同数据顺序与 evaluator | 该 Flat 实现内静态表示/tokenization 的完整效应 |
| `E_P_CONTROLLED_PATCH_RULE` | P3−P1、P3−P2、P2−P1 | 同 BLT backbone、mean patch length=6；P2/P3 通过 patch/FLOP/positivity gate | adaptive rule 的完整受控效应；P3−P2 只在 supported strata 描述 randomized boundary-rule intervention |
| `E_S_SYSTEM_DECOMPOSITION` | `B1−F1`、`P1−F7`、`P1−B1`、`F7−F1` 与预注册差中之差 | 固定 2×2 与共同数据/预算记录 | 层级、粒度和完整系统响应；不是纯 tokenizer/架构因果效应 |
| `E_D_DEPLOYMENT` | D2−D1 | 固定 4096 representation slots、共同 FLOP budget、共同 raw targets | 完整部署系统差异 |

每个 headline estimand 固定使用 final-budget checkpoint、primary cluster-held-out final test、dataset-level `canonical_code_length_BPN`、canonical entities 等权、paired model seeds `17/29/43`。true-suffix continuation 使用同一模型/seed 和共同 raw-prefix manifest，作为 required main-text secondary。

对每个 training seed，先在完全相同的测试 homology clusters 上计算 paired effect，并在该 seed 内做 paired cluster bootstrap 95% CI；family-macro 以 family 为顶层重采样单位；同一原序列的多个 prefix 成组重采样。三 seed 逐个展示效应、CI、均值、范围和方向。cluster bootstrap 不冒充跨训练随机性的显著性。

BPN 相对效应统一为 `δ=100×(BPN_candidate−BPN_reference)/BPN_reference`，负值更好。若三个 seed 的 paired-effect CI 均落入经 justification 的 practical-equivalence 区间且无 harm violation，才可写“在该预算、三个预注册 seeds 和两个受控 backbone 实现下未见实质差异”；若任一必要 CI 同时覆盖实质改善与实质恶化，状态为 `INCONCLUSIVE_UNDERPOWERED`。

#### 3.9.2 主文 Decision Map：简单、可解释、允许弃权

主文结果立方体固定为：

```text
comparison_family × candidate × task × length × compute
```

comparison families 不混排：

- `R_FLAT={F1…F7}`；
- `R_BLT={P1,P2,P3}`；
- `R_SYSTEM={F1…F7,P1,P2,P3,B1}`，只推荐完整系统；
- `D_DEPLOYMENT={D1,D2}`；
- `L_LEARNED_DYNAMIC` 只有在 amendment 后 formalized 才单列，不进入以上池。

每个 cell 的主文只做四步：

1. support gate：至少 100 homology clusters；family 相关至少 20 eligible families；
2. 报告 loss/quality、端到端 cost 和 harm constraints 的效应量与 CI；
3. 展示 validation 前瞻冻结的 Pareto non-dominated set；
4. 输出 `WINNER / TRADE_OFF_SET / NO_RESOLVED_WINNER / INSUFFICIENT_SUPPORT`。

不确定边、harm margin 跨界、候选间实际等价或非传递结果均必须弃权，不能强制给冠军。主文的推荐必须附：适用条件、效应区间、成本、memory/RNA-proxy harm、跨 seed 稳定性和失败条件。

#### 3.9.3 完整 selector 仅为补充材料 secondary

如实现 pairwise dominance graph、regret、Pareto coverage、harm violation、abstention 和 distribution floors：

- 算法、阈值、reference、applicability mask、成本测量脚本与 fixture 在 validation 上冻结；
- final test 只进行一次整体 out-of-sample 评估；
- selector-level mean/worst regret、coverage、harm 和 abstention 属于 secondary；
- 单个 cell winner 始终是 multiplicity-rich secondary；
- 无条件生成 decision map 始终 exploratory；
- selector 失败不否定核心 benchmark，只把论文降级为受控 benchmark/resource + failure map。

#### 3.9.4 Practical margins 的 justification gate

在 final unseal 前为每个 margin 建立 `margin_justification.md`：

- BPN 1%：基于 validation coder重复误差、seed/cluster 变异和相对 uniform/Markov/CTW 基线的实际规模；
- latency 15%：基于同硬件至少 30 次 paired repeats 和部署可感知差异；
- validity/family 2 pp、memorization 1 pp、distribution distance 5%：基于 validation 噪声、已知 evaluator 精度或独立科学依据。

无法给出依据的 margin 仅作 sensitivity，不得产生强 dominance/winner claim。

#### 3.9.5 允许的核心 claim 模板

只有 updated collision matrix、核心数据/evaluator、33 个 core runs、final unseal 和统计/生物控制全部通过后，才允许类似：

> TokBench-RNA provides a homology-aware and compute-accounted evaluation of static tokenization and controlled dynamic segmentation in autoregressive ncRNA models. Within the evaluated Flat Transformer and BLT systems, it quantifies canonical modeling, true-suffix continuation, deployment cost, memorization and RNA-proxy trade-offs, and identifies where method rankings change or remain unresolved across task, length and budget.

禁止：

- “发现普适最佳 RNA tokenizer”；
- “证明 entropy 边界具有生物学意义”；
- “因果证明动态分词优于静态分词”；
- “生成了具有真实功能的 RNA”；
- 从不同公开模型的生态比较归因 tokenizer；
- 用单个平均分或单 seed 宣称 SOTA。

#### 3.9.6 发表潜力与 scope

| 最终证据 | 允许定位 |
|---|---|
| 只比较公开 checkpoints | 技术报告/reference table；发表潜力低 |
| 同一主干比较 NUC/k-mer/BPE | 有实证价值，但与既有工作重叠较大 |
| `CORE_PAPER_SCOPE`：同源/family 隔离、33 个核心 runs、合法 actual codec、true-suffix continuation、共同预算、记忆与 RNA proxy controls、开放 evaluator/manifest | 可形成完整领域 benchmark/resource 论文 |
| 再完成 Track D、350M、release-27 shift、外部 adapter 与 time-to-quality | 增强论文竞争力并支持更完整部署结论 |
| 再形成跨 seed/family/length/budget 稳定规律，或可迁移的系统性失效边界 | 才具备超越“严谨排行榜”的更高层科学结论 |

## 4. 分阶段执行 TODO 与门控

| Phase | 阶段目标与主要任务 | 主要输出 | 验收门 | 失败/降级处理 |
|---|---|---|---|---|
| Phase 0：Goal、authority、prior-art | fresh clone；Goal materialization；旧谱系 ledger；更新 collision matrix；冻结 scope enums | Goal、authority manifest、legacy ledger、collision matrix v1 | broad novelty 正确标 FAIL；窄贡献字段闭合；无 final-test access | authority/来源不闭合则证据化阻塞 |
| Phase 1：数据 benchmark | release 22 主数据；release 23–27 shift；canonicalization；exact/homology/family/clan split；release-27 构成审计 | immutable data、manifests、datasheet、leakage report | 来源/license/accession/hash 闭合；exact=0；80/80 cross-search=0；final sealed | 失败则禁止正式训练 |
| Phase 2：统一 evaluator | actual range coder、Markov/CTW baselines、continuation、generation、bio controls、adapter/schema、sealed gate | evaluator package、fixtures、protocol YAML、empty tables | round-trip、每 nt 一次计分、prefix causality、固定分母、baseline reproducibility 全 PASS | headline 语义未闭合则锁定训练 |
| Phase 3：模型/HPO/成本校准 | Flat/BLT；P1–P3 balance；B1；Track D preflight；Track L pilot；端到端 latency；margin justification | resolved configs、parameter/FLOP census、tokenizers、P2/P3 plan、L1 pilot report、cost protocol | formal seeds 未启动；final access=0；P2/P3 gate 可执行；margin 有独立依据或降为 sensitivity | L1 失败则 reference-only；不阻断核心 |
| Phase 4：核心 100M train/validation | Track R 10×3 + B1×3；只用 train/validation；required main-text continuation validation | 33 core bundles、validation summaries、失败报告 | `33/33` 或负责人前瞻 scope amendment；无换 seed；final access=0 | 核心 arm 失败则先修复或降级项目，不允许用增强赛道补位 |
| Phase 4-G：Publication Scope Gate | 只依据 validation、资源、prior-art 与工程完整性决定 `CORE_ONLY` 或是否启动增强 | scope decision、100M evidence review、frozen go/no-go | scope 在 final unseal 前签字；不得依据 final test 决定扩容 | 100M 区间过宽可先增加训练精度/修复，不盲目上 350M |
| Phase 5：增强 Track D / external / L1 formal（可选） | D1/D2×3；external registry；如 amendment 批准则 L1×3 | enhancement bundles、profile cards、adapter tests | 各模块自己的预算/causality/comparability gate；final access=0 | 缺失只删除对应 claim |
| Phase 6：350M（可选） | 预先固定四臂×3 seeds；只使用 train/validation | 12 scale bundles | 资源、100M validation 和代码/evaluator gate 通过；四臂不随赢家变化 | 资源不足则保留 100M-only paper |
| Phase 7：分析与 FINAL_UNSEAL_LOCK | 冻结本次 scope 内 checkpoint、decoder、query manifests、Decision Map、统计代码、图表模板、claim matrix、collision matrix final | `FINAL_UNSEAL_LOCK`、预填空表/空图 | 所有 artifact 可重放；margin justification 完成；final access=0 | 任一核心项未冻结不得 unseal |
| Phase 8：一次性 final unseal、复现、投稿 | 对本次冻结 scope 内全部内部模型和可运行外部模型同一次评分；clean replay；release | final bundles、主表、Pareto、Decision Map/failure map、paper draft | unseal 后无调参/换模型/改 metric；主结果可重放 | 改变输入/model call/metric semantics 的修复使原 confirmatory test 作废 |

### Phase 0 立即执行顺序

1. `P0-A Fresh authority preflight`：只读核验 host/user/port、jobs、GPU UUID、磁盘、remote refs、目录冲突和旧 outputs；不杀进程、不选择旧副本为 authority。
2. `P0-B Repository bootstrap`：fresh clone、创建 `benchmark-v1`、加入 Goal/schema/license 清单；禁止导入数据、权重、checkpoint、环境或缓存。
3. `P0-C Goal materialization`：将本修订计划生成权威 Goal、SHA-256 和 exact-byte 本地镜像；两份正文逐字节一致。
4. `P0-D Legacy lineage regeneration`：以新 run ID 重建旧报告/ledger/registry，统一标记 `HISTORICAL_ONLY/ORPHAN_RESULT/MISSING`。
5. `P0-E Upstream/source/prior-art audit`：冻结 BLT、Flat、tokenizer、MMseqs2、Infernal、ViennaRNA、range coder、CTW/PPM 和 external baseline 版本/许可；更新 H-Net、DNAChunker、MergeDNA、PatchDNA 与 BiomBenchmark collision fields。
6. `P0-F First delivery closure`：Markdown/schema/hash/secret/large-file 检查；focused commit；push task branch；重读远端 SHA。

## 5. 验收、Final Goal 与终止条件

### 5.1 “只能前进不能后退”的正式定义

“前进”指证据、lineage 和决策状态单调增加，不代表每个 gate 必须 PASS。

合法前进状态包括：

- `PASS_CLOSED`；
- `FAIL_CLOSED_WITH_EVIDENCE`；
- `BLOCKED_EXTERNAL_WITH_EVIDENCE`；
- `BLOCKED_RESOURCE_WITH_EVIDENCE`；
- `TERMINATED_SAFELY_WITH_EVIDENCE`；
- `REDIRECT_PENDING_OWNER_APPROVAL`。

禁止：

- 删除或覆盖失败运行；
- 降低阈值；
- test-informed 调参；
- 修改 split 以消除负结果；
- 把 missing 写成 zero；
- 把 smoke、training loss、单 seed 或 CPU fallback 升格为论文结果；
- 上游 gate 未通过时启动下游；
- 用新 run 覆盖旧 run ID；
- 为了“继续前进”无限重试。

同一 root cause 只允许一次有明确差异的 corrected retry。再次出现同类问题时，必须关闭为 FAIL/BLOCKED 或提交负责人批准转向。

### 5.2 必须通过的测试

- canonical A/C/G/U round-trip 逐字节一致；
- BPE/Unigram 词表只由 train split 构建；
- suffix perturbation 不改变观测 prefix 的编码、prefix 内 patch boundaries、prefix positions 的 logits，以及给定完全相同 prefix 时的 next-step logits；
- k-mer 不跨 packed sample boundary；
- overlapping k-mer 每一步只增加一个 nucleotide；
- illegal overlap transition mass 可复算；
- 在固定 BOS policy 下，改变 masked padding 数量或布局不影响 nucleotide NLL；启用/禁用独立 EOS 报告不改变 nucleotide NLL；BOS 不计分但作为固定条件输入；
- shared scorer 与手工 oracle 在 `1e-7` 内一致；
- 所有 deterministic arms 的 `source RNA→canonical encoder→integer CDF→bitstream→independent decoder→canonical RNA` 逐字节一致；每个 stream 的 actual bits 与 quantized-CDF NLL 差不超过 `64 bits`，model-NLL 与 quantized-NLL 差不超过 `1e-4 bits/nt`；
- 每个真实 nucleotide 在 `canonical_code_length_BPN` 的 canonical target attribution 中恰好出现一次；dataset aggregation 使用总 coded bits/总 nucleotide；
- overlapping k-mer 首 token、每个新碱基、non-overlap tail 和 EOS 的计分规则均有正反 fixture；
- train/eval/generate 使用同一 tokenizer、feature builder 和 patcher；
- offline/realtime entropy boundary 一致；
- P2/P3 总体 patch-count 误差不超过 `0.5%`，各预注册长度层不超过 `2%`，实测 FLOPs 差异不超过 `5%`；
- P2/P3 的 prefix-length×entropy boundary rate、每序列 patch count、patch-length、cap 触发率、GC/low-complexity/source/family balance 与 `0.05≤q≤0.95` support coverage 均通过 3.2 的冻结门；non-supported strata 中 P2 必须逐位等于 P3 boundary；
- Phase 8 使用冻结 `q` 在 final test 上重算 supported-position 与 P3-boundary coverage；任一低于 `80%` 时自动取消 fine boundary-placement claim，禁止 final-test-informed refit，只保留 total adaptive-rule system comparison；
- clean BLT 中 hash/explicit-ngram 参数数量为零；
- 80/80 homology cluster 不跨 split；
- split 后 train→validation/test 直接 cross-search 的 80/80 命中为零；短 RNA 高敏感复核与已知 Rfam family/clan recall 达到 evaluation protocol 的冻结门；
- exact copy 的 novelty 为 0；
- near-homolog 被 80/90% evaluator 捕获；
- 本次冻结投稿 scope 内的每个 formal scientific arm 均有三个冻结 seed；
- 若 Track D 在本次冻结 scope 中启用，D1/D2 必须达到同一累计 `D_FLOP_BUDGET`（离散 overshoot 不超过 `0.5%`），并记录各自 raw-context、raw/valid-nt exposure 与 scheduler FLOP 进度；
- 若 Track D 启用，capacity preflight 必须能用冻结 tokenizer 与 schedule 复算 `D_NOMINAL_FLOP_BUDGET`、`F_cap_D1/F_cap_D2` 和最终共同预算，并证明两臂都保留至少 `5%` 未消费 unique-target 容量；formal run 不发生第二个 pass；
- 若 Track D 启用，`D_long_context_train_view` 只含 train clusters，`D_long_context_validation_view` 只含 validation clusters；D1/D2 使用相同 seeded stratum-interleaved entity order 和冻结的 `2048 context slots + 2048 target-bearing slots` 窗口策略。每臂 primary target repetition histogram 必须全部为 `1`，已评分 target 的重载必须 loss-masked；跨 RNA packing 不计为 context；
- 若 Track D 启用，validation/final raw-target-position manifest 对 D1/D2 必须完全相同，每个目标碱基各计分一次；人工 multi-nt-token fixture 必须证明 `effective_raw_context_nt` 排除当前 token 内部碱基、padding 和跨 RNA packing，并按共同 raw targets 等权聚合；D2 median 未比 D1 高至少 `15%` 时自动撤销“long-context benefit”主张；
- 若 Track D 启用，暴露报告必须能复算 unique entity/cluster、length/source/RNA type/family、GC/low-complexity 与 valid-target 构成；达到 3.3 的暴露偏移门时，自动把上下文解释降级为冻结数据流下的系统效应；
- 固定 `10,000 attempted generations` 后，能从 raw output 独立重算 attempted/decoded/valid 分母、validity、conditional fidelity 和每个 accepted output 的成本；
- validation/final query manifests 可由 seeds `1201/1202` 重建，generation seeds 严格为 `[1101,1102,1103,1104,1105]`；相同 decoder tuple 只对应一个 10,000-attempt bundle；
- prefix 10%/25%/50% 共同切点均向下取整到 6 的倍数，所有模型看到完全相同 raw prefix，并报告实际比例；
- rolling scorer 使每个目标碱基恰好计分一次，且所有模型在 raw-controlled view 中看到相同最近 `4096 raw nt`；该 view 对 Track R/B1/350M 标 length OOD，对 Track D 只标 held-out-long-context sensitivity；
- 每个科研开关都有正反 fixture，证明开关实际改变对应计算；
- 同一 run checkpoint replay 的 logits 和指标在冻结数值容差内一致；
- GPU smoke 无 OOM/NaN，且 `cpu_fallback_count=0`；
- 每个最终分数闭合到 Goal、代码、数据、split、tokenizer、config、checkpoint、decoder、output 和 evaluator hash；
- `FINAL_UNSEAL_LOCK` 前 final-test access log 必须为空；unseal 后 checkpoint、decoder、strata、统计代码、Decision Map/可选 selector、图表模板和 claim matrix 不得变化；
- post-unseal 重算 fixture 必须证明 `FINAL_UNSEAL_RECOMPUTATION` 只读取冻结 logits/outputs；任何改变模型调用、输入或 metric semantics 的修复不得继续使用同一 test 作为 confirmatory evidence；
- Decision Map fixture 必须能输出 `WINNER / TRADE_OFF_SET / NO_RESOLVED_WINNER / INSUFFICIENT_SUPPORT`；如实现完整 selector，另可复算 regret、Pareto coverage、harm violation 与 abstention；
- primary cost fixture 必须证明端到端延迟计入 canonicalization、在线 tokenizer/patcher、packing/transfer、model 与 detokenization；model-only FLOPs、energy 或 accelerator latency 不能替换 primary deployment cost；
- 若实现完整 selector supplement，pairwise/graph fixture 必须覆盖 reference 全支配获胜、resolved trade-off、点估计占优但 CI 不足、nontransitive cycle、harm uncertain、全体实际等价和全部候选不合格；只有逐 seed 区间规则满足时允许 `WINNER`；
- 若实现完整 selector supplement，分布 harm fixture 必须验证每个 metric 使用自己的冻结 `D_floor`，并覆盖 validation reference distances 全零时的 `τ_metric` fallback 与 `HARM_UNCERTAIN_ZERO_REFERENCE`；
- uniform、order-k Markov 与 CTW/PPM 基线在冻结 train/validation artifact 上可重放，uniform A/C/G/U 返回 2 bits/nt（不含 EOS）；
- structure/distribution proxy 能在 length×GC×RNA-type matched view、去除 rRNA/tRNA view、family-balanced view 中独立重算；
- CM hit 与 nearest-train identity/coverage 联合表、RNAGym diagnostic 和 memorization harm 可由保存的原子统计重算；
- `margin_justification.md` 在 final unseal 前存在且与 Decision Map YAML 的阈值一致；
- release-27 shift manifest 能按 source/RNA type/length 分解并证明相对 release-22 train exact/80-80 overlap 为零；
- 若 Track L 启用，必须通过 prefix perturbation、round-trip、参数/FLOP/latency、patch-budget 和独立 formal-seed fixtures；
- clean environment 能重放一个训练 smoke、一个推理和主结果表。

### 5.3 分层成功标准

#### `CORE_PAPER_PASS`

- release-22-derived 数据合法、可追溯、可重建；release 27 已纳入 shift audit；
- exact 和 80/80 homology leakage gate 通过；family/clan split 可重建；
- Track R 10 臂×3 seeds 与 B1×3 seeds，共 33 core runs 完整；
- actual canonical codec、true-suffix continuation、common-FLOP curves、端到端 latency、memorization 和 RNA proxy controls 全部通过 fixture；
- actual codec 明确标为完整 canonical system 码长，并提供 uniform/Markov/CTW 或 PPM 校准；
- family-balanced、non-rRNA/tRNA、length×GC×RNA-type matched、CM-hit×nearest-train-identity 和 RNAGym diagnostics 完成；
- 主文能生成 Main Table 1/2、System Pareto、简化 Decision Map 和 failure map；
- 所有主效应有逐 seed 值、效应量和 CI；实际等价与 underpowered 状态严格区分；
- manifest、evaluator、MethodProfileCard、adapter、result parser、data sheet、model card 和 clean replay 可公开；
- final test 只解封一次，解封后无调参、换 checkpoint、改 cell 或 metric semantics；
- 最终结论不依赖 entropy patching 获胜，也不依赖复杂 selector 成功。

满足以上条件即可形成完整 benchmark/resource 论文，不要求 Track D 或 350M。

#### `ENHANCED_BENCHMARK_PASS`

在 `CORE_PAPER_PASS` 基础上：

- Track D 6 runs 达到共同 FLOP budget、共同 raw targets、context separation/暴露构成/latency gate；
- 350M 12 runs 完成四臂规模趋势复验；
- release-27 shift、外部模型与 time-to-quality 结果完整；
- 若 L1 formalized，独立报告 learned dynamic system，不混入 P1–P3 controlled estimand。

#### `FULL_PROGRAM_PASS`

- 核心 33 + Track D 6 + 350M 12 = 51 个既定 formal runs 完整；
- 可选 Track L 只按独立 amendment 另计；
- Decision Map 的推荐/弃权在 final test 上具有可解释的 coverage、harm 和 stability；
- 形成跨 seed、family-held-out、length OOD/held-out long context 和计算预算稳定的规律，或明确证明某类方法存在可迁移失效边界。

### 5.4 项目转向、降级或终止条件

- release 22 来源、license、accession 或 hash 无法闭合：停止该数据锚点，等待负责人批准以 release 27 为新训练锚点的独立合同；不得静默替换。
- shared scorer、actual codec 或每 nt 一次计分无法通过 oracle：停止正式模型矩阵。
- exact/homology/family split 无法建立：停止训练，优先转为数据 benchmark 恢复。
- 训练、验证和生成路径无法保持相同 tokenizer/patch 语义：停止对应 arm。
- 33 个 core formal runs 无法闭合：停止高水平 benchmark 主张，先修复或以前瞻 amendment 缩小方法集合；Track D/350M 不能补位。
- Track D、350M 或外部模型因资源/许可不可运行：证据化标记并删除相应主张；不自动否定 `CORE_PAPER_PASS`。
- Track L pilot 不满足 prefix causality、round-trip、参数/FLOP/latency 或稳定性：固定为 `REFERENCE_ONLY/NOT_PROMOTED`，不得无限重试。
- 如果 tokenizer 效应只来自参数量、raw context、训练数据、FLOPs 或 decoder search，不归因于 tokenizer。
- 如果所有 headline paired-effect CI 均完全落入有依据的实际等价区间，且 evaluator/harm constraints 合格，可报告限定范围内实际等价；不得泛化到所有 backbone/预算。
- 如果 CI 同时覆盖实质改善与实质恶化，标记 `INCONCLUSIVE_UNDERPOWERED`，不得把未显著写成负结果。
- 如果 true-suffix continuation 与 codec 排名不一致，必须作为主要结果解释，不能只保留对己方有利的指标。
- 如果 structure/CM 改善在 length×GC×RNA-type 匹配后消失，或只由近训练同源序列驱动，撤回 biological-fidelity claim。
- 如果 Decision Map 在 final test 上 regret/harm/abstention 不可迁移，论文降级为受控 benchmark/resource + failure map；核心结果仍可成立。
- 如果投稿前出现等价且更完整工作，转向 homology-aware evaluator、release-22/27 data resource、机制复核或负结果论文，不继续使用首创措辞。
- 任何 post-unseal 修复若改变 tokenizer、mask、model call、decoder、scorer input 或 metric semantics，原 confirmatory test 作废；必须使用新的未暴露 confirmation split 或降级为 post-unseal exploratory analysis。

## 附录 A. Authority、机器接口与状态命名

### A.1 Goal 与代码 authority

Phase 0 正式创建：

- 权威文档：`/home/cunyuliu/tokenizer-benchmark/docs/goals/NCRNA_TOKENIZATION_SEGMENTATION_BENCHMARK_GOAL_V1.md`；
- 本地用户可见镜像：`/Users/liucunyu/Documents/Codex/2026-08-08/volumes-orico-disc-blt-blt-code-2/outputs/NCRNA_TOKENIZATION_SEGMENTATION_BENCHMARK_GOAL_V1.md`；
- 根目录 `GOAL.md` 只指向权威版本，不维护可独立漂移的第二份正文；
- 代码仓库目标：`git@github.com:Cunyu-Liu/tokenizer-benchmark.git`，首个工作分支 `benchmark-v1`；Phase 0 必须重新只读核验 remote refs、服务器目录、GPU 和磁盘，旧快照不构成当前资源授权；
- 新代码从冻结的官方 BLT、公开 Flat Transformer 和 tokenizer 上游逐模块引入并记录来源；`/Volumes/orico-disc/blt` 旧代码、checkpoint 与结果保持只读且仅作历史证据，不整目录复制为新项目起点。

Goal/contract 必须包含 scientific estimands、data/split、CORE/ENHANCEMENT/EXPLORATORY scope、Track R/Bridge/L/D/350M、指标、统计、Decision Map/可选 selector、sealed lifecycle、claim matrix、阶段 gate、失败分支和 amendment log。任何科学关键字段不得由临时命令行覆盖。

### A.2 权威机器接口

- `benchmark_contract.yaml`：Goal/代码/数据/split/tokenizer 版本，scope tier、arm、track、model seed、tuning seed、patch randomization seed、可选 learned-router seed、参数、raw/slot context、valid/raw-nt 与 FLOP 预算、checkpoint policy、decoder、scoring、sealed phase和输出根。
- `dataset_manifest.json` 与 sequence-level Parquet：accession、raw/canonical sequence identity、release、RNA type、taxonomy、Rfam family/clan、homology cluster、canonical entity、multiplicity metadata、排除原因和 split membership。
- `tokenizer_spec.yaml`：tokenizer 类型、trainer seed/config、vocab、k/stride/tail rule、offset、train-only corpus version、工具版本、模型文件、special tokens、round-trip、prefix、canonical path 与 slot-to-raw-context 规则；同一 tokenizer artifact 供三个 formal model seeds 共用。
- `run_manifest.json`：run/parent ID、track/arm、Goal/source/data/split/tokenizer/config 版本、`tuning_seed/model_seed/patch_randomization_seed/generation_seed`、GPU、参数、raw/valid exposure、FLOPs、raw/slot context、checkpoint/output、final-test access state、运行状态和失败原因。
- `ScoreSums`：至少保存 `coded_bits_sum、quantized_cdf_nll_bits_sum、canonical_nll_bits_sum、next_base_nll_bits_sum、valid_nt_count、eos_nll_sum、sequence_count、attempted_count、decoded_count、valid_count、invalid_count、early_eos_count、truncation_count、overshoot_count`，禁止只保存平均 scalar。
- `RNAARAdapter`：所有模型统一实现 canonicalization、encode/decode、prefix preparation、forward、canonical coding/scoring 和 generation；共享 scorer、mask builder 与 artifact writer。
- `decision_map.yaml`：comparison family、candidate/applicability mask、task/length/compute cells、support thresholds、margin justification、Pareto/abstention 规则和冻结版本；如实现完整 selector，再单列 `selector_supplement.yaml`。
- `final_test_access_log` 与 `FINAL_UNSEAL_LOCK`：记录所有读取尝试、锁定 artifact 和唯一 unseal 事件；Phase 4–7 验收要求 access count 为零。

状态不得混用一个 enum：

| 字段 | 允许状态 |
|---|---|
| `run_status` | `NOT_RUN / RUNNING / PASS_CLOSED / FAIL_CLOSED_WITH_EVIDENCE / TERMINATED_SAFELY_WITH_EVIDENCE` |
| `artifact_lineage_status` | `AUTHORITATIVE / HISTORICAL_ONLY / ORPHAN_RESULT / MISSING / REGENERATED_NEW_LINEAGE` |
| `gate_status` | `NOT_EVALUATED / PASS_CLOSED / FAIL_CLOSED_WITH_EVIDENCE / BLOCKED_EXTERNAL_WITH_EVIDENCE / BLOCKED_RESOURCE_WITH_EVIDENCE` |
| `comparability_status` | `STRICTLY_COMPARABLE / REFERENCE_ONLY / PRETRAIN_OVERLAP_EXPECTED_REFERENCE_ONLY / UNAVAILABLE_WITH_EVIDENCE` |
| `result_interpretation_status` | `ESTIMATED / EQUIVALENT_WITHIN_MARGIN / INCONCLUSIVE_UNDERPOWERED / HARM_VIOLATION / INSUFFICIENT_SUPPORT` |
| `task_delivery_status` | `DRAFT / TASK_CLOSED_LOCAL / TASK_CLOSED_PUSHED / REDIRECT_PENDING_OWNER_APPROVAL` |

### A.3 本次审稿修订映射

| 审稿问题 | 已修改位置 | 核心变更 |
|---|---|---|
| 1. 新颖性未关闭 | 1.2、Phase 0 | 加入 H-Net、DNAChunker、MergeDNA、PatchDNA、BiomBenchmark 等 collision matrix；保留窄贡献 |
| 2. 生成定位与 BPN 错位 | 1.1、1.5、2.3、3.7 | true-suffix continuation 升为 required main-text secondary；无条件生成降为 exploratory |
| 3. 项目范围过大 | 1.4、3.5、Phase 4-G、5.3 | 核心 33 runs 可独立发表；Track D/350M 条件启动；完整 51 不再是核心论文前提 |
| 4. Atlas 过度复杂 | 1.5、3.9 | 主文改为分层效应+Pareto+弃权；完整 selector 仅补充材料 |
| 5. 生物证据弱 | 3.1、3.7、5.2 | 增加 length×GC×RNA-type 匹配、CM×近邻、去 rRNA/tRNA、rare-family、RNAGym |
| 6. BPN 语义 | 3.6 | 明确完整 canonical codec 系统；加入 uniform/Markov/CTW/PPM 与存储/构建成本 |
| 7. 因果/普适性措辞 | 2.2、3.2、3.9 | 使用 controlled/randomized intervention/complete-system；结论限定两类 backbone |
| 8. 数据版本 | 1.4、3.1、Phase 1 | release 22 训练锚点不变；release 23–27 shift，单独审计 release 27 构成 |
| 动态变长方法 | 1.6、3.2.2 | 核心 fixed/random/entropy；注册 learned dynamic Track L 与 BIO-DIAG，不阻断核心 |
| 优秀 benchmark 借鉴 | 1.7、3.5、3.8 | 场景矩阵、Profile Card、简单基线、低资源/效率压力、统一 adapter/result parser |

## 附录 B. GPU、监控与恢复 SOP

所有神经训练、验证、推理、生成和神经 evaluator 均为 GPU-only，并记录 GPU physical index/UUID/model、driver/CUDA/PyTorch、model/input/output device、forward/backward、peak VRAM、PID、命令、run ID、日志/checkpoint 路径和 `cpu_fallback_count=0`。

- 启动约 2 分钟和 5 分钟各检查一次；稳定后每 30 分钟只读检查最新 stage、指标尾部和资源快照；
- NaN、Inf、OOM、wrong-device、CPU fallback、磁盘安全线、checkpoint 损坏、PID 所有权异常或 final-test 非授权访问触发安全停止；优先 graceful stop，保留最后完整 checkpoint，不杀无关进程；
- Phase 3 校准前正式 GPU job 串行；校准后 100M 最多两个独占 GPU job 并行；GPU 不共享；350M 一次只运行一个 matched cohort；
- 同时最多一个未限流的大型数据扫描；GPU 等待期可做文档、代码测试、许可、external adapter 和 validation 分析，但不得读取 final test、改变冻结 decoder 或跳过 phase gate；
- 一个 root cause 只允许一次具有明确差异的 corrected retry；再次发生时以 evidence 关闭为 FAIL/BLOCKED，不无限重试。

## 附录 C. Artifact、目录与 Git SOP

大型 artifact 根为 `/mnt/cunyuliu/tokenizer-benchmark`，按 `data/raw`、`data/derived`、`tokenizers`、`weights/reference`、`runs/<run_id>`、`checkpoints/<run_id>`、`manifests/registry`、`tmp/cache` 分离；不从 `/Volumes` 直接训练，不用可变 `latest` 作为权威引用，不删除源 artifact，一个 run ID 只创建一次。

Git 生命周期：从 `benchmark-v1` 建立 `codex/tokenizer-benchmark/<phase>-<task>-<run_id>`，只 stage 本任务文件，运行相关测试与大文件检查，focused commit 后 push task branch；gate 通过后 fast-forward 到 `benchmark-v1` 并重读远端 ref。禁止 force-push、重写历史、自动覆盖 `main` 或上传数据、权重、checkpoint、生成全集与缓存。训练 heartbeat 不产生提交；只在正式 manifest、协议、代码或阶段报告变化时提交。
