# TokBench-RNA Goal Document V2

## Authority

- **Document**: `NCRNA_TOKENIZATION_SEGMENTATION_BENCHMARK_GOAL_V2.md`
- **Version**: V2
- **Status**: ACTIVE
- **Supersedes**: V1 (SHA-256 `9a145302b9a6aaf632cc17165a7b57165e3b1d7f8dfe0f8c9da115480786c509`)
- **SHA-256**: computed at materialization (recorded in the authority manifest)
- **Revision rule**: No silent gate modification. Any change to science, acceptance criteria, thresholds, or phase gating requires an amendment log entry and either owner approval or a new Goal version. Bug fixes that do not alter gates/SHA may be committed as amendments with evidence.

## Amendment Log (V1 -> V2)

| Date | Section | Change | Reason / Evidence |
|---|---|---|---|
| 2026-08-15 | Whole document | V2 replaces V1 as the single authoritative contract | Owner confirmed the new PLAN.md (提示词/PLAN.md, 775 lines) is the authoritative science & engineering execution contract. V1 was a condensed earlier version missing B1 bridge (§3.2.1), Track D (§3.3), the Atlas selector (§3.9), the real canonical codec headline (`canonical_code_length_BPN`, §3.6), the P2 supported-strata conditional random patch (§3.2), and the full Phase 0-8/Appendix A-C execution detail. |
| 2026-08-15 | §3.6 | Real codec headline | Added the actual decodable bitstream metric (`canonical_code_length_BPN`), 64-bit range coder, CDF quantization (2^24, min freq 1, largest-remainder), consistency gates, and independent decoder verification; implemented in `evaluator/codec.py` (12 fixtures PASS). |
| 2026-08-15 | §3.2 | P2 conditional random patch | Added supported-strata conditional random patch (`q(boundary | prefix-length, entropy)`, 8x8 bins, supported 0.05-0.95 sample / non-supported exact P3 replay, >=80% coverage gate); implemented in `model/conditional_patch.py` (12 fixtures PASS). |
| 2026-08-15 | §3.6 | Non-overlap k-mer tail rule | Fixed non-overlap k-mer tokenizer to use canonical tail tokens (no discarded trailing bases): F6 vocab 64->84, F7 4096->5460; lossless round-trip for all lengths (contract 3.6). Old F6 s17 marked FAIL_CLOSED_WITH_EVIDENCE and retrained. |
| 2026-08-15 | Appendix B | Parallelism | Owner-approved operational deviation: use all 6 usable GPUs (GPU0-5) for 100M Phase 4 jobs (up to 6 concurrent) instead of the nominal "<=2 parallel" to accelerate the 51-run matrix; recorded in project memory. |

---

# TokBench-RNA：ncRNA 自回归模型分词与动态分段 Benchmark 科学预注册合同及执行计划

> **当前 authority 状态：`ACTIVE`。** 本文件（Goal V2）是 TokBench-RNA 的唯一有效科学与执行合同，由负责人验收。取代前的任何冲突以本较新、明确验收的版本为准，禁止静默修改。

## 1. 总体结论与已锁定决策

### 1.1 项目转向结论

项目正式从“基于 BLT 提出新的 RNA 分词方法并冲击生成 SOTA”，转向：

> 构建一个序列相似性与家族隔离、数据一致、骨干受控、训练核苷酸暴露量受控、计算量可核算的 ncRNA 自回归建模与生成 benchmark，系统比较静态 tokenization 与动态 segmentation/patching，并判断不同方案在 canonical codec 码长、计算效率、续写质量、记忆风险、家族可识别性和结构分布代理上的真实权衡。

新项目暂定名：

> **TokBench-RNA: A Homology-Aware Benchmark of Tokenization and Segmentation for Autoregressive ncRNA Modeling and Generation**

项目成功不要求 entropy patching 获胜，也不要求提出新架构。一个严格复现的负结果、任务依赖的 tokenizer 排名或“没有全局最优 tokenizer”的证据，都可以成为 benchmark 论文的有效发现。

### 1.2 文献审查后的新颖性边界

第二轮检索确认：

- [BEACON](https://proceedings.neurips.cc/paper_files/paper/2024/hash/a8ea503d91320fcfe12cba61c8a6d285-Abstract-Datasets_and_Benchmarks_Track.html) 已经比较 RNA encoder 模型中的 single nucleotide、BPE、overlapping 6-mer 和 non-overlapping 6-mer，并覆盖 13 个下游任务。
- [GARNET](https://www.nature.com/articles/s41467-024-54812-y) 已经在 RNA 自回归生成模型中比较单碱基、重叠二联体和重叠三联体，且报告三联体较优，但不同模型的训练步数和优化设置不完全一致。
- [PatchDNA](https://openreview.net/forum?id=AFZeojzjoG) 已经在 DNA 中比较 conservation patch、entropy patch 和固定 patch，因此“首次将 entropy patching 用于核酸”不成立。
- [Zero-shot benchmarking of RNA language models](https://academic.oup.com/bib/article/27/2/bbag098/8509095) 已经对 21 个开放 RNA 模型进行结构、分类和突变适应度的统一零样本测评，并建立了 TS-Hard、RfamSample、ArchiveII-Nr 和 RNAGym 等评估子集。
- GenerRNA 已经提供 RNAcentral release 22、BPE-1024 和 350M decoder 模型的公开参照，必须冻结并区分其[更新权重和历史权重](https://huggingface.co/pfnet/GenerRNA)。

因此：

- `G4_BROAD_NOVELTY = FAIL`。
- 禁止宣称“首次 RNA tokenizer benchmark”“生物序列没有分词创新”“首次 RNA 动态分词”“首次核酸 entropy patching”。
- `G4_NARROW_BENCHMARK = CONDITIONAL_PASS_PENDING_UPDATED_COLLISION_MATRIX`。
- 仍可检验的窄缺口是：广谱、多家族 ncRNA 自回归建模与生成中，在共同数据、同源隔离 split、统一预算和核苷酸归一化评估下，对静态 tokenization 与动态 patching 进行受控比较。
- 在冻结论文主张前，必须更新逐工作 collision matrix，并记录检索日期、数据库、完整查询式、纳入/排除标准、发表状态、代码/权重可用性，以及每项工作与本项目在任务、数据、骨干、预算、动态分段、同源隔离、指标和选择规则上的逐字段重叠。
- 投稿前必须重新执行一次系统检索；如果出现完全覆盖该交集的新工作，则转为 evaluator/data resource、机制复核或负结果论文，不通过改名维持“首创”。当前 gate 不等于已经证明新颖。

### 1.3 旧 BLT 项目的证据状态

Goal 文档必须完整记录但不得继承旧结论：

- 旧训练数据 lineage 不闭合，当前可见 cleaned-train artifact 中已经确认至少存在原始 test 记录。
- 当前可见候选代码中，显式 n-gram 查表、生成和 PPL 路径不一致；显式 n-gram 训练可能退化为默认 ID，生成使用全零占位，PPL 又关闭该通道。
- hash byte-groups、explicit lookup n-gram 和 entropy patch 曾被混合，旧结果不能归因于任何单独机制。
- 旧 H100/H800/no-ngram/ngram 结果无法完整闭合到唯一 `code → data → config → checkpoint → decoder → output → evaluator`。
- 当前本地 `outputs/` 中未发现上一轮最终报告和账本。按用户选择，记为 `artifact_lineage_status=MISSING`，后续新产物标记 `REGENERATED_NEW_LINEAGE`，不得冒充恢复旧字节。
- 旧 checkpoint、生成文件和 PPT 数值只能放入“历史记录/排错证据”附录，不能进入新 benchmark 主表、模型选择或先验胜负判断。

### 1.4 已锁定的执行决策

- 数据锚点：RNAcentral release 22。
- Database-release-shift sensitivity：RNAcentral release 23–26 中相对 release 22 新增且与训练集完成 exact/80-80 sequence-similarity 隔离的 accession；不称生物时间泛化。
- Track R：100M 十臂表示受控主矩阵，每臂三个独立训练 seed，共 30 个科学运行；每个 run 使用 `2.0B cumulative_valid_target_nt`、`4096 raw-nt context`，headline 为固定最终预算 checkpoint 的 `canonical_code_length_BPN`。
- Bridge：新增 B1 `BLT patch-size=1`，100M、三个正式训练 seed、每 run `2.0B cumulative_valid_target_nt`；B2 不新增模型，直接复用 F7 `Flat non-overlapping 6-mer`。F1/F7/B1/P1 构成固定的 2×2 系统分解。
- Track D：新增 100M Flat-NUC 与 Flat-BPE 两臂，各三个独立训练 seed；固定 `4096 representation slots` 和同一累计 FLOP 预算，scheduler 按累计 FLOP 比例推进，并报告实际 raw context 与 valid-nucleotide exposure。Track D 是部署辅助赛道，不是纯 tokenizer 因果赛道。
- 350M 阶段：NUC、BPE-1024、固定 patch、entropy patch 四臂，每臂三个训练 seed，共 12 个科学运行。
- 冻结正式训练总数为 `30 Track R + 3 B1 + 6 Track D + 12 350M = 51` 个 run；B2 复用 F7 不重复计数，HPO/smoke/calibration 与 generation seeds 均不计入 51。
- 固定训练 seed：`17、29、43`。
- 独立调参 seed：`101`；调参 run 不得续训为正式 run，正式 seed `17、29、43` 必须从头训练，调参计算单列。
- 唯一跨 tokenizer headline：`canonical_code_length_BPN`；`next_base_BPN` 只在具有精确逐碱基条件分布的兼容模型子集报告。
- 主要任务：固定最终预算 checkpoint 的 canonical code length 与 prefix continuation；best-validation checkpoint 仅作 secondary sensitivity。
- 次要任务：无条件生成、Rfam family recoverability、结构与突变适应度诊断。
- 所有 final test 保持封存，直到 100M、Track D、350M、全部 checkpoint、decoder、数据分层、统计代码、atlas selector、图表模板和 claim–evidence matrix 均冻结；随后所有内部模型与可运行外部模型在同一个 `FINAL_UNSEAL` 阶段一次评分。
- GPU、监控、PID、目录、恢复和 Git 规则全部移至附录 A–C；它们约束执行与复现，但不改变 scientific estimand。

### 1.5 论文核心交付：多变量 Tokenizer 选择图谱

本项目的最终产物不得停留在“tokenizer A 的平均分最高”。论文必须交付一张可复用的 **tokenizer/segmentation 选择图谱（choice atlas）**，回答：

> 在什么任务、什么序列长度和什么计算预算下，应优先选择哪一种 tokenizer 或 patching 方案；它带来的收益、代价、适用边界和不确定性分别是什么？

选择图谱固定为三个条件轴；“生物分辨率”不作为第四个条件轴，而作为必须检查的约束/结果族，避免与 task 重复计数：

| 主轴 | 必须覆盖的条件 |
|---|---|
| 任务 | canonical code length、true-suffix prefix continuation、无条件生成 Pareto |
| 序列长度 | 预注册的短、中、长序列分层及 length OOD |
| 计算 | 相同有效核苷酸 exposure、共同累计 FLOPs 检查点、推理吞吐/显存与有效上下文 |

生物分辨率约束/结果族不构成第四维；在每个适用 cell 中报告单核苷酸/突变敏感性、局部 motif、family/clan recoverability、预测结构分布、低复杂度敏感性和记忆风险。

图谱必须直接检验并回答以下问题，而不是只展示总平均：

- 短序列或单核苷酸敏感任务何时更适合 NUC；
- BPE 在 Track R 的 matched raw context 与 Track D 的 fixed representation slots 下排序如何变化，从而估计额外 raw-context 容量带来的系统增益；该跨 track 差异不写成精确中介比例，也不证明 BPE 边界具有或不具有生物意义；
- overlapping k-mer 是否改善局部模式，同时增加突变敏感性和冗余计算；
- entropy patch 是否把计算分配到重要边界，还是主要追踪低复杂度、GC 变化或数据来源；
- fixed-6、条件匹配 random 与 mean-6 entropy patch 在共同平均 patch 长度、且 P2/P3 patch/FLOP 平衡门通过后是否仍有稳定差异；
- 较低 BPN 是否伴随更严重的训练记忆、家族覆盖下降或结构分布偏移；
- 是否不存在普适最佳 tokenizer，而只存在随任务和预算变化的 Pareto 前沿。

选择图谱中的每个推荐单元必须同时给出：适用条件、候选方法、主要效应量与置信区间、计算代价、生物保真/记忆代价、跨 seed 稳定性、已知失败条件和证据等级。若最终结论是“没有普适最佳 tokenizer”，这仍是有效主结论；但必须由预注册分层、交互效应和反事实对照支持，不能由事后挑选子集得到。

Atlas selector 的决策阈值在看 final test 前固定：

- `canonical_code_length_BPN`：相对差异 `1%` 为实际意义界值；
- 推理成本：相对差异 `15%` 为实际意义界值；
- validity 与 family recoverability：`2` 个绝对百分点为 harm margin；
- memorization：`1` 个绝对百分点为 harm margin；
- 预注册分布距离：相对恶化 `5%` 为 harm margin；
- 每个 cell 至少包含 `100` 个 homology clusters；涉及 family 的 cell 至少包含 `20` 个 eligible families。

Selector 只能输出 `WINNER`、`TRADE_OFF_SET`、`NO_RESOLVED_WINNER` 或 `INSUFFICIENT_SUPPORT`，不得强制填满所有单元。validation 上冻结 selector 后，final test 必须报告平均 regret、最坏 regret、Pareto coverage、harm-violation rate 和 abstention rate；final test 后不得修改 selector、阈值、cell 或 applicability mask。

## 2. 科学问题、估计量与分析层级

### 2.1 唯一核心问题

> 在数据、训练目标和明确的预算约束下，ncRNA 自回归模型的表示粒度与分段规则如何改变实际 canonical codec 码长、可见原始上下文、计算成本、续写、生成记忆和 RNA proxy fidelity；这些权衡能否在 test 之前被一个会主动弃权的选择器稳定预测？

本合同不追求“哪个 tokenizer 平均第一”，而是同时回答表示效应、patch-rule 效应、系统部署效应和选择器可迁移性。任何结论必须限定到实际运行的 Flat Transformer 与 BLT backbones。

### 2.2 三个受控估计量、一个辅助赛道和一个最终选择器

| 对象 | 实验来源 | 回答的问题 | 明确不回答 |
|---|---|---|---|
| `E_R_STATIC_REPRESENTATION` | Track R，F1–F7 | 同 Flat 骨干、同 raw context、同 valid-nt exposure 下，静态表示的完整效应 | 不回答 BLT 是否优于 Flat；不包含 BPE 的额外 raw-context 容量 |
| `E_P_PATCH_RULE` | Track R，P1–P3 | 同 BLT 骨干和 mean patch length=6 下，fixed/random/entropy rule 的差异 | 平衡门失败时不回答边界位置的独立效应 |
| `E_S_SYSTEM_DECOMPOSITION` | F1/F7/B1/P1 固定 2×2 | 层级计算与 1-nt/6-nt 粒度改变时，完整系统响应如何分解 | 不升级为纯 tokenizer 或纯架构因果效应 |
| `E_D_DEPLOYMENT` | Track D，D1/D2 | 同 4096 representation slots 与累计 FLOPs 下，BPE 更长 raw context 的部署系统收益与代价 | 不把 tokenizer、上下文和执行路径的合并差异拆成纯边界效应 |
| `ATLAS_SELECTOR` | validation 拟合、final test 一次确认 | 非生成 selector 的整体 regret/coverage/harm/abstention 能否在 task × length × compute 上迁移，以及 exploratory generation atlas 能否给出 Pareto trade-off 或主动弃权 | selector-level 结果属于 secondary；单个 cell 不升级为 confirmatory claim；不允许 final-test-informed 改规则，generation 不能反向改变非生成 selector |

### 2.3 Endpoint 与证据层级

| 层级 | 冻结内容 | 论文用途 |
|---|---|---|
| Headline confirmatory | fixed final-budget checkpoint；primary cluster-held-out final test；`canonical_code_length_BPN`；canonical entities 等权；seed 17/29/43 paired effects | Main Table 1/2 与核心 benchmark 结论 |
| Secondary | true-suffix continuation code length；350M scale trend；Track D；selector-level final regret/coverage/harm/abstention；family/length 分层；best-validation checkpoint sensitivity | 解释适用范围、选择器迁移与部署权衡，不替换 headline |
| Diagnostic/exploratory | 无条件生成、boundary mechanism、structure/CM proxy、mutation sensitivity、external reference-only models、database-release shift | 形成 failure map、机制假设和限制，不事后升级为 primary |

三个 training seeds 是独立训练重复；测试 cluster、family、prefix 和 generation output 均不是额外模型重复。项目采用效应量与区间估计，不将三个 seeds 包装成高功效显著性检验。

证据层级必须按对象区分：validation-frozen selector 的整体 out-of-sample regret、coverage、harm 和 abstention 是一次性 final-test-confirmed 的 selector-level secondary result；单个 `task × length × compute` cell 的 winner/trade-off 属于 multiplicity-rich secondary result；generation atlas 始终 exploratory。不得从“selector 整体被一次确认”推导出“每个 cell 都是 confirmatory winner”。

### 2.4 主文档阅读顺序

科学合同按以下顺序执行：核心问题与 estimand（本节）→ 数据与 split（3.1）→ Track R/Bridge/Track D/350M 矩阵（3.2–3.5）→ 指标、统计和 atlas（3.6–3.9）→ final-test 生命周期（Phase 0–8）。GPU、Git、PID、目录和恢复规则位于附录，不参与科学 claim 的定义。

## 3. 科学 Benchmark 合同

### 3.1 数据集构建

主训练数据：

- 官方或可验证归档的 RNAcentral release 22。
- 如果官方 release 22 原始快照无法合法、完整地恢复，可使用 GenerRNA 发布的 release-22-derived 数据作为单独标记的重建版本，但不得声称它等于官方原始快照。
- 如果两者都不能闭合 accession、license 和 artifact identity，`gate_status=FAIL_CLOSED_WITH_EVIDENCE`，禁止以未知本地旧数据替代。

Primary 数据规则：

- 大小写统一；
- `T → U`；
- primary alphabet 为 `A/C/G/U`；
- 含其他 IUPAC 字符的记录保留在 QC 账本，但不进入 primary 训练；另建 ambiguity stress subset；
- Track R、B1 和 350M 的主训练长度为 `16–4096 nt`；
- Track D 为回答固定 representation slots 下的上下文容量问题，使用独立但前瞻冻结的 `D_long_context_train_view`：仅包含已经分配到 train clusters 的 `16–16384 nt` canonical entities；D1/D2 使用完全相同的 entities 与冻结顺序，禁止引入 validation/final clusters；
- Track D 另建立 `D_long_context_validation_view`：只包含 validation clusters 中满足相同 `16–16384 nt` 规则的 canonical entities，用于 Track D HPO、selector 拟合、上下文容量检查和 validation-only diagnostics；它与 `D_long_context_train_view`、所有 final test 均严格互斥，不得用 final clusters 补足样本量；
- `4097–16384 nt` final subset 对 Track R/B1/350M 标为 `length_OOD`；对已经在 long-context view 见过该长度范围的 Track D 只能标为 `held_out_long_context`，不得称 length OOD。Atlas applicability mask 必须保留这一区别；
- 更长序列只作描述性资源统计，除非新 Goal 版本前瞻性授权；
- RNA 有方向性，不把 reverse complement 静默视为同一序列；单独报告 reverse-complement 近邻；
- canonical exact duplicate 合并为一个训练实体，但完整保留 accession 和 metadata 映射；
- train、validation、test 之间 canonical exact overlap 必须为零。

同源隔离：

- primary cluster 使用 MMseqs2 `80% identity / 80% query-and-target coverage`；
- `90%/90%` 作为敏感性分析；
- 同一 primary cluster 不能跨 split；
- 工具版本、参数、cluster membership 和全部 hash 冻结；
- 不允许从文件名推断“已经聚类”。
- split 完成后必须直接执行 train→validation/test cross-search，验证不存在满足 primary 80/80 条件的跨 split sequence pair；只比较 cluster ID 不足以关闭 leakage gate；
- 对短 RNA 预注册高敏感 alignment/精确枚举复核，并以已知 Rfam family/clan 测试聚类召回；若 80/80 对短 RNA 的检出不足，相关样本使用更严格的 family-aware 隔离并在 datasheet 报告。

Split：

- 在 family/clan held-out 分配完成后，剩余同源簇按稳定 hash seed `20260808` 分为 `98% train / 1% validation / 1% cluster-held-out test`；
- 分层变量至少包括长度区间、RNA type、来源数据库和 Rfam 标注状态；
- eligible Rfam family 定义为清洗后至少 100 条序列且至少 10 个同源簇；
- eligible family 中 10% 分配给 family-validation、10% 分配给 family-test；
- 有 clan 的 family 另外构建完整 clan-held-out sensitivity split；
- family/clan test 对应的同源簇全部从训练集移除；
- release 23–26 的 database-release-shift sensitivity 只保留相对 release 22 的新 accession，并再次移除对 release 22 train 的 exact 和 80/80 sequence-similarity overlap；不得将数据库版本变化泛称为生物时间泛化；
- primary cluster-held-out test、family test、clan test 和 database-release-shift test 均为 final sealed test；Phase 0–7 只能访问 train、validation 和预注册的未封存 diagnostics，任何 final test 的序列、标签、聚合结果或模型输出都不得读取。Phase 8 是唯一授权 unseal。

训练以去重后的 canonical sequence entity 为等权采样单位；accession multiplicity 只保留为 metadata/QC，并可在预注册的 secondary sensitivity 中单独评估，不得作为 primary 训练重复权重。RNAcentral 收录频率不得称为自然丰度。

评估同时报告：

- canonical-entity-weighted micro average；
- family-balanced macro average；
- RNA type/source/length 分层结果。

不得只用一个总体平均值掩盖 rRNA 或高频 family 的主导效应。

### 3.2 Track R：100M 十臂表示受控主矩阵

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
| P2 | 同一 BLT backbone | supported-strata hybrid matched-random patch | 在平衡门通过时估计受支持区域内的细粒度边界位置价值 |
| P3 | 同一 BLT backbone | causal entropy patch，阈值校准到 mean patch length=6 | 估计 adaptive entropy rule 相对固定/随机的系统效应 |

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
- static track 内可以解释 tokenization effect；
- patch track 内可以解释 patch-rule effect；
- F1 与 P3 等跨骨干比较只能标为 architecture/system comparison，不能称为纯 tokenizer effect。

### 3.2.1 Flat 与 BLT 的三层比较、结果图和桥接模型

本 benchmark 必须把结论分成三层，禁止将三层混成一个总排行榜：

**第一层：Flat 赛道内部的静态 tokenizer 比较**

- 比较对象：F1–F7，即 NUC、BPE、Unigram、overlapping/non-overlapping k-mer；
- 保持不变：Flat causal Transformer 骨干、训练数据、序列顺序、`context_nt`、有效核苷酸 exposure、优化协议和 evaluator；
- 允许结论：在 Flat 骨干中，某种静态 tokenizer 在指定任务、长度和预算下更优或更具性价比；
- 禁止结论：仅凭 Flat 赛道结果推断 BLT 的层级 patching 是否更优。

**第二层：BLT 赛道内部的 patch 边界规则比较**

- 比较对象：P1–P3，即 fixed、patch-length-matched random、causal entropy patch；
- 保持不变：BLT 骨干、参数、平均 patch 数/长度分布口径、训练数据、有效核苷酸 exposure、优化协议和 evaluator；
- 允许结论：在 BLT 骨干中，边界位置是否具有独立价值，以及 entropy 边界是否稳定优于 fixed/random；
- 禁止结论：把 BLT 相对 Flat 的全部差异归因于 entropy 或“动态分词”。

**第三层：Flat 与 BLT 的跨赛道系统比较**

- 比较单位是完整系统，不是单一 tokenizer；
- 两条赛道使用同一数据/split、相同有效核苷酸 exposure、相同 raw-nucleotide context、共同累计 FLOP 检查点、相同测试序列和冻结的解码协议；
- 同时报告参数量、训练 FLOPs、推理 FLOPs、吞吐、延迟、峰值显存、BPN/续写/生成、训练记忆、family recoverability 和结构代理；
- 允许结论：在某个任务、长度和计算预算下，哪一个完整系统处于性能–效率–生物代价 Pareto 前沿；
- 禁止结论：从任意 Flat-vs-BLT 差值直接声称“某 tokenizer 导致提升”。

主结果固定为三类输出：

1. **Main Table 1 — Flat 静态 tokenizer 内部表**：F1–F7 的 headline、逐 seed paired effect、CI、计算和生物代价；
2. **Main Table 2 — BLT patch-rule 内部表**：P1–P3 在共同 mean patch length=6、且 P2/P3 通过分层 patch-budget 平衡门后的 headline、机制与代价；
3. **Main Figure 1 — System Pareto**：预先固定三个 panel：训练累计 FLOPs–`canonical_code_length_BPN`、推理 FLOPs–prefix true-suffix code length、推理成本–生成 validity/family/记忆约束。颜色固定表示方法家族，分面固定表示长度层；跨赛道图只用于系统选择，不用于 tokenizer 因果归因。

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

### 3.3 Track D：100M 部署/上下文容量辅助赛道

Track D 只包含两个 Flat Transformer 系统，各使用三个正式训练 seed：

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
- Phase 3 context-separation gate 在至少 `100` 个 train-only homology clusters 的同一组 frozen raw-target probes 上，以 raw target nucleotide 等权形成 `effective_raw_context_nt` 分布后取 median；D2 median 必须比 D1 高至少 `15%`，并报告 target-level、sequence-level 分布及 effective context `>4096 nt` 的 target 比例。若未通过，Track D 仍可报告固定槽位效率，但删除“更长单序列上下文收益”和对应 atlas 推荐；
- 两臂另报实际接触的 unique entity/cluster 数、length/source/RNA type/family、GC 与 low-complexity 构成及 valid target nt；若任一预注册连续暴露变量的绝对 SMD `>0.1` 或离散构成差超过 `2` 个百分点，则保留 `E_D_DEPLOYMENT`，但上下文解释降级为“该冻结数据流下的完整系统差异”，不得单独归因于更长上下文；
- Track D 不复用 Track R 的中间 checkpoint 冒充 compute-matched training；D1/D2 是单独从头训练的六个科学 run，调参计算另列；
- `D2−D1` 同时改变 tokenizer、可见 raw context 与系统执行路径，只允许称 `deployment/system effect`；它与 Track R 的 F2−F1 共同用于区分“matched raw context 的表示效应”和“固定槽位下更长上下文的系统收益”，不能被解释为边界的纯生物意义。

### 3.4 350M 复验

固定四臂：

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

### 3.5 训练预算与公平性

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
- H100/H800/不同 A100 cohort 不能直接混排速度。

超参数规则：

- 每臂允许完全相同预算的 train/validation-only 调优；
- 固定候选为基础学习率的 `0.5×、1×、2×`，其余 optimizer 配置保持相同；
- 100M 基础学习率初值 `3e-4`，350M 初值 `2e-4`；
- AdamW `β=(0.9,0.95)`、weight decay `0.1`、bf16；
- 每个候选使用独立 `tuning_seed=101`；Track R/Bridge/350M 候选最多 `100M valid target nt`；Track D 候选使用“D1 处理 `100M valid target nt` 所需模型 FLOPs”的同一确定性 pilot 上限；
- 按 validation 指标选择后冻结；
- tuning checkpoint 不得成为正式 checkpoint，也不得续训；正式 seed `17/29/43` 必须以冻结配置从头训练；
- primary 使用固定最终预算 checkpoint；best-validation checkpoint 只作为 secondary sensitivity，二者不得混在 headline 或 atlas selector 中；
- 不允许给失败臂额外调参预算；
- 不允许使用 primary/family/clan final test、database-release-shift sensitivity 或生成主表选择超参数。

### 3.6 统一跨 tokenizer 码长口径

不得把不同含义的 token perplexity 直接横向排列。

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

Track R/B1/350M 的 length-OOD primary 评分固定为 rolling `4096 raw nt` context：对位置 `i`，所有模型只观察相同的最近 `min(4096, i−1)` 个真实 raw nucleotides，每个目标碱基恰好计分一次；窗口重叠不得重复进入分母。该结果称“长序列上的局部窗口 OOD”，不得据此声称模型学习了超过 4096 nt 的长程依赖。Track D 可运行同一 raw-context-controlled sensitivity，但因其 long-context train view，只能标 `held_out_long_context`；其 4096-slot 部署上下文另表报告，不混入 length-OOD claim。

### 3.7 Prefix continuation 与生成协议

冻结的 continuation secondary endpoint：

- 从 sealed test 的真实序列构建 `10% / 25% / 50%` prefix；
- prefix 以 raw nucleotide 定义，而不是 token 数；共同切点固定为 `6 × floor((ratio × sequence_length)/6)`，仅保留至少有 1 nt suffix 的样本，并逐样本报告实际 prefix 长度和实际比例；
- tokenizer 对观测 prefix 独立编码，禁止 BPE token 跨过观测与隐藏后缀边界；
- non-overlapping k-mer、NUC、BPE 和 BLT 必须使用完全相同的共同 raw-prefix 切点，任何模型不得静默多看或少看碱基；
- 给所有模型相同 target raw length；
- 对超过窗口的 prefix，所有模型只观察相同最近 `4096 raw nt`；该条件是 local-window continuation，不称长程 continuation；
- 若最后一个 token 超出目标长度，保留原始完整输出，同时另存固定长度评估视图，并记录 truncation；
- primary continuation endpoint 是真实 suffix 的 `canonical_code_length_BPN`；suffix edit distance、nucleotide accuracy、k-mer recovery、Rfam family/clan recoverability、CM bit score、结构代理偏差和训练最近邻均为 secondary/diagnostic，不把单一真实 suffix 称为唯一合理生成答案。

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

MFE、CM hit、预测 pairing 和 embedding score只能称为 computational proxy，不能写成真实功能、湿实验验证或天然 RNA 证明。

### 3.8 外部 benchmark 与横向模型

内部结果使用 `Main Table 1 Flat`、`Main Table 2 BLT` 和 `Main Figure 1 System Pareto`；外部结果固定分为两张表，禁止混成单一 SOTA 排名。

**External Table 1 — 公共自回归/生成模型参考**

按 best-effort 优先冻结和运行：

- GenerRNA `model_updated.pt`；
- GenerRNA 历史 `model.pt`；
- GARNET 公开 GPT checkpoint；
- EVA 145M 与 437M，在权重、许可和代码可运行时作为外部 single-nucleotide 参考；
- 公共模型的数据/架构/预算不匹配时只称 ecological reference，不用于 tokenizer 因果归因。
- GenerRNA 使用 release-22-derived 语料；在本项目 release-22 primary test 上默认标记 `PRETRAIN_OVERLAP_EXPECTED_REFERENCE_ONLY`，除非其逐序列训练清单能反证污染。更公平的外部参考优先使用确认晚于其训练语料并完成 sequence-similarity 过滤的 database-release-shift subset；训练语料不透明时仍保持 reference-only。
- 任何外部模型不可获得或不可运行，都不得阻断内部 Track R/D/Bridge/350M 的科学完成；只能删除或降低 external-comparison/SOTA claim。

**External Table 2 — family/structure-conditioned 参考**

- RfamGen；
- RNAgg；
- 仅在相同 Rfam family 子集上比较；
- 不与 broad unconditional generation 混排。

Secondary benchmark：

- GARNET 16S/23S 与 231-family continuation；
- RfamSample 的 family recoverability；
- ArchiveII-Nr 的结构分层诊断；
- RNAGym 的 mutation likelihood/fitness；
- TS-Hard 仅在能够定义兼容、冻结的结构 probe 时使用；
- BEACON 用作 prior-art 和可选表示诊断，不把其 13 个任务全部扩入主项目。

所有外部模型必须建立：

- model ID/revision；
- paper/preprint status；
- checkpoint SHA-256；
- code commit；
- tokenizer；
-训练数据和可能 overlap；
- license；
- GPU 运行环境；
- decoder；
- evaluator adapter；
- 使用附录 A 的完整 `comparability_status`，包括 `PRETRAIN_OVERLAP_EXPECTED_REFERENCE_ONLY`，不得退化成模糊“可比/不可比”。

### 3.9 统计分析、Atlas selector 与论文主张

统计采用 **估计优先**，不把三个 training seeds 包装成高功效显著性检验。三个受控估计量固定为：

| Estimand | 正式 contrasts | 共同条件 | 允许解释 |
|---|---|---|---|
| `E_R_STATIC_REPRESENTATION` | Track R 的 F2–F7 分别减 F1 | 同 Flat backbone、4096 raw nt、2B valid target nt、相同数据顺序与评估 | 在该 Flat 实现内，静态表示/tokenization 的完整效应 |
| `E_P_PATCH_RULE` | P3−P1、P3−P2、P2−P1 | 同 BLT backbone、mean patch length=6；P2/P3 须通过分层 patch/FLOP 平衡门 | adaptive patch rule 的总效应；只有平衡门通过时，P3−P2 才可进一步称条件匹配下的边界位置效应 |
| `E_S_SYSTEM_DECOMPOSITION` | `B1−F1`、`P1−F7`、`P1−B1`、`F7−F1` 及预注册差中之差 | F1/F7/B1/P1 的固定 2×2 与共同数据/预算记录 | 层级、粒度和系统响应的分解证据；不是纯 tokenizer/架构因果效应 |

Track D 的 `D2−D1` 是独立的 `E_D_DEPLOYMENT` 辅助估计量；它回答固定 4096 representation slots 和相同累计 FLOPs 下的完整部署系统差异。350M 的 BPE−NUC 与 entropy−fixed 只复验规模趋势，不新增方法选择自由。

每个 confirmatory estimand 均固定使用：final checkpoint、primary cluster-held-out final test、dataset-level `canonical_code_length_BPN`、canonical entities 等权、paired model seed `17/29/43`。prefix continuation、generation、机制、family/length/source 分层和 atlas cells 是层级化 secondary/diagnostic，不能替换 headline。

不确定性与统计单位：

- 对每个 training seed，先在完全相同的测试 homology clusters 上计算 paired effect，并在该 seed 内做 paired cluster bootstrap 95% CI；
- family-macro 结果以 family 为顶层重采样单位；同一原序列的多个 prefix 必须成组重采样；
- 三个 training seeds 是模型训练重复，必须逐 seed 展示 paired effect、均值、范围和方向；cluster bootstrap 只反映测试簇不确定性，不得冒充跨训练随机性的显著性；
- generation seeds 和生成序列只量化单一训练模型下的 Monte Carlo uncertainty，不与 training seed 合并成独立 N；
- 主文不以 cluster-only p 值声称跨训练显著，也不以“未显著”证明等效；如另做假设检验，必须在独立统计附录中预注册检验族与 multiplicity，不得替代效应量与区间；
- BPN 相对效应统一定义为 `δ=100×(BPN_candidate−BPN_reference)/BPN_reference`，因此负值更好；若三个 seed 的 paired-effect 及其 seed-specific cluster-bootstrap CI 均落入 `[-1%, +1%]`，且没有预注册 harm violation，可称“未见实质差异”；若任一必要 CI 同时覆盖 `≤−1%` 与 `≥+1%`，状态必须为 `INCONCLUSIVE_UNDERPOWERED`；
- 必须报告失败运行、唯一 corrected retry、参数、valid/raw nucleotide exposure、FLOPs、GPU、显存和 wall time。

Atlas selector 的统计合同：

- 结果立方体固定为 `comparison_family × candidate × task × length × compute`；comparison family 只是不可混排的预注册候选池，不作为事后分层：`R_FLAT={F1…F7}`、`R_BLT={P1,P2,P3}`、`R_SYSTEM={F1…F7,P1,P2,P3,B1}`、`DEPLOYMENT={D1,D2}`。`SYSTEM_2x2={F1,F7,B1,P1}` 只用于已冻结的机制分解，不重复形成另一个 winner pool；350M C1–C4 只作对应选择的 scale sensitivity，外部 reference-only 模型不进入 selector；
- `R_FLAT/R_BLT/R_SYSTEM` 属于 exposure-indexed `R_ATLAS`；`R_SYSTEM` 的推荐对象是完整系统，绝不作 tokenizer 因果归因。`DEPLOYMENT` 属于 FLOP-scheduled `D_ATLAS`。R/D 使用独立 panel、reference、regret、coverage 和 abstention，不得在同一 cell 选 winner，也不得把 Track R 中途 common-FLOP checkpoint 当成 Track D 式 compute-matched training；
- task 固定为 canonical code length、true-suffix continuation code length和无条件生成 Pareto；长度固定为 `16–127 / 128–511 / 512–4096 / 4097–16384`。最后一层对 Track R/B1/350M 标 `local-window length_OOD`，对 Track D 标 `held_out_long_context`；compute 使用 Phase 3 在 train/validation 上冻结、各 comparison family 全部候选共同可达的绝对累计 FLOP checkpoints；
- canonical code length 与 continuation 构成预注册的非生成 atlas panels；只有 selector 整体 regret/coverage/harm/abstention 是一次性 final-test-confirmed 的 selector-level secondary 结果，单个 cell winner 仍为 secondary。unconditional generation 单独形成 exploratory generation atlas，其 winner/trade-off 只能作 generation-specific exploratory 结论，不得替换 headline、改变非生成 selector 或反向选择模型；
- biological resolution、validity、family recoverability、结构/分布 proxy 和 memorization 是约束/结果族，不是额外 atlas 轴；
- 每个 cell 少于 100 homology clusters，或 family 相关 cell 少于 20 eligible families时，直接输出 `INSUFFICIENT_SUPPORT`；
- Atlas 的 training-compute 轴仍由累计模型训练 FLOPs 定义；但用于 `15%` ε-dominance 的唯一 primary deployment cost 前瞻冻结为同硬件、同 query bundle、同 batch policy 下的 **端到端推理延迟**：非生成任务用 `median milliseconds / scored raw nt`，生成任务用 `median milliseconds / attempted generated raw nt` 并另报每个 valid output 的成本。该 pipeline 必须包含 input canonicalization、BPE/Unigram encode、entropy estimation/patch construction、packing/padding、正式 CPU–accelerator transfer、model forward/generation 与 detokenization；一次性 tokenizer/patcher 训练成本不混入单次部署延迟，而在 amortization sensitivity 中另报；
- 每个 cell 的 primary cost 字段、硬件 cohort、batch/query bundle、warm-up 和测量脚本在 Phase 3 冻结。model-only FLOPs、峰值显存、能耗和不含 tokenizer 的 accelerator latency 只作 secondary Pareto outcomes，不能替换 primary cost 或在看结果后择优。端到端 latency CI 使用 warm-up 后至少 30 次 paired repeats；确定性 FLOPs 作为 secondary cost 时视为退化点区间；
- 对 BPN/continuation task，reference 只用于效应归一化，不享有或失去 winner 资格。对任意候选 A/B，按每个 training seed 分别计算 `Δloss=100×(loss_A−loss_B)/loss_B` 与 `Δcost=100×(primary_cost_A−primary_cost_B)/primary_cost_B`。只有当三个 seed 各自都满足以下二者之一且 A 无 harm violation，才记 A 稳健 ε-dominates B：① `Δloss` 的单侧 95% 上置信界 `≤−1%`；② `Δloss` 的双侧 95% CI 完全落在 `[-1%,+1%]`，且 `Δcost` 的单侧 95% 上置信界 `≤−15%`。loss CI 使用同 seed paired cluster bootstrap，cost CI 使用上述 frozen paired repeats；
- 对每一对 eligible candidates，冻结算法只能标记为 `A_DOMINATES_B`、`B_DOMINATES_A`、`RESOLVED_TRADE_OFF`、`PRACTICAL_EQUIVALENCE` 或 `UNCERTAIN_EDGE`：前两者使用上一条规则；`RESOLVED_TRADE_OFF` 要求一方在 loss 上越过 1% 区间门而另一方在 primary cost 上越过 15% 区间门；`PRACTICAL_EQUIVALENCE` 要求 loss 的双侧 95% CI 完全位于 `[-1%,+1%]` 且 cost 的双侧 95% CI 完全位于 `[-15%,+15%]`；其余均为 `UNCERTAIN_EDGE`，不得把未通过 dominance 当成等价；
- 在 robust-dominance 有向图中，`ROBUST_NONDOMINATED_SET` 唯一定义为所有入度为零的 eligible candidates。只有一个候选对全部其他候选都有出边时输出 `WINNER`，因此 baseline/reference 本身也可以获胜；否则，只要任一 eligible pair 为 `UNCERTAIN_EDGE`，就输出 `NO_RESOLVED_WINNER(reason=UNCERTAIN_DOMINANCE)`；若所有候选两两 practical-equivalent，输出 `NO_RESOLVED_WINNER(reason=PRACTICAL_EQUIVALENCE)`；若 non-dominated set 为空，输出 `NO_RESOLVED_WINNER(reason=NONTRANSITIVE_DOMINANCE)`；其余情况输出完整 `ROBUST_NONDOMINATED_SET` 作为唯一 `TRADE_OFF_SET`，不得事后寻找更小子集。即使该集合只有一个元素，只要它未被证明支配全部候选，也只能标为 singleton `TRADE_OFF_SET`；
- generation task 使用 validation-frozen 的多结果 ε-Pareto pair relation 构建同样的 robust non-dominated set；只有一个候选支配全部 eligible candidates 且无 harm violation时才输出 `WINNER`，其余按上一条的 uncertainty、nontransitivity 与 set 规则输出；
- validity/family recoverability 下降超过 `2` 个绝对百分点、memorization 上升超过 `1` 个绝对百分点、任一预注册分布距离相对恶化超过 `5%`，均为 harm margin；每个分布指标分别冻结 `D_floor[metric]`，取 validation 中该指标非零 reference distances 的第 5 百分位；某预注册 stratum 有至少 100 homology clusters 时可进一步冻结 `D_floor[metric,stratum]`，否则必须回退到 metric-level floor。相对变化分母固定为 `max(reference_distance, applicable_D_floor)`；所有 floor 的指标单位、方向、适用层和数值必须在 final unseal 前写入 selector，不得跨指标共用一个 floor；
- 每个分布指标还必须在 evaluator fixture 阶段、查看 validation 与 final 结果前冻结数值零容差 `τ_metric`。若 validation 的全部 reference distances `≤τ_metric`，该 metric 不构造相对 `D_floor`：candidate distance 也 `≤τ_metric` 时记为该 metric 无可检测恶化；candidate distance `>τ_metric` 时固定触发 `NO_RESOLVED_WINNER(reason=HARM_UNCERTAIN_ZERO_REFERENCE)`，除非合同在 final unseal 前已经用独立科学依据预注册了绝对 harm margin。不得在 final test 后选择 floor、absolute margin 或借用其他 metric 的 floor；
- harm 的不确定性规则固定为：三个 seed-specific one-sided CI 均位于安全侧才判 `NO_HARM_EVIDENCE`；任一 CI 明确越过 harm margin 判 `HARM_VIOLATION`；CI 跨越 margin 但未确认方向时 selector 输出 `NO_RESOLVED_WINNER(reason=HARM_UNCERTAIN)`。只有被 pairwise algorithm 标为 `RESOLVED_TRADE_OFF` 的性能–成本冲突，才可按 robust non-dominated graph 进入 `TRADE_OFF_SET`；不确定边必须弃权；
- 若全部候选两两满足 `PRACTICAL_EQUIVALENCE` 且无 harm violation，固定输出 `NO_RESOLVED_WINNER(reason=PRACTICAL_EQUIVALENCE)`；
- selector 的 reference、候选池、归一化、tie/harm 规则和 applicability mask 只在 validation 与未封存 diagnostics 上冻结；applicability mask 只能由模型语义、数据支持量和预注册任务可用性决定，不能因 validation 表现差而删除困难 cell；final test 后不得新增 cell、改变 threshold 或把外部 reference-only 模型塞入内部 selector；
- final test 报告：对 code-length/continuation cell，单一 winner 的 `regret=100×(loss_selected−loss_best_feasible)/loss_best_feasible`；对 `TRADE_OFF_SET` 同时报 best-member 与 worst-member regret，跨 cell 聚合使用保守的 worst-member regret。对 generation cell，使用 validation 冻结的实用 margin 将各损失归一化，逐 member 计算到 final Pareto frontier 的最小 worst-coordinate shortfall，同样以 worst member 聚合。另报 validation 推荐集合对 final Pareto 集的 coverage、推荐集合 precision/size、harm-violation rate、`NO_RESOLVED_WINNER` rate、`INSUFFICIENT_SUPPORT` rate 和二者之和 abstention rate；regret/coverage/harm 指标均报告 training-seed/cluster-aware uncertainty，generation 另分离 Monte Carlo uncertainty。平均/最坏 regret 只在非弃权 cell 计算且必须与 abstention 同时呈现，禁止用高弃权率或无限放大推荐集合换取低 regret/高 coverage；
- 最终固定输出一张 choice atlas、一张 `Main Figure 1 System Pareto` 和一张 failure/abstention map；平均排行榜只能作为补充表。

发表潜力采用以下证据边界，不以“完成了多少训练”替代：

| 最终证据 | 允许的论文定位 |
|---|---|
| 只比较若干公开 checkpoint | 仅作参考性技术报告；模型差异不能归因于 tokenizer，发表潜力低 |
| 同一主干比较 NUC/k-mer/BPE | 有实证价值，但与 BEACON、GARNET、BiRNA-BERT 和既有 tokenizer 研究重叠较大 |
| 再加入同源隔离、合法且统一的核苷酸评价、fixed/random/entropy patch、exposure 与 common-FLOP 两种公平视图、记忆与生物保真评价 | 形成可辨识的 benchmark 边界，可支持领域 benchmark/resource 论文 |
| 再提供可复用数据 manifest、统一 evaluator、公开模型适配器、模型/配置与独立复现 | 可考虑 NeurIPS Datasets & Benchmarks、Bioinformatics、NAR Genomics and Bioinformatics 等 benchmark/resource 方向；不构成录用保证 |
| 进一步形成跨 seed、family-held-out、length OOD 和计算预算稳定的多变量选择图谱，或证明某类方法存在可迁移的系统性失效边界 | 才具备更高层次的科学结论，而不只是一个平均排行榜 |

允许的核心主张：

> TokBench-RNA evaluates three explicitly bounded quantities under controlled Flat Transformer and BLT backbones: static representation effects within Flat, causal patch-rule effects within BLT, and complete-system Pareto/deployment trade-offs across backbones and context budgets. Its validation-frozen choice atlas recommends or abstains across task × length × compute cells while enforcing memorization and RNA-proxy fidelity constraints.

该句是完成全部实验后才可能获准的 claim 模板，不是当前结果陈述。只有 updated collision matrix 通过、全部受控条件落地且 final test 支持时，才可改为现在时；只有最终检索仍未发现等价工作时，才允许增加限定的 “To our knowledge”。

## 4. 分阶段执行 TODO 与门控

| Phase | 阶段目标与主要任务 | 主要输出 | 验收门 | 并行与失败处理 |
|---|---|---|---|---|
| Phase 0：Goal、authority 与旧谱系重建 | fresh clone 空仓库；建立 `benchmark-v1`；写入完整 Goal；冻结上游代码与许可；重建旧审计 ledger；创建 artifact root 和 registry | Goal、authority manifest、legacy evidence ledger、source/license manifest、Git remote ref | Goal authority 可重读；旧结果全部为 `artifact_lineage_status=HISTORICAL_ONLY` 或 `ORPHAN_RESULT`；无训练、无 final-test access | authority 不闭合则 `BLOCKED_EXTERNAL_WITH_EVIDENCE` |
| Phase 1：数据 benchmark | 获取 release 22 与 23–26；canonicalization；exact dedup；同源聚类；Rfam 标注；cluster/family/clan/release-shift split；数据表和 datasheet | immutable dataset release、split manifests、QC、duplicate/homology clusters、leakage report、data sheet | 来源/license/accession 闭合；80/80 cluster 不跨 split；exact overlap 为零；final test 封存且访问日志为空 | 失败则禁止模型训练 |
| Phase 2：统一 evaluator 与 sealed 协议 | 实现 tokenizer specs、canonical coder、shared scorer、continuation/generation evaluator、external adapter schema、sealed-test gate 与 oracle fixtures | evaluator package、protocol YAML、metric fixtures、初版 claim–evidence matrix | coder/NLL、每 nt 一次计分、prefix causality、生成分母、homology fixtures 全 PASS；final test 未暴露 | 任一 headline 语义未闭合则正式训练锁定 |
| Phase 3：模型实现、HPO 与 GPU 校准 | 构建 Flat/BLT；实现 Track R、B1、Track D；B2 映射到 F7；用 tuning seed 101 做等预算 HPO；参数/FLOP/exposure 计数；按固定公式冻结 D_FLOP_BUDGET、common-FLOP grid、Track D 窗口/target schedule、推理成本重复测量协议与超参数 | resolved configs、parameter census、GPU smoke、frozen hyperparameters、tokenizer artifacts、P2/P3 balance plan、D budget/context/target report、cost-measurement protocol | train/eval/generate parity；GPU fallback=0；参数容差通过；B1 与 D1/D2 可运行；D long-context separation gate有明确 PASS/FAIL；target repetition fixture与cost repeat fixture通过；正式 seed 未启动；final test access=0 | 任一臂 OOM 时统一调整 cohort；不得为单臂降标准；context gate FAIL 时保留效率子赛道但删除 long-context claim |
| Phase 4：Track R 与 Bridge train/validation | Track R 10臂×3 seed、每 run 2B valid nt；B1×3 seed；只运行 train、validation 与未封存 diagnostics；关闭并冻结 fixed final-budget checkpoint，best-validation 仅另存 secondary 指针 | 30 个 Track R bundles、3 个 B1 bundles、validation-only summaries、失败报告 | Full benchmark 要求 33/33 formal run 完成；任一 arm 最终 FAIL 则 `G_TRACK_R_COMPLETE=FAIL`，不得把失败配置当完成；无换 seed；final-test access=0 | 失败后可继续不接触 final test 的恢复/资源工作，但进入 Phase 8 前必须修复，或由负责人以前瞻 amendment 明确降级项目范围 |
| Phase 5：Track D 与外部适配 train/validation | D1/D2 各3 seed按共同 FLOP 预算训练；只在 `D_long_context_validation_view` 上调参与冻结部署 selector；用共同 raw-target-position manifest 做 paired validation；外部模型完成 registry、overlap audit、adapter 和 validation/development scoring | 6 个 Track D bundles、D target/exposure reports、paired validation summaries、外部 registry、adapter tests、validation-only reference summaries | 6/6 Track D run 完成且D1/D2同累计 FLOP；train repetition count=1；validation targets逐位相同；raw context/exposure 可追溯；外部模型明确 comparable/reference/unavailable；final test access=0 | Track D formal run 缺失则 choice-atlas 部署主张不完整；外部不可用不阻断内部 benchmark，只降低 external claim |
| Phase 6：350M train/validation | 预先固定四臂×3 seed、每 run 7B valid nt；只使用 train/validation，不依据 100M final test 选臂 | 12 个规模复验 bundles、validation scale summaries | 四臂集合未改变；full benchmark 要求 12/12 run 完成；资源阻塞是证据化失败而非科学 PASS；final-test access=0 | 资源不足不得静默降 seed 或提前打开 test；进入 Phase 8 前必须补齐，或以前瞻 amendment 降级为 100M-only paper |
| Phase 7：机制、Atlas 与 final-unseal 冻结 | 只在 train/validation/未封存 diagnostics 做 boundary enrichment/swap/confound、分层、统计与 selector 拟合；冻结 checkpoint、decoder、query/prompt manifests及其seed、generation seed列表、cell、阈值、统计代码、图表模板、claim matrix 和 external adapter | `FINAL_UNSEAL_LOCK`、frozen selector、frozen analysis package、预填空表/空图模板、updated collision matrix | 所有 artifact 有不可变版本；selector 与 query/seed manifests 可在 validation 重放；prior-art gate 有逐工作 collision；final test access仍为0 | 未冻结任一项则不得进入 Phase 8 |
| Phase 8：一次性 final unseal、复现与投稿 | 同一事件对所有内部模型和可运行外部模型评分；生成 Main Table 1/2、Main Figure 1、External Table 1/2、atlas/regret/abstention；之后禁止调参；clean replay、发布和写作 | final score bundles、choice atlas、failure map、replay report、代码/manifest/model card/data sheet、论文草稿 | unseal 前访问日志为空；unseal 后无 selector/decoder/cell/checkpoint 变更；主表和 atlas 可重放；许可闭合 | 只允许从已冻结、完整保存的 logits/outputs 做不改变输入或 metric semantics 的 `FINAL_UNSEAL_RECOMPUTATION`；若修复改变 tokenizer、mask、model call、decoder、scorer 输入或指标语义，原 confirmatory test 立即作废，必须启用新的未暴露 confirmation split，或将修正结果降级为 post-unseal analysis；不得在同一 test 上重建 confirmatory claim |

### Phase 0 立即执行顺序

1. `P0-A Fresh authority preflight`

   - 输入：服务器、空 GitHub 仓库、本地旧证据。
   - 检查：host/user/port、现有 jobs、GPU UUID、磁盘、GitHub refs、目录冲突、旧 outputs 缺失状态。
   - 输出：只读 preflight manifest。
   - 验收：不杀进程、不改目录、不选择旧副本作为 authority。

2. `P0-B Repository bootstrap`

   - fresh clone 到代码根；
   - 创建 `benchmark-v1`；
   - 加入 `.gitignore`、Goal 目录、schema 目录、license/third-party 清单；
   - 禁止导入数据、权重、checkpoint、环境或缓存。

3. `P0-C Goal materialization`

   - 将本计划扩展为完整 Goal 正文；
   - 加入所有讨论、事实/推断/未知/待实验标签；
   - 生成 SHA-256；
   - 复制 exact-byte 本地镜像；
   - 验证两份正文完全一致。

4. `P0-D Legacy lineage regeneration`

   - 以新 run ID 重建上一轮报告、evidence ledger、run registry、dataset manifest、evaluation protocol 和 claim matrix；
   - 明确 `parent_delivery_status=MISSING`；
   - 不复用旧 checksum 作为当前文件证据；
   - 历史性能一律排除出新 benchmark 主表。

5. `P0-E Upstream and source audit`

   - 冻结官方 BLT commit；
   - 建立 flat backbone 来源；
   - 审计 tokenizer、MMseqs2、Infernal、ViennaRNA 和 external baseline 许可；
   - 只 port 通过测试的必要代码。

6. `P0-F First delivery closure`

   - 运行 Markdown/schema/hash/secret/large-file 检查；
   - focused commit；
   - push `benchmark-v1`；
   - 重读远端 ref，确认 SHA；
   - 本任务状态只有在 Goal、hash、remote SHA 全闭合后才能为 `TASK_CLOSED_PUSHED`。

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
- 每个 scientific arm 有三个冻结 seed；
- Track D 的 D1/D2 达到同一累计 `D_FLOP_BUDGET`（离散 overshoot 不超过 `0.5%`），并记录各自 raw-context、raw/valid-nt exposure 与 scheduler FLOP 进度；
- Track D capacity preflight 能用冻结 tokenizer 与 schedule 复算 `D_NOMINAL_FLOP_BUDGET`、`F_cap_D1/F_cap_D2` 和最终共同预算，并证明两臂都保留至少 `5%` 未消费 unique-target 容量；formal run 不发生第二个 pass；
- `D_long_context_train_view` 只含 train clusters，`D_long_context_validation_view` 只含 validation clusters；D1/D2 使用相同 seeded stratum-interleaved entity order 和冻结的 `2048 context slots + 2048 target-bearing slots` 窗口策略。每臂 primary target repetition histogram 必须全部为 `1`，已评分 target 的重载必须 loss-masked；跨 RNA packing 不计为 context；
- Track D validation/final 的 raw-target-position manifest 对 D1/D2 完全相同，每个目标碱基各计分一次；人工 multi-nt-token fixture 必须证明 `effective_raw_context_nt` 排除当前 token 内部碱基、padding 和跨 RNA packing，并按共同 raw targets 等权聚合；D2 median 未比 D1 高至少 `15%` 时自动撤销“long-context benefit”主张；
- Track D 暴露报告必须能复算 unique entity/cluster、length/source/RNA type/family、GC/low-complexity 与 valid-target 构成；达到 3.3 的暴露偏移门时，自动把上下文解释降级为冻结数据流下的系统效应；
- 固定 `10,000 attempted generations` 后，能从 raw output 独立重算 attempted/decoded/valid 分母、validity、conditional fidelity 和每个 accepted output 的成本；
- validation/final query manifests 可由 seeds `1201/1202` 重建，generation seeds 严格为 `[1101,1102,1103,1104,1105]`；相同 decoder tuple 只对应一个 10,000-attempt bundle；
- prefix 10%/25%/50% 共同切点均向下取整到 6 的倍数，所有模型看到完全相同 raw prefix，并报告实际比例；
- rolling scorer 使每个目标碱基恰好计分一次，且所有模型在 raw-controlled view 中看到相同最近 `4096 raw nt`；该 view 对 Track R/B1/350M 标 length OOD，对 Track D 只标 held-out-long-context sensitivity；
- 每个科研开关都有正反 fixture，证明开关实际改变对应计算；
- 同一 run checkpoint replay 的 logits 和指标在冻结数值容差内一致；
- GPU smoke 无 OOM/NaN，且 `cpu_fallback_count=0`；
- 每个最终分数闭合到 Goal、代码、数据、split、tokenizer、config、checkpoint、decoder、output 和 evaluator hash；
- `FINAL_UNSEAL_LOCK` 前 final-test access log 必须为空；unseal 后 checkpoint、decoder、strata、统计代码、selector、图表模板和 claim matrix 不得变化；
- post-unseal 重算 fixture 必须证明 `FINAL_UNSEAL_RECOMPUTATION` 只读取冻结 logits/outputs；任何改变模型调用、输入或 metric semantics 的修复不得继续使用同一 test 作为 confirmatory evidence；
- atlas selector fixture 必须能输出 `WINNER / TRADE_OFF_SET / NO_RESOLVED_WINNER / INSUFFICIENT_SUPPORT`，并可复算 regret、Pareto coverage、harm violation 与 abstention；
- Atlas primary cost fixture 必须证明端到端延迟计入 canonicalization、在线 tokenizer/patcher、packing/transfer、model 与 detokenization；model-only FLOPs、energy 或 accelerator latency 不能替换 primary cost；
- atlas 的 pairwise/graph fixture 必须覆盖 reference 全支配获胜、两个 resolved trade-off 节点、点估计占优但 CI 不足、singleton non-dominated 但非全支配、nontransitive cycle、harm uncertain、全体实际等价和全部候选不合格；只有逐 seed 区间规则满足时允许 `WINNER`，`TRADE_OFF_SET` 必须严格等于 frozen graph 的完整 robust non-dominated set；
- 分布 harm fixture 必须验证每个 metric 使用自己的冻结 `D_floor`，并覆盖 validation reference distances 全零时的 `τ_metric` fallback 与 `HARM_UNCERTAIN_ZERO_REFERENCE`；
- clean environment 能重放一个训练 smoke、一个推理和主结果表。

### 5.3 Final Goal 成功标准

项目成功要求：

- release-22-derived 数据 benchmark 可合法、可追溯、可重建；
- exact 和 80/80 homology leakage gate 通过；
- Track R 100M 十臂×三 seed（30 run）完整；
- B1 100M×三 seed（3 run）完整，且与复用 F7 的 B2 一起完成 F1/F7/B1/P1 固定 2×2 system decomposition；
- Track D 100M NUC/BPE×三 seed（6 run）在 train-cluster-only `D_long_context_train_view` 上以单次无放回 schedule 达到容量门允许的相同累计 FLOP budget，并完整报告 within-sequence effective raw context、unique target/entity/cluster exposure、暴露构成、在线 tokenizer 在内的端到端部署延迟和 packing throughput；long-context separation 或暴露构成门未通过时主动删除上下文收益 claim；
- 350M 四臂×三 seed 完整；
- 外部模型以 best-effort 方式形成可追溯 reference；不可运行只降低 external-comparison claim，不阻断内部 benchmark 成功；
- `canonical_code_length_BPN`、continuation、generation、效率、记忆和 RNA proxy 指标口径全部冻结并通过 fixture；
- confirmatory family-held-out 与 Track R/B1/350M 的 length OOD 两个泛化视图完成；Track D 对应结果标 `held_out_long_context`；database-release shift 在数据资格通过时作为固定 sensitivity，不能替代前两项；
- 所有主效应有逐 seed 值、效应量和 CI；
- 形成覆盖 `task × sequence length × compute budget` 的多变量 choice atlas；生物分辨率、validity、family recoverability、结构/分布 proxy 和 memorization 作为每个 cell 的约束/结果族；
- atlas 至少包含 exposure-indexed `R_ATLAS` 与 FLOP-scheduled `D_ATLAS` 两个不可混排面板；R_SYSTEM 推荐明确标为完整系统选择，不能被改写成纯 tokenizer 因果效应；
- atlas 对所有 eligible cells 给出推荐、trade-off、no resolved winner 或 insufficient support，并报告 final-test mean/worst regret、Pareto coverage、harm violation 和 abstention；平均排行榜不能替代这一交付；
- `FINAL_UNSEAL_LOCK` 前访问日志为空，所有内部模型与可运行外部模型在同一次 final unseal 评分，且 unseal 后无调参、selector/cell/阈值或 checkpoint 变更；
- 形成可复现代码、数据 accession/split manifest、模型 registry、evaluator、主表、选择图谱、失败地图、data sheet、model card 和论文草稿；
- 最终结论不依赖 entropy patching 获胜。

### 5.4 项目转向或终止条件

- release 22 来源、license、accession 或 hash 无法闭合：停止该数据锚点，等待负责人批准 release 26 新合同。
- shared scorer 或 BPN 语义无法通过 oracle：停止正式模型矩阵。
- exact/homology split 无法建立：停止训练，优先转为数据 benchmark 恢复。
- 训练、验证和生成路径无法保持相同 tokenizer/patch 语义：停止对应 arm。
- 服务器无法提供同一 GPU cohort 或 350M 三 seed 预算：标记资源阻塞，不把两 seed 结果写成论文级确认。
- external baseline 无法合法获取或运行：明确标记 `UNAVAILABLE_WITH_EVIDENCE`；不得用 PPT 数值替代本地复现；内部 benchmark 可继续，但删除严格 external-SOTA/超越主张。
- 如果 tokenizer 效应只来自参数量、raw context、训练数据、FLOPs 或解码搜索，不归因于 tokenizer。
- 如果所有 headline paired-effect CI 均完全落入预注册实际等价区间，且统计精度、evaluator 与 harm constraints 合格，只允许写“在三个预注册 training seeds、该预算和这两个受控 backbone 实现下，观察到的效应及 cluster-level uncertainty 均位于实际等价区间”；不得泛称 tokenizer families 在一般训练随机性下统计等价；
- 如果 CI 跨越实际意义的改善与恶化区间，必须标记 `INCONCLUSIVE_UNDERPOWERED`，不得把未显著写成等效或负结果；
- 如果平均差异存在但 task×length×compute 的交互效应不稳定，或 validation-frozen selector 在 final test 上 regret/harm/abstention 显示不可迁移，则不得宣称形成稳定的 tokenizer 选择规律；论文降级为受控 benchmark/resource 报告，并将 abstention/failure map 与不确定性作为主要结果；
- 如果投稿前出现等价且更完整的 benchmark，转向序列相似性感知 evaluator、release-22/database-release-shift 数据资源或机制复核；不继续使用“首个完整 benchmark”措辞。
- 如果无法完成数据、evaluator、100M 和 350M 的核心证据链，则停止 SOTA 和高水平 benchmark 论文主张，保留为可复现工程资源。

本计划受 primary-source prior-art 分级审查和 implementation-plan 分阶段验收原则约束：正式文献与预印本、公开代码与经验证科研结果分别建账；每个阶段必须有输入、输出、gate、测试、失败终态和 Git 交付闭环。

## 附录 A. Authority、机器接口与状态命名

### A.1 Goal 与代码 authority

Phase 0 正式创建：

- 权威文档：`/home/cunyuliu/tokenizer-benchmark/docs/goals/NCRNA_TOKENIZATION_SEGMENTATION_BENCHMARK_GOAL_V2.md`；
- 本地用户可见镜像：`/Users/liucunyu/Documents/Codex/2026-08-08/volumes-orico-disc-blt-blt-code-2/outputs/NCRNA_TOKENIZATION_SEGMENTATION_BENCHMARK_GOAL_V2.md`；
- 根目录 `GOAL.md` 只指向权威版本，不维护可独立漂移的第二份正文；
- 代码仓库目标：`git@github.com:Cunyu-Liu/tokenizer-benchmark.git`，首个工作分支 `benchmark-v1`；Phase 0 必须重新只读核验 remote refs、服务器目录、GPU 和磁盘，旧快照不构成当前资源授权；
- 新代码从冻结的官方 BLT、公开 Flat Transformer 和 tokenizer 上游逐模块引入并记录来源；`/Volumes/orico-disc/blt` 旧代码、checkpoint 与结果保持只读且仅作历史证据，不整目录复制为新项目起点。

Goal/contract 必须包含 scientific estimands、data/split、Track R/D/Bridge/350M、指标、统计、selector、sealed lifecycle、claim matrix、阶段 gate、失败分支和 amendment log。任何科学关键字段不得由临时命令行覆盖。

### A.2 权威机器接口

- `benchmark_contract.yaml`：Goal/代码/数据/split/tokenizer 版本，arm、track、model seed、tuning seed、patch randomization seed、参数、raw/slot context、valid/raw-nt 与 FLOP 预算、checkpoint policy、decoder、scoring、sealed phase和输出根。
- `dataset_manifest.json` 与 sequence-level Parquet：accession、raw/canonical sequence identity、release、RNA type、taxonomy、Rfam family/clan、homology cluster、canonical entity、multiplicity metadata、排除原因和 split membership。
- `tokenizer_spec.yaml`：tokenizer 类型、trainer seed/config、vocab、k/stride/tail rule、offset、train-only corpus version、工具版本、模型文件、special tokens、round-trip、prefix、canonical path 与 slot-to-raw-context 规则；同一 tokenizer artifact 供三个 formal model seeds 共用。
- `run_manifest.json`：run/parent ID、track/arm、Goal/source/data/split/tokenizer/config 版本、`tuning_seed/model_seed/patch_randomization_seed/generation_seed`、GPU、参数、raw/valid exposure、FLOPs、raw/slot context、checkpoint/output、final-test access state、运行状态和失败原因。
- `ScoreSums`：至少保存 `coded_bits_sum、quantized_cdf_nll_bits_sum、canonical_nll_bits_sum、next_base_nll_bits_sum、valid_nt_count、eos_nll_sum、sequence_count、attempted_count、decoded_count、valid_count、invalid_count、early_eos_count、truncation_count、overshoot_count`，禁止只保存平均 scalar。
- `RNAARAdapter`：所有模型统一实现 canonicalization、encode/decode、prefix preparation、forward、canonical coding/scoring 和 generation；共享 scorer、mask builder 与 artifact writer。
- `atlas_selector.yaml`：candidate/applicability mask、task/length/compute cells、support thresholds、equivalence/harm margins、reference、tie/abstention、regret/Pareto/harm 计算和冻结版本。
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

## 附录 B. GPU、监控与恢复 SOP

所有神经训练、验证、推理、生成和神经 evaluator 均为 GPU-only，并记录 GPU physical index/UUID/model、driver/CUDA/PyTorch、model/input/output device、forward/backward、peak VRAM、PID、命令、run ID、日志/checkpoint 路径和 `cpu_fallback_count=0`。

- 启动约 2 分钟和 5 分钟各检查一次；稳定后每 30 分钟只读检查最新 stage、指标尾部和资源快照；
- NaN、Inf、OOM、wrong-device、CPU fallback、磁盘安全线、checkpoint 损坏、PID 所有权异常或 final-test 非授权访问触发安全停止；优先 graceful stop，保留最后完整 checkpoint，不杀无关进程；
- Phase 3 校准前正式 GPU job 串行；校准后 100M 最多两个独占 GPU job 并行；GPU 不共享；350M 一次只运行一个 matched cohort（注：2026-08-15 负责人批准操作偏差——使用全部可用 GPU0-5 推进 100M 矩阵，最多 6 并行，详见 Amendment Log）；
- 同时最多一个未限流的大型数据扫描；GPU 等待期可做文档、代码测试、许可、external adapter 和 validation 分析，但不得读取 final test、改变冻结 decoder 或跳过 phase gate；
- 一个 root cause 只允许一次具有明确差异的 corrected retry；再次发生时以 evidence 关闭为 FAIL/BLOCKED，不无限重试。

## 附录 C. Artifact、目录与 Git SOP

大型 artifact 根为 `/mnt/cunyuliu/tokenizer-benchmark`，按 `data/raw`、`data/derived`、`tokenizers`、`weights/reference`、`runs/<run_id>`、`checkpoints/<run_id>`、`manifests/registry`、`tmp/cache` 分离；不从 `/Volumes` 直接训练，不用可变 `latest` 作为权威引用，不删除源 artifact，一个 run ID 只创建一次。

Git 生命周期：从 `benchmark-v1` 建立 `codex/tokenizer-benchmark/<phase>-<task>-<run_id>`，只 stage 本任务文件，运行相关测试与大文件检查，focused commit 后 push task branch；gate 通过后 fast-forward 到 `benchmark-v1` 并重读远端 ref。禁止 force-push、重写历史、自动覆盖 `main` 或上传数据、权重、checkpoint、生成全集与缓存。训练 heartbeat 不产生提交；只在正式 manifest、协议、代码或阶段报告变化时提交。

---

*本文件（Goal V2）是 TokBench-RNA 项目的唯一权威科学与工程合同。不得擅自削弱、降低门槛或跳过阶段，除非通过 amendment 并获负责人批准。*
