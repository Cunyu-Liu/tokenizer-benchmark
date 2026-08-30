# 辩论与重设计决策记录:BLT 架构接受度与外部参考 Registry

- **日期**:2026-08-30
- **作者**:刘存宇(owner)+ 审计顾问
- **状态**:APPROVED_BY_OWNER(本记录签署即批准;时间戳 2026-08-30)
- **性质**:evidence 型 amendment 的决策依据(不改变 33-run 门禁与验收阈值)

## 1. 质疑与裁决

**质疑**:用 BLT 动态分段架构与普通 Flat 架构对比,是否会因 BLT 不被学界广泛认可而被评审拒绝?是否应改用 PatchDNA / H-Net / DNAChunker / MergeDNA 等动态分段工作所使用的架构?

**裁决**:
1. **保持 33-run 核心矩阵不变**;已完成/在跑的 run(16 DONE + F6-s29 在跑约 95%,合计 17/33)继续作为核心证据。
2. BLT 是“实现手段”,不是“论文主张”。主张限定为两层:(a) 同骨干(BLT 系内)受控比较 fixed / matched-random / causal-entropy **分段规则**;(b) 跨骨干只做完整系统 Pareto。
3. “动态分段”不存在唯一公认架构,而是两大家族(见 §2);设计选择应公开声明为“以 BLT 作为该抽象的可复现实例”,并以同构先例(PatchDNA)佐证。

## 2. 外部方法架构事实(2026-08-30 核实)

| 方法 | 架构族 | 骨干/机制 | 出处 | 与 P 赛道同构? |
|---|---|---|---|---|
| PatchDNA | blt_like | BLT 三段式骨干(局部编码器+潜变量 Transformer+局部解码器),仅替换打 patch 规则(PhyloP 保守性/entropy)+ re-patching | bioRxiv 2025-11-28;ICLR 2026(OpenReview AFZeojzjoG) | **是**:同一骨干只变 patch 规则,与我们 P1–P3 同一抽象层次 |
| MergeDNA | blt_like(学习式) | 可学习 token merging 局部编码器 + 潜变量 Transformer + 重建(ToMe 式) | arXiv 2511.14806(2025-11-17) | 部分:潜变量系统,segmentation 为可学习模块 |
| H-Net | hnet_like | 可微分动态 chunk 边界预测 + U-Net 型层级,无词表、端到端 | arXiv 2507.07955;goombalab/hnet + Cartesia 权重 | 否:换一种完整架构 |
| dnaHNet | hnet_like | 基因组版:可微分递归 chunking + U-Net 层级,微生物基因组预训练 | **ICML 2026 Spotlight**(OpenReview 6pN2KNCspk;订正原“ICLR 2026 poster”出处) | 否 |
| DNAChunker | flat + learnable seg | 掩码 DNA LM 上可学习自适应分段(效率导向,RL) | arXiv 2601.03019 | 否 |

## 3. 控制变量铁律(基准设计原则)

- 共同骨干内:比“分段/分词**规则**”(F/P 赛道内)。
- 共同口径下:比“完整系统 Pareto”(跨骨干,含 B1 桥接 2×2,仅系统级结论)。
- 外部模型:只能作为 **Phase 8 external registry** 的 reference 面板;不可在我们的统一数据/核苷酸暴露/FLOP 口径下复训,不进入主表归因与 Decision Map。

## 4. 对评审攻击的防御要点

1. “BLT 不可接受”→ 反驳:受控规则效应与架构系统效应已分离(§3.2 三层结论边界);BLT 为公开、可复现(ACL 2025,权重开源)实例;同构先例 = PatchDNA(ICLR 2026)。
2. “为什么不换 H-Net/MergeDNA”→ 反驳:换架构即引入新的端到端系统混杂,违反控制变量;若需架构无关性证据,后续可选 Track L2(同 Flat 骨干动态分段输入 pilot,不入 33 核心,另行 amendment)。
3. “用外部模型更公平”→ 反驳:外部权重自带固定规模/数据/目标,不可统一口径复训;v3 matrix 与 registry 将其如实标注为 reference-only。

## 5. 产出物

- claims/collision_matrix_v3_20260830.json(架构族 + 两大家族概览)
- claims/external_registry_v1_20260830.json(Phase 8 参考系统注册)
- Goal V3 Amendment Log 追加行;SHA-256 重算并同步 GOAL.md 与 AUTHORITY
- PPT 第 4/6/13 页叙事与来源更新
