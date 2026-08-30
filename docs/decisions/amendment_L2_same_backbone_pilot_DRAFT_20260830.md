# Amendment Draft（草案）:Track L2 — 同骨干动态分段输入 pilot

- **类型**:PROSPECTIVE_AMENDMENT_DRAFT(未批准;批准后方生效)
- **日期**:2026-08-30
- **作者**:刘存宇(owner)
- **状态**:PENDING_OWNER_APPROVAL
- **关联**:
  - 决策记录:`docs/decisions/20260830_blt_framing_and_external_registry.md`
  - 任务队列登记:`docs/data/track_l2_same_backbone_pilot_registration_20260830.json`
  - 治理依据:Goal V3 §1.6 / §3.2.2 变长分段登记范式、附录 B(单 GPU 约束)、§4 Amendment 规则

> 本草案是"前瞻性 amendment":批准后新增独立增强实验 Track L2,**不修改 33-run 核心的门禁、臂、种子、预算与验收阈值**。批准文本(§9)将作为 Goal V3 Amendment Log 的非门禁行追加(按 V3 哈希流程)。

---

## 1. 结论(本 amendment 做什么)

在**同一条 Flat Transformer 骨干**上,把输入侧从"静态词表(NUC)"切换为"变长 patch + 可学习 patch 编码器(动态分段输入)",比较分段方式本身带来的效应,并与 33-run 中 BLT 系(P3 熵规则 vs P1/P2)的规则效应做**方向性对照**。

- 用途:斩断"结论是否只是 BLT 架构特异"的评审质疑,产出**架构无关性敏感性证据**。
- 用途边界:不进入 R_FLAT / R_BLT 主比较家族;不单独作为"动态分段普遍优于静态"的正式主张;不阻塞核心论文。

## 2. 可检验的科学问题与允许/禁止主张

- 问题:在同一骨干、同一数据/暴露/FLOP 口径下,静态 NUC 输入 vs 熵驱动动态分段输入,codec BPN 与 true-suffix continuation 码长是否出现方向稳定差异?
- **允许主张**(敏感性/诊断级别):
  1. "在本骨架内,熵分段输入的效应方向与 BLT 系 P3-vs-基线 的方向一致/不一致(validation 方向性检查;final 数值 Phase 8 解封后给出)"。
  2. "该骨架内固定 patch 与随机边界对照用于分离'规则'与'动态性'的贡献"。
- **禁止主张**:
  1. 以 L2 代替 33-run 的受控规则效应或完整系统 Pareto 结论;
  2. 声称 L2 结果证明了与 BLT 以外的动态分段系统(如 H-Net 系)的可比性;
  3. 在 Phase 8 前引用 final sealed-test 数值。

## 3. 实验设计(两阶段)

### Stage A · 开发 pilot(先跑,用于可行性与参数冻结)

- 规模:每臂 **1 run**,使用独立调参 seed `101`(不续训为正式 run);预算暂定 **500M valid-nt/run**(可行性验证,可低于正式口径)。
- 臂:`L2-NUC`(静态对照,即 F1 基线复用)、`L2-fixed-patch`、`L2-random-patch`、`L2-entropy-patch`。
- 目的:冻结 patch 编码器结构/初始化、entropy 阈值校准(mean patch length=6,train-only)、random 边界分布拟合、FLOP 与参数审计脚本。

### Stage B · 正式(门通过后,owner 二次确认 run 数与资源窗口再启动)

- 规模:4 臂 × 3 seeds(17/29/43)= **12 runs**;预算 **2.0B valid-nt/run**(与核心一致)。
- 触发条件:Stage A 无阻断性失败;owner 书面确认资源窗口(受 2026-08-28 单训练 GPU 约束);平衡门审计通过(§5)。
- 若资源不足:保留 **pilot-only(REFERENCE_ONLY)** 交付敏感性结论,不强行跑 Stage B。

## 4. 骨干与口径(控制变量)

- **骨干**:复用 F1 的 Flat 骨干(相同层数/宽度/注意力/位置编码、`context_nt`、训练序列顺序与有效核苷酸暴露)。
- **输入表示**:
  - 静态对照 L2-NUC = F1(单碱基 token),零新增参数。
  - patch 臂:以边界向量(BLT 式 PatchPolicy 抽象,复用 `model/patch.py`)把序列切为变长 patch;每个 patch 经**可学习 patch 编码器**映射为 token 表示,再进入与 F1 完全相同的 Transformer 主干。
- **patch 编码器约束**:
  - 因果:boundary 与编码只依赖当前+历史前缀,不得读取未来长度(prefix causality);
  - 参数与 FLOP **单独记账**并报告;主干参数保持与 F1 逐项一致;
  - 最终序列须可无损还原(复用 canonical codec 往返校验);
  - 结构建议(草案默认):轻量 1–2 层因果小网络 + patch-length 位置编码;若 Stage A 表明更优结构,须在 Stage A 报告并冻结,不得静默更换。
- **口径**:相同 2.0B(Stage B)/500M(Stage A)valid-nt 暴露预算;共同累计 FLOP 检查点选 best-ckpt(validation);冻结 decoder 与 query manifest(seed 1201/1202 规则);evaluator 全部复用。
- **数据**:同 33-run 的训练/验证切分(不动封存测试);不上长序列视图。

## 5. 平衡与控制(沿用 P2/P3 机制)

- entropy 阈值:只在 train split 校准到 mean patch length=6;冻结 estimator/阈值/最大 patch 长度。
- `L2-random-patch`:q(boundary | causal prefix-length, causal entropy) 只在 train-only 的 L2-entropy 标签上拟合;supported strata(0.05≤q≤0.95)内采样、非支持层精确 replay L2-entropy;三 seed 共用同一冻结随机规则(沿用 patch_randomization_seed 思路,独立于模型 seed)。
- 审计门(Stage A 全程可执行,Stage B 正式生效):总 patch 数相对误差 ≤0.5%、预注册长度分层 ≤2%、实测 FLOPs 差 ≤5%、supported strata 覆盖 ≥80% 训练位置与 L2-entropy 边界。
- 在 L2 内,"熵 vs 随机 vs 固定"与"静态 NUC"分开解释,禁止混称。

## 6. 评估与证据边界

- 指标:actual canonical codec BPN(headline)与 true-suffix continuation 码长(10/25/50%,raw-nt 切点规则同 §3.7);另报告端到端延迟/吞吐与 patch 编码器 FLOP。
- **证据分级**:
  - Phase 8 前:仅用 **validation** 做方向性对照(L2-entropy vs L2-NUC 与 P3-vs-F1 方向一致性),状态标为 SENSITIVITY;
  - final 数值:Phase 8 一次性解封后给出,且不改变 33-run 的主表。
- final_test_access_count == 0(Phase 0–7)。

## 7. 统计与响应定义

- 主响应:同骨干下 L2-entropy − L2-NUC 的 BPN 差(带跨 seed 配对 CI),与 BLT 系(P3 − P1/P2)方向对照。
- 方向一致 → 支持"规则效应非 BLT 架构特异";方向不一致/无差异 → 如实报告,并把该不确定性写进论文 limitations。
- 复合假设不成立时的预注册处理:不做 post-hoc 追认;区间等价走 abstention 表述。

## 8. 资源与时间(受单 GPU 约束)

- Stage A:4 runs × 500M ≈ 数天/张卡(空窗期串行执行,不打断核心排空后的单任务调度)。
- Stage B:12 runs × ~6–7 天 ≈ 2 个月量级(纯串行估算;可按资源窗口分批,但任何时刻 ≤1 个训练进程)。
- 优先级:Stage B 明确排在 33/33 相关决策(Phase 4-G)之后,依据 4-G 的资源账目决定是否启动。

## 9. 批准后追加到 Goal V3 Amendment Log 的文案(候选)

```text
| 2026-08-30 | §1.6/3.2.2/4 | Track L2 same-backbone dynamic-input pilot:新增独立增强实验 L2(同一条 Flat 骨干,静态 NUC vs 变长 patch 输入;Stage A pilot seed101 + 可选 Stage B 3-seeds×4臂),用作架构无关性敏感性证据,不进 R_FLAT/R_BLT 主比较,不改变 33-run 门禁;执行遵守单 GPU 约束与 final access=0 | 草案 docs/decisions/amendment_L2_same_backbone_pilot_DRAFT_20260830.md;任务队列登记 docs/data/track_l2_same_backbone_pilot_registration_20260830.json |
```

## 10. 开放问题(执行前由 owner 拍板)

1. Stage B 是否启动:取决于 4-G 资源账目与单 GPU 时间窗口(12 runs ≈ 2 个月串行)。
2. patch 编码器结构:采纳草案默认轻量因果编码器,或依据 Stage A 结果调整(调整须在 Stage A 冻结并写入本 amendment 修订)。
3. Stage A 预算 500M valid-nt 是否合适:可调,但任何调整均需在批准文本中注明。
4. 是否需要把 L2 加入 Phase 8 external/内部统一评分:倾向"加入内部评分清单",以便 final 数值一次性给出。