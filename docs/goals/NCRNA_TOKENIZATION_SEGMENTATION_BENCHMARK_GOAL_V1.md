# TokBench-RNA Goal Document V1

## Authority

- **Document**: `NCRNA_TOKENIZATION_SEGMENTATION_BENCHMARK_GOAL_V1.md`
- **Version**: V1
- **Status**: ACTIVE
- **SHA-256**: (computed at materialization)
- **Revision rule**: No silent gate modification. Any change to science, acceptance criteria, thresholds, or phase gating requires an amendment log entry and either owner approval or a new Goal version. Bug fixes that do not alter gates/SHA may be committed as amendments with evidence.

---

## 1. Background: Legacy BLT Project Evidence Boundary

The legacy BLT project proposed a new RNA tokenization method and aimed at generation SOTA. This Goal recognizes that the legacy evidence **cannot be inherited** into the new benchmark. Recorded facts:

- Legacy training data lineage is not closed; at least original test records are present inside the currently visible cleaned-train artifact.
- Candidate legacy code mixes explicit n-gram lookup, explicit n-gram training, generation, and PPL paths inconsistently; explicit n-gram training may degenerate to default IDs, generation used all-zero placeholders, and PPL disabled that channel.
- Hash byte-groups, explicit lookup n-gram, and entropy patch were mixed; legacy results cannot be attributed to any single mechanism.
- H100/H800/no-ngram/ngram results cannot be fully closed to a unique `code → data → config → checkpoint → decoder → output → evaluator` chain.
- The prior final report and ledger are **missing** from the current local `outputs/`. Status recorded as `MISSING_OLD_DELIVERY`. Adopted policy: `REGENERATE_NEW_LINEAGE`. Old bytes are not pretended to be restored.
- Legacy checkpoints, generated files, and slide-deck numbers are historical/forensic evidence only. They are excluded from benchmark main tables, model selection, and prior-win judgments.

---

## 2. Project Turnaround Decision

The project turns from "propose a new RNA tokenization method and chase generation SOTA" to:

> Build a homology-isolated, data-consistent, backbone-controlled, exposure-controlled, compute-accountable ncRNA autoregressive modeling and generation benchmark, systematically comparing static tokenization vs dynamic segmentation/patching, and judge the true trade-offs across likelihood, compute efficiency, continuation quality, memorization risk, family validity, and structural distribution fidelity.

Project name: **TokBench-RNA: A Homology-Aware Benchmark of Tokenization and Segmentation for Autoregressive ncRNA Modeling and Generation**.

Success does **not** require entropy patching to win, nor a new architecture. A strictly-reproduced negative result, a task-dependent tokenizer ranking, or evidence of "no globally optimal tokenizer" is a valid benchmark finding.

---

## 3. Novelty Boundary (Second-Round Literature Review)

- **BEACON** (NeurIPS 2024 D&B) already compares single-nucleotide, BPE, overlapping 6-mer, non-overlapping 6-mer in RNA encoder models across 13 downstream tasks.
- **GARNET** (Nat Commun 2024) already compares single-base, overlapping dinucleotide, overlapping trinucleotide in RNA autoregressive generation, reporting trinucleotide better but with not fully matched training steps/optimization.
- **PatchDNA** (OpenReview) already compares conservation patch, entropy patch, fixed patch in DNA. "First entropy patching for nucleic acids" is not supported.
- **Zero-shot RNA LM benchmarking** (BIB, bbag098) uniform zero-shot evaluation of 21 RNA models across structure/classification/mutation fitness, with TS-Hard, RfamSample, ArchiveII-Nr, RNAGym subsets.
- **GenerRNA** provides RNAcentral release 22, BPE-1024, 350M decoder public reference; updated vs historical weights must be frozen and distinguished.

Conclusions:
- `G4_BROAD_NOVELTY = FAIL`.
- Forbidden claims: "first RNA tokenizer benchmark", "no tokenization innovation in biological sequences", "first RNA dynamic tokenization", "first nucleic-acid entropy patching".
- `G4_NARROW_BENCHMARK = CONDITIONAL_PASS`. The narrow verifiable gap: controlled comparison of static tokenization vs dynamic patching for broad, multi-family ncRNA autoregressive modeling and generation under common data, homology-isolated splits, unified budget, and nucleotide-normalized evaluation.
- Before submission, a fresh systematic search is mandatory. If new work fully covers this intersection, pivot to evaluator/data resource, mechanism review, or negative-result paper; do not maintain "first" by renaming.

---

## 4. Final Goal, Core Estimand, Allowed / Forbidden Claims

### Final Goal
Provide a controlled, homology-aware benchmark of tokenization and segmentation choices for broad ncRNA causal language modeling and generation under common data, homology-aware splits, raw-nucleotide exposure, and measured compute.

### Core Estimand
`canonical_path_BPN` and `next_base_BPN` (see §8) over sealed, homology-isolated test sets, plus prefix continuation quality, generation novelty/validity, family recoverability, structural distribution fidelity, memorization risk, and wall-clock/throuthput/memory efficiency under common budgets.

### Allowed Core Claim
> We provide a controlled benchmark of tokenization and segmentation choices for broad ncRNA causal language modeling and generation under common data, homology-aware splits, raw-nucleotide exposure and measured compute, jointly evaluating likelihood, continuation, memorization, family validity and structural distribution fidelity.

"To our knowledge" is allowed only if final search finds no equivalent work and all controlled conditions actually hold.

### Forbidden Claims
- "first RNA tokenizer benchmark", "first RNA dynamic tokenization", "first nucleic-acid entropy patching", "no tokenizing innovation in bio sequences".
- Any SOTA claim that mixes A/B/C tables into one ranking.
- Any attribution of tokenizer effect to parameter count, raw context, training data, FLOPs, or decoding search.
- MFE, CM hit, pairwise prediction, and embedding scores presented as real function, wet-lab validation, or natural-RNA proof (these are computational proxies only; must be labeled `predicted/internal proxy`).

---

## 5. Data Release, Cleaning, Homology Clustering, Split, Sealed Test

### Primary training data
- RNAcentral release 22 (official or verifiable archive).
- If release 22 raw snapshot cannot be legally/fully recovered, GenerRNA release-22-derived data may be used as a separately-labeled reconstructed version, **not** claimed equal to the official raw snapshot.
- If neither can close accession+license+hash, `G_DATA_ANCHOR = FAIL`; do not substitute unknown local legacy data.

### Primary data rules
- Case normalization; `T → U`.
- Primary alphabet `A/C/G/U`.
- Non-ACGU IUPAC records stay in the QC ledger, excluded from primary training; separate ambiguity stress subset.
- Primary training length `16–4096 nt`; `4097–16384 nt` is length-OOD; longer sequences are descriptive resource statistics only unless a future Goal version authorizes.
- RNA is directional; reverse complement is not silently treated as identical; separately report reverse-complement neighbors.
- Canonical exact duplicates merge to one training entity, preserving full accession/metadata mapping.
- Zero canonical exact overlap across train/validation/test.

### Homology isolation
- Primary cluster: MMseqs2 `80% identity / 80% query-and-target coverage`.
- Sensitivity: `90%/90%`.
- No same primary cluster across splits.
- Freeze tool version, parameters, cluster membership, all hashes.
- Do not infer "already clustered" from filenames.

### Split
- After family/clan held-out assignment, remaining homology clusters split by stable hash seed `20260808` into `98% train / 1% validation / 1% cluster-held-out test`.
- Stratify at least by length bin, RNA type, source database, Rfam annotation status.
- Eligible Rfam family: ≥100 cleaned sequences and ≥10 homology clusters.
- 10% of eligible families to family-validation, 10% to family-test.
- Families with clans get a full clan-held-out sensitivity split.
- All homology clusters of family/clan test are removed from training.
- Release 23–26 temporal OOD: only accessions new relative to release 22 and exact/80-80-homology-isolated from release-22 train.
- Primary test, family test, clan test, temporal OOD are all **sealed tests**.

### Reporting
- Train with RNAcentral natural frequency distribution.
- Report natural-frequency micro average, family-balanced macro average, and RNA type/source/length stratified results.
- Do not let one overall mean mask rRNA or high-frequency family dominance.

---

## 6. Experiment Matrices

### 6.1 100M ten-arm matrix
Three training seeds per arm → 30 scientific runs.

| ID | Backbone | Representation / segmentation | Core purpose |
|---|---|---|---|
| F1 | Flat causal Transformer | single-nucleotide NUC | static main baseline |
| F2 | same Flat backbone | BPE, vocab 1024 | GenerRNA-aligned static subword baseline |
| F3 | same Flat backbone | Unigram, vocab 1024 | exclude gains due only to BPE merge algorithm |
| F4 | same Flat backbone | overlapping 3-mer, stride 1 | GARNET-style local representation |
| F5 | same Flat backbone | overlapping 6-mer, stride 1 | BEACON-common 6-mer |
| F6 | same Flat backbone | non-overlapping 3-mer, stride 3 | distinguish overlap vs block tokens |
| F7 | same Flat backbone | non-overlapping 6-mer, stride 6 | k and stride sensitivity |
| P1 | same BLT backbone | fixed-length patch | patch-level architecture baseline |
| P2 | same BLT backbone | length-distribution-matched random patch | is boundary position valuable per se |
| P3 | same BLT backbone | causal entropy patch | is entropy boundary better than fixed/random |

Constraints:
- F1–F7 share layers, width, attention impl, positional encoding, context_nt, training sequence order.
- Do not match params by adjusting backbone depth/width.
- Use factorized/tied embeddings for larger vocab so total trainable params fall in common tolerance.
- P1–P3 use identical BLT params and modules.
- BLT explicit lookup n-gram, hash byte-groups, and legacy dummy n-gram all OFF; tests confirm zero corresponding params.
- P1 fixed length from train-only entropy calibration mean patch length.
- P2 samples from the same train-only entropy calibration full patch-length empirical distribution.
- P2 random boundaries determined by `sequence_id + seed`, prefix-consistent.
- P3 entropy predictor trained only on train split; record separate param count, training FLOPs, checkpoint hash, online/offline cost.
- Static track explains tokenization effect; patch track explains patch-rule effect; F1-vs-P3 cross-backbone is architecture/system comparison only, not a pure tokenizer effect.

### 6.2 350M replication
Four fixed arms:
- C1: NUC flat.
- C2: BPE-1024 flat.
- C3: fixed-patch BLT.
- C4: entropy-patch BLT.

Seeds fixed `17, 29, 43` → 12 scientific runs.
- 350M replicates NUC-vs-BPE and entropy-vs-fixed scale trends.
- "Causal value of entropy boundary position vs random boundary" remains carried by 100M P2/P3 evidence.
- Do not replace the four 350M arms post-hoc based on 100M test/generation results.
- 350M may launch only after 100M data, execution, and evaluator gates pass; it does not require entropy to win at 100M.

---

## 7. External Benchmark Registry and Horizontal Models

Main results split into three tables, NOT merged into one SOTA ranking.

**A. Internal controlled causal table** — F1–F7; P1–P3; 100M and 350M separately; interpret tokenizer/patch-rule effect only within track.

**B. Public autoregressive/generation reference table** — prioritize and run:
- GenerRNA `model_updated.pt`; GenerRNA historical `model.pt`;
- GARNET public GPT checkpoint;
- EVA 145M and 437M (when weights/license/code runnable) as external single-nucleotide reference;
- Public models with mismatched data/arch/budget are `ecological reference` only, no tokenizer causal attribution.

**C. family/structure-conditioned table** — RfamGen; RNAgg; compare only on same Rfam family subsets; not mixed with broad unconditional generation.

Secondary benchmark: GARNET 16S/23S and 231-family continuation; RfamSample family recoverability; ArchiveII-Nr structure-stratified diagnostics; RNAGym mutation likelihood/fitness; TS-Hard only if a compatible frozen structure probe is defined; BEACON as prior-art/optional representation diagnostic (not all 13 tasks expanded into main project).

Every external model must record: model ID/revision; paper/preprint status; checkpoint SHA-256; code commit; tokenizer; training data and possible overlap; license; GPU runtime; decoder; evaluator adapter; status `STRICTLY_COMPARABLE / REFERENCE_ONLY / UNAVAILABLE_WITH_EVIDENCE`.

---

## 8. Metric Definitions

### 8.1 Unified likelihood
Two explicitly distinguished metrics:

1. `canonical_path_BPN` — cumulative token NLL over the deterministic, lossless tokenizer's canonical token sequence, divided by original canonical nucleotide count. Compares compression efficiency of the encoding path; NOT equal to string probability marginalized over all tokenization paths.

2. `next_base_BPN` —
\[
-\frac{\sum_i\log_2 p(x_i\mid x_{<i})}{N_{\mathrm{canonical\ nt}}}
\]
Only for models giving exact per-base conditional distributions. Each real nucleotide scored exactly once.

Rules:
- BOS, padding, packing separator excluded from numerator and denominator.
- EOS reported separately as `EOS_NLL`, not mixed into `next_base_BPN`.
- Dataset BPN uses total NLL / total nucleotides, not unweighted mean of sequence BPN.
- BPE/Unigram without exact prefix marginalization report only `canonical_path_BPN`, never labeled `next_base_BPN`.
- Overlapping k-mer adds one new nucleotide per step; overlap generation permits only four legal transitions consistent with existing suffix; report probability mass assigned by model to illegal transitions.
- Full-vocabulary overlap path scored separately as `overlap_path_BPN`.
- Spot-check canonical-path results with an actual arithmetic coder; if cross-entropy disagrees with decodable bitstream, do not use "compression" wording.
- Token PPL only as training diagnostic within same tokenizer; never in cross-tokenizer ranking.

### 8.2 Prefix continuation and generation protocol
Primary continuation:
- From sealed-test real sequences build `10% / 25% / 50%` prefixes (raw-nucleotide defined, not token count).
- Tokenizer independently encodes the observed prefix; BPE tokens must not cross observed/hidden suffix boundary.
- Same target raw length to all models.
- If last token overshoots target length, keep original full output, also save a fixed-length evaluation view, record truncation.
- Report suffix edit distance, nucleotide accuracy, k-mer recovery, Rfam family/clan recoverability, CM bit score, structural proxy bias, nearest training neighbors.

Decoder protocol:
- Validation-only grid: temperature `{0.7,0.9,1.1}` × top-p `{0.90,0.95,1.00}`, top-k fixed 0, plus greedy.
- Same query count and seed per grid cell.
- Main table reports three preregistered common points: `conservative=(0.8,0.95)`, `balanced=(1.0,0.95)`, `exploratory=(1.1,1.0)`.
- May select one Pareto point per model on validation with the same search budget; freeze before test.
- Test executes frozen settings only.
- Each training seed generates ≥10,000 valid sequences under each final condition; five generation seeds, ≥2,000 valid sequences each.
- Training seed is an independent model replicate; 10,000 generated sequences do NOT masquerade as 10,000 independent model replicates.

Generation metrics: exact uniqueness; exact train/val/test memorization; 100%/90%/80% identity novelty with alignment coverage; cluster-level uniqueness; nearest-training-neighbor identity/coverage; Rfam 15.1 `cmscan --cut_ga` family/clan hit coverage and bit score; family/clan coverage; length/GC/dinucleotide/k-mer distributions; ViennaRNA MFE, ensemble diversity, paired fraction; validity rate, illegal-char rate, EOS completeness rate, truncation/overshoot; memorization–validity Pareto; 95% homology-cluster bootstrap CI.

MFE, CM hit, predicted pairing, embedding score are computational proxies only.

---

## 9. Parameter, Exposure, FLOPs, Context, Hardware Fairness

- 100M target params `98M–102M`; 350M target `343M–357M`.
- Same-track matched pairs: backbone non-embedding params identical.
- Total trainable params differ ≤2%; vocab/embedding params listed separately.
- 100M main budget: `2.0B cumulative_valid_target_nt` per run.
- 350M main budget: `7.0B cumulative_valid_target_nt` per run.
- Primary context: `4096 canonical nucleotides`.
- Scheduler/warmup/checkpoint/stop keyed on valid target nucleotide count, not optimizer step or tokenizer token count.
- All arms use same raw sequence order; padding/BOS/separator/overlap repeat windows do not add nucleotide exposure.
- Main scientific track is data-matched.
- Same trajectory also reports a compute-performance curve on common cumulative-FLOP checkpoints, not represented as separately-tuned compute-matched training.
- Entropy tokenizer/patcher build and online cost are not zero.
- Same hardware cohort required before comparing wall-clock/throughput/VRAM; H100/H800/different-A100 cohorts cannot be mixed directly for speed.

Hyperparameters:
- Each arm may tune with identical budget on train/validation only.
- Fixed candidates: base LR `0.5×,1×,2×`; other optimizer settings identical.
- 100M base LR init `3e-4`; 350M init `2e-4`.
- AdamW `β=(0.9,0.95)`, weight decay `0.1`, bf16.
- Each candidate seed 17, ≤100M valid target nt.
- Select by validation metric then freeze.
- No extra tuning budget for failed arms; no test/family-test/temporal-OOD/generation main-table hyperparameter selection.

---

## 10. GPU-Only, Monitoring, Parallelism, `/mnt` Governance

### GPU-only
All neural training/validation/inference/generation/PPL/BPN and neural evaluator are GPU-only, `cpu_fallback_count=0`. Allowed CPU: hashing, data cleaning, MMseqs2, Infernal, ViennaRNA, bootstrap, Git, deterministic fixtures.

Record: `CUDA_VISIBLE_DEVICES`; GPU physical index, UUID, model; driver/CUDA/PyTorch; model/input/output device; successful forward/backward; peak VRAM; `cpu_fallback_count=0`; PID, command, run ID, log, checkpoint path.

### Monitoring
- Check ~2 min after launch; ~5 min after launch; every 30 min once stable.
- Read only latest stage, metric tail, resource snapshot (not re-read full logs).
- NaN/Inf/OOM/wrong-device/CPU-fallback/disk watermark/checkpoint corruption/PID-ownership anomaly/sealed-test access trigger immediate safety stop; prefer graceful stop preserving last complete checkpoint; never kill unrelated processes.

### Parallelism
- Before Phase 3 calibration: formal GPU jobs serial.
- After calibration: at most two 100M benchmark GPU jobs in parallel, on different explicitly-reserved GPUs; no GPU sharing.
- 350M: one matched cohort at a time.
- At most one unthrottled `/mnt` full scan at a time.
- During GPU wait: parallel docs, code tests, license checks, external adapters, analysis of completed validation. Forbidden during wait: reading sealed test early, changing decoder, launching not-yet-unlocked downstream.

### `/mnt` governance
- `data/raw` immutable release; `data/derived` versioned by parent hash + pipeline commit; `tokenizers` vocab+spec; `weights/reference` external public models; `runs/<run_id>` config/logs/metrics/generation; `checkpoints/<run_id>` model+optimizer state; `manifests`+`registry` small authoritative indices; `tmp`+`cache` non-authoritative rebuildable.
- Raw data/external weights: `copy → size/hash verify → destination manifest → atomic cutover`.
- Do not delete source by default; no mutable `latest` as authoritative reference.
- Checkpoints: write temp file, atomic rename after completion, compute SHA-256.
- A run ID is created once only.
- Disk safe space ≥1.5× parallel-run peak persistent need.
- Do not train directly from `/Volumes`.

---

## 11. Git Branch, Commit, Push, Large-File Boundary

- Each code-bearing task starts from `benchmark-v1` as `codex/tokenizer-benchmark/<phase>-<task>-<run_id>`.
- Stage only task allowlist; run tests, secret scan, large-file check.
- Focused commit; push task branch.
- After gate passes, fast-forward to `benchmark-v1` and push; re-read remote SHA.
- No force-push, history rewrite, auto-overwrite of main, or large-artifact upload.
- Training runs and read-only audits do not commit per heartbeat; they form focused commits when producing new official manifests/protocol/phase reports.
- Large data, external weights, checkpoints, generation corpus, caches stay in `/mnt/cunyuliu/tokenizer-benchmark`, not GitHub.

---

## 12. Phased TODO, Dependency DAG, Gates, Acceptance, Failure Branches

| Phase | Goal & main tasks | Outputs | Gate | Parallel & failure |
|---|---|---|---|---|
| Phase 0 | Goal/authority/legacy reconstruction; fresh clone; `benchmark-v1`; freeze upstream; rebuild legacy audit; `/mnt` root+registry | Goal, hash, authority manifest, legacy ledger, source/license manifest, Git remote ref | Goal byte-identical local/remote/mirror; remote SHA re-readable; all legacy `HISTORICAL_ONLY`/orphan; no training | Goal/bootstrap/lit-review/light code audit parallel; full-disk hash serial; any authority not closed → `BLOCKED_EXTERNAL` |
| Phase 1 | Fetch releases 22 & 23–26; canonicalize; exact dedup; homology cluster; Rfam annotate; cluster/family/clan/time split; datasheet | immutable release, split manifests, QC, dup/homology clusters, leakage report, data sheet | sources/release/license/accession/hash close; 80/80 cluster not across split; exact overlap zero; test sealed | download+tests parallel; heavy hash/MMseqs2/Arrow serial. Failure → no model training |
| Phase 2 | Tokenizer specs, shared scorer, continuation/generation evaluator, external adapter schema, sealed-test gate; oracle fixtures | evaluator package, protocol YAML, metric fixtures, claim–evidence matrix | all scoring/causality/round-trip/homology fixtures PASS; test unexposed; metrics independently recomputable | evaluator vs external weight manifest parallel; BPN semantics not closed → formal training locked |
| Phase 3 | Clean flat + clean BLT; ten arms; 20-step smoke, 200-step calibration per arm; param/FLOP/exposure counts; equal-budget LR pilot | resolved configs, parameter census, GPU smoke, throughput/memory report, frozen hyps | train/eval/generate parity; no future leakage; GPU fallback=0; common context+effective batch runnable; param tolerance | default single run; after calibration ≤2 100M jobs on distinct GPUs; OOM → adjust whole cohort, not lower one arm |
| Phase 4 | 10×3 seed, 2B valid nt; validation checkpoint; frozen decoder; one-shot sealed test; main likelihood/continuation/generation/efficiency tables | 30 run bundles, checkpoints, main tables, CI, Pareto, failure report | 30/30 scientific runs or config gate FAIL; no silent seed swap; all results traceable to full manifest | ≤2 independent 100M jobs; GPU wait only non-sealed docs/tests/external adapters/analysis of completed validation |
| Phase 5 | GPU-run GenerRNA/GARNET/EVA; family-conditioned RfamGen/RNAgg; GARNET/RfamSample/ArchiveII-Nr/RNAGym adapters | external registry, reference tables, overlap audit, adapter tests | GenerRNA updated/history + GARNET at least close; non-comparable explicitly reference-only; no tokenizer causal overreach | parallel with Phase 4 training, frozen evaluator, separate GPU, separate output root |
| Phase 6 | Four×3 seed, 7B valid nt; frozen selection/data/metrics/stat | 12 scale runs, scale plots, two primary contrasts | 12/12 runs closed; same GPU cohort/FSDP; all three seeds reported | one 350M cohort at a time; insufficient resources → `BLOCKED_RESOURCE`, no silent two-seed paper-level claim |
| Phase 7 | boundary enrichment/swap, random boundary, low-complexity/source confound, length/family/type stratification; tokenizer fragmentation & compute allocation | mechanism figures, failure library, tokenizer/patch stats, robustness table | mechanism conclusions across ≥2 independent stratification and multiple seeds; null mechanisms reported honestly | no added complex architecture; if mechanism null, keep benchmark paper |
| Phase 8 | clean replay; code cleanup; data sheet/model card; benchmark release; final prior-art update; paper & limitations | reproducible code, manifests, checkpoint registry, main-table replay, draft, release checklist | clean-env replay of one training smoke, one checkpoint inference, all main tables; GitHub==paper commit/hash; license closed | publish accession+scripts+split manifest if raw cannot be released; drop non-replayable results from main conclusions |

---

## 13. Statistics and Paper Claims

### 100M primary contrasts
BPE vs NUC; Unigram vs NUC; overlap-3 vs NUC; overlap-6 vs NUC; non-overlap-3 vs NUC; non-overlap-6 vs NUC; entropy vs fixed; entropy vs random; random vs fixed. Paired cluster-level effects; Holm correction on nine primary contrasts.

### 350M primary contrasts
BPE vs NUC; entropy vs fixed.

### Required reporting
- raw value per training seed; seed mean and variation; homology-cluster bootstrap 95% CI; family/type/length/source stratified effects; absolute and relative differences; failed runs and corrected retries; params, valid nucleotides, FLOPs, GPU, VRAM, wall time.

---

## 14. Claim–Evidence Matrix

Every claim is tied to evidence class:
- `empirical_measured` — reproduced measured numbers with full manifest closure.
- `empirical_metric` — measured but dependent on defined metric.
- `computational_proxy` — MFE/CM/pairing/embedding scores, not real function.
- `descriptive_statistic` — data description only.
- `historical_evidence` — legacy, excluded from main tables.
- `development_only` — smoke/calibration/pilot, not a scientific claim.

No claim may be upgraded without the underlying evidence class. See `claims/claim_evidence_matrix.json` (schema in `schemas/`).

---

## 15. Project Turnaround, Termination, Re-initiation

- Release 22 source/license/accession/hash not closable: stop that anchor; await owner approval for a release-26 new contract.
- Shared scorer or BPN semantics cannot pass oracle: stop formal model matrix.
- Exact/homology split cannot be built: stop training; prioritize data-benchmark recovery.
- Training/validation/generation keep same tokenizer/patch semantics: else stop the corresponding arm.
- Server cannot provide same GPU cohort or 350M three-seed budget: mark resource-blocked; do not write two-seed results as paper-level confirmation.
- Required external baseline cannot be legally obtained: mark unavailable; do not substitute PPT numbers for local reproduction.
- If tokenizer effect originates only from params/raw context/data/FLOPs/decoding search, do not attribute to tokenizer.
- If no tokenizer differs significantly, project may still publish as controlled negative result and evaluator/resource benchmark.
- If a more complete equivalent benchmark appears before submission, pivot to homology-aware evaluator, release-22/temporal-OOD data resource, or mechanism review; do not keep "first complete benchmark" wording.
- If the core evidence chain (data, evaluator, 100M, 350M) cannot complete, stop SOTA/high-level benchmark claims; retain as reproducible engineering resource.

---

## 16. Amendment Log

No amendments yet. (Append entries here with date, author, section, reason, evidence; never silently modify gates.)

---

## 17. Status Registry Reference

Status enum: `NOT_RUN`, `DEVELOPMENT_ONLY`, `PASS_CLOSED`, `FAIL_CLOSED_WITH_EVIDENCE`, `BLOCKED_EXTERNAL_WITH_EVIDENCE`, `TERMINATED_SAFELY_WITH_EVIDENCE`, `REDIRECT_PENDING_OWNER_APPROVAL`.

---

*This document is the single authoritative scientific and engineering contract for the TokBench-RNA project. It must not be weakened, gates lowered, or phases skipped without an amendment and owner approval.*