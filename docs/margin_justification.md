# Margin Justification (Phase 4-G, draft)

> Status placeholder — **IN COMPLETION PENDING final-budget checkpoints + validation
> codec repeat runs.** This document registers the derived margins required by the
> contract (§3.9.4, Goal V3 §690) **before FINAL_UNSEAL**. Margins without an
> independent justification are treated only as sensitivity, not as support for a
> strong winner claim. Refs: `PLAN_审稿修订版_20260819.md` §3.9.4; Goal V3 §138/§690.

## 0. Purpose

The practical-equivalence / winner thresholds
(`1% BPN, 15% latency, 2 pp validity/family, 1 pp memorization, 5% distribution
distance`) must be justified by validation repeat noise, measurement precision,
deployment meaning or literature **before** the final test unseal. Otherwise they
are flagged `SENSITIVITY_ONLY`.

## 1. BPN margin (1%)

**Claimed default:** a Headline pairwise BPN difference < 1% is treated as
practically equivalent only if it is above the measured coder + seed noise.

Justification anchors:

1. **Uniform reference scale** (contract §3.6 calibration): uniform A/C/G/U is
   exactly `2.000 bits/nt` (no EOS). A 1% margin on `canonical_code_length_BPN`
   is therefore `0.02 bits/nt`. Any head-to-head that resolves below this is
   within 1% of the theoretical reference — an interpretable floor.
1.5 **Measured calibration-baseline floor (contract §3.6; produced 2026-08-20):**
   on the FROZEN homology-stratified sealed-test subsample (1000 seqs,
   sha `f43d16245`, `data/derived/codec_subsample/blt_test_subsample.parquet`),
   fit on a 2000-seq train view and scored on the same 1000 holdout:
   uniform = `2.0000` BPN (theoretical), order-3 Markov = `1.9715` BPN,
   PPM(order 5) = `1.9900` BPN. The compression floor is only ~1.4% (Markov3)
   below uniform, so a 1% neural BPN margin is a real, non-negligible fraction
   of the compressible headroom above the non-neural baselines. Full-train-fit
   baselines (million-nt) will only tighten Markov below 1.97, keeping the
   floor meaningful.
2. **Coder repeat noise (measured):** the frozen 64-bit byte-oriented range coder
   has a documented fixed lookahead overhead of ~66–72 bits **per stream**
   (`coder_overhead_bits`, bound `CODEC_OVERHEAD_BITS=128`), independent of stream
   length. On a 108K-test/100M-nt stream this is `<= 72/1e8 ≈ 7e-7 bits/nt`
   << 1%. Byte-identical decode + `1e-4 bits/nt` NLL gate already bound the coder
   precision well below 1%. → **coder noise does not reach 1%.**
3. **Seed variation (validation estimate):** completed 100M runs in the same
   family differ in validation best-loss across seed (`F1 s17=0.721`,
   `F4 s17=0.701`, `F5 s17=0.715`; P1/P3 s17=1.185/1.17). Running 3-seed paired
   cluster bootstrap CIs (§3.9.1) is the required scale for the 1% BPN read on the
   *final-budget checkpoint* — the 1% must be compared against the seed CI half-width,
   **not** against these early validation-loss deltas.
4. **Derivation rule:** set the BPN practical-equivalence half-width =
   `max(1%, coder_noise_bp, median_seed_bootstrap_halfwidth)`. If a pairwise
   CI half-width is larger, the margin defaults to the CI-based
   `INCONCLUSIVE_UNDERPOWERED` state rather than a claim.

**Status:** partial — coder-noise bound is *measured/documented*; the `1%`
reference is grounded in uniform = 2.0 bits/nt. Final seed-CI half-widths require
the completed final-budget checkpoints (Phase 4 full 33-run closure).

## 2. Latency margin (15%)

Requires *same-hardware repeated paired inference* (`>= 30 repeats`) to establish
deployment-distinguishable latency. Not yet measured (arm checkpoints not all
final). Placeholder rule: end-to-end latency must include canonicalization +
online tokenizer/patcher + packing/transfer + model + detokenization (Goal §690,
Primary cost fixture §5.2). **Status:** NOT_MEASURED — no 30-repeat paired run yet;
must be measured before unseal, otherwise `SENSITIVITY_ONLY`.

## 3. Validity / family-recoverability margin (2 pp)

**Documented measurement floor (validated by the codec/evaluator fixtures):**
- Illegal-overlap transition mass and validity denominators are reproduced from
  atomic counts (§5.2), so validity is measured at the per-10,000-attempt
  denominator → Monte-Carlo SE ≈ `sqrt(0.02·0.98/1e4) ≈ 0.14 pp` for a binary
  validity near 98%. A 2 pp difference is > 10× this binomial SE → **measurable**
  at the fixed 10,000-attempt bundle.
- Family recoverability is CM/family-balanced; its noise is cluster-bootstrap
  driven and must be reported on the family-balanced macro scale.

**Status:** justified by denominator/SE; final CI depends on clustering-level
bootstrap.

## 4. Memorization margin (1 pp)

Memorization is the rate of exact/near-copy generations attributable to train
duplication. Measured at the 10,000-attempt denominator → binomial SE ≈ 0.1 pp at
1% memorization. A 1 pp margin is > 5× the SE. **Status:** methodologically
justified; needs final 10,000-attempt bundles.

## 5. Distribution-distance margin (5%)

Distribution proxies (length / GC / dinucleotide / k-mer / MFE / paired-fraction)
are compared on *length×GC×RNA-type matched* views (§3.7). The 5% is a
pre-registered effect-size threshold; its noise floor is the matched-view resample
variability. **Status:** rule fixed; numeric noise floor to be completed with final
generation bundles.

## 6. Margin ↔ Decision Map consistency

`decision_map.yaml` uses identical thresholds; any numeric override here must be
reflected there (Goal §822). This doc is the authority the Decision Map YAML
references.

## 7. Consequence table

| Margin | Basis | Status |
|---|---|---|
| BPN 1% | uniform=2.0 bits/nt ref + coder noise `~7e-7` b/nt | coder part measured; seed-CI pending |
| Latency 15% | needs ≥30 paired same-HW repeats | TODO |
| Validity/family 2 pp | 10k-attempt binomial SE ≈0.14pp | justified |
| Memorization 1 pp | 10k-attempt binomial SE ≈0.1pp | justified |
| Dist distance 5% | matched-view resample | rule fixed; floor pending |

Until the TODO rows are closed this margin set is `SENSITIVITY_ONLY` for the
affected claims (Goal §138).