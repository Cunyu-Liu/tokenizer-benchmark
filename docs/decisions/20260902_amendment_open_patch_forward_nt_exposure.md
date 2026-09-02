# Amendment 2026-09-02: open-patch causal forward, nt-based exposure, E_S reframing

> Owner approval: 2026-09-02 conversation (audit findings A-D; "每一个决策都选 A").
> Authority: this file is the evidence record; the GOAL V3 Amendment Log row is the
> binding summary. Status: APPROVED_BY_OWNER (2026-09-02).

## Background (audit findings, 2026-09-02 server verification)

- **A. B1 == F1 (bit-identical).** The "BLT" implementation folds nt into
  patches by mean-pooling and runs the SAME trunk as the flat backbone; with
  patch-size=1 the fold/unfold is the identity map. Loading the trained F1 s17
  checkpoint into `BLTCausalLM` with patch=1 boundaries reproduces FlatCausalLM
  logits bit-for-bit (max diff 0.0; 98,247,296 params, identical state-dict
  keys). The contracted `B1 - F1` "hierarchy/system effect" therefore does not
  exist architecturally; the only training difference is lr (B1 3e-4 vs F 6e-4).
- **B. Within-patch training leak.** The closed-patch fold made each patch
  embedding the mean of ALL its nt (including future nt); for mid-patch
  positions the target x_{i+1} was inside the model input. The evaluator
  already acknowledged this ("a single full-sequence forward leaks future nt
  within the open patch") and recomputed per-prefix at O(T^2) cost — training
  and validation, however, optimised the leaky objective (P1 fixed-6: 5/6 of
  scored positions leaked; val-loss-based LR pilots contaminated).
- **C. Exposure counted tokens, not nt.** `count_valid_nt` counted target
  tokens. F2/F3/F6/F7 (multi-nt tokens) consumed 3-6x the contracted 2.0B
  valid nt (F7 ~11.2B nt ≈ 5.6x, stopping at 1.867B tokens on data exhaustion).
- **D. Mean-pool is order-blind within a patch** (bag-of-nucleotides): patch
  "ACG" == "GCA". The 62% val-loss gap of P arms vs F1 is dominated by this
  and by the leak, not by "dynamic segmentation" per se.

## Decisions (owner, 2026-09-02)

1. **Leak fix — open-patch causal forward (deviation from literal option A,
   documented).** The literal instruction "score only patch-final positions"
   was rejected during implementation: it would leave the 5/6 mid-patch
   EVALUATION positions conditioned on open-patch inputs never seen in
   training (P1 fixed-6 worst hit), biasing E_P against P1. The implemented
   fix achieves the option-A intent (training objective == exact evaluation
   quantity, zero leak): position i's input is the open-patch running mean
   (mean of its patch's nt embeddings from the patch start up to i). This is
   exactly the causal input the per-base evaluator conditions on. Length-1
   patches are taken bit-identically from the raw embedding, so B1 (patch=1)
   remains bit-identical to the Flat forward. Within a patch the running means
   are an INVERTIBLE linear reparameterisation of the raw embeddings
   (m_k*(k-s+1) - m_{k-1}*(k-s) = emb[x_k]): no information is lost, so the
   P arms are reframed as a **boundary-rule input-parameterisation study**
   (not a compression/bottleneck study). Trunk runs over all T nt positions
   in training and evaluation; the old folded runs are historical only.
2. **Exposure fix — count nt, not tokens.** `_window` now emits per-target nt
   weights (`tokenizer.token_nt_counts`), `count_valid_nt` sums weights, and
   training/validation use nt-weighted cross-entropy (a 6-nt token counts 6x
   a 1-nt token). Every arm now consumes exactly 2.0B valid nt. The six DONE
   cells of F2/F3/F6/F7 (s17/s29) are marked
   `FAIL_CLOSED_WITH_EVIDENCE` + `superseded` and re-run under the fix;
   F1/F4/F5 (1 nt per token) remain valid. LR 6e-4 (F arms) / 3e-4 (B1, P
   arms) retained for continuity; the pilots were run under the old objective
   and this is recorded as a limitation.
3. **E_S reframing.** With B1 == F1 (bit-level, permanently enforced by
   `tests/test_open_patch_and_exposure.py::test_b1_patch1_bit_identical_to_flat`),
   `B1 - F1` may only be reported as an lr-sensitivity datapoint (F1@3e-4).
   The 2x2 diagonal and `(P1-B1) - (F7-F1)` are downgraded to a descriptive
   decomposition of input parameterisation within ONE trunk family. All
   "BLT backbone / hierarchy system effect" wording is replaced by
   "patched-input variant of the same trunk". B1 s17 (currently RUNNING,
   ~900M/2B nt) is NOT interrupted (single-GPU and no-interruption
   constraints) and completes as the F1-lr-sensitivity run.
4. **Track L2 redundancy.** `PatchInputFlatCausalLM` computes the same
   open-patch forward as the P arms (test-enforced,
   `test_l2_patched_input_matches_blt_forward`); the registered L2 pilot
   would duplicate P1-P3 at smaller budget. Recommendation: do NOT launch L2
   as registered; owner to recast or cancel (pending owner decision, not
   actioned here).
5. **Codec BLT path batched.** `all_log_probs_next_base` (single forward, all
   positions) replaces the O(T^2) per-prefix loop for sequences within
   context_nt (>4096 nt keeps the exact per-prefix rolling-window path).
   This removes the root cause of the 2026-08-25 codec stall (P1 s17, 23h,
   0-byte JSON). Base-0 keeps the adapter no-context prior.

## Evidence

- Code: `model/backbone.py` (`open_patch_running_mean`, new BLT/L2 forwards),
  `model/dataset.py` (nt weights), `model/train.py` (nt-weighted CE),
  `evaluator/blt_adapter.py` (`all_log_probs_next_base`),
  `evaluator/codec_scoring.py` (batched BLT path + legacy-adapter fallback).
- Tests: `tests/test_open_patch_and_exposure.py` (7 new: bit-identity,
  suffix-perturbation invariance, no-target-leak, patch-final consistency,
  running-mean invertibility, nt exposure counting, L2==BLT forward); full
  suite 251 passed / 20 skipped / 0 failed after fixes (incl. GPU run of the
  BLT adapter and codec fixtures).
- Superseded manifests: F2 s17, F3 s17, F6 s17/s29, F7 s17/s29
  (`superseded.effective = 2026-09-02`); closure ledger after marking:
  3 DONE / 1 RUNNING / 14 FAIL_CLOSED (superseded) / 15 NOT_STARTED.
- E2E: F1 s17 real checkpoint reproduces bit-identical logits through the new
  BLT(patch=1) forward; F7 smoke under the fix consumes 60,811 nt for a 60,000
  nt budget (previously ~6x that in raw nt).


## Follow-up owner decision (2026-09-02, same session)

1. **Track L2: CANCELLED before any run.** With the open-patch forward,
   `PatchInputFlatCausalLM` is bit-identical to the P-arm forward
   (test-enforced); the registered pilot would duplicate P1-P3 at smaller
   budget. Registration marked `CANCELLED_OWNER_2026-09-02` /
   `CANCELLED_BEFORE_ANY_RUN`; zero GPU compute spent; code retained as
   equivalence evidence. A future L2 would need a genuinely different
   parameterisation (e.g. a learnable patch encoder) and a fresh prospective
   amendment.
2. **Single-GPU training constraint: KEPT** (2026-08-28 owner rule stands:
   max 1 training process / 1 physical GPU at any time). The ~6-month
   re-closure horizon under this constraint is accepted ("慢慢跑").
3. **Execution order** (scheduler-driven, no manual queue changes):
   re-run of the six superseded F cells (F2/F3/F6/F7) and the remaining
   F1/F4/F5 seeds first (E_R is the intact primary deliverable), then the
   P1-P3 re-runs under the open-patch forward. B1 s17 finishes undisturbed.
