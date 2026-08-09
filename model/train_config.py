"""Training config generator for the 100M ten-arm and 350M four-arm matrices.

Implements contract 3.2-3.4:
  - Same-track matched arms share identical backbone non-embedding params.
  - Total trainable params within 2% of the nominal target (embedding factorized
    for large vocabularies so BPE-1024 / 6-mer arms stay in-tolerance).
  - Budgets counted in cumulative valid target nt (100M: 2.0B, 350M: 7.0B).
  - LR tuning candidates = base_lr * {0.5, 1.0, 2.0}, seed 17, <=100M nt each.
  - AdamW (0.9, 0.95), weight_decay 0.1, bf16, context 4096 canonical nt.
"""
from __future__ import annotations

from dataclasses import dataclass

from .arms import ArmSpec, ARMS_100M, ARMS_350M, SEEDS
from .backbone import FlatCausalLM, BLTCausalLM
from .census import count_params

# Contract 3.4 constants
CONTEXT_NT = 4096
D_FF_MULT = 4
HEAD_DIM = 64
TARGET_100M = 100_000_000
TARGET_350M = 350_000_000
TOL = 0.02
BUDGET_100M_NT = 2_000_000_000
BUDGET_350M_NT = 7_000_000_000
BASE_LR_100M = 3e-4
BASE_LR_350M = 2e-4
LR_CANDIDATES = (0.5, 1.0, 2.0)
TUNE_SEED = 17
TUNE_BUDGET_NT = 100_000_000
# Frozen per-arm LR factor selected by validation metric on seed 17 (contract 3.4:
# select by a validation metric, then freeze). Evidence: Phase 3 LR pilot aggregate
# /mnt/cunyuliu/tokenizer-benchmark/runs/phase3_lr_report_aggregate.json
# (2026-08-09, 10-arm 100M matrix, 300 optimizer steps / ~1M valid nt per candidate,
# all_cpu_fallback_zero=True, all_finite=True). 9/10 arms pick 2.0x; P3 picks 1.0x.
LR_FACTOR_SELECTED = {
    "F1": 2.0, "F2": 2.0, "F3": 2.0, "F4": 2.0, "F5": 2.0,
    "F6": 2.0, "F7": 2.0, "P1": 2.0, "P2": 2.0, "P3": 1.0,
}


@dataclass(frozen=True)
class ArchConfig:
    d_model: int
    n_layers: int
    n_heads: int
    max_len: int = CONTEXT_NT
    d_ff_mult: int = D_FF_MULT

    @property
    def d_ff(self) -> int:
        return int(self.d_model * self.d_ff_mult)


@dataclass(frozen=True)
class EmbedConfig:
    embed_dim: int
    factorized: bool
    tied: bool = True


@dataclass(frozen=True)
class OptimConfig:
    lr: float
    betas: tuple = (0.9, 0.95)
    weight_decay: float = 0.1
    dtype: str = "bf16"
    max_grad_norm: float = 1.0


@dataclass(frozen=True)
class RunConfig:
    run_id: str
    scale: str                 # "100M" | "350M"
    arm: ArmSpec
    seed: int
    arch: ArchConfig
    embed: EmbedConfig
    optim: OptimConfig
    budget_nt: int             # cumulative valid target nt
    batch_nt: int              # effective nt per optimizer step
    warmup_nt: int
    context_nt: int = CONTEXT_NT
    entropy_predictor_params: int = 0

    @property
    def batch_size_seq(self) -> int:
        return max(1, self.batch_nt // self.context_nt)

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "scale": self.scale,
            "arm": self.arm.id,
            "backbone": self.arm.backbone,
            "tokenizer_type": self.arm.tokenizer_type,
            "vocab_size": self.arm.vocab_size,
            "k": self.arm.k,
            "stride": self.arm.stride,
            "seed": self.seed,
            "arch": {
                "d_model": self.arch.d_model,
                "n_layers": self.arch.n_layers,
                "n_heads": self.arch.n_heads,
                "d_ff": self.arch.d_ff,
                "max_len": self.arch.max_len,
            },
            "embed": {
                "embed_dim": self.embed.embed_dim,
                "factorized": self.embed.factorized,
                "tied": self.embed.tied,
            },
            "optimizer": {
                "lr": self.optim.lr,
                "betas": list(self.optim.betas),
                "weight_decay": self.optim.weight_decay,
                "dtype": self.optim.dtype,
                "max_grad_norm": self.optim.max_grad_norm,
            },
            "budget_nt": self.budget_nt,
            "batch_nt": self.batch_nt,
            "batch_size_seq": self.batch_size_seq,
            "warmup_nt": self.warmup_nt,
            "context_nt": self.context_nt,
            "entropy_predictor_params": self.entropy_predictor_params,
            "note": self.arm.note,
        }


# --- parameter algebra (must match model/backbone.py exactly) ---

def _non_emb_params(d_model: int, n_layers: int, d_ff: int) -> int:
    # per Block: attn(qkv 3D^2 + out D^2) + mlp(2 D d_ff) + ln1 + ln2 (4D)
    block = 4 * d_model * d_model + 2 * d_model * d_ff + 4 * d_model
    return n_layers * block + 2 * d_model


def _embedding_params_plain(vocab: int, d_model: int, max_len: int) -> int:
    return vocab * d_model + max_len * d_model


def _embedding_params_fact(vocab: int, d_model: int, emb: int, max_len: int) -> int:
    # tok_emb(vocab*emb) + up(emb*D) + down(D*emb) + pos_emb(max_len*D); lm_head tied
    return vocab * emb + 2 * emb * d_model + max_len * d_model


def solve_arch(target: int, max_len: int = CONTEXT_NT) -> ArchConfig:
    """Smallest (d_model, n_layers) with total params (vocab=4) within tol."""
    best = None
    for d_model in range(128, 4096, HEAD_DIM):
        d_ff = int(d_model * D_FF_MULT)
        n_heads = d_model // HEAD_DIM
        for n_layers in range(1, 48):
            non_emb = _non_emb_params(d_model, n_layers, d_ff)
            emb = _embedding_params_plain(4, d_model, max_len)
            total = non_emb + emb
            if target * (1 - TOL) <= total <= target * (1 + TOL):
                return ArchConfig(d_model, n_layers, n_heads, max_len, D_FF_MULT)
    raise RuntimeError("no architecture found for target=%d" % target)


def solve_embed(vocab: int, arch: ArchConfig, target: int, max_len: int) -> EmbedConfig:
    """Choose plain or factorized embedding so total params stay within tol."""
    non_emb = _non_emb_params(arch.d_model, arch.n_layers, arch.d_ff)
    base = non_emb + max_len * arch.d_model
    plain = base + vocab * arch.d_model
    if target * (1 - TOL) <= plain <= target * (1 + TOL):
        return EmbedConfig(embed_dim=arch.d_model, factorized=False)
    for emb in range(64, arch.d_model, 64):
        tot = non_emb + _embedding_params_fact(vocab, arch.d_model, emb, max_len)
        if target * (1 - TOL) <= tot <= target * (1 + TOL):
            return EmbedConfig(embed_dim=emb, factorized=True)
    raise RuntimeError("no embedding dimension for vocab=%d target=%d" % (vocab, target))


def _run_id(arm_id: str, seed: int, scale: str, batch_nt: int) -> str:
    return "tokbench_%s_%s_s%d_nt%d_%d" % (arm_id, scale, seed, batch_nt, seed)


def target_for_scale(scale: str) -> int:
    return TARGET_350M if scale == "350M" else TARGET_100M


def budget_for_scale(scale: str) -> int:
    return BUDGET_350M_NT if scale == "350M" else BUDGET_100M_NT


def base_lr_for_scale(scale: str) -> float:
    return BASE_LR_350M if scale == "350M" else BASE_LR_100M


def resolved_config(arm_id: str, seed: int, scale: str = "100M",
                    batch_nt: int = 32768) -> RunConfig:
    """Resolve a fully-specified training config for one arm/seed/scale."""
    if scale not in ("100M", "350M"):
        raise ValueError("scale must be '100M' or '350M'")
    arm = arm_(arm_id)
    target = target_for_scale(scale)
    arch = solve_arch(target)
    embed = solve_embed(arm.vocab_size, arch, target, arch.max_len)
    budget = budget_for_scale(scale)
    # Contract 3.4: LR frozen by validation selection (LR_FACTOR_SELECTED).
    # Untuned arms (e.g. 350M C1-C4 until their pilot runs) default to base 1.0x.
    lr = base_lr_for_scale(scale) * LR_FACTOR_SELECTED.get(arm_id, 1.0)
    entropy_params = 0
    if arm.tokenizer_type == "entropy_patch":
        # P3/C4: entropy boundary predictor trained on train split only.
        # The recorded independent param count must match the predictor that
        # the training loop actually uses (contract 3.2).
        from .entropy_predictor import EntropyPredictor
        entropy_params = count_params(EntropyPredictor()).total_params
    return RunConfig(
        run_id=_run_id(arm_id, seed, scale, batch_nt),
        scale=scale,
        arm=arm,
        seed=seed,
        arch=arch,
        embed=embed,
        optim=OptimConfig(lr=lr),
        budget_nt=budget,
        batch_nt=batch_nt,
        warmup_nt=int(0.005 * budget),
        entropy_predictor_params=entropy_params,
    )


def build_model(cfg: RunConfig):
    """Construct the torch backbone for a resolved config (not yet on GPU)."""
    if cfg.arm.backbone == "flat":
        return FlatCausalLM(
            vocab_size=cfg.arm.vocab_size, d_model=cfg.arch.d_model,
            n_layers=cfg.arch.n_layers, n_heads=cfg.arch.n_heads,
            max_len=cfg.arch.max_len, embed_dim=cfg.embed.embed_dim,
            tied_embed=cfg.embed.tied)
    if cfg.arm.backbone == "blt":
        return BLTCausalLM(
            vocab_size=cfg.arm.vocab_size, d_model=cfg.arch.d_model,
            n_layers=cfg.arch.n_layers, n_heads=cfg.arch.n_heads,
            max_len=cfg.arch.max_len, embed_dim=cfg.embed.embed_dim,
            tied_embed=cfg.embed.tied, patcher=None,
            default_patch_len=8)
    raise ValueError("unknown backbone %s" % cfg.arm.backbone)


def verify_arm(cfg: RunConfig) -> bool:
    """Dry verify: total params within tol and non-embedding reported correctly."""
    model = build_model(cfg)
    c = count_params(model)
    target = target_for_scale(cfg.scale)
    ok_total = target * (1 - TOL) <= c.total_params <= target * (1 + TOL)
    assert ok_total, "arm %s total=%d out of [%d,%d]" % (
        cfg.arm.id, c.total_params, int(target * (1 - TOL)), int(target * (1 + TOL)))
    assert c.embedding_params > 0
    return True


def all_100M(seed: int | None = None) -> list[RunConfig]:
    seeds = [seed] if seed is not None else SEEDS
    return [resolved_config(a.id, s) for a in ARMS_100M for s in seeds]


def all_350M(seed: int | None = None) -> list[RunConfig]:
    seeds = [seed] if seed is not None else SEEDS
    return [resolved_config(a.id, s, scale="350M", batch_nt=65536) for a in ARMS_350M for s in seeds]


def lr_tuning_candidates(cfg: RunConfig) -> list[float]:
    """Contract 3.4: base_lr * {0.5, 1.0, 2.0}, tuned on seed 17 only.

    Anchored to the scale base LR (not the frozen, validation-selected lr), so
    the candidate grid is identical regardless of the final LR freeze.
    """
    if cfg.seed != TUNE_SEED:
        raise ValueError("LR tuning only on seed %d" % TUNE_SEED)
    return [base_lr_for_scale(cfg.scale) * f for f in LR_CANDIDATES]


def arm_(id_: str) -> ArmSpec:
    for a in ARMS_100M + ARMS_350M:
        if a.id == id_:
            return a
    raise KeyError(id_)