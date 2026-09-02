"""Clean flat causal Transformer (F-arm) and clean BLT backbone (P-arm).

- Flat: token-level causal Transformer over a static tokenizer embedding.
- BLT: latent-bottleneck patched Transformer (entropy/fixed/random patch).
Both are GPU-only. No explicit n-gram / hash byte-group lookup anywhere;
the patcher is the ONLY mechanism controlling segmentation (P-arm).

Design keeps the two backbones parameter-matched at the non-embedding layers
so cross-backbone comparisons are architecture/system, not pure tokenizer.

Gradient checkpointing (on by default) trades a little compute for a large
drop in peak activation memory, which is required to run the shared context
(4096 nt) on the 40GB A100 cohort without changing the science (contract 3.4:
"共同 context 和有效 batch 可运行").
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as ckpt


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.0, max_len=4096):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("mask", torch.triu(
            torch.ones(1, 1, max_len, max_len) * float("-inf"),
            diagonal=1))

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        def _r(t):
            return t.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        q, k, v = _r(q), _r(k), _r(v)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        att = att + self.mask[:, :, :T, :T]
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.out(y)


class Block(nn.Module):
    def __init__(self, d_model, n_heads, ff_mult=4, dropout=0.0, max_len=4096):
        super().__init__()
        d_ff = int(d_model * ff_mult)
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout, max_len)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.GELU(),
            nn.Linear(d_ff, d_model, bias=False),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class _EmbeddingHead(nn.Module):
    """Optional factorized/tied embedding + head for large vocabularies (3.4)."""
    def __init__(self, vocab_size, d_model, embed_dim=None, tied=True):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embed_dim = embed_dim if embed_dim is not None else d_model
        self.factorized = self.embed_dim != d_model
        self.tok_emb = nn.Embedding(vocab_size, self.embed_dim)
        if self.factorized:
            self.up = nn.Linear(self.embed_dim, d_model, bias=False)
            self.down = nn.Linear(d_model, self.embed_dim, bias=False)
        else:
            self.up = nn.Identity()
            self.down = nn.Identity()
        self.lm_head = nn.Linear(self.embed_dim, vocab_size, bias=False)
        if tied:
            self.lm_head.weight = self.tok_emb.weight

    def embed(self, ids):
        return self.tok_emb(ids)

    def head(self, h):
        return self.lm_head(h)


class FlatCausalLM(nn.Module):
    """F-arm: causal LM over static token embeddings."""
    def __init__(self, vocab_size, d_model, n_layers, n_heads,
                 max_len=4096, dropout=0.0, tied_embed=True, embed_dim=None,
                 use_checkpoint=True):
        super().__init__()
        self.d_model = d_model
        self.use_checkpoint = use_checkpoint
        self.embed_head = _EmbeddingHead(
            vocab_size, d_model, embed_dim=embed_dim, tied=tied_embed)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.normal_(self.pos_emb, std=0.02)
        self.blocks = nn.ModuleList(
            Block(d_model, n_heads, dropout=dropout, max_len=max_len)
            for _ in range(n_layers))
        self.ln_f = nn.LayerNorm(d_model)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, token_ids, targets=None):
        B, T = token_ids.shape
        x = self.embed_head.embed(token_ids)
        x = self.embed_head.up(x) + self.pos_emb[:, :T, :]
        for blk in self.blocks:
            if self.use_checkpoint and self.training:
                x = ckpt(blk, x, use_reentrant=False)
            else:
                x = blk(x)
        x = self.ln_f(x)
        h = self.embed_head.down(x)
        logits = self.embed_head.head(h)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1),
                ignore_index=-100)
        return logits, loss


class EntropyPatcher(nn.Module):
    """Causal entropy boundary predictor (P3). Trained on train split only.

    Predicts per-position boundary probability from the nucleotide context.
    Fixed/random patch use the same shallow predictor with a fixed policy.
    """
    def __init__(self, d_model=64, d_hidden=128, max_len=4096):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(4, d_model)
        self.pos = nn.Parameter(torch.zeros(1, max_len, d_model))
        self.lstm = nn.LSTM(d_model, d_hidden, batch_first=True, bidirectional=False)
        self.head = nn.Linear(d_hidden, 1)

    def forward(self, nt_ids):
        B, T = nt_ids.shape
        x = self.embed(nt_ids) + self.pos[:, :T, :]
        out, _ = self.lstm(x)
        return torch.sigmoid(self.head(out).squeeze(-1))  # boundary prob per pos



def open_patch_running_mean(embed_out, boundary):
    """Per-position open-patch mean embedding (exact causal, no leak).

    Position i's value is the mean of the nucleotide embeddings from its
    patch start s_i up to i (inclusive). The running means within a patch are
    an invertible linear reparameterisation of the raw per-nt embeddings
    (m_k*(k-s+1) - m_{k-1}*(k-s) = emb[x_k]), so no information is discarded;
    crucially the value at i never contains x_j for j > i (amendment
    2026-09-02: train/eval causal-input alignment).
    """
    B, T, E = embed_out.shape
    starts = (boundary > 0.5).long()
    starts[:, 0] = 1
    idx = torch.arange(T, device=embed_out.device).view(1, T).expand(B, T)
    marker = torch.where(starts > 0, idx, torch.full_like(idx, -1))
    last_start = torch.cummax(marker, dim=1).values          # B,T patch starts
    cnt = (idx - last_start + 1).clamp(min=1).float()        # open length
    e32 = embed_out.float()
    csum = torch.cumsum(e32, dim=1)
    cs_shift = torch.cat([torch.zeros_like(csum[:, :1]), csum[:, :-1]], dim=1)
    seg_sum = csum - torch.gather(
        cs_shift, 1, last_start.unsqueeze(-1).expand(-1, -1, E))  # sum emb[s..i]
    pooled = seg_sum / cnt.unsqueeze(-1)
    # length-1 patches (incl. the whole B1 patch=1 bridge) must be
    # BIT-identical to the raw embedding: cumsum differencing loses
    # the last ULP, which would break B1 == Flat equivalence.
    solo = (cnt.unsqueeze(-1) == 1.0)
    pooled = torch.where(solo, e32, pooled)
    return pooled.to(embed_out.dtype)


class BLTCausalLM(nn.Module):
    """P-arm: patched latent Transformer with a boundary predictor.

    Fold nt -> patches (by boundary policy), run a causal Transformer over
    patch embeddings, unfold to nucleotide logits via a linear head.
    No explicit n-gram / hash lookup: segmentation driven only by the patch
    policy (fixed/random/entropy).
    """
    def __init__(self, vocab_size, d_model, n_layers, n_heads,
                 max_len=4096, dropout=0.0, tied_embed=True, embed_dim=None,
                 patcher=None, default_patch_len=6, use_checkpoint=True):
        super().__init__()
        self.d_model = d_model
        self.use_checkpoint = use_checkpoint
        self.embed_head = _EmbeddingHead(
            vocab_size, d_model, embed_dim=embed_dim, tied=tied_embed)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.normal_(self.pos_emb, std=0.02)
        self.blocks = nn.ModuleList(
            Block(d_model, n_heads, dropout=dropout, max_len=max_len)
            for _ in range(n_layers))
        self.ln_f = nn.LayerNorm(d_model)
        self.patcher = patcher
        self.default_patch_len = default_patch_len
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, nt_ids, boundary, targets=None):
        """Open-patch causal LM forward (amendment 2026-09-02).

        Runs the SAME trunk as FlatCausalLM over all T nt positions; the only
        difference is the input parameterisation: position i sees the open
        patch running mean (mean of its patch's nt embeddings from the patch
        start up to i). This is exactly the causal input the per-base
        evaluator conditions on, so training optimises the scored quantity
        with no within-patch future-nt leak and no train/eval mismatch.
        logits[:, i] is the conditional for x_{i+1}.
        """
        B, T = nt_ids.shape
        seq_emb = self.embed_head.embed(nt_ids)          # B,T,emb_dim
        pooled = open_patch_running_mean(seq_emb, boundary)
        x = self.embed_head.up(pooled) + self.pos_emb[:, :T, :]
        for blk in self.blocks:
            if self.use_checkpoint and self.training:
                x = ckpt(blk, x, use_reentrant=False)
            else:
                x = blk(x)
        x = self.ln_f(x)
        h = self.embed_head.down(x)
        logits = self.embed_head.head(h)                 # B,T,vocab
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1),
                ignore_index=-100)
        return logits, loss

class PatchInputFlatCausalLM(FlatCausalLM):
    """L2 pilot (approved amendment 2026-08-30, Track L2 Stage A).

    SAME Flat trunk / embedding / head as FlatCausalLM (this is a subclass with
    no added modules, so trainable parameters are bit-for-bit identical to the
    F1 static-token arm). The only difference is INPUT resolution: the sequence
    is folded into variable-length patches whose mean nucleotide embeddings are
    fed to the trunk, i.e. dynamic segmentation input on one Flat backbone.
    Boundaries come from a PatchPolicy (fixed / random / entropy) at call time.
    """

    def forward(self, nt_ids, boundary, targets=None):
        """Open-patch forward, identical computation to BLTCausalLM.forward.

        Track L2 shares the Flat trunk/embedding/head and differs from F1 only
        in input parameterisation (boundary-reset running means), same as the
        P arms (see open_patch_running_mean docstring and amendment
        2026-09-02).
        """
        B, T = nt_ids.shape
        seq_emb = self.embed_head.embed(nt_ids)
        pooled = open_patch_running_mean(seq_emb, boundary)
        x = self.embed_head.up(pooled) + self.pos_emb[:, :T, :]
        for blk in self.blocks:
            if self.use_checkpoint and self.training:
                x = ckpt(blk, x, use_reentrant=False)
            else:
                x = blk(x)
        x = self.ln_f(x)
        h = self.embed_head.down(x)
        logits = self.embed_head.head(h)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1),
                ignore_index=-100)
        return logits, loss
