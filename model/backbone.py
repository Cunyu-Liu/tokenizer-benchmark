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


class BLTCausalLM(nn.Module):
    """P-arm: patched latent Transformer with a boundary predictor.

    Fold nt -> patches (by boundary policy), run a causal Transformer over
    patch embeddings, unfold to nucleotide logits via a linear head.
    No explicit n-gram / hash lookup: segmentation driven only by the patch
    policy (fixed/random/entropy).
    """
    def __init__(self, vocab_size, d_model, n_layers, n_heads,
                 max_len=4096, dropout=0.0, tied_embed=True, embed_dim=None,
                 patcher=None, default_patch_len=8, use_checkpoint=True):
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

    def _segments(self, nt_ids, boundary):
        """Boundary[b,i] in {0,1}: whether position i starts a new patch."""
        B, T = nt_ids.shape
        starts = (boundary > 0.5).long()
        starts[:, 0] = 1
        seg_ids = torch.cumsum(starts, dim=1) - 1
        return seg_ids

    def forward(self, nt_ids, boundary, targets=None):
        B, T = nt_ids.shape
        seg = self._segments(nt_ids, boundary)
        # patch embedding = mean of nucleotide embeddings in segment
        mask = F.one_hot(seg, num_classes=seg.max() + 1).permute(0, 2, 1).float()
        seq_emb = self.embed_head.embed(nt_ids)  # B,T,emb_dim
        patch_emb = mask @ seq_emb  # B, n_patch, emb_dim
        patch_emb = self.embed_head.up(patch_emb)  # B, n_patch, d_model
        n_patch = patch_emb.size(1)
        x = patch_emb + self.pos_emb[:, :n_patch, :]
        for blk in self.blocks:
            if self.use_checkpoint and self.training:
                x = ckpt(blk, x, use_reentrant=False)
            else:
                x = blk(x)
        x = self.ln_f(x)
        # unfold patch logits back to nt positions
        h = self.embed_head.down(x)          # B, n_patch, emb_dim
        logits_patch = self.embed_head.head(h)  # B, n_patch, vocab
        # for each nt position, use its patch's logits (gather on patch dim)
        vocab = logits_patch.size(-1)
        per_nt = torch.gather(
            logits_patch, 1, seg.unsqueeze(-1).expand(-1, -1, vocab))  # B,T,vocab
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                per_nt.view(-1, per_nt.size(-1)), targets.view(-1),
                ignore_index=-100)
        return per_nt, loss