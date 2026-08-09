"""Clean flat causal Transformer (F-arm) and clean BLT backbone (P-arm).

- Flat: token-level causal Transformer over a static tokenizer embedding.
- BLT: latent-bottleneck patched Transformer (entropy/fixed/random patch).
Both are GPU-only. No explicit n-gram / hash byte-group lookup anywhere;
the patcher is the ONLY mechanism controlling segmentation (P-arm).

Design keeps the two backbones parameter-matched at the non-embedding layers
so cross-backbone comparisons are architecture/system, not pure tokenizer.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


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


class FlatCausalLM(nn.Module):
    """F-arm: causal LM over static token embeddings."""
    def __init__(self, vocab_size, d_model, n_layers, n_heads,
                 max_len=4096, dropout=0.0, tied_embed=True):
        super().__init__()
        self.d_model = d_model
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.normal_(self.pos_emb, std=0.02)
        self.blocks = nn.ModuleList(
            Block(d_model, n_heads, dropout=dropout, max_len=max_len)
            for _ in range(n_layers))
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        if tied_embed:
            self.lm_head.weight = self.tok_emb.weight
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
        x = self.tok_emb(token_ids) + self.pos_emb[:, :T, :]
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
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
                 max_len=4096, dropout=0.0, tied_embed=True,
                 patcher=None, default_patch_len=8):
        super().__init__()
        self.d_model = d_model
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.normal_(self.pos_emb, std=0.02)
        self.blocks = nn.ModuleList(
            Block(d_model, n_heads, dropout=dropout, max_len=max_len)
            for _ in range(n_layers))
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        if tied_embed:
            self.lm_head.weight = self.tok_emb.weight
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
        # patch embedding = mean of token embeddings in segment
        mask = F.one_hot(seg, num_classes=seg.max() + 1).permute(0, 2, 1).float()
        seq_emb = self.tok_emb(nt_ids)  # B,T,C
        patch_emb = mask @ seq_emb  # B, n_patch, C
        n_patch = patch_emb.size(1)
        x = patch_emb + self.pos_emb[:, :n_patch, :]
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        # unfold patch logits back to nt positions
        logits_patch = self.lm_head(x)  # B, n_patch, vocab
        # for each nt position, use its patch's logits
        per_nt = logits_patch[:, seg]  # B,T,vocab
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                per_nt.view(-1, per_nt.size(-1)), targets.view(-1),
                ignore_index=-100)
        return per_nt, loss
