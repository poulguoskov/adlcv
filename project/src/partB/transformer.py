"""Transformer encoder for spatial reasoning across image features.

After the ResNet+FiLM stage, image features are local — each spatial position
mostly summarizes its own region. The transformer's self-attention layers let
every position attend to every other position, enabling global reasoning about
where in the scene the object should go.

This is an encoder-only stack (no decoder, no [CLS] token, no classification
head). Input and output are sequences of the same length and dimension. The
caller is responsible for flattening spatial dims to a sequence before calling.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn


class MultiHeadSelfAttention(nn.Module):
    """Standard multi-head self-attention."""

    def __init__(self, embed_dim: int, num_heads: int) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})")

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.to_qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.to_out = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, N, D) → (B, N, D)"""
        qkv = self.to_qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = rearrange(q, "b n (h d) -> (b h) n d", h=self.num_heads)
        k = rearrange(k, "b n (h d) -> (b h) n d", h=self.num_heads)
        v = rearrange(v, "b n (h d) -> (b h) n d", h=self.num_heads)

        scores = (q @ k.transpose(1, 2)) * self.scale
        attn = F.softmax(scores, dim=-1)
        out = attn @ v

        out = rearrange(out, "(b h) n d -> b n (h d)", h=self.num_heads)
        return self.to_out(out)


class TransformerBlock(nn.Module):
    """Pre-norm encoder block: attention → MLP, both with residuals."""

    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_ratio * embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_ratio * embed_dim, embed_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout(self.attn(self.norm1(x)))
        x = x + self.dropout(self.mlp(self.norm2(x)))
        return x


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed (non-learnable) sinusoidal positional encoding from the original paper."""

    def __init__(self, embed_dim: int, max_seq_len: int = 1024) -> None:
        super().__init__()
        pe = torch.zeros(max_seq_len, embed_dim)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2, dtype=torch.float) * -(math.log(10000.0) / embed_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class TransformerEncoder(nn.Module):
    """Stack of N encoder blocks with positional encoding.

    Input/output: (B, seq_len, embed_dim).
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_layers: int,
        max_seq_len: int = 1024,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.pos_enc = SinusoidalPositionalEncoding(embed_dim, max_seq_len)
        self.blocks = nn.ModuleList(
            [TransformerBlock(embed_dim, num_heads, dropout=dropout) for _ in range(num_layers)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pos_enc(x)
        for block in self.blocks:
            x = block(x)
        return x


if __name__ == "__main__":
    encoder = TransformerEncoder(
        embed_dim=1024,
        num_heads=8,
        num_layers=2,
        max_seq_len=1024,
    )

    n_params = sum(p.numel() for p in encoder.parameters())
    print(f"Transformer encoder")
    print(f"  Trainable params: {n_params:,}")

    seq = torch.rand(2, 1024, 1024)  # (B, seq_len = 32*32, embed_dim)
    out = encoder(seq)
    print(f"\nInput:  {tuple(seq.shape)}")
    print(f"Output: {tuple(out.shape)}  (expected: (2, 1024, 1024))")