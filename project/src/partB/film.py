"""FiLM (Feature-wise Linear Modulation) for class conditioning.

Reference: Perez et al., "FiLM: Visual Reasoning with a General Conditioning
Layer" (AAAI 2018).

Given image features [B, C, H, W] and a class embedding [B, D], produce
modulated features [B, C, H, W] where the class controls a per-channel scale
and shift:

    gamma, beta = MLP(class_embedding)        # both (B, C)
    output = (1 + gamma) * normalize(features) + beta

The (1 + gamma) form means zero output from the MLP leaves features unchanged,
so the model only learns deviations from "do nothing." InstanceNorm before the
modulation removes per-channel statistics so gamma and beta operate on a clean
baseline.
"""
from __future__ import annotations

import torch
from torch import nn


class FiLM(nn.Module):
    """Per-channel affine modulation conditioned on a class embedding.

    Args:
        class_dim: Dimensionality of the class embedding (e.g. 512 for CLIP).
        feature_channels: Number of channels in the feature map being modulated.
    """

    def __init__(self, class_dim: int, feature_channels: int) -> None:
        super().__init__()
        self.norm = nn.InstanceNorm2d(feature_channels, affine=False)
        self.mlp = nn.Linear(class_dim, feature_channels * 2)

    def forward(
        self,
        features: torch.Tensor,
        class_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """Modulate per-channel.

        Args:
            features: (B, C, H, W) feature map.
            class_embedding: (B, class_dim) embedding.

        Returns:
            (B, C, H, W) modulated features.
        """
        gamma_beta = self.mlp(class_embedding)        # (B, 2C)
        gamma, beta = gamma_beta.chunk(2, dim=1)      # (B, C) each
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)     # (B, C, 1, 1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)       # (B, C, 1, 1)
        normalized = self.norm(features)
        return (1.0 + gamma) * normalized + beta


if __name__ == "__main__":
    film = FiLM(class_dim=512, feature_channels=1024)
    n_params = sum(p.numel() for p in film.parameters())
    print(f"FiLM module")
    print(f"  Trainable params: {n_params:,}")

    features = torch.rand(2, 1024, 32, 32)
    class_emb = torch.rand(2, 512)

    out = film(features, class_emb)
    print(f"\nFeatures in:        {tuple(features.shape)}")
    print(f"Class embedding in: {tuple(class_emb.shape)}")
    print(f"Modulated out:      {tuple(out.shape)}  (expected: (2, 1024, 32, 32))")

    print(f"\nFeature stats before/after modulation:")
    print(f"  before: mean={features.mean():.4f}, std={features.std():.4f}")
    print(f"  after:  mean={out.mean():.4f},  std={out.std():.4f}")