"""CNN decoder: 1024-channel features → 8-channel logits.

Maps the post-transformer features back to the target tensor shape.
The output is logits — softmax is applied later in the loss function,
not here. Spatial resolution stays at 32x32 throughout.
"""
from __future__ import annotations

import torch
from torch import nn


class HeatmapDecoder(nn.Module):
    """Three 3x3 conv layers projecting from feature_dim to num_scales channels.

    Input:  (B, feature_dim, H, W)
    Output: (B, num_scales, H, W) logits.
    """

    def __init__(
        self,
        feature_dim: int = 1024,
        hidden_dim: int = 512,
        num_scales: int = 8,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(feature_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, num_scales, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


if __name__ == "__main__":
    decoder = HeatmapDecoder(feature_dim=1024, hidden_dim=512, num_scales=8)

    n_params = sum(p.numel() for p in decoder.parameters())
    print(f"Heatmap decoder")
    print(f"  Trainable params: {n_params:,}")

    features = torch.rand(2, 1024, 32, 32)
    out = decoder(features)
    print(f"\nInput:  {tuple(features.shape)}")
    print(f"Output: {tuple(out.shape)}  (expected: (2, 8, 32, 32))")