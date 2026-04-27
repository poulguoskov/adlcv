"""The full PlacementModel composing all Part B components.

Forward pass:
    image (B, 3, 512, 512) + class_embedding (B, 512)
        → ResNet-50 C4 → (B, 1024, 32, 32)
        → FiLM(features, class_embedding) → (B, 1024, 32, 32)
        → flatten spatial → (B, 1024, 1024)  [seq_len = 32*32]
        → Transformer encoder → (B, 1024, 1024)
        → reshape to spatial → (B, 1024, 32, 32)
        → CNN decoder → (B, 8, 32, 32) logits
"""
from __future__ import annotations

import torch
from torch import nn

from src.config import (
    BACKBONE_FEATURE_DIM,
    CLASS_EMBED_DIM,
    GRID_SIZE,
    NUM_ATTENTION_HEADS,
    NUM_SCALES,
    NUM_TRANSFORMER_LAYERS,
)
from src.partB.backbone import FrozenResNet50C4
from src.partB.decoder import HeatmapDecoder
from src.partB.film import FiLM
from src.partB.transformer import TransformerEncoder


class PlacementModel(nn.Module):
    """End-to-end heatmap predictor for object placement."""

    def __init__(
        self,
        feature_dim: int = BACKBONE_FEATURE_DIM,
        class_dim: int = CLASS_EMBED_DIM,
        num_scales: int = NUM_SCALES,
        grid_size: int = GRID_SIZE,
        num_layers: int = NUM_TRANSFORMER_LAYERS,
        num_heads: int = NUM_ATTENTION_HEADS,
    ) -> None:
        super().__init__()
        self.grid_size = grid_size
        self.feature_dim = feature_dim

        self.backbone = FrozenResNet50C4()
        self.film = FiLM(class_dim=class_dim, feature_channels=feature_dim)
        self.transformer = TransformerEncoder(
            embed_dim=feature_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            max_seq_len=grid_size * grid_size,
        )
        self.decoder = HeatmapDecoder(
            feature_dim=feature_dim,
            hidden_dim=feature_dim // 2,
            num_scales=num_scales,
        )

    def forward(
        self,
        image: torch.Tensor,
        class_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """Compute placement heatmap logits.

        Args:
            image: (B, 3, H, W) RGB image in [0, 1].
            class_embedding: (B, class_dim) text embedding.

        Returns:
            (B, num_scales, grid_size, grid_size) logits.
        """
        features = self.backbone(image)
        features = self.film(features, class_embedding)

        B, C, H, W = features.shape
        seq = features.flatten(2).transpose(1, 2)   # (B, H*W, C)
        seq = self.transformer(seq)
        features = seq.transpose(1, 2).reshape(B, C, H, W)

        return self.decoder(features)


if __name__ == "__main__":
    model = PlacementModel()

    n_total = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"PlacementModel")
    print(f"  Total params:     {n_total:,}")
    print(f"  Trainable params: {n_trainable:,}")
    print(f"  Frozen params:    {n_total - n_trainable:,}  (ResNet-50 backbone)")

    image = torch.rand(2, 3, 512, 512)
    class_emb = torch.rand(2, 512)

    out = model(image, class_emb)
    print(f"\nInput image:           {tuple(image.shape)}")
    print(f"Input class embedding: {tuple(class_emb.shape)}")
    print(f"Output logits:         {tuple(out.shape)}  (expected: (2, 8, 32, 32))")