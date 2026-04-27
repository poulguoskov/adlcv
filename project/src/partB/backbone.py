"""Frozen ResNet-50 backbone for image feature extraction.

Uses ImageNet-pretrained weights from torchvision. We extract the C4 stage
output, which has 1024 channels at 32x32 spatial resolution for a 512x512 input.
This resolution matches the GRID_SIZE used in Part A's targets.

The backbone is frozen — all parameters have requires_grad=False — so it acts
as a fixed feature extractor. Only the downstream model components are trained.
"""
from __future__ import annotations

import torch
from torch import nn
from torchvision.models import ResNet50_Weights, resnet50


class FrozenResNet50C4(nn.Module):
    """ResNet-50 with all layers up to C4, frozen.

    Input:  (B, 3, 512, 512) image tensor, normalized to [0, 1].
    Output: (B, 1024, 32, 32) feature map.
    """

    def __init__(self) -> None:
        super().__init__()
        weights = ResNet50_Weights.DEFAULT
        net = resnet50(weights=weights)

        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3  # outputs C4

        for p in self.parameters():
            p.requires_grad = False
        self.eval()

        self.register_buffer("imagenet_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("imagenet_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def train(self, mode: bool = True) -> "FrozenResNet50C4":
        """Override: keep BatchNorm in eval mode regardless of training flag."""
        return super().train(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.imagenet_mean) / self.imagenet_std
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


if __name__ == "__main__":
    model = FrozenResNet50C4()
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"ResNet-50 C4 backbone")
    print(f"  Total params:     {n_total:,}")
    print(f"  Trainable params: {n_trainable:,}  (should be 0)")

    x = torch.rand(2, 3, 512, 512)
    with torch.no_grad():
        out = model(x)
    print(f"\nInput shape:  {tuple(x.shape)}")
    print(f"Output shape: {tuple(out.shape)}  (expected: (2, 1024, 32, 32))")