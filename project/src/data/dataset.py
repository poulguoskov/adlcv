"""PyTorch Dataset for training Part B's heatmap prediction model.

Each item: (image_tensor, class_embedding, target_heatmap, fg_class_name).
Targets are pre-computed by Part A and saved to disk; this Dataset only loads them.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from src.config import IMG_SIZE
from src.data.images import load_scene_image


class HeatmapDataset(Dataset):
    """Loads (image, class_embedding, target_heatmap) triplets.

    Expects an index file (JSON) listing all samples, plus pre-computed target
    .npz files and a cached class-embedding .pt dict. All produced by Part A's
    preprocessing pipeline.
    """

    def __init__(
        self,
        index_path: Path,
        places365_root: Path,
        class_embeddings_path: Path,
        img_size: int = IMG_SIZE,
    ) -> None:
        with open(index_path, "r") as f:
            self.index: list[dict] = json.load(f)

        self.places365_root = Path(places365_root)
        self.class_embeddings: dict[str, torch.Tensor] = torch.load(
            class_embeddings_path, weights_only=False
        )
        self.img_size = img_size

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
        item = self.index[idx]

        img_np = load_scene_image(item["bg_path"], root=self.places365_root)
        img = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0

        target_data = np.load(item["target_path"])
        target = torch.from_numpy(target_data["target"]).float()

        fg_class = item["fg_class"]
        class_embed = self.class_embeddings[fg_class].float()

        return img, class_embed, target, fg_class