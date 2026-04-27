"""Loading background images from the Places365 subset.

Bboxes in the dataset are normalized to a 512x512 center crop, so all
image loading goes through center_crop_512 to ensure consistency.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from src.config import IMG_SIZE, PLACES365_ROOT


def center_crop_to_square(img: Image.Image, size: int = IMG_SIZE) -> Image.Image:
    """Resize the shortest side to `size`, then center crop to size x size.

    Matches the preprocessing implied by Marco's annotation pipeline.
    """
    w, h = img.size
    short = min(w, h)
    scale = size / short
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    img = img.resize((new_w, new_h), Image.BILINEAR)

    left = (new_w - size) // 2
    top = (new_h - size) // 2
    return img.crop((left, top, left + size, top + size))


def load_scene_image(bg_path: str, root: Path = PLACES365_ROOT) -> np.ndarray:
    """Load a background image, center-crop to 512x512, return as HxWx3 uint8.

    Args:
        bg_path: Relative path from the dataset, e.g.
            "data_large_standard/k/kitchen/00002986.jpg".
        root: Root directory containing the Places365 subset.

    Returns:
        np.ndarray of shape (IMG_SIZE, IMG_SIZE, 3), dtype uint8.

    Raises:
        FileNotFoundError: if the image does not exist on disk.
    """
    full_path = root / bg_path
    if not full_path.exists():
        raise FileNotFoundError(f"Image not found: {full_path}")

    img = Image.open(full_path).convert("RGB")
    img = center_crop_to_square(img)
    return np.array(img)


if __name__ == "__main__":
    img = load_scene_image("data_large_standard/k/kitchen/00002986.jpg")
    print(f"Loaded image: shape={img.shape}, dtype={img.dtype}")
    print(f"Pixel range:  [{img.min()}, {img.max()}]")