"""Build the [8, 32, 32] target heatmap for one (scene, class) pair.

This is Part A's core contribution: aggregate ~1000 candidate bboxes into a
single probability distribution over (scale, y, x) on a discrete grid.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import (
    GRID_SIZE,
    NUM_SCALES,
    SCORE_TEMPERATURE,
    SIGMA_S,
    SIGMA_XY,
)
from src.partA.scale_bins import assign_scale_bin


def softmax_with_temperature(x: np.ndarray, temperature: float) -> np.ndarray:
    """Stable softmax with temperature scaling."""
    x = np.asarray(x, dtype=np.float64) / max(temperature, 1e-8)
    x = x - x.max()
    exp_x = np.exp(x)
    return exp_x / (exp_x.sum() + 1e-12)


def bbox_to_grid_position(
    bbox: list[float],
    grid_size: int,
    scale_bin_edges: np.ndarray,
) -> tuple[float, float, int]:
    """Convert a normalized [x, y, w, h] bbox to (gx, gy, scale_bin).

    gx, gy are continuous (not yet rounded to integer cells) so the Gaussian
    splat can land between cells.
    """
    x, y, w, h = bbox
    cx = x + w / 2.0
    cy = y + h / 2.0
    gx = cx * (grid_size - 1)
    gy = cy * (grid_size - 1)
    log_area = np.log(max(w * h, 1e-12))
    s_bin = assign_scale_bin(log_area, scale_bin_edges)
    return gx, gy, s_bin


def build_target_tensor(
    rows: pd.DataFrame,
    scale_bin_edges: np.ndarray,
    grid_size: int = GRID_SIZE,
    num_scales: int = NUM_SCALES,
    sigma_xy: float = SIGMA_XY,
    sigma_s: float = SIGMA_S,
    score_temperature: float = SCORE_TEMPERATURE,
    score_higher_is_better: bool = True,
) -> np.ndarray:
    """Build the [num_scales, grid_size, grid_size] target tensor.

    Vectorized: all bboxes' kernels are computed in one numpy operation.
    """
    if len(rows) == 0:
        raise ValueError("Cannot build target tensor from empty rows.")

    bboxes = np.asarray(rows["bbox"].tolist(), dtype=np.float64)  # (N, 4)
    raw_scores = np.asarray(rows["image_reward_score"].values, dtype=np.float64)

    score_for_weight = raw_scores if score_higher_is_better else -raw_scores
    weights = softmax_with_temperature(score_for_weight, temperature=score_temperature)

    cx = bboxes[:, 0] + bboxes[:, 2] / 2.0
    cy = bboxes[:, 1] + bboxes[:, 3] / 2.0
    gx = cx * (grid_size - 1)  # (N,)
    gy = cy * (grid_size - 1)  # (N,)

    log_areas = np.log(np.clip(bboxes[:, 2] * bboxes[:, 3], 1e-12, None))
    s_bins = np.searchsorted(scale_bin_edges, log_areas, side="right") - 1
    s_bins = np.clip(s_bins, 0, num_scales - 1).astype(np.float64)  # (N,)

    grid_x = np.arange(grid_size, dtype=np.float64)
    grid_y = np.arange(grid_size, dtype=np.float64)
    grid_s = np.arange(num_scales, dtype=np.float64)

    # Per-bbox 1D kernels: (N, grid_size) and (N, num_scales)
    kx = np.exp(-((grid_x[None, :] - gx[:, None]) ** 2) / (2.0 * sigma_xy ** 2))
    ky = np.exp(-((grid_y[None, :] - gy[:, None]) ** 2) / (2.0 * sigma_xy ** 2))
    ks = np.exp(-((grid_s[None, :] - s_bins[:, None]) ** 2) / (2.0 * sigma_s ** 2))

    # Weighted sum: combine kernels via einsum
    # Final shape: (num_scales, grid_size, grid_size)
    target = np.einsum("n,ns,ny,nx->syx", weights, ks, ky, kx)

    total = target.sum()
    if total > 0:
        target /= total

    return target.astype(np.float32)


if __name__ == "__main__":
    from src.data.annotations import get_pair_rows, load_as_dataframe
    from src.partA.scale_bins import load_edges

    df = load_as_dataframe("train")
    edges = load_edges()

    rows = get_pair_rows(
        df,
        bg_path="data_large_standard/k/kitchen/00002986.jpg",
        fg_class="bottle",
        label=1,
    )
    print(f"Pair: kitchen/bottle, {len(rows)} positives")

    target = build_target_tensor(rows, scale_bin_edges=edges)
    print(f"Target shape: {target.shape}")
    print(f"Sum:          {target.sum():.6f}  (should be ~1.0)")
    print(f"Min:          {target.min():.6f}")
    print(f"Max:          {target.max():.6f}")
    print(f"Per-scale-bin total mass:")
    for s in range(target.shape[0]):
        print(f"  bin {s}: {target[s].sum():.4f}")