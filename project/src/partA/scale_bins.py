"""Compute global scale bin edges for the target tensor's scale dimension.

Each bbox has a log-area = log(w * h). We split the global range of log-areas
into NUM_SCALES equal-width bins. These bin edges are computed once over the
whole training set and then reused everywhere — Part A's target construction
and Part C's likelihood lookup must use the same edges.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.config import NUM_SCALES, SCALE_BIN_EDGES_PATH


def compute_scale_bin_edges(
    df: pd.DataFrame,
    num_scales: int = NUM_SCALES,
    label: int = 1,
) -> np.ndarray:
    """Compute equal-width bin edges over log(w*h) for all positive bboxes.

    Args:
        df: Full annotations DataFrame.
        num_scales: Number of scale bins (returns num_scales + 1 edges).
        label: Filter to label=1 (positives) only.

    Returns:
        np.ndarray of shape (num_scales + 1,) with bin edges in log-area space.
    """
    positives = df[df["label"] == label]
    bboxes = np.array(positives["bbox"].tolist())  # (N, 4)
    w, h = bboxes[:, 2], bboxes[:, 3]
    log_areas = np.log(np.clip(w * h, 1e-12, None))

    min_la, max_la = log_areas.min(), log_areas.max()
    if np.isclose(min_la, max_la):
        max_la = min_la + 1e-3

    return np.linspace(min_la, max_la, num_scales + 1)


def assign_scale_bin(log_area: float, edges: np.ndarray) -> int:
    """Assign one log_area value to a scale bin index in [0, num_scales)."""
    bin_idx = np.searchsorted(edges, log_area, side="right") - 1
    return int(np.clip(bin_idx, 0, len(edges) - 2))


def save_edges(edges: np.ndarray, path: Path = SCALE_BIN_EDGES_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, edges)


def load_edges(path: Path = SCALE_BIN_EDGES_PATH) -> np.ndarray:
    return np.load(path)


if __name__ == "__main__":
    from src.data.annotations import load_as_dataframe

    print("Loading annotations...")
    df = load_as_dataframe("train")

    print(f"Computing scale bin edges (over {(df['label'] == 1).sum():,} positives)...")
    edges = compute_scale_bin_edges(df)

    print(f"\nScale bin edges (log-area space):")
    for i, e in enumerate(edges):
        area_pct = np.exp(e) * 100  # area as % of image
        print(f"  edge {i}: log_area={e:.3f}  →  area={area_pct:.2f}% of image")

    save_edges(edges)
    print(f"\nSaved to: {SCALE_BIN_EDGES_PATH}")