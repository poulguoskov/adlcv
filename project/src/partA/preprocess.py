"""Batch preprocessing: build target tensors for every (scene, class) pair."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.config import (
    LABEL_FILTER,
    PREPROCESSED_DIR,
    SCALE_BIN_EDGES_PATH,
    TARGETS_TRAIN_DIR,
)
from src.partA.scale_bins import load_edges
from src.partA.targets import build_target_tensor


def build_class_vocab(df: pd.DataFrame) -> dict[str, int]:
    """Assign integer ids to each unique fg_class. Sorted alphabetically."""
    classes = sorted(df["fg_class"].unique())
    return {cls: i for i, cls in enumerate(classes)}


def preprocess_split(
    df: pd.DataFrame,
    output_dir: Path,
    scale_bin_edges: np.ndarray,
    label: int = LABEL_FILTER,
    min_positives: int = 1,
) -> None:
    """Process every (scene, class) pair into a saved target tensor + index."""
    output_dir.mkdir(parents=True, exist_ok=True)

    class_to_id = build_class_vocab(df)
    print(f"Number of classes: {len(class_to_id)}")

    sub = df[df["label"] == label].reset_index(drop=True)
    print(f"Positive rows: {len(sub):,}")

    grouped = sub.groupby(["bg_path", "fg_class"], sort=False)
    print(f"Number of (scene, class) pairs: {len(grouped):,}")

    index: list[dict] = []

    for sample_id, ((bg_path, fg_class), rows) in enumerate(
        tqdm(grouped, desc="Building targets")
    ):
        if len(rows) < min_positives:
            continue

        try:
            target = build_target_tensor(rows, scale_bin_edges=scale_bin_edges)
        except Exception as e:
            print(f"\nFailed for ({bg_path}, {fg_class}): {e}")
            continue

        save_path = output_dir / f"{sample_id:07d}.npz"
        np.savez_compressed(
            save_path,
            target=target,
            bg_path=np.array(bg_path),
            fg_class=np.array(fg_class),
            class_id=np.int64(class_to_id[fg_class]),
            num_rows=np.int64(len(rows)),
        )

        index.append({
            "sample_id": sample_id,
            "bg_path": bg_path,
            "fg_class": fg_class,
            "class_id": class_to_id[fg_class],
            "target_path": str(save_path),
            "num_rows": len(rows),
        })

    with open(output_dir / "index.json", "w") as f:
        json.dump(index, f, indent=2)

    with open(PREPROCESSED_DIR / "class_to_id.json", "w") as f:
        json.dump(class_to_id, f, indent=2)

    print(f"\nSaved {len(index)} target tensors to {output_dir}")
    print(f"Saved index.json and class_to_id.json")


if __name__ == "__main__":
    from src.data.annotations import load_as_dataframe

    print("Loading annotations...")
    df = load_as_dataframe("train")

    edges = load_edges(SCALE_BIN_EDGES_PATH)

    preprocess_split(df, TARGETS_TRAIN_DIR, scale_bin_edges=edges)