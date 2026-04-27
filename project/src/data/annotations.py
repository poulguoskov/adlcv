"""Loading and filtering the HiddenObjects HuggingFace dataset.

Each row in the dataset is one (scene, object_class, candidate_bbox) triplet
with a reward score. See https://huggingface.co/datasets/marco-schouten/hidden-objects
for the full schema.
"""
from __future__ import annotations

import pandas as pd
from datasets import Dataset, load_dataset

from src.config import HF_DATASET_NAME


def load_hf_split(split: str = "train") -> Dataset:
    """Load one split of the HiddenObjects dataset from HuggingFace.

    Args:
        split: "train" or "test".

    Returns:
        Raw HuggingFace Dataset (15.9M rows for train, 1.7M for test).
    """
    return load_dataset(HF_DATASET_NAME, split=split)


def load_as_dataframe(split: str = "train") -> pd.DataFrame:
    """Load a split as a pandas DataFrame.

    Much faster than HF for repeated filtering. Loads ~3.6 GB into memory.
    """
    return load_hf_split(split).to_pandas()


def get_pair_rows(
    df: pd.DataFrame,
    bg_path: str,
    fg_class: str,
    label: int | None = 1,
) -> pd.DataFrame:
    """Return rows for a single (scene, class) pair.

    Args:
        df: Full annotations DataFrame.
        bg_path: Background image relative path, e.g.
            "data_large_standard/k/kitchen/00002986.jpg".
        fg_class: Object class, e.g. "bottle".
        label: 1 for positives, 0 for negatives, None for both.

    Returns:
        Filtered DataFrame, index reset.
    """
    mask = (df["bg_path"] == bg_path) & (df["fg_class"] == fg_class)
    if label is not None:
        mask &= df["label"] == label
    return df[mask].reset_index(drop=True)


def list_unique_pairs(df: pd.DataFrame, min_positives: int = 1) -> pd.DataFrame:
    """Return a DataFrame of unique (bg_path, fg_class) pairs.

    Args:
        df: Full annotations DataFrame.
        min_positives: Only return pairs with at least this many label=1 rows.

    Returns:
        DataFrame with columns [bg_path, fg_class, n_positives].
    """
    pairs = (
        df[df["label"] == 1]
        .groupby(["bg_path", "fg_class"])
        .size()
        .reset_index(name="n_positives")
    )
    return pairs[pairs["n_positives"] >= min_positives].reset_index(drop=True)


if __name__ == "__main__":
    df = load_as_dataframe("train")
    print(f"Train rows: {len(df):,}")
    print(f"Columns:    {df.columns.tolist()}")

    pairs = list_unique_pairs(df, min_positives=1)
    print(f"Unique (scene, class) pairs with >=1 positive: {len(pairs):,}")

    sub = get_pair_rows(
        df,
        bg_path="data_large_standard/k/kitchen/00002986.jpg",
        fg_class="bottle",
    )
    print(f"\nSample pair: kitchen/bottle, label=1 → {len(sub)} rows")
    print(sub[["bbox", "image_reward_score"]].head(3))