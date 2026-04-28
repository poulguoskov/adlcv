"""Poster figure: top-2 in-distribution / OOC pairs by log-likelihood difference.

Selects the two (image, bbox) pairs where the trained model most confidently
distinguishes the correct in-distribution class from the swapped OOC class.
Renders a single row of 4 panels for inclusion on the poster.

Usage:
    uv run python -m src.partC.poster_figure
    # or with a different results JSON:
    uv run python -m src.partC.poster_figure --results outputs/checkpoints/partC_results_jacob_20ep.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import zoom

from src.config import CHECKPOINTS_DIR, FIGURES_DIR
from src.data.images import load_scene_image
from src.partC.inference import PlacementScorer


def select_top_pairs(
    in_dist_results: list[dict],
    ooc_results: list[dict],
    n_pairs: int = 2,
) -> list[tuple[dict, dict, float]]:
    """Pick the n pairs with the largest log-likelihood gap (in-dist - OOC).

    Pairs are matched by index — in_dist_results[i] and ooc_results[i] share
    the same image and bbox, only the class differs.

    Returns:
        List of (in_dist_example, ooc_example, log_lik_diff) tuples, sorted
        by diff descending. Pairs with NaN/inf scores are filtered out.
    """
    if len(in_dist_results) != len(ooc_results):
        raise ValueError(
            f"Length mismatch: in_dist={len(in_dist_results)}, ooc={len(ooc_results)}"
        )

    candidates = []
    for in_ex, ooc_ex in zip(in_dist_results, ooc_results):
        ll_in = in_ex["log_likelihood"]
        ll_ooc = ooc_ex["log_likelihood"]

        if any(np.isnan(x) or np.isinf(x) for x in [ll_in, ll_ooc]):
            continue
        if in_ex["bg_path"] != ooc_ex["bg_path"]:
            continue

        diff = ll_in - ll_ooc
        candidates.append((in_ex, ooc_ex, diff))

    candidates.sort(key=lambda x: x[2], reverse=True)
    return candidates[:n_pairs]


def render_panel(
    ax: plt.Axes,
    example: dict,
    scorer: PlacementScorer,
) -> None:
    """Render one panel: image + predicted heatmap + bbox + caption."""
    img = load_scene_image(example["bg_path"])
    heatmap = scorer.predict_heatmap(img, example["fg_class"])
    heatmap_2d = heatmap.sum(axis=0)
    heatmap_up = zoom(heatmap_2d, 512 / 32, order=1)

    ax.imshow(img)
    ax.imshow(heatmap_up, cmap="hot", alpha=0.5)

    x, y, w, h = example["bbox"]
    rect = plt.Rectangle(
        (x * 512, y * 512), w * 512, h * 512,
        fill=False, edgecolor="cyan", linewidth=2.5,
    )
    ax.add_patch(rect)

    is_anom = example.get("is_anomalous", False)
    ll = example["log_likelihood"]

    if is_anom and "original_class" in example:
        label = "OOC"
        title = (
            f"[{label}] class = {example['fg_class']}\n"
            f"(was {example['original_class']})\n"
            f"log-lik = {ll:.2f}"
        )
    else:
        label = "in-dist"
        title = (
            f"[{label}] class = {example['fg_class']}\n"
            f" \n"
            f"log-lik = {ll:.2f}"
        )

    ax.set_title(title, fontsize=11)
    ax.axis("off")


def plot_poster_figure(
    results_path: Path,
    output_path: Path,
    n_pairs: int = 2,
    device: str = "cpu",
) -> None:
    """Build the poster comparison figure from a results JSON."""
    with open(results_path) as f:
        results_data = json.load(f)

    in_dist_results = results_data["in_dist_results"]
    ooc_results = results_data["ooc_results"]
    ckpt_path = Path(results_data["checkpoint"])

    print(f"Loaded {len(in_dist_results)} in-dist + {len(ooc_results)} OOC results")
    print(f"Checkpoint: {ckpt_path}")

    print("Selecting top pairs by log-likelihood difference...")
    pairs = select_top_pairs(in_dist_results, ooc_results, n_pairs=n_pairs)

    if len(pairs) < n_pairs:
        raise RuntimeError(f"Only found {len(pairs)} valid pairs, need {n_pairs}")

    print(f"\nTop {n_pairs} pairs:")
    for i, (in_ex, ooc_ex, diff) in enumerate(pairs):
        print(f"  {i + 1}. {in_ex['bg_path']}")
        print(f"     in-dist: {in_ex['fg_class']:15s}  log-lik = {in_ex['log_likelihood']:.2f}")
        print(f"     OOC:     {ooc_ex['fg_class']:15s}  log-lik = {ooc_ex['log_likelihood']:.2f}")
        print(f"     diff:    {diff:.2f}")

    print("\nLoading scorer for heatmap regeneration...")
    scorer = PlacementScorer(checkpoint_path=ckpt_path, device=device)

    n_panels = n_pairs * 2
    fig, axes = plt.subplots(1, n_panels, figsize=(4.5 * n_panels, 5.2))

    for i, (in_ex, ooc_ex, _) in enumerate(pairs):
        render_panel(axes[2 * i], in_ex, scorer)
        render_panel(axes[2 * i + 1], ooc_ex, scorer)

    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved figure to {output_path}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        type=Path,
        default=CHECKPOINTS_DIR / "partC_results_m1_full.json",
        help="Path to evaluation results JSON",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=FIGURES_DIR / "poster_partC_top_pairs.png",
        help="Where to save the figure",
    )
    parser.add_argument(
        "--n-pairs",
        type=int,
        default=2,
        help="Number of in-dist/OOC pairs to show (default: 2 = 4 panels)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Inference device (cpu / mps / cuda)",
    )
    args = parser.parse_args()

    if not args.results.exists():
        print(f"Results file not found: {args.results}")
        print("Run src.partC.evaluate first.")
        raise SystemExit(1)

    plot_poster_figure(
        results_path=args.results,
        output_path=args.output,
        n_pairs=args.n_pairs,
        device=args.device,
    )