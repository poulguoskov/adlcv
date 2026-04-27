"""Project-wide configuration: paths, hyperparameters, constants.

Edit this file to point at different data locations or change hyperparameters.
Everything in the project should import from here, never hardcode paths.
"""
from __future__ import annotations

from pathlib import Path

# --- Paths ---
# Project root is the parent of the parent of this file: project/src/config.py -> project/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
CHECKPOINTS_DIR = OUTPUTS_DIR / "checkpoints"
FIGURES_DIR = OUTPUTS_DIR / "figures"

# Background images (Places365 subset from Marco's Drive)
PLACES365_ROOT = DATA_DIR / "places365_subset"

# Generated heatmap targets (from Part A preprocessing)
PREPROCESSED_DIR = DATA_DIR / "preprocessed"
TARGETS_TRAIN_DIR = PREPROCESSED_DIR / "train"
TARGETS_TEST_DIR = PREPROCESSED_DIR / "test"
SCALE_BIN_EDGES_PATH = PREPROCESSED_DIR / "scale_bin_edges.npy"
CLASS_EMBEDDINGS_PATH = PREPROCESSED_DIR / "class_embeddings.pt"

# --- HuggingFace dataset ---
HF_DATASET_NAME = "marco-schouten/hidden-objects"

# --- Image processing ---
IMG_SIZE = 512  # bbox normalization assumes a 512x512 center crop

# --- Part A: target heatmap construction ---
GRID_SIZE = 32           # spatial resolution of target tensor
NUM_SCALES = 8           # number of scale bins
SIGMA_XY = 1.25          # spatial Gaussian width (in grid cells)
SIGMA_S = 0.6            # scale Gaussian width (in scale bins)
SCORE_TEMPERATURE = 0.3  # softmax temperature for reward weighting
LABEL_FILTER = 1         # only use positives (label=1) for targets

# --- Part B: model architecture ---
BACKBONE_FEATURE_DIM = 1024  # ResNet-50 C4 output channels
CLASS_EMBED_DIM = 512        # CLIP text encoder output dim
NUM_TRANSFORMER_LAYERS = 2
NUM_ATTENTION_HEADS = 8

# --- Part B: training ---
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
NUM_EPOCHS = 20  # we'll do shorter runs first


def ensure_dirs() -> None:
    """Create all output directories if they don't exist."""
    for d in [DATA_DIR, OUTPUTS_DIR, CHECKPOINTS_DIR, FIGURES_DIR, PREPROCESSED_DIR]:
        d.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    print("Project root:", PROJECT_ROOT)
    print("Data dir:    ", DATA_DIR)
    print("Outputs dir: ", OUTPUTS_DIR)
    print()
    print("Existing files:")
    for path in [PLACES365_ROOT, PREPROCESSED_DIR]:
        marker = "✓" if path.exists() else "✗"
        print(f"  {marker} {path}")