# ADLCV Project — HiddenObjects spatial priors

Implementation of the ADLCV course project on diffusion-distilled spatial priors for object placement, following the group's overall approach with a cleaner code structure.

The project addresses three connected tasks:
- **Part A** — build per-(scene, class) target distributions from the dataset's bbox annotations
- **Part B** — train a neural network to predict these distributions from images alone
- **Part C** — use the trained model to detect out-of-context object placements

Each part feeds the next: Part A produces the supervision targets for Part B, and Part B produces the model used in Part C.

## Quick start

You only need `uv` installed. Everything else (Python, dependencies, virtual environment) is handled automatically by `uv run`.

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and enter the repo
git clone https://github.com/poulguoskov/adlcv.git
cd adlcv/project

# Download the Places365 subset from Marco's Drive link
# Place it at: project/data/places365_subset/data_large_standard/...
# (the dataset's bg_path field expects this prefix)

# Run any script — uv handles the rest
uv run python -m src.partA.scale_bins
```

`uv run` resolves dependencies from `pyproject.toml`, creates a venv if needed, and runs the script. No manual `pip install` step.

## Folder structure

```
project/
├── README.md                      # this file
├── data/                          # gitignored
│   ├── places365_subset/          # Places365 background images (manual download)
│   └── preprocessed/              # generated Part A artifacts
├── outputs/                       # gitignored
│   ├── checkpoints/               # trained model weights
│   ├── figures/                   # plots and galleries
│   └── train.log                  # training stdout
├── notebooks/                     # exploratory work, sanity checks
└── src/
    ├── config.py                  # all paths and hyperparameters in one place
    ├── data/
    │   ├── annotations.py         # load HF dataset, filter pairs
    │   ├── images.py              # load Places365 backgrounds, center crop
    │   └── dataset.py             # PyTorch Dataset for training
    ├── partA/
    │   ├── scale_bins.py          # global log-area bin edges
    │   ├── targets.py             # build [8, 32, 32] target tensor for one pair
    │   ├── preprocess.py          # batch-build all targets, save to disk
    │   └── text_embeddings.py     # cache CLIP embeddings for the 50 classes
    ├── partB/
    │   ├── backbone.py            # frozen ResNet-50 (C4 layer)
    │   ├── film.py                # FiLM class conditioning
    │   ├── transformer.py         # encoder-only transformer
    │   ├── decoder.py             # CNN decoder to [8, 32, 32]
    │   ├── model.py               # composes backbone + FiLM + transformer + decoder
    │   └── train.py               # training loop with KL divergence loss
    └── partC/
        ├── inference.py           # PlacementScorer: bbox → log-likelihood
        ├── test_sets.py           # build in-distribution + OOC test sets
        ├── evaluate.py            # AUROC + PR-AUC + curves
        └── visualize.py           # histograms, ROC, PR, qualitative galleries
```

Every module under `src/` has a `__main__` block so it can be run standalone for testing. Use `uv run python -m src.partA.targets` (etc.) to test each component.

## Pipeline overview

The whole project boils down to one core function: `p(bbox | image, class)` — the probability that an object goes at a specific bbox, given a scene and a class. The three parts are different ways of building or using this function.

```
Raw HF dataset (15.9M bbox annotations)
        │
        ▼
   Part A preprocess
        │
        ▼
[8, 32, 32] target tensors per (scene, class) pair
        │
        ▼
   Part B training
        │
        ▼
Trained PlacementModel (image + class → predicted heatmap)
        │
        ▼
   Part C evaluation
        │
        ▼
AUROC for out-of-context detection
```

## Part A — target heatmap construction

For each (scene, class) pair in the dataset (~25k pairs total), aggregate the ~1000 candidate bboxes into a single `[8, 32, 32]` probability distribution over (scale, y, x).

Each bbox contributes a Gaussian blob centered at its grid position, weighted by its `image_reward_score`. The blobs are summed and normalized to sum to 1.

**Run order:**

```bash
# 1. Compute global scale bin edges (log-area space, 8 equal-width bins).
uv run python -m src.partA.scale_bins
# Saves: data/preprocessed/scale_bin_edges.npy

# 2. Smoke test for one (scene, class) pair.
uv run python -m src.partA.targets

# 3. Batch-build all 25k targets.
uv run python -m src.partA.preprocess
# Saves: data/preprocessed/train/*.npz + index.json + class_to_id.json

# 4. Cache CLIP text embeddings for the 50 classes.
uv run python -m src.partA.text_embeddings
# Saves: data/preprocessed/class_embeddings.pt
```

After all four, Part A is complete. Total runtime: ~3 minutes on M1.

**Key design choices:**

- `score_higher_is_better=True` (ImageReward is higher = better, the group's `data_preprocess.py` had this inverted as `False` which gives the opposite supervision signal — likely a bug)
- Bilinear-style Gaussian splatting in 3D (location + scale)
- Global scale bin edges computed across all positives, ensuring consistency across all (scene, class) targets

## Part B — the placement model

A neural network that takes an image and a class embedding, and predicts the same `[8, 32, 32]` distribution that Part A produced. Trained with KL divergence between predicted and target distributions.

**Architecture** (forward pass for one image):

```
image [B, 3, 512, 512]
  ↓ frozen ResNet-50 (C4)
features [B, 1024, 32, 32]
  ↓ FiLM conditioning on class_embedding [B, 512]
modulated features [B, 1024, 32, 32]
  ↓ flatten to tokens
tokens [B, 1024, 1024]
  ↓ Transformer encoder (2 layers, 8 heads)
tokens [B, 1024, 1024]
  ↓ reshape back
features [B, 1024, 32, 32]
  ↓ CNN decoder
logits [B, 8, 32, 32]
```

Loss: KL divergence over the flattened 8192-dim distribution per sample.

**Important fix vs. the group's code:** ImageNet normalization is applied inside the backbone (the group's code feeds raw [0, 1] images, which gives ResNet garbage features). Also, see the CLIP embedding bug fix below.

**Run order:**

```bash
# Smoke test (200 samples, 2 epochs, ~1 min on M1).
# Edit src/partB/train.py __main__ to use subset_size=200, num_epochs=2
uv run python -m src.partB.train

# Full training (~5 hours on M1, batch size 8, 5 epochs).
# Edit __main__ to use subset_size=None, num_epochs=5, checkpoint_name="m1_full"
caffeinate -i uv run python -m src.partB.train 2>&1 | tee outputs/train.log
```

`caffeinate -i` keeps the M1 awake during long runs. Skip on Linux.

Each individual model component (backbone, FiLM, transformer, decoder) can be tested standalone:

```bash
uv run python -m src.partB.backbone     # tests the ResNet
uv run python -m src.partB.film         # tests FiLM
uv run python -m src.partB.transformer  # tests the encoder
uv run python -m src.partB.decoder      # tests the decoder
uv run python -m src.partB.model        # tests the full model
```

## Part C — out-of-context detection

For a query `(image, class, bbox)`, compute its log-likelihood under the trained Part B model. Low log-likelihood = anomalous placement.

**Test sets:**

- **In-distribution**: 500 high-reward positives sampled from Marco's `test` split (held-out scenes the model never saw)
- **OOC**: same 500 examples, but with the class swapped to something from a different category (e.g. `car on highway` → `scissors on highway`). Tests scene-class compatibility — same image, same bbox, only the class is wrong.

**Metrics:** AUROC, PR-AUC, score histograms, qualitative galleries.

**Run order:**

```bash
# 1. Build the test sets.
uv run python -m src.partC.test_sets
# Saves: data/preprocessed/partC_test_sets/in_distribution.json + ooc_class_swap.json

# 2. Sanity-check inference on one bbox.
uv run python -m src.partC.inference

# 3. Score both test sets, compute AUROC.
# (Edit src/partC/evaluate.py __main__ to point at the right checkpoint)
uv run python -m src.partC.evaluate

# 4. Generate figures.
uv run python -m src.partC.visualize
# Saves to outputs/figures/
```

## The CLIP embedding bug

Worth flagging because the group's code has the same bug.

**Problem:** the original code reads `outputs.last_hidden_state[:, 0, :]` from CLIP's text encoder. That's the BOS token's hidden state, which is the same regardless of input text. So every class gets the same 512-dim embedding, and the model can't distinguish classes at all.

**Symptom:** loss decreases normally during training (the model learns a class-agnostic spatial prior averaged over the dataset). But Part C OOC detection gets AUROC = 0.500 exactly — the model literally cannot tell `car` from `scissors`.

**Fix:** use `outputs.pooler_output` instead (the EOS token after layernorm — CLIP's canonical sentence-level embedding), and use the standard prompt template `"a photo of a {class}"` since that's what CLIP was trained on.

```python
# WRONG
embedding = outputs.last_hidden_state[:, 0, :]

# RIGHT
prompt = f"a photo of a {cls}"
tokens = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
outputs = text_model(**tokens)
embedding = outputs.pooler_output.squeeze(0)
```

To diagnose this in the group's code: compare predicted heatmaps for the same image with two different classes. If they're identical bit-for-bit (L2 distance = 0), the bug is present.

## Configuration

All paths and hyperparameters live in `src/config.py`. Edit there, never hardcode.

Key values:

```python
GRID_SIZE = 32                # spatial grid (32x32 = 1024 cells)
NUM_SCALES = 8                # scale bins
SIGMA_XY = 1.25               # Gaussian splat width in spatial dim
SIGMA_S = 0.6                 # Gaussian splat width in scale dim
SCORE_TEMPERATURE = 0.3       # softmax temperature for reward weighting
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
NUM_EPOCHS = 20
```

## Notes on running

**Disk usage:**
- Places365 subset: ~5 GB (depends on download)
- Preprocessed targets: ~720 MB
- Trained checkpoint: ~160 MB per save (best + last)

**M1 training speed:**
- Smoke test (200 samples × 2 epochs): ~1 min
- Full run (25k samples × 5 epochs): ~5 hours, batch size 8

**HPC migration:**
- Increase batch size to 32+
- Use AMP (automatic mixed precision) — already wired up for CUDA in `train.py`
- Adjust `num_workers` based on the queue's CPU allocation
- Expected: ~30 min per epoch on a single A100

## Status

- ✅ Part A complete (targets generated, embeddings cached)
- 🟡 Part B training in progress
- ✅ Part C pipeline complete (will produce final numbers once Part B finishes)