"""Pre-compute and cache CLIP text embeddings for the 50 fg_classes.

We use openai/clip-vit-base-patch32 — same model the group used. Each class name
gets a 512-dim embedding from CLIP's text encoder. These are saved as a dict
{class_name: tensor} that the training Dataset looks up per sample.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from transformers import CLIPTextModel, CLIPTokenizer

from src.config import CLASS_EMBEDDINGS_PATH, PREPROCESSED_DIR


CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"


def compute_class_embeddings(
    class_names: list[str],
    model_name: str = CLIP_MODEL_NAME,
    device: str = "cpu",
) -> dict[str, torch.Tensor]:
    """Run CLIP's text encoder on each class name, return a name→embedding dict.

    Uses CLIP's pooler_output — the hidden state at the EOS token after final
    layernorm. This is the canonical sentence-level text embedding from CLIP.

    NOTE: We previously used last_hidden_state[:, 0, :] (the BOS token state),
    which is identical for every input — a silent bug that gives every class
    the same embedding.

    Returns:
        Dict mapping class_name → 512-dim tensor (CPU, float32).
    """
    tokenizer = CLIPTokenizer.from_pretrained(model_name)
    text_model = CLIPTextModel.from_pretrained(model_name).to(device)
    text_model.eval()

    embeddings: dict[str, torch.Tensor] = {}

    with torch.no_grad():
        for cls in class_names:
            prompt = f"a photo of a {cls}"  # CLIP standard prompt template
            tokens = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
            tokens = {k: v.to(device) for k, v in tokens.items()}
            outputs = text_model(**tokens)
            embedding = outputs.pooler_output.squeeze(0).cpu().float()
            embeddings[cls] = embedding

    return embeddings


def save_embeddings(
    embeddings: dict[str, torch.Tensor],
    path: Path = CLASS_EMBEDDINGS_PATH,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(embeddings, path)


def load_embeddings(path: Path = CLASS_EMBEDDINGS_PATH) -> dict[str, torch.Tensor]:
    return torch.load(path, weights_only=False)


if __name__ == "__main__":
    class_to_id_path = PREPROCESSED_DIR / "class_to_id.json"
    with open(class_to_id_path) as f:
        class_to_id = json.load(f)

    class_names = sorted(class_to_id.keys())
    print(f"Loaded {len(class_names)} classes from {class_to_id_path.name}")
    print(f"First 5: {class_names[:5]}")

    print(f"\nComputing CLIP embeddings (model: {CLIP_MODEL_NAME})...")
    embeddings = compute_class_embeddings(class_names)

    save_embeddings(embeddings)
    print(f"Saved to: {CLASS_EMBEDDINGS_PATH}")

    sample_cls = class_names[0]
    sample_emb = embeddings[sample_cls]
    print(f"\nSample: '{sample_cls}' → shape={tuple(sample_emb.shape)}, dtype={sample_emb.dtype}")
    print(f"First 5 values: {sample_emb[:5].tolist()}")