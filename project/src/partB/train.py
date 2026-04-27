"""Training loop for the placement model.

Uses KL-divergence between the predicted distribution (softmax over the
flattened 8*32*32 cells) and the target distribution (already normalized to
sum to 1). Trains only the unfrozen parameters: FiLM, transformer, decoder.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from src.config import (
    BATCH_SIZE,
    CHECKPOINTS_DIR,
    CLASS_EMBEDDINGS_PATH,
    LEARNING_RATE,
    NUM_EPOCHS,
    PLACES365_ROOT,
    TARGETS_TRAIN_DIR,
)
from src.data.dataset import HeatmapDataset
from src.partB.model import PlacementModel


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def kl_divergence_loss(pred_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """KL(target || pred) over the joint (scale, y, x) distribution.

    Args:
        pred_logits: (B, S, H, W) raw model output.
        target: (B, S, H, W) probability distribution summing to 1 per sample.
    """
    B = pred_logits.size(0)
    pred_log_probs = F.log_softmax(pred_logits.reshape(B, -1), dim=1)
    target_probs = target.reshape(B, -1)
    target_probs = target_probs / target_probs.sum(dim=1, keepdim=True).clamp(min=1e-12)
    return F.kl_div(pred_log_probs, target_probs, reduction="batchmean")


def train_one_epoch(
    model: PlacementModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    epoch: int,
    use_amp: bool,
    scaler: torch.amp.GradScaler | None,
) -> float:
    """Run one epoch of training. Returns average loss across batches."""
    model.train()
    running_loss = 0.0
    n_batches = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch}")
    for images, class_embeds, targets, _ in pbar:
        images = images.to(device, non_blocking=True)
        class_embeds = class_embeds.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if use_amp and scaler is not None:
            with torch.amp.autocast("cuda"):
                logits = model(images, class_embeds)
                loss = kl_divergence_loss(logits, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images, class_embeds)
            loss = kl_divergence_loss(logits, targets)
            loss.backward()
            optimizer.step()

        running_loss += loss.item()
        n_batches += 1
        pbar.set_postfix(loss=f"{running_loss / n_batches:.4f}")

    return running_loss / max(n_batches, 1)


def train(
    num_epochs: int = NUM_EPOCHS,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = LEARNING_RATE,
    subset_size: int | None = None,
    num_workers: int = 4,
    checkpoint_name: str = "model",
) -> None:
    """Main training entrypoint.

    Args:
        num_epochs: Number of epochs to train.
        batch_size: Per-step batch size.
        learning_rate: AdamW learning rate.
        subset_size: If set, only train on the first N samples (smoke testing).
        num_workers: DataLoader worker processes.
        checkpoint_name: Filename prefix for saved checkpoints.
    """
    device = get_device()
    print(f"Device: {device}")

    index_path = TARGETS_TRAIN_DIR / "index.json"
    dataset = HeatmapDataset(
        index_path=index_path,
        places365_root=PLACES365_ROOT,
        class_embeddings_path=CLASS_EMBEDDINGS_PATH,
    )
    print(f"Full dataset size: {len(dataset):,}")

    if subset_size is not None:
        dataset = Subset(dataset, range(min(subset_size, len(dataset))))
        print(f"Using subset of size: {len(dataset)}")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
        persistent_workers=(num_workers > 0),
    )

    model = PlacementModel().to(device)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_trainable:,}")

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
    )

    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    epoch_losses: list[float] = []
    best_loss = float("inf")
    best_path = CHECKPOINTS_DIR / f"{checkpoint_name}_best.pt"
    last_path = CHECKPOINTS_DIR / f"{checkpoint_name}_last.pt"

    t0 = time.time()
    for epoch in range(1, num_epochs + 1):
        avg_loss = train_one_epoch(
            model, loader, optimizer, device, epoch,
            use_amp=use_amp, scaler=scaler,
        )
        epoch_losses.append(avg_loss)
        elapsed = time.time() - t0
        print(f"Epoch {epoch}: avg_loss={avg_loss:.4f} | elapsed={elapsed/60:.1f} min")

        ckpt = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch_losses": epoch_losses,
            "config": {
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "subset_size": subset_size,
            },
        }
        torch.save(ckpt, last_path)
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(ckpt, best_path)
            print(f"  Saved best checkpoint (loss={best_loss:.4f})")

    print(f"\nDone. Total time: {(time.time() - t0)/60:.1f} min")
    print(f"Best loss: {best_loss:.4f}")
    print(f"Checkpoints: {best_path}, {last_path}")


if __name__ == "__main__":
    train(
        num_epochs=5,
        batch_size=8,
        learning_rate=1e-4,
        subset_size=None,         # full dataset
        num_workers=4,
        checkpoint_name="m1_full",
    )