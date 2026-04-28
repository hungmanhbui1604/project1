import argparse
import os
import random

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import wandb
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from data import PADDataset
from metrics import compute_pad_metrics
from models import get_model
from schedulers import get_scheduler
from transforms import get_transforms

# ---------------------------------------------------------------------------
# DDP Utilities
# ---------------------------------------------------------------------------


def setup_ddp() -> tuple[int, int]:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", device_id=local_rank)
    return local_rank, dist.get_world_size()


def cleanup_ddp() -> None:
    dist.destroy_process_group()


def is_main() -> bool:
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _unwrap(module):
    return module.module if isinstance(module, DDP) else module


# ---------------------------------------------------------------------------
# Optimizer Selection
# ---------------------------------------------------------------------------


def get_optimizer(opt_name: str, parameters: list, opt_cfg: dict):
    if opt_name == "adamw":
        return torch.optim.AdamW(
            parameters, lr=opt_cfg["lr"], weight_decay=opt_cfg["weight_decay"]
        )

    raise ValueError("Unknown optimizer: " + opt_name)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    epoch: int,
) -> dict[str, float]:
    model.eval()

    total_loss = 0.0
    all_probs, all_labels = [], []

    pbar = tqdm(
        val_loader,
        desc=f"Epoch {epoch:03d} [val]",
        leave=False,
        unit="batch",
        disable=not is_main(),
    )

    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.autocast(device_type="cuda"):
            _, logits = model(images, branch="b")
            loss = F.binary_cross_entropy_with_logits(logits.squeeze(1), labels.float())

        total_loss += loss.item()

        probs = torch.sigmoid(logits)
        all_probs.append(probs.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    avg_loss = total_loss / len(val_loader)

    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)
    metrics = compute_pad_metrics(all_labels, all_probs)

    return {
        "loss": avg_loss,
        **metrics,
    }


# ---------------------------------------------------------------------------
# Checkpoint Management
# ---------------------------------------------------------------------------


def load_checkpoint(
    path: str,
    model: DDP,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    scaler: torch.amp.GradScaler,
) -> tuple[int, float]:
    start_epoch = 1
    best_ace = float("inf")

    if os.path.isfile(path):
        if is_main():
            print(f"=> Loading checkpoint '{path}'")
        ckpt_dict = torch.load(path, map_location="cpu")
        _unwrap(model).load_state_dict(ckpt_dict["model"])
        if "optimizer" in ckpt_dict:
            optimizer.load_state_dict(ckpt_dict["optimizer"])
        if "scheduler" in ckpt_dict:
            scheduler.load_state_dict(ckpt_dict["scheduler"])
        if "scaler" in ckpt_dict:
            scaler.load_state_dict(ckpt_dict["scaler"])
        if "epoch" in ckpt_dict:
            start_epoch = ckpt_dict["epoch"] + 1
        if "ace" in ckpt_dict:
            best_ace = ckpt_dict["ace"]

        if is_main():
            print(f"=> Loaded checkpoint (epoch {start_epoch - 1})")
    else:
        if is_main():
            print(f"=> No checkpoint found at '{path}'")

    return start_epoch, best_ace


def save_checkpoint(
    path: str,
    epoch: int,
    model: DDP,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    scaler: torch.amp.GradScaler,
    ace: float,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model": _unwrap(model).state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "ace": ace,
        },
        path,
    )
    tqdm.write(f"  [checkpoint] saved → {path}")

    if wandb.run is not None:
        artifact = wandb.Artifact(
            name=f"checkpoint-epoch{epoch:03d}",
            type="checkpoint",
            metadata={"epoch": epoch, "ace": ace},
        )
        artifact.add_file(path)
        wandb.log_artifact(artifact)
        tqdm.write("  [wandb] checkpoint artifact logged")


def save_best(
    ckpt_dir: str, best_name: str, epoch: int, model: DDP, ace: float
) -> None:
    path = os.path.join(ckpt_dir, best_name)
    torch.save(
        {
            "epoch": epoch,
            "model": _unwrap(model).state_dict(),
            "ace": ace,
        },
        path,
    )
    tqdm.write(f"  [best model] ACE={ace:.2%} saved → {path}")

    if wandb.run is not None:
        artifact = wandb.Artifact(
            name="best-model",
            type="model",
            metadata={"epoch": epoch, "ace": ace},
        )
        artifact.add_file(path)
        wandb.log_artifact(artifact)
        wandb.run.summary["best_val_ace"] = ace
        wandb.run.summary["best_val_ace_epoch"] = epoch
        tqdm.write("  [wandb] best-model artifact logged")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_one_epoch(
    model: DDP,
    train_loader: DataLoader,
    train_sampler: DistributedSampler,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    epoch: int,
) -> float:
    model.train()
    train_sampler.set_epoch(epoch)

    total_loss = 0.0

    pbar = tqdm(
        train_loader,
        desc=f"Epoch {epoch:03d} [train]",
        leave=False,
        unit="batch",
        disable=not is_main(),
    )

    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type="cuda"):
            _, logits = model(images, branch="b")
            loss = F.binary_cross_entropy_with_logits(logits.squeeze(1), labels.float())

        scaler.scale(loss).backward()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        old_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        new_scale = scaler.get_scale()
        if new_scale >= old_scale:
            scheduler.step()

        loss_val = loss.item()
        lr_val = scheduler.get_last_lr()[0]
        total_loss += loss_val

        pbar.set_postfix(loss=f"{loss_val:.4f}", lr=f"{lr_val:.2e}")

    return total_loss / len(train_loader)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(cfg: dict, no_wandb: bool = False, checkpoint: str = None) -> None:
    general_cfg = cfg["general"]
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    opt_cfg = cfg["optimizer"]
    sched_cfg = cfg["scheduler"]
    output_cfg = cfg["output"]
    wandb_cfg = cfg["wandb"]

    # ── DDP init ────────────────────────────────────────────────────────────
    local_rank, world_size = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")

    set_seed(general_cfg["seed"] + local_rank)

    if is_main():
        print(f"Device: {device}  |  world_size: {world_size}")

    # ── Wandb ─────────────────────────────────────────────────────────────
    if is_main() and not no_wandb and wandb_cfg.get("api_key"):
        wandb.login(key=wandb_cfg["api_key"])
        wandb.init(project=wandb_cfg["project"], config=cfg)

    # ── Transforms ────────────────────────────────────────────────────────
    train_transform, eval_transform, _ = get_transforms(data_cfg["transform_name"])

    # ── Datasets ──────────────────────────────────────────────────────────
    train_dataset = PADDataset(
        split_path=data_cfg["split_path"],
        split="train",
        transform=train_transform,
    )
    val_dataset = PADDataset(
        split_path=data_cfg["split_path"],
        split="val",
        transform=eval_transform,
    )

    if is_main():
        print(f"\n{train_dataset}")
        print(f"{val_dataset}")

    # ── Dataloaders ───────────────────────────────────────────────────────
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=local_rank,
        shuffle=True,
        seed=general_cfg["seed"],
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg["pad_batch_size"],
        sampler=train_sampler,
        num_workers=train_cfg["num_workers"],
        pin_memory=train_cfg["pin_memory"],
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg["pad_batch_size"],
        shuffle=False,
        num_workers=train_cfg["num_workers"],
        pin_memory=train_cfg["pin_memory"],
    )

    # ── Model ─────────────────────────────────────────────────────────────
    model = get_model(model_cfg["model_name"], model_cfg).to(device)
    if model_cfg.get("ckpt_path"):
        model.load_state_dict(
            torch.load(model_cfg["ckpt_path"], map_location="cpu")["model"]
        )
        tqdm.write(f"  [model] loaded from {model_cfg['ckpt_path']}")
    model = DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=True,
    )

    if is_main():
        n_params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"[model] {model_cfg['model_name']}  ({n_params:.2f}M params)")

    # ── Optimizer, Scheduler, Scaler ──────────────────────────────────────
    optimizer = get_optimizer(
        opt_name=opt_cfg["opt_name"], parameters=model.parameters(), opt_cfg=opt_cfg
    )

    scheduler = get_scheduler(
        sched_name=sched_cfg["sched_name"],
        optimizer=optimizer,
        iters=len(train_loader),
        epochs=train_cfg["epochs"],
        sched_cfg=sched_cfg,
    )

    scaler = torch.amp.GradScaler("cuda")

    # ── Training loop ─────────────────────────────────────────────────────
    start_epoch = 1
    best_ace = float("inf")

    if checkpoint is not None:
        start_epoch, best_ace = load_checkpoint(
            checkpoint, model, optimizer, scheduler, scaler
        )

    if is_main():
        if not wandb.run:
            history = {"epoch": [], "train_loss": [], "val_ace": []}

        os.makedirs(output_cfg["checkpoint_dir"], exist_ok=True)

        print("\n" + "=" * 60)
        print("Starting PAD training")
        print("=" * 60)

    epoch_pbar = tqdm(
        range(start_epoch, train_cfg["epochs"] + 1),
        desc="Training",
        unit="epoch",
        disable=not is_main(),
    )

    for epoch in epoch_pbar:
        avg_loss = train_one_epoch(
            model,
            train_loader,
            train_sampler,
            optimizer,
            scheduler,
            scaler,
            device,
            epoch,
        )

        dist.barrier()

        if is_main():
            metrics = evaluate(_unwrap(model), val_loader, device, epoch)

            epoch_pbar.set_postfix(
                loss=f"{avg_loss:.4f}",
                ace=f"{metrics['ace']:.4f}",
            )
            tqdm.write(
                f"Epoch {epoch:03d} | "
                f"loss: {avg_loss:.4f} | "
                f"val accuracy: {metrics['accuracy']:.2%} | "
                f"val ACE: {metrics['ace']:.2%} "
                f"(APCER={metrics['apcer']:.2%}, BPCER={metrics['bpcer']:.2%}) "
                f"(thr={metrics['threshold']:.4f})"
            )

            if wandb.run is not None:
                wandb.log(
                    {
                        "train/loss": avg_loss,
                        "epoch": epoch,
                        "val/loss": metrics["loss"],
                        "val/ace": metrics["ace"],
                        "val/apcer": metrics["apcer"],
                        "val/bpcer": metrics["bpcer"],
                        "val/threshold": metrics["threshold"],
                        "val/accuracy": metrics["accuracy"],
                    }
                )
            else:
                history["epoch"].append(epoch)
                history["train_loss"].append(avg_loss)
                history["val_ace"].append(metrics["ace"])

            if metrics["ace"] < best_ace:
                best_ace = metrics["ace"]
                save_best(
                    output_cfg["checkpoint_dir"],
                    output_cfg["best_model_name"],
                    epoch,
                    model,
                    metrics["ace"],
                )

            if epoch % train_cfg["checkpoint_interval"] == 0:
                ckpt_path = os.path.join(
                    output_cfg["checkpoint_dir"], f"checkpoint_epoch{epoch:03d}.pth"
                )
                save_checkpoint(
                    ckpt_path,
                    epoch,
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    metrics["ace"],
                )

        dist.barrier()

    if is_main():
        print("=" * 60)
        print(f"Training complete. Best val ACE: {best_ace:.2%}")
        print("=" * 60)

        if wandb.run is not None:
            wandb.finish()
        else:
            import matplotlib.pyplot as plt

            fig, ax1 = plt.subplots(figsize=(10, 5))
            ax2 = ax1.twinx()

            ax1.plot(history["epoch"], history["train_loss"], "g-", label="Train Loss")
            ax2.plot(history["epoch"], history["val_ace"], "b-", label="Val ACE")

            ax1.set_xlabel("Epoch")
            ax1.set_ylabel("Train Loss", color="g")
            ax2.set_ylabel("Val ACE", color="b")

            lines_1, labels_1 = ax1.get_legend_handles_labels()
            lines_2, labels_2 = ax2.get_legend_handles_labels()
            ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right")

            plt.title("PAD Training History")
            plot_path = os.path.join(
                output_cfg["checkpoint_dir"], "training_history.png"
            )
            plt.savefig(plot_path)
            plt.close()
            print(f"Saved training history plot to {plot_path}")

    cleanup_ddp()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PAD Training")
    parser.add_argument(
        "--config",
        type=str,
        default="pad_config.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable Weights & Biases logging",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    main(cfg, no_wandb=args.no_wandb, checkpoint=args.checkpoint)
