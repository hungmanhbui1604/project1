import argparse
import os
import random

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import wandb
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from data import (
    AuthenticationEvaluationDataset,
    RecogTrainingDataset,
    UniqueFingerprintDataset,
)
from losses import ArcFaceLoss
from metrics import compute_authentication_metrics
from models import get_model
from schedulers import get_scheduler
from transforms import get_transforms

# ---------------------------------------------------------------------------
# Utilities
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


def save_best(
    ckpt_dir: str,
    best_name: str,
    epoch: int,
    model: DDP,
    eer: float,
) -> None:
    path = os.path.join(ckpt_dir, best_name)
    torch.save(
        {
            "epoch": epoch,
            "model": _unwrap(model).state_dict(),
            "eer": eer,
        },
        path,
    )
    tqdm.write(f"  [best model] EER={eer:.2%} saved → {path}")

    artifact = wandb.Artifact(
        name="best-model",
        type="model",
        metadata={"epoch": epoch, "eer": eer},
    )
    artifact.add_file(path)
    wandb.log_artifact(artifact)
    wandb.run.summary["best_val_eer"] = eer
    wandb.run.summary["best_val_eer_epoch"] = epoch
    tqdm.write("  [wandb] best-model artifact logged")


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------


def get_optimizer(opt_name: str, parameters: list, opt_cfg: dict):
    if opt_name == "adam":
        return torch.optim.Adam(
            parameters, lr=opt_cfg["lr"], weight_decay=opt_cfg["weight_decay"]
        )
    
    raise ValueError("Unknown optimizer: " + opt_name)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


@torch.no_grad()
def get_embeddings(
    model: torch.nn.Module,
    val_unique_loader: DataLoader,
    device: torch.device,
    epoch: int,
) -> torch.Tensor:
    model.eval()

    embed_dim = _unwrap(model).branch_a_head.out_features
    n_unique_images = len(val_unique_loader.dataset)

    local_embeddings = torch.zeros((n_unique_images, embed_dim), device=device)
    mask = torch.zeros(n_unique_images, device=device, dtype=torch.bool)

    pbar = tqdm(
        val_unique_loader,
        desc=f"Epoch {epoch:03d} [val extract]",
        leave=False,
        unit="batch",
        disable=not is_main(),
    )

    for idxs, imgs in pbar:
        imgs = imgs.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda"):
            embs, _ = model(imgs, branch="a")
        embs = F.normalize(embs, dim=1).float()

        local_embeddings[idxs] = embs
        mask[idxs] = True

    counts = torch.zeros(n_unique_images, device=device)
    counts[mask] = 1.0

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(local_embeddings, op=dist.ReduceOp.SUM)
        dist.all_reduce(counts, op=dist.ReduceOp.SUM)

    global_embeddings = local_embeddings / counts.clamp_min(1.0).unsqueeze(1)
    return global_embeddings


@torch.no_grad()
def evaluate(
    val_loader: DataLoader,
    global_embeddings: torch.Tensor,
    device: torch.device,
    epoch: int,
) -> tuple[float, float]:
    all_scores, all_labels = [], []

    pbar = tqdm(
        val_loader,
        desc=f"Epoch {epoch:03d} [val evaluate]",
        leave=False,
        unit="batch",
        disable=not is_main(),
    )

    for idx_a, idx_b, labels in pbar:
        idx_a = idx_a.to(device, non_blocking=True)
        idx_b = idx_b.to(device, non_blocking=True)

        emb_a = global_embeddings[idx_a]
        emb_b = global_embeddings[idx_b]

        cos_sim = (emb_a * emb_b).sum(dim=1)

        all_scores.append(cos_sim.cpu().numpy())
        all_labels.append(labels.numpy())

    metrics = compute_authentication_metrics(
        np.concatenate(all_scores), np.concatenate(all_labels)
    )

    return metrics


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_one_epoch(
    model: DDP,
    arcface_loss: DDP,
    train_loader: DataLoader,
    train_sampler: DistributedSampler,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    epoch: int,
) -> float:
    model.train()
    arcface_loss.train()

    train_sampler.set_epoch(epoch)

    total_loss = 0.0
    all_params = list(model.parameters()) + list(arcface_loss.parameters())

    pbar = tqdm(
        train_loader,
        desc=f"[train] Epoch {epoch:03d}",
        leave=False,
        unit="batch",
        disable=not is_main(),
    )

    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type="cuda"):
            embeddings, _ = model(images, branch="a")
            loss, _ = arcface_loss(embeddings, labels)

        scaler.scale(loss).backward()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
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


def main(cfg: dict) -> None:
    general_cfg = cfg["general"]
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    loss_cfg = cfg["loss"]
    opt_cfg = cfg["optimizer"]
    sched_cfg = cfg["scheduler"]
    output_cfg = cfg["output"]
    wandb_cfg = cfg["wandb"]
    eval_cfg = cfg["evaluation"]

    # ── DDP init ────────────────────────────────────────────────────────────
    local_rank, world_size = setup_ddp()
    device = torch.device("cuda", local_rank)

    set_seed(general_cfg["seed"] + local_rank)

    if is_main():
        print(f"Device: {device}  |  world_size: {world_size}")

    # ── Wandb ─────────────────────────────────────────────────────────────
    if is_main():
        if wandb_cfg.get("api_key"):
            wandb.login(key=wandb_cfg["api_key"])
            wandb.init(project=wandb_cfg["project"], config=cfg)
        else:
            raise ValueError("Missing wandb api key in config")

    # ── Transforms ────────────────────────────────────────────────────────
    transform = get_transforms(data_cfg["transform_name"])
    train_transform = transform["train"]
    eval_transform = transform["test"]

    # ── Datasets ──────────────────────────────────────────────────────────
    train_dataset = RecogTrainingDataset(
        split_path=data_cfg["split_path"],
        transform=train_transform,
    )
    val_dataset = AuthenticationEvaluationDataset(
        split_path=data_cfg["split_path"],
        split="val",
        n_genuine_impressions=data_cfg["n_genuine_impressions"],
        n_impostor_impressions=data_cfg["n_impostor_impressions"],
        impostor_mode=data_cfg["impostor_mode"],
        n_impostor_subset=data_cfg["n_impostor_subset"],
        seed=general_cfg["seed"],
    )
    unique_val_dataset = UniqueFingerprintDataset(
        idx_to_path=val_dataset.idx_to_path, transform=eval_transform
    )

    if is_main():
        print(f"\n{train_dataset}")
        print(f"{val_dataset}")

    # ── Dataloaders ───────────────────────────────────────────────────────
    global_batch_size = train_cfg["recog_batch_size"]
    local_batch_size = max(1, global_batch_size // world_size)

    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=local_rank,
        shuffle=True,
        seed=general_cfg["seed"],
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=local_batch_size,
        sampler=train_sampler,
        num_workers=train_cfg["num_workers"],
        pin_memory=train_cfg["pin_memory"],
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=eval_cfg["auth_batch_size"],
        shuffle=False,
        num_workers=train_cfg["num_workers"],
        pin_memory=train_cfg["pin_memory"],
    )

    unique_val_sampler = DistributedSampler(
        unique_val_dataset,
        num_replicas=world_size,
        rank=local_rank,
        shuffle=False,
        drop_last=False,
    )
    unique_val_loader = DataLoader(
        unique_val_dataset,
        batch_size=local_batch_size,
        sampler=unique_val_sampler,
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
        print(f"[model] {model_cfg['model_name']} ({n_params:.2f}M params)")

    # ── Loss ──────────────────────────────────────────────────────────────
    n_ids = train_dataset.n_ids
    arcface_loss = ArcFaceLoss(
        embed_dim=model_cfg["branch_a_num_classes"],
        num_classes=n_ids,
        margin=loss_cfg["margin"],
        scale=loss_cfg["scale"],
    ).to(device)
    arcface_loss = DDP(arcface_loss, device_ids=[local_rank], output_device=local_rank)

    # ── Optimizer, Scheduler, Scaler ──────────────────────────────────────
    all_parameters = list(model.parameters()) + list(arcface_loss.parameters())
    optimizer = get_optimizer(opt_cfg["opt_name"], all_parameters, opt_cfg)
    
    scheduler = get_scheduler(
        sched_cfg["sched_name"],
        optimizer,
        iters=len(train_loader),
        epochs=train_cfg["epochs"],
        sched_cfg=sched_cfg,
    )

    scaler = torch.amp.GradScaler("cuda")

    # ── Training loop ─────────────────────────────────────────────────────
    start_epoch = 1
    best_eer = float("inf")

    if is_main():
        os.makedirs(output_cfg["checkpoint_dir"], exist_ok=True)

        print("\n" + "=" * 60)
        print("Starting recognition training")
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
            arcface_loss,
            train_loader,
            train_sampler,
            optimizer,
            scheduler,
            scaler,
            device,
            epoch,
        )

        dist.barrier()

        global_embeddings = get_embeddings(
            _unwrap(model), unique_val_loader, device, epoch
        )

        if is_main():
            metrics = evaluate(val_loader, global_embeddings, device, epoch)

            epoch_pbar.set_postfix(
                loss=f"{avg_loss:.4f}",
                eer=f"{metrics['eer']:.2%}",
            )
            tqdm.write(
                f"Epoch {epoch:03d} | avg loss: {avg_loss:.4f} | "
                f"val EER: {metrics['eer']:.2%}  (thr={metrics['eer_threshold']:.4f}) | "
                f"val TAR@FAR=0.1: {metrics['tar_at_far_0.1']:.2%} | "
                f"val TAR@FAR=0.01: {metrics['tar_at_far_0.01']:.2%} | "
                f"val TAR@FAR=0.001: {metrics['tar_at_far_0.001']:.2%}"
            )

            wandb.log(
                {
                    "train/loss": avg_loss,
                    "val/eer": metrics["eer"],
                    "val/eer_threshold": metrics["eer_threshold"],
                    "val/tar_at_far_0.1": metrics["tar_at_far_0.1"],
                    "val/tar_at_far_0.01": metrics["tar_at_far_0.01"],
                    "val/tar_at_far_0.001": metrics["tar_at_far_0.001"],
                    "epoch": epoch,
                }
            )

            if metrics["eer"] < best_eer:
                best_eer = metrics["eer"]
                save_best(
                    output_cfg["checkpoint_dir"],
                    output_cfg["best_model_name"],
                    epoch,
                    model,
                    metrics["eer"],
                )

        dist.barrier()

    if is_main():
        print("=" * 60)
        print(f"Training complete. Best val EER: {best_eer:.2%}")
        print("=" * 60)

        wandb.finish()

    cleanup_ddp()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recognition Training")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/recog_config.yaml",
        help="Path to YAML config file",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    main(cfg)
