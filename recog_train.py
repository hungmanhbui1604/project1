import argparse
import os
import random

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import yaml
from sklearn.metrics import roc_curve
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

import wandb
from data import RecogEvaluationDataset, RecogTrainingDataset, UniqueImageDataset
from loss import ArcFaceLoss
from model import get_model
from schedulers import cosine_warmup_scheduler
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
# EER Computation
# ---------------------------------------------------------------------------


@torch.no_grad()
def compute_eer(scores: np.ndarray, labels: np.ndarray):
    fmr, tar, thrs = roc_curve(labels, scores, pos_label=1)
    fnmr = 1.0 - tar

    diff = fmr - fnmr
    idx1 = np.where(diff >= 0)[0][0]
    idx0 = idx1 - 1 if idx1 > 0 else idx1

    x0, y0 = fmr[idx0], fnmr[idx0]
    x1, y1 = fmr[idx1], fnmr[idx1]

    if idx0 == idx1:
        eer = (x0 + y0) / 2
        eer_thr = thrs[idx0]
    else:
        t = (y0 - x0) / ((x1 - x0) - (y1 - y0))
        eer = x0 + t * (x1 - x0)
        eer_thr = thrs[idx0] + t * (thrs[idx1] - thrs[idx0])

    return float(eer), float(eer_thr)


# ---------------------------------------------------------------------------
# Embedding Extraction & Evaluation
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
            emb, _ = model(imgs, branch="a")
        emb = F.normalize(emb, dim=1).float()

        local_embeddings[idxs] = emb
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

    eer, thr = compute_eer(np.concatenate(all_scores), np.concatenate(all_labels))

    return eer, thr


# ---------------------------------------------------------------------------
# Checkpoint Management
# ---------------------------------------------------------------------------


def load_checkpoint(
    path: str,
    model: DDP,
    arcface_loss: DDP,
    optimizer: AdamW,
    scheduler: LambdaLR,
    scaler: torch.amp.GradScaler,
) -> tuple[int, float]:
    start_epoch = 1
    best_eer = float("inf")

    if os.path.isfile(path):
        if is_main():
            print(f"=> Loading checkpoint '{path}'")
        ckpt_dict = torch.load(path, map_location="cpu")
        _unwrap(model).load_state_dict(ckpt_dict["model"])
        _unwrap(arcface_loss).load_state_dict(ckpt_dict["arcface"])
        if "optimizer" in ckpt_dict:
            optimizer.load_state_dict(ckpt_dict["optimizer"])
        if "scheduler" in ckpt_dict:
            scheduler.load_state_dict(ckpt_dict["scheduler"])
        if "scaler" in ckpt_dict:
            scaler.load_state_dict(ckpt_dict["scaler"])
        if "epoch" in ckpt_dict:
            start_epoch = ckpt_dict["epoch"] + 1
        if "eer" in ckpt_dict:
            best_eer = ckpt_dict["eer"]

        if is_main():
            print(f"=> Loaded checkpoint (epoch {start_epoch - 1})")
    else:
        if is_main():
            print(f"=> No checkpoint found at '{path}'")

    return start_epoch, best_eer


def save_checkpoint(
    path: str,
    epoch: int,
    model: DDP,
    arcface_loss: DDP,
    optimizer: AdamW,
    scheduler: LambdaLR,
    scaler: torch.amp.GradScaler,
    eer: float,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model": _unwrap(model).state_dict(),
            "arcface": _unwrap(arcface_loss).state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "eer": eer,
        },
        path,
    )
    tqdm.write(f"  [checkpoint] saved → {path}")

    if wandb.run is not None:
        artifact = wandb.Artifact(
            name=f"checkpoint-epoch{epoch:03d}",
            type="checkpoint",
            metadata={"epoch": epoch, "eer": eer},
        )
        artifact.add_file(path)
        wandb.log_artifact(artifact)
        tqdm.write("  [wandb] checkpoint artifact logged")


def save_best(
    ckpt_dir: str,
    best_name: str,
    epoch: int,
    model: DDP,
    arcface_loss: DDP,
    eer: float,
) -> None:
    path = os.path.join(ckpt_dir, best_name)
    torch.save(
        {
            "epoch": epoch,
            "model": _unwrap(model).state_dict(),
            "arcface": _unwrap(arcface_loss).state_dict(),
            "eer": eer,
        },
        path,
    )
    tqdm.write(f"  [best model] EER={eer:.4f} saved → {path}")

    if wandb.run is not None:
        artifact = wandb.Artifact(
            name="best-model",
            type="model",
            metadata={"epoch": epoch, "val_eer": eer},
        )
        artifact.add_file(path)
        wandb.log_artifact(artifact)
        wandb.run.summary["best_val_eer"] = eer
        wandb.run.summary["best_val_eer_epoch"] = epoch
        tqdm.write("  [wandb] best-model artifact logged")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_one_epoch(
    model: DDP,
    arcface_loss: DDP,
    train_loader: DataLoader,
    train_sampler: DistributedSampler,
    optimizer: AdamW,
    scheduler: LambdaLR,
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


def main(cfg: dict, no_wandb: bool = False, checkpoint: str = None) -> None:
    general_cfg = cfg["general"]
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    training_cfg = cfg["training"]
    loss_cfg = cfg["loss"]
    opt_cfg = cfg["optimizer"]
    sched_cfg = cfg["scheduler"]
    output_cfg = cfg["output"]
    wandb_cfg = cfg["wandb"]
    evaluation_cfg = cfg["evaluation"]

    # ── DDP init ────────────────────────────────────────────────────────────
    local_rank, world_size = setup_ddp()
    device = torch.device("cuda", local_rank)

    set_seed(general_cfg["seed"] + local_rank)

    if is_main():
        print(f"Device: {device}  |  world_size: {world_size}")

    # ── Wandb ─────────────────────────────────────────────────────────────
    if is_main() and not no_wandb and wandb_cfg.get("api_key"):
        wandb.login(key=wandb_cfg["api_key"])
        wandb.init(project=wandb_cfg.get("project", "DualViT-Recognition"), config=cfg)

    # ── Transforms ────────────────────────────────────────────────────────
    train_transform, eval_transform = get_transforms("all")

    # ── Datasets ──────────────────────────────────────────────────────────
    train_dataset = RecogTrainingDataset(
        split_path=data_cfg["split_path"],
        transform=train_transform,
    )
    val_dataset = RecogEvaluationDataset(
        split_path=data_cfg["split_path"],
        split="val",
        n_genuine_impressions=data_cfg["n_genuine_impressions"],
        n_impostor_impressions=data_cfg["n_impostor_impressions"],
        impostor_mode=data_cfg["impostor_mode"],
        n_impostor_subset=None
        if data_cfg.get("n_impostor_subset") in ("None", None, "null")
        else data_cfg["n_impostor_subset"],
        seed=general_cfg["seed"],
    )
    unique_val_dataset = UniqueImageDataset(
        idx_to_path=val_dataset.idx_to_path, transform=eval_transform
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
        batch_size=training_cfg["recog_batch_size"],
        sampler=train_sampler,
        num_workers=training_cfg["num_workers"],
        pin_memory=training_cfg["pin_memory"],
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=evaluation_cfg["recog_batch_size"],
        shuffle=False,
        num_workers=training_cfg["num_workers"],
        pin_memory=training_cfg["pin_memory"],
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
        batch_size=training_cfg["recog_batch_size"],
        sampler=unique_val_sampler,
        num_workers=training_cfg["num_workers"],
        pin_memory=training_cfg["pin_memory"],
    )

    # ── Model ─────────────────────────────────────────────────────────────
    model = get_model(**model_cfg).to(device)
    if model_cfg.get("ckpt_path"):
        model.load_state_dict(torch.load(model_cfg["ckpt_path"], map_location="cpu")["model"])
        tqdm.write(f"  [model] loaded from {model_cfg['ckpt_path']}")
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    if is_main():
        n_params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"[model] DualViT  ({n_params:.2f}M params)")

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
    optimizer = AdamW(
        list(model.parameters()) + list(arcface_loss.parameters()),
        lr=opt_cfg["lr"],
        weight_decay=opt_cfg["weight_decay"],
    )

    total_iters = training_cfg["epochs"] * len(train_loader)
    warmup_iters = sched_cfg["warmup_epochs"] * len(train_loader)
    scheduler = cosine_warmup_scheduler(
        optimizer,
        warmup_iters=warmup_iters,
        total_iters=total_iters,
        min_lr=sched_cfg["min_lr"],
    )

    scaler = torch.amp.GradScaler("cuda")

    # ── Training loop ─────────────────────────────────────────────────────
    start_epoch = 1
    best_eer = float("inf")

    if checkpoint is not None:
        start_epoch, best_eer = load_checkpoint(
            checkpoint, model, arcface_loss, optimizer, scheduler, scaler
        )

    if is_main():
        if not wandb.run:
            history = {"epoch": [], "train_loss": [], "val_eer": []}

        os.makedirs(output_cfg["checkpoint_dir"], exist_ok=True)

        print("\n" + "=" * 60)
        print("Starting recognition training")
        print("=" * 60)

    epoch_pbar = tqdm(
        range(start_epoch, training_cfg["epochs"] + 1),
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
            eer, thr = evaluate(val_loader, global_embeddings, device, epoch)

            epoch_pbar.set_postfix(
                loss=f"{avg_loss:.4f}", eer=f"{eer:.4f}", thr=f"{thr:.4f}"
            )
            tqdm.write(
                f"Epoch {epoch:03d} | avg loss: {avg_loss:.4f} | "
                f"val EER: {eer:.4f}  (threshold={thr:.4f})"
            )

            if wandb.run is not None:
                wandb.log(
                    {
                        "train/loss_epoch": avg_loss,
                        "val/eer": eer,
                        "val/threshold": thr,
                        "epoch": epoch,
                    }
                )
            else:
                history["epoch"].append(epoch)
                history["train_loss"].append(avg_loss)
                history["val_eer"].append(eer)

            if eer < best_eer:
                best_eer = eer
                save_best(
                    output_cfg["checkpoint_dir"],
                    output_cfg["best_model_name"],
                    epoch,
                    model,
                    arcface_loss,
                    eer,
                )

            if epoch % training_cfg["checkpoint_interval"] == 0:
                ckpt_path = os.path.join(
                    output_cfg["checkpoint_dir"], f"checkpoint_epoch{epoch:03d}.pth"
                )
                save_checkpoint(
                    ckpt_path,
                    epoch,
                    model,
                    arcface_loss,
                    optimizer,
                    scheduler,
                    scaler,
                    eer,
                )

        dist.barrier()

    if is_main():
        print("=" * 60)
        print(f"Training complete. Best val EER: {best_eer:.4f}")
        print("=" * 60)

        if wandb.run is not None:
            wandb.finish()
        else:
            import matplotlib.pyplot as plt

            fig, ax1 = plt.subplots(figsize=(10, 5))
            ax2 = ax1.twinx()

            ax1.plot(history["epoch"], history["train_loss"], "g-", label="Train Loss")
            ax2.plot(history["epoch"], history["val_eer"], "b-", label="Val EER")

            ax1.set_xlabel("Epoch")
            ax1.set_ylabel("Train Loss", color="g")
            ax2.set_ylabel("Val EER", color="b")

            lines_1, labels_1 = ax1.get_legend_handles_labels()
            lines_2, labels_2 = ax2.get_legend_handles_labels()
            ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right")

            plt.title("Recognition Training History")
            plot_path = os.path.join(
                output_cfg["checkpoint_dir"], "training_history.png"
            )
            plt.savefig(plot_path)
            plt.close()
            print(f"Saved training history plot to {plot_path}")

    cleanup_ddp()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DualViT Recognition Training")
    parser.add_argument(
        "--config",
        type=str,
        default="recog_config.yaml",
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
