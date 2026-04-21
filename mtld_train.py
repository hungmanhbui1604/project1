import argparse
import os
import random

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
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
from data import (
    PADDataset,
    RecogEvaluationDataset,
    RecogTrainingDataset,
    UniqueImageDataset,
)
from loss import ArcFaceLoss
from model import DualViT
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
# Distillation Losses
# ---------------------------------------------------------------------------


def feature_distillation_loss(
    student_feat: torch.Tensor, teacher_feat: torch.Tensor
) -> torch.Tensor:
    """
    MSE loss on L2-normalised, spatially-averaged features.

    Casts to float32 for numerical stability under mixed-precision.
    """
    if student_feat.dim() == 3:
        s = student_feat.float().mean(dim=1)  # (B, C)
        t = teacher_feat.float().mean(dim=1)
    else:
        s = student_feat.float()
        t = teacher_feat.float()

    s = F.normalize(s, dim=-1)
    t = F.normalize(t, dim=-1)
    return F.mse_loss(s, t)


def compute_distillation_losses(
    student_feats: dict[str, torch.Tensor],
    recog_t_feats: dict[str, torch.Tensor],
    pad_t_feats: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, float]]:
    breakdown = {}

    # ── Layer 4: distil from both teachers (averaged) ──────────
    l_layer4_recog = feature_distillation_loss(
        student_feats["layer4"], recog_t_feats["layer4"]
    )
    l_layer4_pad = feature_distillation_loss(
        student_feats["layer4"], pad_t_feats["layer4"]
    )
    l_layer4 = 0.5 * (l_layer4_recog + l_layer4_pad)
    breakdown["distill/layer4"] = l_layer4.item()

    # ── Layer 8 ────────────────────────────────────────────────
    l_a_layer8 = feature_distillation_loss(
        student_feats["a_layer8"], recog_t_feats["a_layer8"]
    )
    l_b_layer8 = feature_distillation_loss(
        student_feats["b_layer8"], pad_t_feats["b_layer8"]
    )
    breakdown["distill/a_layer8"] = l_a_layer8.item()
    breakdown["distill/b_layer8"] = l_b_layer8.item()

    # ── Layer 12 ───────────────────────────────────────────────
    l_a_layer12 = feature_distillation_loss(
        student_feats["a_layer12"], recog_t_feats["a_layer12"]
    )
    breakdown["distill/a_layer12"] = l_a_layer12.item()

    distill_loss = l_layer4 + l_a_layer8 + l_b_layer8 + l_a_layer12
    breakdown["distill/total"] = distill_loss.item()

    return distill_loss, breakdown


# ---------------------------------------------------------------------------
# EER Computation
# ---------------------------------------------------------------------------


@torch.no_grad()
def compute_eer(
    scores: np.ndarray,
    labels: np.ndarray,
) -> tuple[float, float]:
    """Compute Equal Error Rate from similarity scores and binary labels."""
    fmr, tar, thrs = roc_curve(labels, scores, pos_label=1)
    fnmr = 1.0 - tar

    asc_idx = np.argsort(thrs)
    thrs = thrs[asc_idx]
    fmr = fmr[asc_idx]
    fnmr = fnmr[asc_idx]

    diff = np.abs(fmr - fnmr)
    eer_idx = int(np.argmin(diff))
    eer = float((fmr[eer_idx] + fnmr[eer_idx]) / 2.0)
    eer_thr = float(thrs[eer_idx])

    return eer, eer_thr


# ---------------------------------------------------------------------------
# Recognition Evaluation (Branch A)
# ---------------------------------------------------------------------------


@torch.no_grad()
def get_embeddings(
    model: DualViT,
    val_unique_loader: DataLoader,
    device: torch.device,
    epoch: int,
) -> torch.Tensor:
    """Extract recognition embeddings (branch A) for all unique images."""
    model.eval()

    embed_dim = _unwrap(model).branch_a_head.out_features
    n_unique_images = len(val_unique_loader.dataset)

    local_embeddings = torch.zeros((n_unique_images, embed_dim), device=device)
    mask = torch.zeros(n_unique_images, device=device, dtype=torch.bool)

    pbar = tqdm(
        val_unique_loader,
        desc=f"Epoch {epoch:03d} [recog extract]",
        leave=False,
        unit="batch",
        disable=not is_main(),
    )

    for idxs, imgs in pbar:
        imgs = imgs.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda"):
            emb_a, _ = model(imgs, branch="a")
        emb_a = F.normalize(emb_a, dim=1).float()

        local_embeddings[idxs] = emb_a
        mask[idxs] = True

    counts = torch.zeros(n_unique_images, device=device)
    counts[mask] = 1.0

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(local_embeddings, op=dist.ReduceOp.SUM)
        dist.all_reduce(counts, op=dist.ReduceOp.SUM)

    global_embeddings = local_embeddings / counts.clamp_min(1.0).unsqueeze(1)
    return global_embeddings


@torch.no_grad()
def evaluate_recog(
    val_loader: DataLoader,
    global_embeddings: torch.Tensor,
    device: torch.device,
    epoch: int,
) -> tuple[float, float]:
    """Evaluate recognition via cosine similarity and EER on validation pairs."""
    all_scores, all_labels = [], []

    pbar = tqdm(
        val_loader,
        desc=f"Epoch {epoch:03d} [recog eval]",
        leave=False,
        unit="batch",
        disable=not is_main(),
    )

    for idx_a, idx_b, labels in pbar:
        emb_a = global_embeddings[idx_a]
        emb_b = global_embeddings[idx_b]

        cos_sim = (emb_a * emb_b).sum(dim=1)

        all_scores.append(cos_sim.cpu().numpy())
        all_labels.append(labels.numpy())

    eer, thr = compute_eer(np.concatenate(all_scores), np.concatenate(all_labels))

    return eer, thr


# ---------------------------------------------------------------------------
# PAD Evaluation (Branch B)
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate_pad(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    epoch: int,
) -> dict[str, float]:
    """Evaluate PAD (branch B). Compute APCER, BPCER, and ACE."""
    model.eval()

    if len(val_loader.dataset) == 0:
        return {"val/pad_loss": 0.0, "val/apcer": 0.0, "val/bpcer": 0.0, "val/ace": 0.0}

    total_loss = 0.0
    all_preds, all_labels = [], []

    pbar = tqdm(
        val_loader,
        desc=f"Epoch {epoch:03d} [pad eval]",
        leave=False,
        unit="batch",
        disable=not is_main(),
    )

    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.autocast(device_type="cuda"):
            _, logits = model(images, branch="b")
            loss = F.cross_entropy(logits, labels)

        total_loss += loss.item()

        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.append(preds)
        all_labels.append(labels.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    avg_loss = total_loss / len(val_loader)

    # APCER: spoof samples classified as live
    spoof_mask = all_labels == 1
    apcer = (all_preds[spoof_mask] == 0).mean() if spoof_mask.any() else 0.0

    # BPCER: live samples classified as spoof
    live_mask = all_labels == 0
    bpcer = (all_preds[live_mask] == 1).mean() if live_mask.any() else 0.0

    ace = (apcer + bpcer) / 2.0

    return {
        "val/pad_loss": avg_loss,
        "val/apcer": float(apcer),
        "val/bpcer": float(bpcer),
        "val/ace": float(ace),
    }


# ---------------------------------------------------------------------------
# Teacher Loading
# ---------------------------------------------------------------------------


def load_teacher(
    checkpoint_path: str, model_cfg: dict, device: torch.device, branch: str
) -> DualViT:
    """
    Load a pre-trained DualViT teacher from checkpoint.
    """
    if branch == "a":
        model = DualViT(
            model_name=model_cfg["model_name"],
            pretrained=model_cfg["pretrained"],
            branch_a_num_classes=model_cfg["branch_a_num_classes"],
            branch_b_num_classes=model_cfg["branch_b_num_classes"],
            head_hidden_dim=model_cfg["head_hidden_dim"],
            head_drop_rate=model_cfg["head_drop_rate"],
        ).to(device)
    else:
        model = DualViT(
            model_name=model_cfg["model_name"],
            pretrained=model_cfg["pretrained"],
            branch_a_num_classes=model_cfg["branch_a_num_classes"],
            branch_b_num_classes=model_cfg["branch_b_num_classes"],
            head_hidden_dim=model_cfg["head_hidden_dim"],
            head_drop_rate=model_cfg["head_drop_rate"],
        ).to(device)
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


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
    best_avg_ace_eer = float("inf")

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
        if "best_avg_ace_eer" in ckpt_dict:
            best_avg_ace_eer = ckpt_dict["best_avg_ace_eer"]

        if is_main():
            print(f"=> Loaded checkpoint (epoch {start_epoch - 1})")
    else:
        if is_main():
            print(f"=> No checkpoint found at '{path}'")

    return start_epoch, best_avg_ace_eer


def save_checkpoint(
    path: str,
    epoch: int,
    model: DDP,
    arcface_loss: DDP,
    optimizer: AdamW,
    scheduler: LambdaLR,
    scaler: torch.amp.GradScaler,
    best_avg_ace_eer: float,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model": _unwrap(model).state_dict(),
            "arcface": _unwrap(arcface_loss).state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "best_avg_ace_eer": best_avg_ace_eer,
        },
        path,
    )
    tqdm.write(f"  [checkpoint] saved → {path}")

    if wandb.run is not None:
        artifact = wandb.Artifact(
            name=f"checkpoint-epoch{epoch:03d}",
            type="checkpoint",
            metadata={"epoch": epoch, "best_avg_ace_eer": best_avg_ace_eer},
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
    metrics: dict,
) -> None:
    path = os.path.join(ckpt_dir, best_name)
    torch.save(
        {
            "epoch": epoch,
            "model": _unwrap(model).state_dict(),
            "arcface": _unwrap(arcface_loss).state_dict(),
            "metrics": metrics,
        },
        path,
    )
    avg = metrics["val/avg_ace_eer"]
    tqdm.write(f"  [best model] avg(ACE,EER)={avg:.4f} saved → {path}")

    if wandb.run is not None:
        artifact = wandb.Artifact(
            name="best-model",
            type="model",
            metadata={"epoch": epoch, **metrics},
        )
        artifact.add_file(path)
        wandb.log_artifact(artifact)
        wandb.run.summary["best_val_avg_ace_eer"] = avg
        wandb.run.summary["best_val_avg_ace_eer_epoch"] = epoch
        tqdm.write("  [wandb] best-model artifact logged")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_one_epoch(
    model: DDP,
    arcface_loss: DDP,
    recog_teacher: DualViT,
    pad_teacher: DualViT,
    recog_loader: DataLoader,
    recog_sampler: DistributedSampler,
    pad_loader: DataLoader,
    pad_sampler: DistributedSampler,
    optimizer: AdamW,
    scheduler: LambdaLR,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    epoch: int,
    steps_per_epoch: int,
    recog_weight: float = 1.0,
    pad_weight: float = 1.0,
    distill_weight: float = 1.0,
) -> dict[str, float]:
    model.train()
    arcface_loss.train()

    recog_sampler.set_epoch(epoch)
    pad_sampler.set_epoch(epoch)

    total_loss = 0.0
    total_recog_loss = 0.0
    total_pad_loss = 0.0
    total_distill_loss = 0.0

    all_params = list(model.parameters()) + list(arcface_loss.parameters())

    pbar = tqdm(
        zip(recog_loader, pad_loader),
        desc=f"[train] Epoch {epoch:03d}",
        total=steps_per_epoch,
        leave=False,
        unit="step",
        disable=not is_main(),
    )

    for (recog_images, recog_labels), (pad_images, pad_labels) in pbar:
        recog_images = recog_images.to(device, non_blocking=True)
        recog_labels = recog_labels.to(device, non_blocking=True)
        pad_images = pad_images.to(device, non_blocking=True)
        pad_labels = pad_labels.to(device, non_blocking=True)

        # Combine both batches for a single forward pass
        combined = torch.cat([recog_images, pad_images], dim=0)
        n_recog = recog_images.size(0)

        optimizer.zero_grad(set_to_none=True)

        # ── Teacher forward (frozen, no grad, autocast for speed) ─────
        with torch.no_grad(), torch.autocast(device_type="cuda"):
            _, _, recog_t_feats = recog_teacher(
                combined, return_features=True, branch="a"
            )
            _, _, pad_t_feats = pad_teacher(combined, return_features=True, branch="b")

        # ── Student forward ────────
        with torch.autocast(device_type="cuda"):
            emb_a, emb_b, student_feats = model(combined, return_features=True)

            # Task losses (split outputs by task)
            recog_emb = emb_a[:n_recog]
            pad_logits = emb_b[n_recog:]
            recog_loss, _ = arcface_loss(recog_emb, recog_labels)
            pad_loss = F.cross_entropy(pad_logits, pad_labels)

            # Distillation losses (computed on full combined batch)
            distill_loss, distill_info = compute_distillation_losses(
                student_feats,
                recog_t_feats,
                pad_t_feats,
            )

            loss = (
                recog_weight * recog_loss
                + pad_weight * pad_loss
                + distill_weight * distill_loss
            )

        scaler.scale(loss).backward()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)

        old_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        new_scale = scaler.get_scale()
        if new_scale >= old_scale:
            scheduler.step()

        total_loss += loss.item()
        total_recog_loss += recog_loss.item()
        total_pad_loss += pad_loss.item()
        total_distill_loss += distill_loss.item()

        lr_val = scheduler.get_last_lr()[0]
        pbar.set_postfix(
            total=f"{loss.item():.4f}",
            recog=f"{recog_loss.item():.4f}",
            pad=f"{pad_loss.item():.4f}",
            dist=f"{distill_loss.item():.4f}",
            lr=f"{lr_val:.2e}",
        )

    return {
        "train/loss": total_loss / steps_per_epoch,
        "train/recog_loss": total_recog_loss / steps_per_epoch,
        "train/pad_loss": total_pad_loss / steps_per_epoch,
        "train/distill_loss": total_distill_loss / steps_per_epoch,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(cfg: dict, no_wandb: bool = False, checkpoint: str = None) -> None:
    general_cfg = cfg["general"]
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    teacher_cfg = cfg["teachers"]
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
        wandb.init(project=wandb_cfg.get("project", "DualViT-MTLD"), config=cfg)

    # ── Transforms ────────────────────────────────────────────────────────
    train_transform, eval_transform = get_transforms("all")

    # ── Recognition Datasets ─────────────────────────────────────────────
    recog_train_dataset = RecogTrainingDataset(
        split_path=data_cfg["recog_split_path"],
        transform=train_transform,
    )
    recog_val_dataset = RecogEvaluationDataset(
        split_path=data_cfg["recog_split_path"],
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
        idx_to_path=recog_val_dataset.idx_to_path, transform=eval_transform
    )

    # ── PAD Datasets ─────────────────────────────────────────────────────
    pad_train_dataset = PADDataset(
        split_path=data_cfg["pad_split_path"],
        split="train",
        transform=train_transform,
    )
    pad_val_dataset = PADDataset(
        split_path=data_cfg["pad_split_path"],
        split="val",
        transform=eval_transform,
    )

    if is_main():
        print(f"\n{recog_train_dataset}")
        print(f"{recog_val_dataset}")
        print(f"{pad_train_dataset}")
        print(f"{pad_val_dataset}")

    # ── Dataloaders ───────────────────────────────────────────────────────
    recog_train_sampler = DistributedSampler(
        recog_train_dataset,
        num_replicas=world_size,
        rank=local_rank,
        shuffle=True,
        seed=general_cfg["seed"],
    )
    recog_train_loader = DataLoader(
        recog_train_dataset,
        batch_size=training_cfg["recog_batch_size"],
        sampler=recog_train_sampler,
        num_workers=training_cfg["num_workers"],
        pin_memory=training_cfg["pin_memory"],
    )

    pad_train_sampler = DistributedSampler(
        pad_train_dataset,
        num_replicas=world_size,
        rank=local_rank,
        shuffle=True,
        seed=general_cfg["seed"],
    )
    pad_train_loader = DataLoader(
        pad_train_dataset,
        batch_size=training_cfg["pad_batch_size"],
        sampler=pad_train_sampler,
        num_workers=training_cfg["num_workers"],
        pin_memory=training_cfg["pin_memory"],
    )

    # Validation loaders
    recog_val_loader = DataLoader(
        recog_val_dataset,
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

    pad_val_loader = DataLoader(
        pad_val_dataset,
        batch_size=evaluation_cfg["pad_batch_size"],
        shuffle=False,
        num_workers=training_cfg["num_workers"],
        pin_memory=training_cfg["pin_memory"],
    )

    # ── Teachers (frozen) ─────────────────────────────────────────────────
    if is_main():
        print(
            f"\n[teacher] Loading recog teacher from {teacher_cfg['recog_checkpoint']}"
        )
    recog_teacher = load_teacher(
        teacher_cfg["recog_checkpoint"],
        model_cfg=model_cfg,
        device=device,
        branch="a",
    )

    if is_main():
        print(f"[teacher] Loading PAD teacher from {teacher_cfg['pad_checkpoint']}")
    pad_teacher = load_teacher(
        teacher_cfg["pad_checkpoint"],
        model_cfg=model_cfg,
        device=device,
        branch="b",
    )

    if is_main():
        n_recog_t = sum(p.numel() for p in recog_teacher.parameters()) / 1e6
        n_pad_t = sum(p.numel() for p in pad_teacher.parameters()) / 1e6
        print(f"[teacher] Recog teacher: {n_recog_t:.2f}M params (frozen)")
        print(f"[teacher] PAD teacher:   {n_pad_t:.2f}M params (frozen)")

    # ── Student Model ─────────────────────────────────────────────────────
    model = DualViT(
        model_name=model_cfg["model_name"],
        pretrained=model_cfg["pretrained"],
        branch_a_num_classes=model_cfg["branch_a_num_classes"],
        branch_b_num_classes=model_cfg["branch_b_num_classes"],
        head_hidden_dim=model_cfg["head_hidden_dim"],
        head_drop_rate=model_cfg["head_drop_rate"],
    ).to(device)
    if model_cfg.get("ckpt_path"):
        model.load_state_dict(torch.load(model_cfg["ckpt_path"], map_location="cpu")["model"])
        tqdm.write(f"  [model] loaded from {model_cfg['ckpt_path']}")
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    if is_main():
        n_params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"[student] DualViT  ({n_params:.2f}M params)")

    # ── Loss ──────────────────────────────────────────────────────────────
    n_ids = recog_train_dataset.n_ids
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

    steps_per_epoch = min(len(recog_train_loader), len(pad_train_loader))
    total_iters = training_cfg["epochs"] * steps_per_epoch
    warmup_iters = sched_cfg["warmup_epochs"] * steps_per_epoch
    scheduler = cosine_warmup_scheduler(
        optimizer,
        warmup_iters=warmup_iters,
        total_iters=total_iters,
        min_lr=sched_cfg["min_lr"],
    )

    scaler = torch.amp.GradScaler("cuda")

    recog_weight = loss_cfg.get("recog_weight", 1.0)
    pad_weight = loss_cfg.get("pad_weight", 1.0)
    distill_weight = loss_cfg.get("distill_weight", 1.0)

    if is_main():
        print(
            f"\nLoss weights: recog={recog_weight}, pad={pad_weight}, "
            f"distill={distill_weight}"
        )

    # ── Training loop ─────────────────────────────────────────────────────
    start_epoch = 1
    best_avg_ace_eer = float("inf")

    if checkpoint is not None:
        start_epoch, best_avg_ace_eer = load_checkpoint(
            checkpoint, model, arcface_loss, optimizer, scheduler, scaler
        )

    if is_main():
        if not wandb.run:
            history = {
                "epoch": [],
                "loss": [],
                "recog_loss": [],
                "pad_loss": [],
                "distill_loss": [],
                "val_eer": [],
                "val_ace": [],
                "val_avg": [],
            }

        os.makedirs(output_cfg["checkpoint_dir"], exist_ok=True)

        print("\n" + "=" * 70)
        print("Starting MTLD training")
        print("=" * 70)

    epoch_pbar = tqdm(
        range(start_epoch, training_cfg["epochs"] + 1),
        desc="Training",
        unit="epoch",
        disable=not is_main(),
    )

    for epoch in epoch_pbar:
        train_metrics = train_one_epoch(
            model,
            arcface_loss,
            recog_teacher,
            pad_teacher,
            recog_train_loader,
            recog_train_sampler,
            pad_train_loader,
            pad_train_sampler,
            optimizer,
            scheduler,
            scaler,
            device,
            epoch,
            steps_per_epoch,
            recog_weight=recog_weight,
            pad_weight=pad_weight,
            distill_weight=distill_weight,
        )

        dist.barrier()

        # ── Recognition evaluation ────────────────────────────────────────
        global_embeddings = get_embeddings(
            _unwrap(model), unique_val_loader, device, epoch
        )

        if is_main():
            eer, thr = evaluate_recog(
                recog_val_loader, global_embeddings, device, epoch
            )

            # ── PAD evaluation ────────────────────────────────────────────
            pad_metrics = evaluate_pad(_unwrap(model), pad_val_loader, device, epoch)

            ace = pad_metrics["val/ace"]
            avg_ace_eer = (ace + eer) / 2.0

            metrics = {
                "val/eer": eer,
                "val/threshold": thr,
                **pad_metrics,
                "val/avg_ace_eer": avg_ace_eer,
            }

            epoch_pbar.set_postfix(
                total=f"{train_metrics['train/loss']:.4f}",
                recog=f"{train_metrics['train/recog_loss']:.4f}",
                pad=f"{train_metrics['train/pad_loss']:.4f}",
                eer=f"{eer:.4f}",
                ace=f"{ace:.4f}",
                avg=f"{avg_ace_eer:.4f}",
            )
            tqdm.write(
                f"Epoch {epoch:03d} | "
                f"total: {train_metrics['train/loss']:.4f} | "
                f"recog: {train_metrics['train/recog_loss']:.4f} | "
                f"pad: {train_metrics['train/pad_loss']:.4f} | "
                f"distill: {train_metrics['train/distill_loss']:.4f} | "
                f"EER: {eer:.4f} (thr={thr:.4f}) | "
                f"ACE: {ace:.4f} "
                f"(APCER={pad_metrics['val/apcer']:.4f}, BPCER={pad_metrics['val/bpcer']:.4f}) | "
                f"avg: {avg_ace_eer:.4f}"
            )

            if wandb.run is not None:
                wandb.log(
                    {
                        **train_metrics,
                        "epoch": epoch,
                        **metrics,
                    }
                )
            else:
                history["epoch"].append(epoch)
                history["loss"].append(train_metrics["train/loss"])
                history["recog_loss"].append(train_metrics["train/recog_loss"])
                history["pad_loss"].append(train_metrics["train/pad_loss"])
                history["distill_loss"].append(train_metrics["train/distill_loss"])
                history["val_eer"].append(eer)
                history["val_ace"].append(ace)
                history["val_avg"].append(avg_ace_eer)

            if avg_ace_eer < best_avg_ace_eer:
                best_avg_ace_eer = avg_ace_eer
                save_best(
                    output_cfg["checkpoint_dir"],
                    output_cfg["best_model_name"],
                    epoch,
                    model,
                    arcface_loss,
                    metrics,
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
                    best_avg_ace_eer,
                )

        dist.barrier()

    # ── Cleanup ───────────────────────────────────────────────────────────

    if is_main():
        print("=" * 70)
        print(f"Training complete. Best val avg(ACE, EER): {best_avg_ace_eer:.4f}")
        print("=" * 70)

        if wandb.run is not None:
            wandb.finish()
        else:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, 4, figsize=(24, 5))

            # Task losses
            axes[0].plot(history["epoch"], history["loss"], "k-", label="Total Loss")
            axes[0].plot(
                history["epoch"], history["recog_loss"], "g-", label="Recog Loss"
            )
            axes[0].plot(history["epoch"], history["pad_loss"], "r-", label="PAD Loss")
            axes[0].set_xlabel("Epoch")
            axes[0].set_ylabel("Loss")
            axes[0].legend()
            axes[0].set_title("Task Losses")

            # Distillation losses
            axes[1].plot(
                history["epoch"], history["distill_loss"], "c-", label="Distillation"
            )
            axes[1].set_xlabel("Epoch")
            axes[1].set_ylabel("Loss")
            axes[1].legend()
            axes[1].set_title("Distillation Losses")

            # EER & ACE
            axes[2].plot(history["epoch"], history["val_eer"], "b-", label="Val EER")
            axes[2].plot(history["epoch"], history["val_ace"], "m-", label="Val ACE")
            axes[2].set_xlabel("Epoch")
            axes[2].set_ylabel("Error Rate")
            axes[2].legend()
            axes[2].set_title("Validation Metrics")

            # Combined
            axes[3].plot(
                history["epoch"], history["val_avg"], "k-", label="avg(ACE, EER)"
            )
            axes[3].set_xlabel("Epoch")
            axes[3].set_ylabel("avg(ACE, EER)")
            axes[3].legend()
            axes[3].set_title("Best Model Selection Metric")

            plt.tight_layout()
            plot_path = os.path.join(
                output_cfg["checkpoint_dir"], "training_history.png"
            )
            plt.savefig(plot_path)
            plt.close()
            print(f"Saved training history plot to {plot_path}")

    cleanup_ddp()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Multi-Task Multi-Layer Distillation Training"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="mtld_config.yaml",
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
