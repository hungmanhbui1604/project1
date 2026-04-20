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
from model import DualSwinTransformerTiny, SwinTransformerTiny
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
# Teacher Feature Extraction
# ---------------------------------------------------------------------------


@torch.no_grad()
def teacher_forward(
    model: SwinTransformerTiny, x: torch.Tensor
) -> dict[str, torch.Tensor]:
    feats = {}
    x, H, W = model.patch_embed(x)
    x, H, W = model.stages[0](x, H, W)
    x, H, W = model.stages[1](x, H, W)
    feats["stage2"] = x
    x, H, W = model.stages[2](x, H, W)
    feats["stage3"] = x
    x, H, W = model.stages[3](x, H, W)
    feats["stage4"] = x
    x = model.norm(x)
    x = model.avgpool(x.transpose(1, 2)).flatten(1)
    x = model.head(x)
    feats["output"] = x
    return feats


# ---------------------------------------------------------------------------
# Student Feature-Hook Manager
# ---------------------------------------------------------------------------


class FeatureHooks:
    def __init__(self):
        self.features: dict[str, torch.Tensor] = {}
        self._hooks: list[torch.utils.hooks.RemovableHook] = []

    def register(self, module: nn.Module, name: str) -> None:
        def hook(mod, inp, out):
            # SwinTransformerStage.forward returns (x, H, W)
            self.features[name] = out[0] if isinstance(out, tuple) else out

        self._hooks.append(module.register_forward_hook(hook))

    def clear(self) -> None:
        self.features.clear()

    def remove_all(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


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


def embedding_distillation_loss(
    student_emb: torch.Tensor, teacher_emb: torch.Tensor
) -> torch.Tensor:
    """Cosine-distance loss: mean(1 − cos_sim) over the batch."""
    s = F.normalize(student_emb.float(), dim=-1)
    t = F.normalize(teacher_emb.float(), dim=-1)
    return (1.0 - (s * t).sum(dim=-1)).mean()


def logit_distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 4.0,
) -> torch.Tensor:
    """Hinton-style KL-divergence loss with temperature scaling."""
    s = F.log_softmax(student_logits.float() / temperature, dim=-1)
    t = F.softmax(teacher_logits.float() / temperature, dim=-1)
    return F.kl_div(s, t, reduction="batchmean") * (temperature**2)


def compute_distillation_losses(
    student_hooks: dict[str, torch.Tensor],
    student_emb_a: torch.Tensor,
    student_emb_b: torch.Tensor,
    recog_t_feats: dict[str, torch.Tensor],
    pad_t_feats: dict[str, torch.Tensor],
    temperature: float = 4.0,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    breakdown = {}

    # ── Shared stage 2: distil from both teachers (averaged) ──────────
    l_s2_recog = feature_distillation_loss(
        student_hooks["stage2"], recog_t_feats["stage2"]
    )
    l_s2_pad = feature_distillation_loss(student_hooks["stage2"], pad_t_feats["stage2"])
    l_stage2 = 0.5 * (l_s2_recog + l_s2_pad)
    breakdown["distill/stage2"] = l_stage2.item()

    # ── Branch A intermediates ← recog teacher ────────────────────────
    l_a_s3 = feature_distillation_loss(
        student_hooks["a_stage3"], recog_t_feats["stage3"]
    )
    l_a_s4 = feature_distillation_loss(
        student_hooks["a_stage4"], recog_t_feats["stage4"]
    )
    breakdown["distill/a_stage3"] = l_a_s3.item()
    breakdown["distill/a_stage4"] = l_a_s4.item()

    # ── Branch B intermediates ← PAD teacher ──────────────────────────
    l_b_s3 = feature_distillation_loss(student_hooks["b_stage3"], pad_t_feats["stage3"])
    l_b_s4 = feature_distillation_loss(student_hooks["b_stage4"], pad_t_feats["stage4"])
    breakdown["distill/b_stage3"] = l_b_s3.item()
    breakdown["distill/b_stage4"] = l_b_s4.item()

    # ── Output distillation ───────────────────────────────────────────
    l_emb_a = embedding_distillation_loss(student_emb_a, recog_t_feats["output"])
    l_emb_b = logit_distillation_loss(student_emb_b, pad_t_feats["output"], temperature)
    breakdown["distill/emb_a"] = l_emb_a.item()
    breakdown["distill/emb_b"] = l_emb_b.item()

    inter_loss = l_stage2 + l_a_s3 + l_a_s4 + l_b_s3 + l_b_s4
    output_loss = l_emb_a + l_emb_b

    breakdown["distill/inter_total"] = inter_loss.item()
    breakdown["distill/output_total"] = output_loss.item()

    return inter_loss, output_loss, breakdown


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
    model: DualSwinTransformerTiny,
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
            emb_a, _ = model(imgs)
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
            _, logits = model(images)
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
    checkpoint_path: str, embed_dim: int, device: torch.device
) -> SwinTransformerTiny:
    """
    Load a pre-trained SwinTransformerTiny teacher from checkpoint.

    The checkpoint is expected to contain a 'model' key with the state dict
    (as saved by recog_train.py / pad_train.py).
    """
    model = SwinTransformerTiny(embed_dim=embed_dim).to(device)
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
    best_metric = float("inf")

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
        if "best_metric" in ckpt_dict:
            best_metric = ckpt_dict["best_metric"]

        if is_main():
            print(f"=> Loaded checkpoint (epoch {start_epoch - 1})")
    else:
        if is_main():
            print(f"=> No checkpoint found at '{path}'")

    return start_epoch, best_metric


def save_checkpoint(
    path: str,
    epoch: int,
    model: DDP,
    arcface_loss: DDP,
    optimizer: AdamW,
    scheduler: LambdaLR,
    scaler: torch.amp.GradScaler,
    best_metric: float,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model": _unwrap(model).state_dict(),
            "arcface": _unwrap(arcface_loss).state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "best_metric": best_metric,
        },
        path,
    )
    tqdm.write(f"  [checkpoint] saved → {path}")

    if wandb.run is not None:
        artifact = wandb.Artifact(
            name=f"checkpoint-epoch{epoch:03d}",
            type="checkpoint",
            metadata={"epoch": epoch, "best_metric": best_metric},
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
    recog_teacher: SwinTransformerTiny,
    pad_teacher: SwinTransformerTiny,
    hooks: FeatureHooks,
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
    inter_distill_weight: float = 0.5,
    output_distill_weight: float = 1.0,
    temperature: float = 4.0,
) -> dict[str, float]:
    model.train()
    arcface_loss.train()

    recog_sampler.set_epoch(epoch)
    pad_sampler.set_epoch(epoch)

    total_loss = 0.0
    total_recog_loss = 0.0
    total_pad_loss = 0.0
    total_inter_distill = 0.0
    total_output_distill = 0.0

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
            recog_t_feats = teacher_forward(recog_teacher, combined)
            pad_t_feats = teacher_forward(pad_teacher, combined)

        # ── Student forward (DDP, hooks capture intermediates) ────────
        hooks.clear()
        with torch.autocast(device_type="cuda"):
            emb_a, emb_b = model(combined)

            # Task losses (split outputs by task)
            recog_emb = emb_a[:n_recog]
            pad_logits = emb_b[n_recog:]
            recog_loss, _ = arcface_loss(recog_emb, recog_labels)
            pad_loss = F.cross_entropy(pad_logits, pad_labels)

            # Distillation losses (computed on full combined batch)
            inter_loss, output_loss, distill_info = compute_distillation_losses(
                hooks.features,
                emb_a,
                emb_b,
                recog_t_feats,
                pad_t_feats,
                temperature,
            )

            loss = (
                recog_weight * recog_loss
                + pad_weight * pad_loss
                + inter_distill_weight * inter_loss
                + output_distill_weight * output_loss
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
        total_inter_distill += inter_loss.item()
        total_output_distill += output_loss.item()

        lr_val = scheduler.get_last_lr()[0]
        pbar.set_postfix(
            total=f"{loss.item():.4f}",
            recog=f"{recog_loss.item():.4f}",
            pad=f"{pad_loss.item():.4f}",
            dist=f"{(inter_loss.item() + output_loss.item()):.4f}",
            lr=f"{lr_val:.2e}",
        )

    return {
        "train/loss": total_loss / steps_per_epoch,
        "train/recog_loss": total_recog_loss / steps_per_epoch,
        "train/pad_loss": total_pad_loss / steps_per_epoch,
        "train/inter_distill_loss": total_inter_distill / steps_per_epoch,
        "train/output_distill_loss": total_output_distill / steps_per_epoch,
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
        wandb.init(project=wandb_cfg.get("project", "DualSwin-MTLD"), config=cfg)

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
        embed_dim=teacher_cfg["recog_embed_dim"],
        device=device,
    )

    if is_main():
        print(f"[teacher] Loading PAD teacher from {teacher_cfg['pad_checkpoint']}")
    pad_teacher = load_teacher(
        teacher_cfg["pad_checkpoint"],
        embed_dim=teacher_cfg["pad_embed_dim"],
        device=device,
    )

    if is_main():
        n_recog_t = sum(p.numel() for p in recog_teacher.parameters()) / 1e6
        n_pad_t = sum(p.numel() for p in pad_teacher.parameters()) / 1e6
        print(f"[teacher] Recog teacher: {n_recog_t:.2f}M params (frozen)")
        print(f"[teacher] PAD teacher:   {n_pad_t:.2f}M params (frozen)")

    # ── Student Model ─────────────────────────────────────────────────────
    model = DualSwinTransformerTiny(
        embed_dim_a=model_cfg["embed_dim_a"],
        embed_dim_b=model_cfg["embed_dim_b"],
    ).to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    if is_main():
        n_params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"[student] DualSwinTransformerTiny  ({n_params:.2f}M params)")

    # ── Register hooks on student for intermediate feature capture ────────
    student = _unwrap(model)
    hooks = FeatureHooks()
    hooks.register(student.shared_stage2, "stage2")
    hooks.register(student.branch_a_stage3, "a_stage3")
    hooks.register(student.branch_a_stage4, "a_stage4")
    hooks.register(student.branch_b_stage3, "b_stage3")
    hooks.register(student.branch_b_stage4, "b_stage4")

    # ── Loss ──────────────────────────────────────────────────────────────
    n_ids = recog_train_dataset.n_ids
    arcface_loss = ArcFaceLoss(
        embed_dim=model_cfg["embed_dim_a"],
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

    # ── Loss weights ──────────────────────────────────────────────────────
    recog_weight = loss_cfg.get("recog_weight", 1.0)
    pad_weight = loss_cfg.get("pad_weight", 1.0)
    inter_distill_weight = loss_cfg.get("inter_distill_weight", 0.5)
    output_distill_weight = loss_cfg.get("output_distill_weight", 1.0)
    temperature = loss_cfg.get("temperature", 4.0)

    if is_main():
        print(
            f"\nLoss weights: recog={recog_weight}, pad={pad_weight}, "
            f"inter_distill={inter_distill_weight}, output_distill={output_distill_weight}, "
            f"temperature={temperature}"
        )

    # ── Training loop ─────────────────────────────────────────────────────
    start_epoch = 1
    best_metric = float("inf")

    if checkpoint is not None:
        start_epoch, best_metric = load_checkpoint(
            checkpoint, model, arcface_loss, optimizer, scheduler, scaler
        )

    if is_main():
        if not wandb.run:
            history = {
                "epoch": [],
                "loss": [],
                "recog_loss": [],
                "pad_loss": [],
                "inter_distill": [],
                "output_distill": [],
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
            hooks,
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
            inter_distill_weight=inter_distill_weight,
            output_distill_weight=output_distill_weight,
            temperature=temperature,
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
                f"inter_d: {train_metrics['train/inter_distill_loss']:.4f} | "
                f"out_d: {train_metrics['train/output_distill_loss']:.4f} | "
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
                history["inter_distill"].append(
                    train_metrics["train/inter_distill_loss"]
                )
                history["output_distill"].append(
                    train_metrics["train/output_distill_loss"]
                )
                history["val_eer"].append(eer)
                history["val_ace"].append(ace)
                history["val_avg"].append(avg_ace_eer)

            if avg_ace_eer < best_metric:
                best_metric = avg_ace_eer
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
                    best_metric,
                )

        dist.barrier()

    # ── Cleanup ───────────────────────────────────────────────────────────
    hooks.remove_all()

    if is_main():
        print("=" * 70)
        print(f"Training complete. Best val avg(ACE, EER): {best_metric:.4f}")
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
                history["epoch"], history["inter_distill"], "c-", label="Intermediate"
            )
            axes[1].plot(
                history["epoch"], history["output_distill"], "m-", label="Output"
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
