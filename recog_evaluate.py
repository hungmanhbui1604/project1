import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import RecogEvaluationDataset, UniqueImageDataset
from metrics import compute_recog_metrics
from models import get_model
from transforms import get_transforms

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Embedding Extraction & Evaluation
# ---------------------------------------------------------------------------


@torch.no_grad()
def collect_scores(
    model: torch.nn.Module,
    loader: DataLoader,
    unique_loader: DataLoader,
    device: torch.device,
    embed_dim: int,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()

    n_unique_images = len(unique_loader.dataset)

    embeddings = torch.zeros((n_unique_images, embed_dim), device=device)
    for idxs, imgs in tqdm(unique_loader, desc="Extracting embeddings", unit="batch"):
        imgs = imgs.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda"):
            emb, _ = model(imgs, branch="a")
        emb = F.normalize(emb, dim=1).float()
        embeddings[idxs] = emb

    all_scores, all_labels = [], []
    for idx_a, idx_b, labels in tqdm(loader, desc="Inference", unit="batch"):
        idx_a = idx_a.to(device, non_blocking=True)
        idx_b = idx_b.to(device, non_blocking=True)

        emb_a = embeddings[idx_a]
        emb_b = embeddings[idx_b]

        cos_sim = (emb_a * emb_b).sum(dim=1).cpu().numpy()

        all_scores.append(cos_sim)
        all_labels.append(labels.numpy())

    return np.concatenate(all_scores), np.concatenate(all_labels)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    general_cfg = cfg["general"]
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    eval_cfg = cfg["evaluation"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Checkpoint path ──────────────────────────────────────────────────
    ckpt_path = args.checkpoint_path

    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # ── Transforms ───────────────────────────────────────────────────────
    _, eval_transform, _ = get_transforms(data_cfg["transform_name"])

    # ── Datasets ─────────────────────────────────────────────────────────
    dataset = RecogEvaluationDataset(
        split_path=args.split_path,
        split="test",
        n_genuine_impressions=data_cfg["n_genuine_impressions"],
        n_impostor_impressions=data_cfg["n_impostor_impressions"],
        impostor_mode=data_cfg["impostor_mode"],
        n_impostor_subset=data_cfg["n_impostor_subset"],
        seed=general_cfg["seed"],
    )
    print(f"\n{dataset}")

    unique_dataset = UniqueImageDataset(
        idx_to_path=dataset.idx_to_path,
        transform=eval_transform,
    )

    # ── Dataloaders ──────────────────────────────────────────────────────
    loader = DataLoader(
        dataset,
        batch_size=eval_cfg["recog_batch_size"],
        shuffle=False,
        num_workers=train_cfg["num_workers"],
        pin_memory=train_cfg["pin_memory"],
    )

    unique_loader = DataLoader(
        unique_dataset,
        batch_size=eval_cfg["recog_batch_size"],
        shuffle=False,
        num_workers=train_cfg["num_workers"],
        pin_memory=train_cfg["pin_memory"],
    )

    # ── Model ────────────────────────────────────────────────────────────
    model = get_model(model_cfg["model_name"], model_cfg).to(device)
    embed_dim = model_cfg["branch_a_num_classes"]
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(
        f"[model] {model_cfg['model_name']} ({n_params:.2f}M params, embed_dim={embed_dim})"
    )

    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])

    # ── Evaluate ─────────────────────────────────────────────────────────
    print("\nEvaluating on recognition test set...")

    scores, labels = collect_scores(model, loader, unique_loader, device, embed_dim)
    metrics = compute_recog_metrics(scores, labels)

    # ── Report ───────────────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print(f"Split path: {args.split_path}")
    print(
        f"Total pairs: {len(dataset):,} "
        f"(genuine={dataset.n_genuine:,}, impostor={dataset.n_impostor:,})"
    )
    print("-" * 50)
    print(
        f"EER           : {metrics['eer']:.4f} (threshold={metrics['eer_threshold']:.4f})"
    )
    print(f"AUC (ROC)     : {metrics['auc']:.4f}")
    print(f"TAR@FAR=0.1   : {metrics['tar_at_far_0.1']:.4f}")
    print(f"TAR@FAR=0.01  : {metrics['tar_at_far_0.01']:.4f}")
    print(f"TAR@FAR=0.001 : {metrics['tar_at_far_0.001']:.4f}")
    print("=" * 50)

    # ── Save results to JSON ──────────────────────────────────────────────────
    summary = {
        "split_path": args.split_path,
        "n_pairs": len(dataset),
        "n_genuine": dataset.n_genuine,
        "n_impostor": dataset.n_impostor,
        "eer": metrics["eer"],
        "eer_threshold": metrics["eer_threshold"],
        "auc": metrics["auc"],
        "tar_at_far_0.1": metrics["tar_at_far_0.1"],
        "tar_at_far_0.01": metrics["tar_at_far_0.01"],
        "tar_at_far_0.001": metrics["tar_at_far_0.001"],
    }
    json_path = os.path.join(args.output_dir, "recog_metrics.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved → {json_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recognition Evaluation")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the PAD YAML config",
    )
    parser.add_argument(
        "--split-path",
        required=True,
        help="Path to the PAD split JSON file",
    )
    parser.add_argument(
        "--output-dir",
        default="results/recog/",
        help="Directory for metrics JSON and plot PNGs",
    )
    parser.add_argument(
        "--checkpoint-path",
        required=True,
        help="Path to the checkpoint",
    )
    args = parser.parse_args()
    main(args)
