import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import (
    AuthenticationEvaluationDataset,
    IdentificationEvaluationDataset,
    UniqueFingerprintDataset,
)
from metrics import compute_authentication_metrics, compute_identification_metrics
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
def collect_authentication_scores(
    model: torch.nn.Module,
    authentication_loader: DataLoader,
    unique_loader: DataLoader,
    device: torch.device,
    embed_dim: int,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()

    n_unique_images = len(unique_loader.dataset)

    embeddings = torch.zeros((n_unique_images, embed_dim), device=device)
    for idxs, imgs in tqdm(unique_loader, desc="Extracting Embeddings", unit="batch"):
        imgs = imgs.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda"):
            emb = model.branch_forward(imgs, branch="a")
        emb = F.normalize(emb, dim=1).float()
        embeddings[idxs] = emb

    all_scores, all_labels = [], []
    for idx_a, idx_b, labels in tqdm(
        authentication_loader, desc="Authentication Inference", unit="batch"
    ):
        idx_a = idx_a.to(device, non_blocking=True)
        idx_b = idx_b.to(device, non_blocking=True)

        emb_a = embeddings[idx_a]
        emb_b = embeddings[idx_b]

        cos_sim = (emb_a * emb_b).sum(dim=1).cpu().numpy()

        all_scores.append(cos_sim)
        all_labels.append(labels.numpy())

    return np.concatenate(all_scores), np.concatenate(all_labels)


@torch.no_grad()
def collect_identification_scores(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    dataset: torch.utils.data.Dataset,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()

    all_features = []
    all_labels = []
    all_indices = []

    with torch.no_grad():
        for imgs, labels, idx in tqdm(loader, desc="Identification Inference", unit="batch"):
            imgs = imgs.to(device, non_blocking=True)

            features = model.branch_forward(imgs, branch="a")

            features = F.normalize(features, dim=1).cpu()

            all_features.append(features)
            all_labels.extend(labels.numpy())
            all_indices.extend(idx.numpy())

    all_features = torch.cat(all_features, dim=0).numpy()
    all_labels = np.array(all_labels)
    all_indices = np.array(all_indices)

    sort_order = np.argsort(all_indices)
    all_features = all_features[sort_order]
    all_labels = all_labels[sort_order]

    n_gal = dataset.n_gallery
    gallery_feats = all_features[:n_gal]
    gallery_labels = all_labels[:n_gal]

    probe_feats = all_features[n_gal:]
    probe_labels = all_labels[n_gal:]

    sim_mat = np.dot(probe_feats, gallery_feats.T)
    return sim_mat, probe_labels, gallery_labels


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
    authentication_dataset = AuthenticationEvaluationDataset(
        split_path=args.split_path,
        split="test",
        n_genuine_impressions=data_cfg["n_genuine_impressions"],
        n_impostor_impressions=data_cfg["n_impostor_impressions"],
        impostor_mode=data_cfg["impostor_mode"],
        n_impostor_subset=data_cfg["n_impostor_subset"],
        seed=general_cfg["seed"],
    )
    print(f"\n{authentication_dataset}")

    unique_dataset = UniqueFingerprintDataset(
        idx_to_path=authentication_dataset.idx_to_path,
        transform=eval_transform,
    )

    identification_dataset = IdentificationEvaluationDataset(
        split_path=args.split_path,
        split="test",
        gallery_per_id=data_cfg["gallery_per_id"],
        probe_per_id=data_cfg["probe_per_id"],
        transform=eval_transform,
        seed=general_cfg["seed"],
    )
    print(f"\n{identification_dataset}")

    # ── Dataloaders ──────────────────────────────────────────────────────
    authentication_loader = DataLoader(
        authentication_dataset,
        batch_size=eval_cfg["recog_batch_size"],
        shuffle=False,
        num_workers=train_cfg["num_workers"],
        pin_memory=train_cfg["pin_memory"],
    )

    unique_loader = DataLoader(
        unique_dataset,
        batch_size=train_cfg["recog_batch_size"],
        shuffle=False,
        num_workers=train_cfg["num_workers"],
        pin_memory=train_cfg["pin_memory"],
    )

    identification_loader = DataLoader(
        identification_dataset,
        batch_size=train_cfg["recog_batch_size"],
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

    scores, labels = collect_authentication_scores(
        model, authentication_loader, unique_loader, device, embed_dim
    )
    authentication_metrics = compute_authentication_metrics(scores, labels)

    sim_mat, probe_labels, gallery_labels = collect_identification_scores(
        model, identification_loader, device, identification_dataset
    )
    identification_metrics = compute_identification_metrics(
        sim_mat, probe_labels, gallery_labels
    )

    # ── Report ───────────────────────────────────────────────────────────
    print(f"\nSplit path: {args.split_path}")

    print("\n" + "=" * 50)
    print("Authentication Metrics:")
    print("-" * 50)
    print(
        f"Total pairs: {len(authentication_dataset):,} "
        f"(genuine={authentication_dataset.n_genuine:,}, impostor={authentication_dataset.n_impostor:,})"
    )
    print("-" * 50)
    print(
        f"EER           : {authentication_metrics['eer']:.2%} (threshold={authentication_metrics['eer_threshold']:.4f})"
    )
    print(f"AUC (ROC)     : {authentication_metrics['auc']:.2%}")
    print(f"TAR@FAR=0.1   : {authentication_metrics['tar_at_far_0.1']:.2%}")
    print(f"TAR@FAR=0.01  : {authentication_metrics['tar_at_far_0.01']:.2%}")
    print(f"TAR@FAR=0.001 : {authentication_metrics['tar_at_far_0.001']:.2%}")
    print("=" * 50)
    print("Identification Metrics:")
    print("-" * 50)
    print(f"Identities: {identification_dataset.n_ids:,}")
    print(
        f"Gallery samples: {identification_dataset.n_gallery:,}, Probe samples: {identification_dataset.n_probes:,}"
    )
    print("-" * 50)
    print(f"Rank-1 Accuracy : {identification_metrics['rank_1']:.2%}")
    print(f"Rank-5 Accuracy : {identification_metrics['rank_5']:.2%}")
    print(f"Rank-10 Accuracy: {identification_metrics['rank_10']:.2%}")
    print("=" * 50)

    # ── Save results to JSON ──────────────────────────────────────────────────
    summary = {
        "split_path": args.split_path,
        "authentication": {
            "n_pairs": len(authentication_dataset),
            "n_genuine": authentication_dataset.n_genuine,
            "n_impostor": authentication_dataset.n_impostor,
            "eer": authentication_metrics["eer"],
            "eer_threshold": authentication_metrics["eer_threshold"],
            "auc": authentication_metrics["auc"],
            "tar_at_far_0.1": authentication_metrics["tar_at_far_0.1"],
            "tar_at_far_0.01": authentication_metrics["tar_at_far_0.01"],
            "tar_at_far_0.001": authentication_metrics["tar_at_far_0.001"],
        },
        "identification": {
            "n_ids": identification_dataset.n_ids,
            "n_gallery": identification_dataset.n_gallery,
            "n_probes": identification_dataset.n_probes,
            "rank_1": identification_metrics["rank_1"],
            "rank_5": identification_metrics["rank_5"],
            "rank_10": identification_metrics["rank_10"],
        },
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
