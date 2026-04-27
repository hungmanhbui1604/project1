import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import auc, roc_curve
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import RecogEvaluationDataset, UniqueImageDataset
from model import get_model
from transforms import get_transforms

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def compute_recog_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
) -> dict:
    # ROC
    fmr, tar, thrs = roc_curve(labels, scores, pos_label=1)
    fnmr = 1.0 - tar

    # Remove sklearn's artificial first point
    if len(thrs) > 1 and np.isinf(thrs[0]):
        fmr = fmr[1:]
        tar = tar[1:]
        fnmr = fnmr[1:]
        thrs = thrs[1:]

    # EER (interpolation)
    diff = fmr - fnmr
    idx1_candidates = np.where(diff >= 0)[0]

    if len(idx1_candidates) == 0:
        # fallback: closest point
        eer_idx = int(np.argmin(np.abs(diff)))
        eer = (fmr[eer_idx] + fnmr[eer_idx]) / 2.0
        eer_thr = thrs[eer_idx]
    else:
        idx1 = idx1_candidates[0]
        idx0 = idx1 - 1 if idx1 > 0 else idx1

        x0, y0 = fmr[idx0], fnmr[idx0]
        x1, y1 = fmr[idx1], fnmr[idx1]

        if idx0 == idx1:
            eer = (x0 + y0) / 2.0
            eer_thr = thrs[idx0]
        else:
            den = (x1 - x0) - (y1 - y0)
            if abs(den) < 1e-12:
                eer = (x0 + y0) / 2.0
                eer_thr = thrs[idx0]
            else:
                t = (y0 - x0) / den
                eer = x0 + t * (x1 - x0)
                eer_thr = thrs[idx0] + t * (thrs[idx1] - thrs[idx0])

    # AUC
    auc_roc = float(auc(fmr, tar))

    # TAR@FAR (interpolation)
    def interp_tar_at_far(target_far: float) -> float:
        # exact match
        mask = (fmr == target_far)
        if mask.any():
            return float(tar[mask].max())

        idx = np.searchsorted(fmr, target_far, side="right")

        if idx == 0:
            return float(tar[0])
        if idx >= len(fmr):
            return float(tar[-1])

        x0, x1 = fmr[idx - 1], fmr[idx]
        y0, y1 = tar[idx - 1], tar[idx]

        # avoid division by zero
        if abs(x1 - x0) < 1e-12:
            return float(y0)

        t = (target_far - x0) / (x1 - x0)
        return float(y0 + t * (y1 - y0))

    tar_at_far = {
        0.1: interp_tar_at_far(0.1),
        0.01: interp_tar_at_far(0.01),
        0.001: interp_tar_at_far(0.001),
    }

    return {
        "EER": float(eer),
        "EER_threshold": float(eer_thr),
        "AUC": auc_roc,
        "thresholds": thrs.tolist(),
        "FMR": fmr.tolist(),
        "TAR": tar.tolist(),
        "FNMR": fnmr.tolist(),
        "TAR@FAR=0.1": tar_at_far[0.1],
        "TAR@FAR=0.01": tar_at_far[0.01],
        "TAR@FAR=0.001": tar_at_far[0.001],
    }


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
    print(f"[model] {model_cfg['model_name']} ({n_params:.2f}M params, embed_dim={embed_dim})")

    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model"])

    # ── Evaluate ─────────────────────────────────────────────────────────
    print("\nEvaluating on recognition test set...")

    scores, labels = collect_scores(
        model, loader, unique_loader, device, embed_dim
    )
    metrics = compute_recog_metrics(scores, labels)

    # ── Report ───────────────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print(f"Split path: {args.split_path}")
    print(
        f"Total pairs: {len(dataset):,} "
        f"(genuine={dataset.n_genuine:,}, impostor={dataset.n_impostor:,})"
    )
    print("-" * 50)
    print(f"EER: {metrics['EER']:.4f} (threshold={metrics['EER_threshold']:.4f})")
    print(f"AUC (ROC): {metrics['AUC']:.4f}")
    print(f"TAR @ FAR=0.1: {metrics['TAR@FAR=0.1']:.4f}")
    print(f"TAR @ FAR=0.01: {metrics['TAR@FAR=0.01']:.4f}")
    print(f"TAR @ FAR=0.001: {metrics['TAR@FAR=0.001']:.4f}")
    print("=" * 50)

    # ── Save results to JSON ──────────────────────────────────────────────────
    summary = {
        "split_path": args.split_path,
        "n_pairs": len(dataset),
        "n_genuine": dataset.n_genuine,
        "n_impostor": dataset.n_impostor,
        "EER": metrics["EER"],
        "EER_threshold": metrics["EER_threshold"],
        "AUC": metrics["AUC"],
        "TAR@FAR=0.1": metrics["TAR@FAR=0.1"],
        "TAR@FAR=0.01": metrics["TAR@FAR=0.01"],
        "TAR@FAR=0.001": metrics["TAR@FAR=0.001"],
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
