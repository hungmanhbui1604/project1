import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import auc, roc_curve
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import RecogEvaluationDataset, UniqueImageDataset
from model import DualSwinTransformerTiny, SwinTransformerTiny
from transforms import get_transforms

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _style():
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 11,
        }
    )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def compute_recog_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
) -> dict:
    # fmr, fnmr, tar
    fmr, tar, thrs = roc_curve(labels, scores, pos_label=1)
    fnmr = 1.0 - tar

    asc_idx = np.argsort(thrs)
    thrs = thrs[asc_idx]
    fmr = fmr[asc_idx]
    tar = tar[asc_idx]
    fnmr = fnmr[asc_idx]

    # EER
    diff = np.abs(fmr - fnmr)
    eer_idx = int(np.argmin(diff))
    eer = float((fmr[eer_idx] + fnmr[eer_idx]) / 2.0)
    eer_thr = float(thrs[eer_idx])

    # AUC
    auc_roc = float(auc(fmr, tar))

    # TAR@FAR
    tar_at_far = {}
    for far_target in (0.1, 0.01, 0.001):
        mask = fmr <= far_target
        tar_at_far[far_target] = float(tar[mask].max()) if mask.any() else 0.0

    return {
        "thresholds": thrs.tolist(),
        "FMR": fmr.tolist(),
        "FNMR": fnmr.tolist(),
        "TAR": tar.tolist(),
        "EER": eer,
        "EER_threshold": eer_thr,
        "AUC": auc_roc,
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
    is_dual: bool,
    embed_dim: int,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()

    n_unique_images = len(unique_loader.dataset)

    embeddings = torch.zeros((n_unique_images, embed_dim), device=device)
    for idxs, imgs in tqdm(unique_loader, desc="Extracting embeddings", unit="batch"):
        imgs = imgs.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda"):
            outputs = model(imgs)
            if is_dual:
                emb = outputs[0]
            else:
                emb = outputs
        emb = F.normalize(emb, dim=1).float()
        embeddings[idxs] = emb

    all_scores, all_labels = [], []
    for idx_a, idx_b, labels in tqdm(loader, desc="Inference", unit="batch"):
        emb_a = embeddings[idx_a]
        emb_b = embeddings[idx_b]

        cos_sim = (emb_a * emb_b).sum(dim=1).cpu().numpy()

        all_scores.append(cos_sim)
        all_labels.append(labels.numpy())

    return np.concatenate(all_scores), np.concatenate(all_labels)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def plot_roc(metrics: dict, output_dir: str) -> str:
    fmr = np.array(metrics["FMR"])
    tar = np.array(metrics["TAR"])
    eer = metrics["EER"]
    auc_val = metrics["AUC"]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fmr, tar, lw=2, color="#2563EB", label=f"ROC  (AUC={auc_val:.4f})")
    ax.axvline(eer, color="grey", ls="--", lw=1, label=f"EER={eer:.4f}")

    # EER operating point on the curve
    ax.scatter(
        [eer],
        [1 - eer],
        zorder=6,
        s=80,
        color="grey",
        marker="x",
        label=f"EER point ({eer:.4f}, {1 - eer:.4f})",
    )

    # TAR@FAR operating points
    colors = ["#DC2626", "#D97706", "#16A34A"]
    for far_t, color in zip([0.1, 0.01, 0.001], colors):
        key = f"TAR@FAR={far_t}"
        tar_val = metrics[key]
        ax.scatter(
            [far_t],
            [tar_val],
            zorder=5,
            s=60,
            color=color,
            label=f"TAR@FAR={far_t} = {tar_val:.4f}",
        )

    ax.set_xlabel("FMR  (False Match Rate)")
    ax.set_ylabel("TAR  (True Accept Rate)")
    ax.set_title("ROC Curve — Fingerprint Recognition")
    ax.legend(fontsize=9)
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.grid(alpha=0.3)

    path = os.path.join(output_dir, "roc_curve.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] ROC curve saved → {path}")
    return path


def plot_det(metrics: dict, output_dir: str) -> str:
    """DET curve: FNMR vs FMR on a log-log scale (ISO/IEC 19795-1 style)."""
    fmr = np.array(metrics["FMR"])
    fnmr = np.array(metrics["FNMR"])
    eer = metrics["EER"]

    # clip to avoid log(0)
    fmr_plot = np.clip(fmr, 1e-5, 1.0)
    fnmr_plot = np.clip(fnmr, 1e-5, 1.0)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fmr_plot, fnmr_plot, lw=2, color="#7C3AED", label="DET")
    ax.plot(
        [eer],
        [eer],
        "o",
        color="#DC2626",
        zorder=6,
        label=f"EER = {eer:.4f}",
        markersize=8,
    )
    ax.plot([1e-5, 1.0], [1e-5, 1.0], "--", color="grey", lw=1, alpha=0.6)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1e-5, 1.0)
    ax.set_ylim(1e-5, 1.0)
    ax.set_xlabel("FMR  (False Match Rate)")
    ax.set_ylabel("FNMR  (False Non-Match Rate)")
    ax.set_title("DET Curve — Fingerprint Recognition")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, which="both")

    path = os.path.join(output_dir, "det_curve.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] DET curve saved → {path}")
    return path


def plot_score_dist(
    scores: np.ndarray,
    labels: np.ndarray,
    eer_thr: float,
    output_dir: str,
) -> str:
    genuine_scores = scores[labels == 1]
    impostor_scores = scores[labels == 0]

    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(scores.min(), scores.max(), 80)
    ax.hist(
        genuine_scores,
        bins=bins,
        alpha=0.6,
        color="#16A34A",
        label=f"Genuine  (n={len(genuine_scores):,})",
        density=True,
    )
    ax.hist(
        impostor_scores,
        bins=bins,
        alpha=0.6,
        color="#DC2626",
        label=f"Impostor (n={len(impostor_scores):,})",
        density=True,
    )
    ax.axvline(
        eer_thr, color="black", ls="--", lw=1.5, label=f"EER threshold = {eer_thr:.4f}"
    )

    ax.set_xlabel("Cosine Similarity")
    ax.set_ylabel("Density")
    ax.set_title("Score Distributions — Fingerprint Recognition")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    path = os.path.join(output_dir, "score_distributions.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Score distributions saved → {path}")
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="SwinT Recognition Evaluation")
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

    cfg = load_config(args.config)
    general_cfg = cfg["general"]
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    training_cfg = cfg["training"]
    evaluation_cfg = cfg["evaluation"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Checkpoint path ──────────────────────────────────────────────────
    ckpt_path = args.checkpoint_path

    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # ── Transforms ───────────────────────────────────────────────────────
    _, eval_transform = get_transforms("all")

    # ── Datasets ─────────────────────────────────────────────────────────
    dataset = RecogEvaluationDataset(
        split_path=args.split_path,
        split="test",
        n_genuine_impressions=data_cfg["n_genuine_impressions"],
        n_impostor_impressions=data_cfg["n_impostor_impressions"],
        impostor_mode=data_cfg["impostor_mode"],
        n_impostor_subset=None
        if data_cfg["n_impostor_subset"] in ("None", None, "null")
        else data_cfg["n_impostor_subset"],
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
        batch_size=evaluation_cfg["recog_batch_size"],
        shuffle=False,
        num_workers=training_cfg["num_workers"],
        pin_memory=training_cfg["pin_memory"],
    )

    unique_loader = DataLoader(
        unique_dataset,
        batch_size=training_cfg["recog_batch_size"],
        shuffle=False,
        num_workers=training_cfg["num_workers"],
        pin_memory=training_cfg["pin_memory"],
    )

    # ── Model ────────────────────────────────────────────────────────────
    if "embed_dim_a" in model_cfg:
        model = DualSwinTransformerTiny(
            embed_dim_a=model_cfg["embed_dim_a"],
            embed_dim_b=model_cfg["embed_dim_b"],
        ).to(device)
        is_dual = True
        embed_dim = model_cfg["embed_dim_a"]
        print("[model] DualSwinTransformerTiny")
    else:
        model = SwinTransformerTiny(
            embed_dim=model_cfg["embed_dim"],
        ).to(device)
        is_dual = False
        embed_dim = model_cfg["embed_dim"]
        print("[model] SwinTransformerTiny")

    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model"])

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"({n_params:.2f}M params)")

    # ── Evaluate ─────────────────────────────────────────────────────────
    print("\nEvaluating on recognition test set...")

    scores, labels = collect_scores(
        model, loader, unique_loader, device, is_dual, embed_dim
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
        "metrics": metrics,
    }
    json_path = os.path.join(args.output_dir, "recog_metrics.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved → {json_path}")

    # ── Plots ────────────────────────────────────────────────────────────────
    _style()
    plot_roc(metrics, args.output_dir)
    plot_det(metrics, args.output_dir)
    plot_score_dist(scores, labels, metrics["EER_threshold"], args.output_dir)

    print(f"\nAll outputs written to: {os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    main()
