import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import PADDataset
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


def plot_confusion_matrix(
    metrics: dict,
    output_dir: str,
    title: str = "Confusion Matrix",
) -> str:
    _style()
    cm = np.array(
        [[metrics["TN"], metrics["FP"]], [metrics["FN"], metrics["TP"]]],
        dtype=int,
    )

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred: Live", "Pred: Spoof"], fontsize=11)
    ax.set_yticklabels(["True: Live", "True: Spoof"], fontsize=11)
    ax.set_title(title, pad=14)

    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(1)
    thresh = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                f"{cm[i, j]:,}\n({cm_norm[i, j]:.1%})",
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=12,
            )

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()

    path = os.path.join(output_dir, "pad_confusion_matrix.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] confusion matrix saved → {path}")
    return path


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


@torch.no_grad()
def collect_preds(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    is_dual: bool,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()

    all_preds, all_labels = [], []

    for images, labels in tqdm(loader, desc="Inference", unit="batch"):
        images = images.to(device, non_blocking=True)

        with torch.autocast(device_type="cuda"):
            outputs = model(images)
            if is_dual:
                logits = outputs[1]
            else:
                logits = outputs

        preds = logits.argmax(dim=1).cpu().numpy()

        all_preds.append(preds)
        all_labels.append(labels.cpu().numpy())

    return np.concatenate(all_preds), np.concatenate(all_labels)


def compute_pad_metrics(preds: np.ndarray, labels: np.ndarray) -> dict:
    n_total = len(labels)
    n_live = int((labels == 0).sum())
    n_spoof = int((labels == 1).sum())

    # True Positives / True Negatives / False Positives / False Negatives
    # Positive class = spoof (1)
    tp = int(((preds == 1) & (labels == 1)).sum())  # spoof correctly classified
    tn = int(((preds == 0) & (labels == 0)).sum())  # live  correctly classified
    fp = int(((preds == 1) & (labels == 0)).sum())  # live misclassified as spoof
    fn = int(((preds == 0) & (labels == 1)).sum())  # spoof misclassified as live

    accuracy = (tp + tn) / n_total if n_total > 0 else 0.0
    apcer = fn / n_spoof if n_spoof > 0 else 0.0  # spoof→live errors
    bpcer = fp / n_live if n_live > 0 else 0.0  # live→spoof errors
    ace = (apcer + bpcer) / 2.0

    return {
        "n_total": n_total,
        "n_live": n_live,
        "n_spoof": n_spoof,
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "accuracy": float(accuracy),
        "APCER": float(apcer),
        "BPCER": float(bpcer),
        "ACE": float(ace),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="SwinT PAD Evaluation")
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
        default="results/pad/",
        help="Directory for metrics JSON and plot PNGs",
    )
    parser.add_argument(
        "--checkpoint-path",
        required=True,
        help="Path to the checkpoint",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_cfg = cfg["model"]
    training_cfg = cfg["training"]
    evaluation_cfg = cfg["evaluation"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Checkpoint path ──────────────────────────────────────────────────
    if not os.path.isfile(args.checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint_path}")

    # ── Transforms ───────────────────────────────────────────────────────
    _, eval_transform = get_transforms("all")

    # ── Dataset ──────────────────────────────────────────────────────────
    dataset = PADDataset(
        split_path=args.split_path,
        split="test",
        transform=eval_transform,
    )

    loader = DataLoader(
        dataset,
        batch_size=evaluation_cfg["pad_batch_size"],
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
        print("[model] DualSwinTransformerTiny")
    else:
        model = SwinTransformerTiny(
            embed_dim=model_cfg["embed_dim"],
        ).to(device)
        is_dual = False
        print("[model] SwinTransformerTiny")

    print(f"Loading checkpoint: {args.checkpoint_path}")
    ckpt = torch.load(args.checkpoint_path, map_location="cpu")
    model.load_state_dict(ckpt["model"])

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"({n_params:.2f}M params)")

    # ── Evaluate ─────────────────────────────────────────────────────────
    print("\nEvaluating on PAD test set...")
    preds, labels = collect_preds(model, loader, device, is_dual)
    metrics = compute_pad_metrics(preds, labels)

    # ── Report ───────────────────────────────────────────────────────────
    print("=" * 50)
    print(f"Split path: {args.split_path}")
    print(
        f"Samples: {metrics['n_total']:,} (live={metrics['n_live']:,}, spoof={metrics['n_spoof']:,})"
    )
    print("-" * 50)
    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"APCER    : {metrics['APCER']:.4f}  (argmax threshold)")
    print(f"BPCER    : {metrics['BPCER']:.4f}  (argmax threshold)")
    print(f"ACE      : {metrics['ACE']:.4f}  (argmax threshold)")
    print("-" * 50)
    print("=" * 50)

    # ── Save JSON ────────────────────────────────────────────────────────
    results = {
        "checkpoint": args.checkpoint_path,
        "split_path": args.split_path,
        "metrics": metrics,
    }

    json_path = os.path.join(args.output_dir, "pad_metrics.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → {json_path}")

    # ── Plots ────────────────────────────────────────────────────────────
    _style()
    plot_confusion_matrix(metrics, args.output_dir)

    print(f"\nAll outputs written to: {os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    main()
