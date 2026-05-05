import argparse
import json
import os

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import PADDataset
from metrics import compute_pad_metrics
from models import get_model
from transforms import get_transforms

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


@torch.no_grad()
def collect_probs(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()

    all_probs, all_labels = [], []

    for images, labels in tqdm(loader, desc="Inference", unit="batch"):
        images = images.to(device, non_blocking=True)

        with torch.autocast(device_type="cuda"):
            logits = model.branch_forward(images, branch="b")

        probs = torch.sigmoid(logits)

        all_probs.append(probs.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    return np.concatenate(all_probs), np.concatenate(all_labels)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    eval_cfg = cfg["evaluation"]
    data_cfg = cfg["data"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Checkpoint path ──────────────────────────────────────────────────
    if not os.path.isfile(args.checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint_path}")

    # ── Transforms ───────────────────────────────────────────────────────
    _, eval_transform, _ = get_transforms(data_cfg["transform_name"])

    # ── Dataset ──────────────────────────────────────────────────────────
    dataset = PADDataset(
        split_path=args.split_path,
        split="test",
        transform=eval_transform,
    )

    loader = DataLoader(
        dataset,
        batch_size=eval_cfg["pad_batch_size"],
        shuffle=False,
        num_workers=train_cfg["num_workers"],
        pin_memory=train_cfg["pin_memory"],
    )

    # ── Model ────────────────────────────────────────────────────────────
    model = get_model(model_cfg["model_name"], model_cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[model] {model_cfg['model_name']} ({n_params:.2f}M params)")

    print(f"Loading checkpoint: {args.checkpoint_path}")
    ckpt = torch.load(args.checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])

    # ── Evaluate ─────────────────────────────────────────────────────────
    print("\nEvaluating on PAD test set...")
    probs, labels = collect_probs(model, loader, device)
    metrics = compute_pad_metrics(probs, labels)

    # ── Report ───────────────────────────────────────────────────────────
    n_live = sum(1 for (_, label) in dataset.samples if label == 0)
    n_spoof = len(dataset) - n_live
    print("=" * 50)
    print(f"Split path: {args.split_path}")
    print("-" * 50)
    print(f"Total samples: {len(dataset)} (live={n_live}, spoof={n_spoof})")
    print("-" * 50)
    print(f"Threshold : {metrics['threshold']:.4f}")
    print(f"Accuracy  : {metrics['accuracy']:.2%}")
    print(f"ACE       : {metrics['ace']:.2%}")
    print(f"APCER     : {metrics['apcer']:.2%}")
    print(f"BPCER     : {metrics['bpcer']:.2%}")
    print("=" * 50)

    # ── Save JSON ────────────────────────────────────────────────────────
    summary = {
        "split_path": args.split_path,
        "n_samples": len(dataset),
        "n_live": n_live,
        "n_spoof": n_spoof,
        "threshold": metrics["threshold"],
        "accuracy": metrics["accuracy"],
        "ace": metrics["ace"],
        "apcer": metrics["apcer"],
        "bpcer": metrics["bpcer"],
    }

    json_path = os.path.join(args.output_dir, "pad_metrics.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved → {json_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PAD Evaluation")
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

    main(args)
