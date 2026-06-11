import argparse
import json
import os

import numpy as np
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import PADDataset
from metrics import compute_pad_metrics
from transforms import get_transforms
from trt_runner import TensorRTRunner


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def evaluate_pad_trt(engine, cfg, pad_split_path):
    eval_cfg = cfg["evaluation"]
    data_cfg = cfg["data"]

    _, eval_transform, _ = get_transforms(data_cfg["transform_name"])

    dataset = PADDataset(
        split_path=pad_split_path,
        split="test",
        transform=eval_transform,
    )

    loader = DataLoader(
        dataset,
        batch_size=eval_cfg["pad_batch_size"],
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    all_probs = []
    all_labels = []

    for images, labels in tqdm(loader, desc="TensorRT PAD Inference", unit="batch"):
        images_np = images.numpy().astype(np.float32)

        # branch_b_out is PAD / spoof logit output
        _, branch_b_out = engine.infer(images_np)

        logits = branch_b_out.reshape(-1)
        probs = sigmoid(logits)

        all_probs.append(probs)
        all_labels.append(labels.numpy())

    probs = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)

    metrics = compute_pad_metrics(probs, labels)

    n_live = sum(1 for _, label in dataset.samples if label == 0)
    n_spoof = len(dataset) - n_live

    summary = {
        "split_path": pad_split_path,
        "split": "test",
        "n_samples": len(dataset),
        "n_live": int(n_live),
        "n_spoof": int(n_spoof),
        "threshold": float(metrics["threshold"]),
        "accuracy": float(metrics["accuracy"]),
        "ace": float(metrics["ace"]),
        "apcer": float(metrics["apcer"]),
        "bpcer": float(metrics["bpcer"]),
    }

    return summary


def run_pad_evaluation(
    engine_path="dmv_fp16.engine",
    config_path="config.yaml",
    pad_split_path=None,
    output_dir="results/dmv_trt_pad"
):
    cfg = load_config(config_path)

    if pad_split_path is None:
        pad_split_path = cfg["data"]["pad_split_path"]

    os.makedirs(output_dir, exist_ok=True)

    engine = TensorRTRunner(engine_path)

    pad_summary = evaluate_pad_trt(
        engine=engine,
        cfg=cfg,
        pad_split_path=pad_split_path,
    )

    summary = {
        "engine_path": engine_path,
        "config_path": config_path,
        "pad": pad_summary,
    }

    json_path = os.path.join(output_dir, "pad_trt_metrics.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print("PAD TensorRT Evaluation")
    print("=" * 60)
    print(f"Samples   : {pad_summary['n_samples']:,}")
    print(f"Live/Spoof: {pad_summary['n_live']:,} / {pad_summary['n_spoof']:,}")
    print(f"Threshold : {pad_summary['threshold']:.4f}")
    print(f"Accuracy  : {pad_summary['accuracy']:.2%}")
    print(f"ACE       : {pad_summary['ace']:.2%}")
    print(f"APCER     : {pad_summary['apcer']:.2%}")
    print(f"BPCER     : {pad_summary['bpcer']:.2%}")
    print("\nSaved:", json_path)
    print("=" * 60)

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine-path", default="dmv_fp16.engine")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--pad-split-path", default=None)
    parser.add_argument("--output-dir", default="results/dmv_trt_pad")
    args = parser.parse_args()

    run_pad_evaluation(
        engine_path=args.engine_path,
        config_path=args.config,
        pad_split_path=args.pad_split_path,
        output_dir=args.output_dir
    )
