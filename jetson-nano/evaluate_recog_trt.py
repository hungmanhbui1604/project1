import argparse
import json
import os

import numpy as np
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import (
    AuthenticationEvaluationDataset,
    IdentificationEvaluationDataset,
    UniqueFingerprintDataset,
)
from metrics import compute_authentication_metrics, compute_identification_metrics
from transforms import get_transforms
from trt_runner import TensorRTRunner


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def l2_normalize_np(x, axis=1, eps=1e-12):
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(norm, eps)


def collect_authentication_scores_trt(engine, authentication_loader, unique_loader, embed_dim):
    n_unique_images = len(unique_loader.dataset)
    embeddings = np.zeros((n_unique_images, embed_dim), dtype=np.float32)

    for idxs, imgs in tqdm(unique_loader, desc="TensorRT Extracting Embeddings", unit="batch"):
        imgs_np = imgs.numpy().astype(np.float32)

        # branch_a_out is recognition embedding output
        branch_a_out, _ = engine.infer(imgs_np)

        embs = branch_a_out.reshape(imgs_np.shape[0], -1)
        embs = l2_normalize_np(embs)

        embeddings[idxs.numpy()] = embs

    all_scores = []
    all_labels = []

    for idx_a, idx_b, labels in tqdm(
        authentication_loader,
        desc="TensorRT Authentication",
        unit="batch",
    ):
        emb_a = embeddings[idx_a.numpy()]
        emb_b = embeddings[idx_b.numpy()]
        cos_sim = np.sum(emb_a * emb_b, axis=1)

        all_scores.append(cos_sim)
        all_labels.append(labels.numpy())

    return np.concatenate(all_scores), np.concatenate(all_labels)


def collect_identification_scores_trt(engine, loader, dataset):
    all_embs = []
    all_labels = []
    all_indices = []

    for imgs, labels, idx in tqdm(loader, desc="TensorRT Identification", unit="batch"):
        imgs_np = imgs.numpy().astype(np.float32)

        # branch_a_out is recognition embedding output
        branch_a_out, _ = engine.infer(imgs_np)

        embs = branch_a_out.reshape(imgs_np.shape[0], -1)
        embs = l2_normalize_np(embs)

        all_embs.append(embs)
        all_labels.extend(labels.numpy())
        all_indices.extend(idx.numpy())

    all_embs = np.concatenate(all_embs, axis=0)
    all_labels = np.array(all_labels)
    all_indices = np.array(all_indices)

    sort_order = np.argsort(all_indices)
    all_embs = all_embs[sort_order]
    all_labels = all_labels[sort_order]

    n_gallery = dataset.n_gallery

    gallery_embs = all_embs[:n_gallery]
    gallery_labels = all_labels[:n_gallery]

    probe_embs = all_embs[n_gallery:]
    probe_labels = all_labels[n_gallery:]

    sim_mat = np.dot(probe_embs, gallery_embs.T)

    return sim_mat, probe_labels, gallery_labels


def evaluate_recognition_trt(engine, cfg, recog_split_path):
    general_cfg = cfg["general"]
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    eval_cfg = cfg["evaluation"]

    _, eval_transform, _ = get_transforms(data_cfg["transform_name"])

    authentication_dataset = AuthenticationEvaluationDataset(
        split_path=recog_split_path,
        split="test",
        n_genuine_impressions=data_cfg["n_genuine_impressions"],
        n_impostor_impressions=data_cfg["n_impostor_impressions"],
        impostor_mode=data_cfg["impostor_mode"],
        n_impostor_subset=data_cfg["n_impostor_subset"],
        seed=general_cfg["seed"],
    )

    unique_dataset = UniqueFingerprintDataset(
        idx_to_path=authentication_dataset.idx_to_path,
        transform=eval_transform,
    )

    identification_dataset = IdentificationEvaluationDataset(
        split_path=recog_split_path,
        split="test",
        gallery_per_id=data_cfg["gallery_per_id"],
        probe_per_id=data_cfg["probe_per_id"],
        transform=eval_transform,
        seed=general_cfg["seed"],
    )

    authentication_loader = DataLoader(
        authentication_dataset,
        batch_size=eval_cfg["auth_batch_size"],
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    unique_loader = DataLoader(
        unique_dataset,
        batch_size=eval_cfg["recog_batch_size"],
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    identification_loader = DataLoader(
        identification_dataset,
        batch_size=eval_cfg["recog_batch_size"],
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    embed_dim = model_cfg["branch_a_num_classes"]

    scores, labels = collect_authentication_scores_trt(
        engine=engine,
        authentication_loader=authentication_loader,
        unique_loader=unique_loader,
        embed_dim=embed_dim,
    )

    authentication_metrics = compute_authentication_metrics(scores, labels)

    sim_mat, probe_labels, gallery_labels = collect_identification_scores_trt(
        engine=engine,
        loader=identification_loader,
        dataset=identification_dataset,
    )

    identification_metrics = compute_identification_metrics(
        sim_mat,
        probe_labels,
        gallery_labels,
    )

    summary = {
        "split_path": recog_split_path,
        "split": "test",
        "authentication": {
            "n_pairs": len(authentication_dataset),
            "n_genuine": authentication_dataset.n_genuine,
            "n_impostor": authentication_dataset.n_impostor,
            "eer": float(authentication_metrics["eer"]),
            "eer_threshold": float(authentication_metrics["eer_threshold"]),
            "auc": float(authentication_metrics["auc"]),
            "tar_at_far_0.1": float(authentication_metrics["tar_at_far_0.1"]),
            "tar_at_far_0.01": float(authentication_metrics["tar_at_far_0.01"]),
            "tar_at_far_0.001": float(authentication_metrics["tar_at_far_0.001"]),
        },
        "identification": {
            "n_ids": identification_dataset.n_ids,
            "n_gallery": identification_dataset.n_gallery,
            "n_probes": identification_dataset.n_probes,
            "rank_1": float(identification_metrics["rank_1"]),
            "rank_5": float(identification_metrics["rank_5"]),
            "rank_10": float(identification_metrics["rank_10"]),
        },
    }

    return summary


def run_recognition_evaluation(
    engine_path="dmv_fp16.engine",
    config_path="config.yaml",
    recog_split_path=None,
    output_dir="results/dmv_trt_recog",
):
    cfg = load_config(config_path)

    if recog_split_path is None:
        recog_split_path = cfg["data"]["recog_split_path"]

    os.makedirs(output_dir, exist_ok=True)

    engine = TensorRTRunner(engine_path)

    recog_summary = evaluate_recognition_trt(
        engine=engine,
        cfg=cfg,
        recog_split_path=recog_split_path
    )

    summary = {
        "engine_path": engine_path,
        "config_path": config_path,
        "recognition": recog_summary,
    }

    json_path = os.path.join(output_dir, "recog_trt_metrics.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    auth = recog_summary["authentication"]
    ident = recog_summary["identification"]

    print("\n" + "=" * 60)
    print("Recognition TensorRT Evaluation")
    print("=" * 60)

    print("\nAuthentication:")
    print(f"Pairs        : {auth['n_pairs']:,}")
    print(f"Genuine/Imp. : {auth['n_genuine']:,} / {auth['n_impostor']:,}")
    print(f"EER          : {auth['eer']:.2%}")
    print(f"EER threshold: {auth['eer_threshold']:.4f}")
    print(f"AUC          : {auth['auc']:.2%}")
    print(f"TAR@FAR=0.1  : {auth['tar_at_far_0.1']:.2%}")
    print(f"TAR@FAR=0.01 : {auth['tar_at_far_0.01']:.2%}")
    print(f"TAR@FAR=0.001: {auth['tar_at_far_0.001']:.2%}")

    print("\nIdentification:")
    print(f"Identities: {ident['n_ids']:,}")
    print(f"Gallery   : {ident['n_gallery']:,}")
    print(f"Probes    : {ident['n_probes']:,}")
    print(f"Rank-1    : {ident['rank_1']:.2%}")
    print(f"Rank-5    : {ident['rank_5']:.2%}")
    print(f"Rank-10   : {ident['rank_10']:.2%}")

    print("\nSaved:", json_path)
    print("=" * 60)

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine-path", default="dmv_fp16.engine")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--recog-split-path", default=None)
    parser.add_argument("--output-dir", default="results/dmv_trt_recog")
    args = parser.parse_args()

    run_recognition_evaluation(
        engine_path=args.engine_path,
        config_path=args.config,
        recog_split_path=args.recog_split_path,
        output_dir=args.output_dir
    )
