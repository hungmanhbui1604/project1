import numpy as np


def _binary_roc_curve(labels, scores, pos_label=1):
    """
    Fast NumPy replacement for sklearn.metrics.roc_curve.

    labels: array-like
        Ground-truth labels.
    scores: array-like
        Higher score means more likely positive.
    pos_label:
        Label treated as positive.

    Returns:
        fpr, tpr, thresholds

    This version is O(N log N) because it sorts once, instead of looping over
    every threshold and scanning the whole array each time.
    """
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=np.float64)

    if labels.shape[0] != scores.shape[0]:
        raise ValueError("labels and scores must have the same length")

    if labels.size == 0:
        raise ValueError("labels and scores must not be empty")

    y_true = (labels == pos_label).astype(np.int32)

    n_pos = int(y_true.sum())
    n_neg = int(y_true.size - n_pos)

    if n_pos == 0 or n_neg == 0:
        raise ValueError("ROC needs both positive and negative samples")

    # Sort scores descending.
    order = np.argsort(scores, kind="mergesort")[::-1]
    y_true = y_true[order]
    scores = scores[order]

    # Find the last index of each unique score group.
    distinct_value_indices = np.where(np.diff(scores))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    # Cumulative TP/FP at each threshold.
    tps = np.cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps

    thresholds = scores[threshold_idxs]

    # Add first point: threshold = +inf, no predicted positives.
    tps = np.r_[0, tps]
    fps = np.r_[0, fps]
    thresholds = np.r_[np.inf, thresholds]

    tpr = tps.astype(np.float64) / float(n_pos)
    fpr = fps.astype(np.float64) / float(n_neg)

    return fpr, tpr, thresholds


def _auc(x, y):
    """
    NumPy trapezoidal AUC.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    order = np.argsort(x, kind="mergesort")
    x = x[order]
    y = y[order]

    return float(np.trapz(y, x))


def compute_pad_metrics(probabilities: np.ndarray, labels: np.ndarray) -> dict:
    """
    PAD metrics.

    Assumption:
        label 1 = spoof / attack
        label 0 = live / bona fide
        higher probability = more likely spoof

    Returns:
        threshold, accuracy, ace, apcer, bpcer
    """
    probabilities = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels).reshape(-1)

    fpr, tpr, thresholds = _binary_roc_curve(
        labels=labels,
        scores=probabilities,
        pos_label=1,
    )

    # For spoof-positive ROC:
    # TPR = spoof correctly detected as spoof
    # FPR = live incorrectly detected as spoof
    apcer = 1.0 - tpr
    bpcer = fpr
    ace = 0.5 * (apcer + bpcer)

    idx = int(np.argmin(ace))
    threshold = thresholds[idx]

    # Avoid +inf as final decision threshold.
    if np.isinf(threshold):
        threshold = thresholds[1] if len(thresholds) > 1 else 0.5

    predictions = (probabilities >= threshold).astype(labels.dtype)
    accuracy = float(np.mean(predictions == labels))

    return {
        "threshold": float(threshold),
        "accuracy": accuracy,
        "ace": float(ace[idx]),
        "apcer": float(apcer[idx]),
        "bpcer": float(bpcer[idx]),
    }


def _interpolate_eer(fmr, fnmr, thresholds):
    """
    Compute EER where FMR and FNMR cross.
    """
    diff = fmr - fnmr

    crossing = np.where(diff >= 0)[0]

    if crossing.size == 0:
        idx = int(np.argmin(np.abs(diff)))
        eer = 0.5 * (fmr[idx] + fnmr[idx])
        eer_thr = thresholds[idx]
        return float(eer), float(eer_thr)

    idx1 = int(crossing[0])

    if idx1 == 0:
        eer = 0.5 * (fmr[0] + fnmr[0])
        eer_thr = thresholds[0]
        return float(eer), float(eer_thr)

    idx0 = idx1 - 1

    x0, x1 = fmr[idx0], fmr[idx1]
    y0, y1 = fnmr[idx0], fnmr[idx1]
    t0, t1 = thresholds[idx0], thresholds[idx1]

    den = (x1 - x0) - (y1 - y0)

    if abs(den) < 1e-12:
        eer = 0.5 * (x0 + y0)
        eer_thr = t0
        return float(eer), float(eer_thr)

    alpha = (y0 - x0) / den
    alpha = float(np.clip(alpha, 0.0, 1.0))

    eer = x0 + alpha * (x1 - x0)
    eer_thr = t0 + alpha * (t1 - t0)

    return float(eer), float(eer_thr)


def _interp_y_at_x(x, y, target_x):
    """
    Interpolate y at target_x for monotonic x.
    Used for TAR@FAR.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    # If exact FAR exists, use best TAR at that FAR.
    exact = x == target_x
    if np.any(exact):
        return float(np.max(y[exact]))

    idx = np.searchsorted(x, target_x, side="right")

    if idx <= 0:
        return float(y[0])

    if idx >= len(x):
        return float(y[-1])

    x0, x1 = x[idx - 1], x[idx]
    y0, y1 = y[idx - 1], y[idx]

    if abs(x1 - x0) < 1e-12:
        return float(max(y0, y1))

    alpha = (target_x - x0) / (x1 - x0)
    return float(y0 + alpha * (y1 - y0))


def compute_authentication_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
) -> dict:
    """
    Authentication metrics.

    labels:
        1 = genuine pair
        0 = impostor pair

    scores:
        Higher score means more likely genuine.
    """
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels).reshape(-1)

    fmr, tar, thresholds = _binary_roc_curve(
        labels=labels,
        scores=scores,
        pos_label=1,
    )

    fnmr = 1.0 - tar

    # Remove artificial +inf threshold point.
    if len(thresholds) > 1 and np.isinf(thresholds[0]):
        fmr = fmr[1:]
        tar = tar[1:]
        fnmr = fnmr[1:]
        thresholds = thresholds[1:]

    eer, eer_thr = _interpolate_eer(fmr, fnmr, thresholds)
    auc_roc = _auc(fmr, tar)

    return {
        "eer": float(eer),
        "eer_threshold": float(eer_thr),
        "auc": float(auc_roc),
        "thresholds": thresholds.tolist(),
        "fmr": fmr.tolist(),
        "tar": tar.tolist(),
        "fnmr": fnmr.tolist(),
        "tar_at_far_0.1": _interp_y_at_x(fmr, tar, 0.1),
        "tar_at_far_0.01": _interp_y_at_x(fmr, tar, 0.01),
        "tar_at_far_0.001": _interp_y_at_x(fmr, tar, 0.001),
    }


def compute_identification_metrics(
    sim_mat: np.ndarray,
    probe_labels: np.ndarray,
    gallery_labels: np.ndarray,
    top_k=(1, 5, 10),
) -> dict:
    """
    Fast Rank-k identification metrics.

    Instead of sorting the full gallery for every probe, this only extracts
    the top max(k) candidates using argpartition, then sorts that small subset.
    """
    sim_mat = np.asarray(sim_mat, dtype=np.float32)
    probe_labels = np.asarray(probe_labels)
    gallery_labels = np.asarray(gallery_labels)

    if sim_mat.ndim != 2:
        raise ValueError("sim_mat must be a 2D matrix")

    if sim_mat.shape[0] != probe_labels.shape[0]:
        raise ValueError("sim_mat rows must match number of probe labels")

    if sim_mat.shape[1] != gallery_labels.shape[0]:
        raise ValueError("sim_mat columns must match number of gallery labels")

    n_gallery = sim_mat.shape[1]
    max_k = min(max(top_k), n_gallery)

    # Get top max_k indices without full sorting.
    top_indices_unsorted = np.argpartition(
        -sim_mat,
        kth=max_k - 1,
        axis=1,
    )[:, :max_k]

    top_scores = np.take_along_axis(sim_mat, top_indices_unsorted, axis=1)

    # Sort only the selected top max_k candidates.
    order = np.argsort(-top_scores, axis=1)
    top_indices = np.take_along_axis(top_indices_unsorted, order, axis=1)

    pred_labels = gallery_labels[top_indices]
    matches = pred_labels == probe_labels[:, None]

    metrics = {}
    for k in top_k:
        k_eff = min(k, n_gallery)
        metrics[f"rank_{k}"] = float(np.mean(np.any(matches[:, :k_eff], axis=1)))

    return metrics
