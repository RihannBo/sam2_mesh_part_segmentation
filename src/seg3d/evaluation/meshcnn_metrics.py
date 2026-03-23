"""
MeshCNN-style evaluation metrics for mesh segmentation.

Why Hungarian matching is required:
-----------------------------------
In mesh (and semantic) segmentation, predicted and ground-truth segment IDs are
arbitrary: the model might assign label 0 to "leg" while the GT uses 3 for "leg".
Evaluating by direct label comparison would wrongly penalize correct segmentations
that use different ID spaces. The Hungarian algorithm finds a one-to-one mapping
from predicted segment IDs to ground-truth IDs that maximizes the total overlap
(count of correctly assigned faces). This yields a fair evaluation that is
invariant to label permutation and allows different numbers of segments (with
unmatched segments counting as errors).
"""

import numpy as np
from scipy.optimize import linear_sum_assignment


# Sentinel value for predicted segments that could not be matched to any GT segment.
# Chosen so it never equals any ground-truth label after alignment.
UNMATCHED_SENTINEL = -1


def align_labels_hungarian(pred_labels: np.ndarray, gt_labels: np.ndarray) -> np.ndarray:
    """
    Align predicted segment labels to ground-truth labels via maximum-overlap
    matching (Hungarian algorithm).

    Predicted and GT label IDs are arbitrary and unordered. This function finds
    an optimal one-to-one assignment from predicted segment IDs to GT segment IDs
    that maximizes the total number of matching faces. Unmatched predicted
    segments are assigned a sentinel value so they count as incorrect in
    downstream metrics.

    Parameters
    ----------
    pred_labels : np.ndarray
        Integer array of shape (num_faces,) with predicted per-face labels.
    gt_labels : np.ndarray
        Integer array of shape (num_faces,) with ground-truth per-face labels.

    Returns
    -------
    pred_labels_aligned : np.ndarray
        Integer array of shape (num_faces,) with predicted labels remapped to
        GT label space. Unmatched predictions are set to UNMATCHED_SENTINEL.
    """
    pred_labels = np.asarray(pred_labels, dtype=np.int64)
    gt_labels = np.asarray(gt_labels, dtype=np.int64)

    if pred_labels.shape != gt_labels.shape:
        raise ValueError("pred_labels and gt_labels must have the same shape")

    n_faces = pred_labels.size
    if n_faces == 0:
        return np.array([], dtype=np.int64)

    pred_unique = np.unique(pred_labels)
    gt_unique = np.unique(gt_labels)

    # Overlap matrix: overlap[i, j] = number of faces with pred_unique[i] and gt_unique[j].
    # Vectorized via index mapping and bincount (O(n_faces)).
    pred_idx = np.searchsorted(pred_unique, pred_labels)
    gt_idx = np.searchsorted(gt_unique, gt_labels)
    n_p, n_g = len(pred_unique), len(gt_unique)
    flat_idx = np.ravel_multi_index((pred_idx, gt_idx), (n_p, n_g))
    overlap = np.bincount(flat_idx, minlength=n_p * n_g).reshape(n_p, n_g)

    # Hungarian algorithm minimizes cost; we maximize overlap so use cost = -overlap.
    # Pad to square if needed (unmatched rows/cols get zero overlap).
    n_p, n_g = overlap.shape
    if n_p <= n_g:
        cost = -overlap
        row_ind, col_ind = linear_sum_assignment(cost)
        # row_ind[i] -> col_ind[i] maps pred_unique index to gt_unique index
        pred_to_gt_idx = {row_ind[k]: col_ind[k] for k in range(len(row_ind))}
    else:
        cost = -overlap
        row_ind, col_ind = linear_sum_assignment(cost)
        pred_to_gt_idx = {row_ind[k]: col_ind[k] for k in range(len(row_ind))}

    # Map each predicted unique label to its assigned GT label or UNMATCHED.
    pred_id_to_aligned = {}
    for i, p in enumerate(pred_unique):
        if i in pred_to_gt_idx:
            j = pred_to_gt_idx[i]
            pred_id_to_aligned[p] = gt_unique[j]
        else:
            pred_id_to_aligned[p] = UNMATCHED_SENTINEL

    # Build aligned label array without modifying the input.
    pred_aligned = np.empty_like(pred_labels)
    for i, p in enumerate(pred_labels):
        pred_aligned[i] = pred_id_to_aligned[p]

    return pred_aligned


def compute_accuracy_and_miou(
    pred_labels_aligned: np.ndarray, gt_labels: np.ndarray
) -> tuple[float, float]:
    """
    Compute face-wise accuracy and mean Intersection over Union (mIoU) given
    aligned predicted labels and ground-truth labels.

    Assumes pred_labels_aligned has already been aligned to GT label space (e.g.
    via align_labels_hungarian). Unmatched predictions (UNMATCHED_SENTINEL) are
    treated as incorrect.

    Parameters
    ----------
    pred_labels_aligned : np.ndarray
        Integer array of shape (num_faces,) with predicted labels in GT space.
    gt_labels : np.ndarray
        Integer array of shape (num_faces,) with ground-truth labels.

    Returns
    -------
    accuracy : float
        Fraction of faces where pred_labels_aligned == gt_labels (in [0, 1]).
    miou : float
        Mean IoU over all ground-truth segments (in [0, 1]). Empty GT yields 0.0.
    """
    pred_labels_aligned = np.asarray(pred_labels_aligned, dtype=np.int64)
    gt_labels = np.asarray(gt_labels, dtype=np.int64)

    if pred_labels_aligned.shape != gt_labels.shape:
        raise ValueError("pred_labels_aligned and gt_labels must have the same shape")

    n_faces = gt_labels.size
    if n_faces == 0:
        return 0.0, 0.0

    # Face-wise accuracy: correct when aligned prediction matches GT.
    accuracy = np.mean(pred_labels_aligned == gt_labels)

    # Mean IoU: for each GT segment, IoU = |intersection| / |union|.
    gt_unique = np.unique(gt_labels)
    ious = []
    for g in gt_unique:
        pred_g = pred_labels_aligned == g
        gt_g = gt_labels == g
        intersection = np.sum(pred_g & gt_g)
        union = np.sum(pred_g | gt_g)
        if union == 0:
            ious.append(0.0)
        else:
            ious.append(intersection / union)

    miou = np.mean(ious) if ious else 0.0
    return float(accuracy), float(miou)


def evaluate_meshcnn_metrics(
    pred_labels: np.ndarray, gt_labels: np.ndarray
) -> dict[str, float]:
    """
    Evaluate mesh segmentation with MeshCNN-style metrics: face-wise accuracy
    and mean IoU, after optimally aligning predicted labels to ground-truth
    via the Hungarian algorithm.

    Parameters
    ----------
    pred_labels : np.ndarray
        Integer array of shape (num_faces,) with predicted per-face labels.
    gt_labels : np.ndarray
        Integer array of shape (num_faces,) with ground-truth per-face labels.

    Returns
    -------
    dict with keys:
        - "accuracy": float in [0, 1]
        - "miou": float in [0, 1]
    """
    pred_aligned = align_labels_hungarian(pred_labels, gt_labels)
    accuracy, miou = compute_accuracy_and_miou(pred_aligned, gt_labels)
    return {"accuracy": accuracy, "miou": miou}
