from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn.functional as F


def evaluate_predictions_fast(
    pred_masks_list: List[torch.Tensor | None],
    gt_instance_masks: torch.Tensor,
    threshold: float = 0.0,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, Any]:
    """
    Fast vectorized IoU/Dice evaluation on GPU.

    Args:
        pred_masks_list: List of predicted masks per view, each is (P, H, W) tensor or None
        gt_instance_masks: Ground truth (T, H, W) tensor with instance IDs
        threshold: Threshold for binarizing predictions (logits > threshold)
        device: Device to use for computation

    Returns:
        dict with mean/std IoU & Dice and per-view values.
    """
    T, H, W = gt_instance_masks.shape

    # Get unique GT instance IDs (excluding background)
    gt_ids = torch.unique(gt_instance_masks)
    gt_ids = gt_ids[gt_ids != 0]
    G = len(gt_ids)

    if G == 0:
        return {
            "mean_iou": 0.0,
            "mean_dice": 0.0,
            "std_iou": 0.0,
            "std_dice": 0.0,
            "iou_per_view": [0.0] * T,
            "dice_per_view": [0.0] * T,
        }

    gt_instance_masks_gpu = gt_instance_masks.to(device)
    gt_ids_gpu = gt_ids.to(device)

    ious_per_view: List[float] = []
    dices_per_view: List[float] = []

    for t in range(T):
        # GT binary masks for this view: (G, H, W)
        gt_bin = (gt_instance_masks_gpu[t : t + 1] == gt_ids_gpu[:, None, None]).squeeze(
            1
        )

        # Predictions for this view
        if t >= len(pred_masks_list) or pred_masks_list[t] is None:
            ious_per_view.append(0.0)
            dices_per_view.append(0.0)
            continue

        pred = pred_masks_list[t]  # (P, H, W) or (P, 1, H, W) or (H, W)
        if not isinstance(pred, torch.Tensor):
            pred = torch.from_numpy(pred)
        pred = pred.to(device)

        # Normalize shapes
        if pred.dim() == 2:
            pred = pred.unsqueeze(0)  # (1, H, W)
        elif pred.dim() == 4:
            pred = pred[:, 0, :, :]  # (P, 1, H, W) -> (P, H, W)

        # Binarize once
        pred_bin = pred > threshold

        H_pred, W_pred = pred_bin.shape[1], pred_bin.shape[2]
        H_gt, W_gt = gt_bin.shape[1], gt_bin.shape[2]
        if H_pred != H_gt or W_pred != W_gt:
            pred_bin_4d = pred_bin.float().unsqueeze(0)
            pred_bin = (
                F.interpolate(pred_bin_4d, size=(H_gt, W_gt), mode="nearest").squeeze(0)
                > 0.5
            )

        # Vectorized IoU/Dice over all GT × Pred pairs
        gt_expanded = gt_bin[:, None, :, :]  # (G, 1, H, W)
        pred_expanded = pred_bin[None, :, :, :]  # (1, P, H, W)

        intersection = (gt_expanded & pred_expanded).sum(dim=(2, 3)).float()  # (G, P)
        union = (gt_expanded | pred_expanded).sum(dim=(2, 3)).float()  # (G, P)
        iou_matrix = intersection / union.clamp(min=1.0)

        gt_area = gt_bin.sum(dim=(1, 2))  # (G,)
        pred_area = pred_bin.sum(dim=(1, 2))  # (P,)
        dice_numerator = 2.0 * intersection
        dice_denominator = gt_area[:, None] + pred_area[None, :]
        dice_matrix = dice_numerator / dice_denominator.clamp(min=1.0)

        best_iou_per_gt = iou_matrix.max(dim=1).values
        best_dice_per_gt = dice_matrix.max(dim=1).values

        ious_per_view.append(float(best_iou_per_gt.mean().item()))
        dices_per_view.append(float(best_dice_per_gt.mean().item()))

    ious_arr = np.array(ious_per_view)
    dices_arr = np.array(dices_per_view)
    return {
        "mean_iou": float(ious_arr.mean()),
        "mean_dice": float(dices_arr.mean()),
        "std_iou": float(ious_arr.std()),
        "std_dice": float(dices_arr.std()),
        "iou_per_view": ious_per_view,
        "dice_per_view": dices_per_view,
    }


def predict_from_view0_all_frames_finetuned_gt(
    predictor,
    normal_imgs: torch.Tensor,
    point_imgs: torch.Tensor,
    gt_instance_masks: torch.Tensor,
    view_dirs: torch.Tensor,
) -> List[torch.Tensor | None]:
    """
    Dual-modality finetuned predictor prompted with GT instance masks from view 0,
    then propagated to all frames.
    """
    state = predictor.init_state(
        normal_imgs=normal_imgs,
        point_imgs=point_imgs,
        view_dirs=view_dirs,
    )

    gt_view0 = gt_instance_masks[0]
    gt_ids = torch.unique(gt_view0)
    gt_ids = gt_ids[gt_ids != 0]

    for obj_id in gt_ids.tolist():
        mask0 = gt_view0 == obj_id
        if mask0.any():
            predictor.add_new_mask(state, frame_idx=0, obj_id=int(obj_id), mask=mask0)

    num_frames = state["num_frames"]
    pred_masks_list: List[torch.Tensor | None] = [None] * num_frames
    for frame_idx, obj_ids_step, video_res_masks in predictor.propagate_in_video(state):
        pred_masks_list[frame_idx] = video_res_masks[:, 0, :, :].detach()

    return pred_masks_list

