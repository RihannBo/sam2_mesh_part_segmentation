"""
Automatic Mask Generator (AMG) modality ablation.

Compares SAM2 AMG proposals from three input modalities:
  - normal maps
  - point maps
  - matte images

Evaluation:
  - Only view 0 of each mesh in `MultiViewSAM2Dataset`
  - GT: per-instance masks from `gt_instance_masks[0]`
  - Metrics per GT instance: IoU, Dice vs best-overlap AMG proposal
  - Aggregation:
      * per-mesh (view-0-instance-averaged) metrics
      * dataset-level modality metrics

Outputs (under `visualization_results/evaluation_metrics/`):
  - `amg_modality_raw_results.csv`
      * one row per GT instance in view 0 of each mesh
  - `amg_mesh_level_results.csv`
      * one row per mesh × modality
  - `amg_modality_summary.csv`
      * one row per modality
  - `amg_modality_ablation.log`
      * human-readable experiment log
  - Plots:
      * `amg_modality_metrics.png`
      * `amg_iou_distribution.png`
      * `amg_mask_counts.png`
      * `amg_instance_success_rate.png`
      * `amg_iou_success_curve.png`
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple
import csv
import json
import logging
from datetime import datetime
import random
import time

import numpy as np
import torch
from omegaconf import OmegaConf
import matplotlib.pyplot as plt
from tqdm import tqdm


def setup_paths_and_logging() -> Dict[str, Any]:
    """
    Resolve project root, set up sys.path, logging, and device.

    This file lives at:
      <PROJECT_ROOT>/src/seg3d/ablation_studies/amg_modality_ablation.py
    so PROJECT_ROOT is three levels up from this file's directory.
    """
    import sys

    project_root = Path(__file__).resolve().parents[3]  # .../seg3d
    sam2_root = project_root / "sam2"

    src_root = project_root / "src"
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))
    if str(sam2_root) not in sys.path:
        sys.path.insert(0, str(sam2_root))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    logger = logging.getLogger("amg_modality_ablation")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    output_dir = project_root / "visualization_results" / "evaluation_metrics"
    output_dir.mkdir(parents=True, exist_ok=True)

    return {
        "PROJECT_ROOT": project_root,
        "SAM2_ROOT": sam2_root,
        "OUTPUT_DIR": output_dir,
        "logger": logger,
        "device": device,
    }


def build_dataset(project_root: Path, num_views: int = 12):
    from seg3d.dataset.multiview_dataset import MultiViewSAM2Dataset

    dataset_root = project_root / "training_dataset"
    dataset = MultiViewSAM2Dataset(dataset_root=str(dataset_root), num_views=num_views)
    return dataset


def build_amg_generator(
    project_root: Path,
    device: torch.device,
) -> Tuple[Any, Dict[str, Any]]:
    """
    Build a single `MaskProposalGenerator` instance configured for AMG.
    """
    from seg3d.models.mask_proposal_generator import MaskProposalGenerator

    amg_cfg_dict: Dict[str, Any] = {
        "checkpoint": str(
            project_root / "sam2" / "checkpoints" / "sam2.1_hiera_large.pt"
        ),
        "model_config": "configs/sam2.1/sam2.1_hiera_l.yaml",
        "points_per_side": 128,
        "pred_iou_thresh": 0.85,
        "stability_score_thresh": 0.85,
        "stability_score_offset": 1.0,
        "min_mask_region_area": 300,
        "box_nms_thresh": 0.7,
        "crop_nms_thresh": 0.5,
        "crop_n_layers": 0,
        "use_m2m": False,
        "multimask_output": True,
    }
    cfg = OmegaConf.create(amg_cfg_dict)
    generator = MaskProposalGenerator(config=cfg, device=str(device))
    return generator, amg_cfg_dict


def tensor_view0_to_uint8_hwc(img_tensor: torch.Tensor) -> np.ndarray:
    """
    Convert a (3,H,W) tensor in [0,1] to uint8 HWC [0,255].
    """
    if img_tensor.dim() != 3:
        raise ValueError(f"Expected (3,H,W) tensor, got {img_tensor.shape}")
    with torch.no_grad():
        t = img_tensor.detach().cpu()
        img_min = float(t.min().item())
        img_max = float(t.max().item())
        assert img_min >= 0.0 and img_max <= 1.0, (
            f"Input image tensor must be in [0,1], got min={img_min}, max={img_max}"
        )
        img = t.clamp(0.0, 1.0).numpy()
    img = np.transpose(img, (1, 2, 0))  # H, W, C
    img = (img * 255.0).round().astype(np.uint8)
    return img


def compute_instance_metrics_for_modality(
    gt_label_map: np.ndarray,
    pred_masks: np.ndarray,
) -> Tuple[np.ndarray, List[float], List[float], List[int], List[int], List[int], int]:
    """
    For a single mesh, view 0 and single modality.

    Args:
        gt_label_map: (H,W) int, 0=background, 1..N=instances
        pred_masks: (P,H,W) bool array of AMG proposals; P may be 0

    Returns:
        (ious, dices, gt_areas, best_pred_areas, best_pred_indices)
        plus num_pred_masks (P).
        All list lengths equal the number of GT instances in this mesh.
    """
    gt_ids = np.unique(gt_label_map)
    gt_ids = gt_ids[gt_ids != 0]

    if gt_ids.size == 0:
        return gt_ids, [], [], [], [], [], 0

    H, W = gt_label_map.shape
    if pred_masks.size == 0:
        # No predictions: IoU/Dice = 0 for all instances
        ious = [0.0] * int(gt_ids.size)
        dices = [0.0] * int(gt_ids.size)
        gt_areas = [int((gt_label_map == gid).sum()) for gid in gt_ids]
        best_pred_areas = [0] * int(gt_ids.size)
        best_pred_indices = [-1] * int(gt_ids.size)
        return gt_ids, ious, dices, gt_areas, best_pred_areas, best_pred_indices, 0

    # Ensure boolean and shapes
    pred_masks = pred_masks.astype(bool)
    num_pred = pred_masks.shape[0]
    pred_flat = pred_masks.reshape(num_pred, -1).astype(np.float32)
    pred_areas = pred_flat.sum(axis=1)  # (P,)

    ious: List[float] = []
    dices: List[float] = []
    gt_areas: List[int] = []
    best_pred_areas: List[int] = []
    best_pred_indices: List[int] = []

    # Vectorize over predictions per GT instance
    gt_flat_all = gt_label_map.reshape(-1)
    for gid in gt_ids:
        gt_mask = (gt_flat_all == gid)
        gt_area = int(gt_mask.sum())
        if gt_area == 0:
            ious.append(0.0)
            dices.append(0.0)
            gt_areas.append(0)
            best_pred_areas.append(0)
            best_pred_indices.append(-1)
            continue

        gt_mask_f = gt_mask.astype(np.float32)[None, :]  # (1, HW)

        intersection = (gt_mask_f * pred_flat).sum(axis=1)  # (P,)
        union = gt_area + pred_areas - intersection
        iou = intersection / np.maximum(union, 1.0)
        dice = (2.0 * intersection) / np.maximum(gt_area + pred_areas, 1.0)

        best_idx = int(iou.argmax())
        best_iou = float(iou[best_idx])
        best_dice = float(dice[best_idx])

        ious.append(best_iou)
        dices.append(best_dice)
        gt_areas.append(gt_area)
        best_pred_areas.append(int(pred_areas[best_idx]))
        best_pred_indices.append(best_idx)

    return gt_ids, ious, dices, gt_areas, best_pred_areas, best_pred_indices, int(num_pred)


def write_csv(path: Path, fieldnames: List[str], rows: List[Dict[str, Any]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def make_plots(
    output_dir: Path,
    modalities: List[str],
    modality_summary: Dict[str, Dict[str, float]],
    instance_ious_all: Dict[str, List[float]],
    avg_raw_pred_masks_per_modality: Dict[str, float],
    mesh_num_pred_masks_raw: Dict[str, List[int]],
) -> None:
    """
    Generate and save all required plots from aggregated statistics.
    """
    # Plot 1 — Quantitative Comparison (IoU & Dice)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    x = np.arange(len(modalities))

    mean_ious = [modality_summary[m]["Mean_IoU"] for m in modalities]
    std_ious = [modality_summary[m]["Std_IoU"] for m in modalities]
    mean_dices = [modality_summary[m]["Mean_Dice"] for m in modalities]
    std_dices = [modality_summary[m]["Std_Dice"] for m in modalities]

    axes[0].bar(x, mean_ious, yerr=std_ious, capsize=5)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([m.capitalize() for m in modalities])
    axes[0].set_ylabel("Mean IoU")
    axes[0].set_title("AMG Modality – Mean IoU ± Std")

    axes[1].bar(x, mean_dices, yerr=std_dices, capsize=5)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([m.capitalize() for m in modalities])
    axes[1].set_ylabel("Mean Dice")
    axes[1].set_title("AMG Modality – Mean Dice ± Std")

    fig.tight_layout()
    fig.savefig(output_dir / "amg_modality_metrics.png", dpi=200)
    plt.close(fig)

    # Plot 2 — IoU Distribution
    fig, ax = plt.subplots(figsize=(7, 5))
    bins = np.linspace(0.0, 1.0, 21)
    for m in modalities:
        values = np.array(instance_ious_all[m], dtype=float)
        if values.size == 0:
            continue
        ax.hist(
            values,
            bins=bins,
            alpha=0.4,
            density=True,
            label=m.capitalize(),
        )
    ax.set_xlabel("IoU")
    ax.set_ylabel("Density")
    ax.set_title("IoU Distribution per Modality (AMG)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "amg_iou_distribution.png", dpi=200)
    plt.close(fig)

    # Plot 3 — Mask Count Statistics
    fig, ax = plt.subplots(figsize=(7, 5))
    avg_counts = [avg_raw_pred_masks_per_modality[m] for m in modalities]
    ax.bar(x, avg_counts)
    ax.set_xticks(x)
    ax.set_xticklabels([m.capitalize() for m in modalities])
    ax.set_ylabel("Average # Predicted Masks (view 0)")
    ax.set_title("AMG – Average Proposal Count per Modality")
    fig.tight_layout()
    fig.savefig(output_dir / "amg_mask_counts.png", dpi=200)
    plt.close(fig)

    # Plot 4 — Instance Detection Success (IoU >= 0.5)
    fig, ax = plt.subplots(figsize=(7, 5))
    success_rates = []
    for m in modalities:
        vals = np.array(instance_ious_all[m], dtype=float)
        if vals.size == 0:
            success_rates.append(0.0)
        else:
            success_rates.append(float((vals >= 0.5).mean()))
    ax.bar(x, success_rates)
    ax.set_xticks(x)
    ax.set_xticklabels([m.capitalize() for m in modalities])
    ax.set_ylabel("Success Rate (IoU ≥ 0.5)")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("AMG – Instance Detection Success")
    fig.tight_layout()
    fig.savefig(output_dir / "amg_instance_success_rate.png", dpi=200)
    plt.close(fig)

    # Plot 5 — IoU Success Curve
    thresholds = np.arange(0.1, 1.0, 0.1)
    fig, ax = plt.subplots(figsize=(7, 5))
    for m in modalities:
        vals = np.array(instance_ious_all[m], dtype=float)
        if vals.size == 0:
            curve = np.zeros_like(thresholds, dtype=float)
        else:
            curve = np.array([(vals >= t).mean() for t in thresholds], dtype=float)
        ax.plot(thresholds, curve, marker="o", label=m.capitalize())
    ax.set_xlabel("IoU Threshold")
    ax.set_ylabel("Success Rate")
    ax.set_title("AMG – IoU Success Curves")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "amg_iou_success_curve.png", dpi=200)
    plt.close(fig)

    # Plot 6 — Proposal Count Distribution per Mesh
    max_count = 0
    for v in mesh_num_pred_masks_raw.values():
        if len(v) > 0:
            max_count = max(max_count, max(v))
    if max_count > 0:
        fig, ax = plt.subplots(figsize=(7, 5))
        bins_counts = np.linspace(0, max_count + 1, 21)
        for m in modalities:
            counts = np.array(mesh_num_pred_masks_raw[m], dtype=float)
            if counts.size == 0:
                continue
            ax.hist(
                counts,
                bins=bins_counts,
                alpha=0.4,
                label=m.capitalize(),
            )
        ax.set_xlabel("# Predicted Masks per Mesh (view 0)")
        ax.set_ylabel("Number of Meshes")
        ax.set_title("AMG – Proposal Count Distribution")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / "amg_proposal_count_distribution.png", dpi=200)
        plt.close(fig)


def run_ablation(
    max_meshes: int | None = None,
) -> None:
    # Deterministic seeds for reproducibility
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    ctx = setup_paths_and_logging()
    project_root: Path = ctx["PROJECT_ROOT"]
    output_dir: Path = ctx["OUTPUT_DIR"]
    logger: logging.Logger = ctx["logger"]
    device: torch.device = ctx["device"]

    logger.info("Building dataset...")
    dataset = build_dataset(project_root, num_views=12)
    num_meshes_total = len(dataset)
    if max_meshes is not None:
        num_to_eval = min(max_meshes, num_meshes_total)
    else:
        num_to_eval = num_meshes_total

    logger.info(
        f"Dataset root: {project_root / 'training_dataset'} | "
        f"num meshes total = {num_meshes_total}, evaluating = {num_to_eval}"
    )

    logger.info("Building AMG MaskProposalGenerator...")
    amg_generator, amg_cfg_dict = build_amg_generator(project_root, device)

    modalities = ["normal", "point", "matte"]

    raw_rows: List[Dict[str, Any]] = []
    mesh_rows: List[Dict[str, Any]] = []
    summary_per_modality: Dict[str, Dict[str, float]] = {}

    mesh_mean_ious: Dict[str, List[float]] = {m: [] for m in modalities}
    mesh_mean_dices: Dict[str, List[float]] = {m: [] for m in modalities}
    mesh_std_ious: Dict[str, List[float]] = {m: [] for m in modalities}
    mesh_std_dices: Dict[str, List[float]] = {m: [] for m in modalities}
    mesh_num_instances: Dict[str, List[int]] = {m: [] for m in modalities}
    mesh_num_pred_masks_raw: Dict[str, List[int]] = {m: [] for m in modalities}
    mesh_num_pred_masks_used: Dict[str, List[int]] = {m: [] for m in modalities}

    instance_ious_all: Dict[str, List[float]] = {m: [] for m in modalities}
    instance_iou_rows: List[Dict[str, Any]] = []

    total_gt_instances = 0
    total_pred_masks_raw: Dict[str, int] = {m: 0 for m in modalities}
    total_pred_masks_used: Dict[str, int] = {m: 0 for m in modalities}
    mesh_runtimes: List[float] = []

    start_time = datetime.now()
    exp_start_time_s = time.time()

    # Preselect meshes for qualitative examples
    num_example_meshes = min(20, num_to_eval)
    example_indices = set(random.sample(range(num_to_eval), num_example_meshes)) if num_example_meshes > 0 else set()
    examples_dir = output_dir / "amg_examples"
    examples_dir.mkdir(parents=True, exist_ok=True)
    sanity_done = False

    # Fixed-seed selection for 15-mesh qualitative triplets (normal/point/matte on matte)
    num_triplet_meshes = min(15, num_to_eval)
    rng_triplet = np.random.RandomState(40)
    triplet_indices = set(
        rng_triplet.choice(num_to_eval, size=num_triplet_meshes, replace=False)
    ) if num_triplet_meshes > 0 else set()

    pbar = tqdm(range(num_to_eval), desc="AMG modality ablation")
    for mesh_idx in pbar:
        logger.info(f"Processing mesh {mesh_idx + 1} / {num_to_eval}")
        mesh_start_s = time.time()
        sample = dataset[mesh_idx]
        mesh_id = sample.get("mesh_id", f"mesh_{mesh_idx}")

        gt_inst_masks = sample["gt_instance_masks"]  # (T,H,W)
        gt_view0 = gt_inst_masks[0].detach().cpu().numpy()
        assert gt_view0.dtype in (np.int32, np.int64, np.int16, np.uint16), (
            f"GT view0 must be integer type, got {gt_view0.dtype}"
        )
        gt_ids = np.unique(gt_view0)
        gt_ids = gt_ids[gt_ids != 0]
        if gt_ids.size > 0:
            expected_ids = np.arange(1, gt_ids.size + 1, dtype=gt_ids.dtype)
            if not np.array_equal(gt_ids, expected_ids):
                logger.warning(
                    f"Non-sequential GT instance IDs for mesh {mesh_id}: {gt_ids}"
                )
        num_instances = int(gt_ids.size)
        total_gt_instances += num_instances

        # Prepare modality images for view 0
        normal_img_np = tensor_view0_to_uint8_hwc(sample["normal_imgs"][0])
        point_img_np = tensor_view0_to_uint8_hwc(sample["point_imgs"][0])
        matte_available = ("matte_imgs" in sample) and (
            sample["matte_imgs"] is not None
        )
        matte_img_np = (
            tensor_view0_to_uint8_hwc(sample["matte_imgs"][0])
            if matte_available
            else None
        )

        pred_masks_by_modality: Dict[str, np.ndarray] = {}

        # Normal
        with torch.no_grad():
            bmasks_normal = amg_generator(
                image=normal_img_np, view_idx=0, fg_mask=None
            )
        pred_masks_by_modality["normal"] = bmasks_normal

        # Point
        with torch.no_grad():
            bmasks_point = amg_generator(
                image=point_img_np, view_idx=0, fg_mask=None
            )
        pred_masks_by_modality["point"] = bmasks_point

        # Matte (if available)
        if matte_img_np is not None:
            with torch.no_grad():
                bmasks_matte = amg_generator(
                    image=matte_img_np, view_idx=0, fg_mask=None
                )
        else:
            bmasks_matte = np.zeros((0,) + gt_view0.shape, dtype=bool)
        pred_masks_by_modality["matte"] = bmasks_matte

        # Per-modality metrics for this mesh
        for modality, bmasks in pred_masks_by_modality.items():
            # Raw proposals from AMG
            if bmasks is None or bmasks.size == 0:
                raw_pred_count = 0
                bmasks_np = np.zeros((0,) + gt_view0.shape, dtype=bool)
            else:
                bmasks_np = bmasks.astype(bool)
                raw_pred_count = int(bmasks_np.shape[0])

            # Apply safety cap on number of proposals per image to keep runtime reasonable
            if raw_pred_count > 200:
                logger.warning(
                    f"AMG produced {raw_pred_count} masks for mesh {mesh_id}, modality {modality}; capping to 200"
                )
                bmasks_used = bmasks_np[:200]
            else:
                bmasks_used = bmasks_np

            used_pred_count = int(bmasks_used.shape[0])

            # Track raw vs used proposal counts
            total_pred_masks_raw[modality] += raw_pred_count
            total_pred_masks_used[modality] += used_pred_count
            mesh_num_pred_masks_raw[modality].append(raw_pred_count)
            mesh_num_pred_masks_used[modality].append(used_pred_count)

            (
                gt_ids_mod,
                ious,
                dices,
                gt_areas,
                best_pred_areas,
                best_pred_indices,
                num_pred_masks,
            ) = compute_instance_metrics_for_modality(gt_view0, bmasks_used)

            if len(ious) == 0:
                mean_iou = 0.0
                mean_dice = 0.0
                std_iou = 0.0
                std_dice = 0.0
                num_inst_this_mesh = 0
            else:
                arr_iou = np.array(ious, dtype=float)
                arr_dice = np.array(dices, dtype=float)
                mean_iou = float(arr_iou.mean())
                mean_dice = float(arr_dice.mean())
                std_iou = float(arr_iou.std())
                std_dice = float(arr_dice.std())
                num_inst_this_mesh = len(ious)

            mesh_mean_ious[modality].append(mean_iou)
            mesh_mean_dices[modality].append(mean_dice)
            mesh_std_ious[modality].append(std_iou)
            mesh_std_dices[modality].append(std_dice)
            mesh_num_instances[modality].append(num_inst_this_mesh)

            instance_ious_all[modality].extend(ious)

            # Raw rows: one per GT instance in this mesh and modality
            for inst_idx, gid in enumerate(gt_ids_mod):
                row = {
                    "Mesh_ID": mesh_id,
                    "Mesh_Index": mesh_idx,
                    "Modality": modality,
                    "GT_Instance_ID": int(gid),
                    "IoU": float(ious[inst_idx]) if inst_idx < len(ious) else 0.0,
                    "Dice": float(dices[inst_idx]) if inst_idx < len(dices) else 0.0,
                    "GT_Area": int(gt_areas[inst_idx])
                    if inst_idx < len(gt_areas)
                    else 0,
                    "Pred_Area": int(best_pred_areas[inst_idx])
                    if inst_idx < len(best_pred_areas)
                    else 0,
                    # Log raw proposal count so instance rows match summary counts
                    "Num_Predicted_Masks": int(raw_pred_count),
                    "Best_Pred_Mask_ID": int(best_pred_indices[inst_idx])
                    if inst_idx < len(best_pred_indices)
                    else -1,
                }
                raw_rows.append(row)
                instance_iou_rows.append(
                    {
                        "Modality": modality,
                        "IoU": float(ious[inst_idx]) if inst_idx < len(ious) else 0.0,
                        "Mesh_ID": mesh_id,
                        "Instance_ID": int(gid),
                    }
                )

            mesh_rows.append(
                {
                    "Mesh_ID": mesh_id,
                    "Mesh_Index": mesh_idx,
                    "Modality": modality,
                    "Mean_IoU": mean_iou,
                    "Mean_Dice": mean_dice,
                    "Std_IoU": std_iou,
                    "Std_Dice": std_dice,
                    "Num_Instances": num_inst_this_mesh,
                    # Store raw proposal count at mesh level as well
                    "Num_Predicted_Masks": int(raw_pred_count),
                    "Mesh_Runtime": 0.0,  # placeholder, overwritten below
                }
            )

        mesh_runtime_s = time.time() - mesh_start_s
        mesh_runtimes.append(mesh_runtime_s)
        # Back-fill runtime into mesh_rows entries created for this mesh
        # Use the known number of modalities for simplicity and robustness
        start_row_idx = len(mesh_rows) - len(modalities)
        for r_idx in range(start_row_idx, len(mesh_rows)):
            mesh_rows[r_idx]["Mesh_Runtime"] = mesh_runtime_s

        # Sanity visualization for first mesh
        if not sanity_done:
            base_img = (
                matte_img_np
                if matte_img_np is not None
                else normal_img_np
            )
            sanity_masks = pred_masks_by_modality.get("normal")
            if sanity_masks is None or sanity_masks.size == 0:
                sanity_masks = np.zeros((0,) + base_img.shape[:2], dtype=bool)
            else:
                sanity_masks = sanity_masks.astype(bool)
            top_k = min(10, sanity_masks.shape[0])
            sanity_masks = sanity_masks[:top_k]

            # Build colored overlay
            overlay = base_img.astype(np.float32) / 255.0
            H, W, _ = overlay.shape
            rng = np.random.RandomState(mesh_idx)
            colors = rng.rand(max(top_k, 1), 3)
            for k_idx in range(top_k):
                mask_k = sanity_masks[k_idx]
                if not mask_k.any():
                    continue
                overlay[mask_k] = (
                    0.5 * overlay[mask_k] + 0.5 * colors[k_idx]
                )
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.imshow(overlay)
            ax.axis("off")
            fig.tight_layout()
            fig.savefig(output_dir / "sanity_check_amg.png", dpi=200)
            plt.close(fig)
            sanity_done = True

        # Qualitative examples
        if mesh_idx in example_indices:
            base_pairs = [
                ("normal", normal_img_np),
                ("point", point_img_np),
                ("matte", matte_img_np if matte_img_np is not None else normal_img_np),
            ]
            rng_examples = np.random.RandomState(mesh_idx)
            for modality_name, base_img_mod in base_pairs:
                if base_img_mod is None:
                    continue
                masks_mod = pred_masks_by_modality.get(modality_name)
                if masks_mod is None or masks_mod.size == 0:
                    masks_mod = np.zeros(
                        (0, base_img_mod.shape[0], base_img_mod.shape[1]),
                        dtype=bool,
                    )
                else:
                    masks_mod = masks_mod.astype(bool)
                H, W, _ = base_img_mod.shape
                overlay = base_img_mod.astype(np.float32) / 255.0
                P = masks_mod.shape[0]
                colors = rng_examples.rand(max(P, 1), 3)
                for p_idx in range(P):
                    mask_p = masks_mod[p_idx]
                    if not mask_p.any():
                        continue
                    overlay[mask_p] = (
                        0.5 * overlay[mask_p] + 0.5 * colors[p_idx]
                    )
                fig, ax = plt.subplots(figsize=(5, 5))
                ax.imshow(overlay)
                ax.axis("off")
                fig.tight_layout()
                fig.savefig(
                    examples_dir / f"mesh_{mesh_idx:03d}_{modality_name}.png",
                    dpi=200,
                )
                plt.close(fig)

            # GT visualization
            gt_vis = np.zeros((*gt_view0.shape, 3), dtype=np.float32)
            gt_ids_nonzero = np.unique(gt_view0)
            gt_ids_nonzero = gt_ids_nonzero[gt_ids_nonzero != 0]
            colors_gt = rng_examples.rand(max(len(gt_ids_nonzero), 1), 3)
            for idx_gid, gid in enumerate(gt_ids_nonzero):
                mask_gid = gt_view0 == gid
                gt_vis[mask_gid] = colors_gt[idx_gid]
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.imshow(gt_vis)
            ax.axis("off")
            fig.tight_layout()
            fig.savefig(
                examples_dir / f"mesh_{mesh_idx:03d}_gt.png",
                dpi=200,
            )
            plt.close(fig)

        # Update running progress statistics in tqdm
        running_stats = {}
        for m in modalities:
            if len(mesh_mean_ious[m]) > 0:
                running_stats[f"IoU({m})"] = f"{np.mean(mesh_mean_ious[m]):.3f}"
                running_stats[f"Dice({m})"] = f"{np.mean(mesh_mean_dices[m]):.3f}"
        pbar.set_postfix(running_stats)

        # Triplet qualitative plots (normal/point/matte over matte, view 0) for 15 meshes (seed 40)
        if mesh_idx in triplet_indices and matte_img_np is not None:
            base_img = matte_img_np.astype(np.float32) / 255.0
            overlays = []
            rng_trip = np.random.RandomState(mesh_idx)
            for modality_name in ["normal", "point", "matte"]:
                masks_mod = pred_masks_by_modality.get(modality_name)
                if masks_mod is None or masks_mod.size == 0:
                    masks_mod = np.zeros(
                        (0, base_img.shape[0], base_img.shape[1]),
                        dtype=bool,
                    )
                else:
                    masks_mod = masks_mod.astype(bool)
                overlay = base_img.copy()
                P = masks_mod.shape[0]
                colors = rng_trip.rand(max(P, 1), 3)
                for p_idx in range(P):
                    mask_p = masks_mod[p_idx]
                    if not mask_p.any():
                        continue
                    overlay[mask_p] = (
                        0.5 * overlay[mask_p] + 0.5 * colors[p_idx]
                    )
                overlays.append(overlay)

            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            titles = ["Normal", "Point", "Matte"]
            for ax, img, title in zip(axes, overlays, titles):
                ax.imshow(img)
                ax.set_title(title)
                ax.axis("off")
            fig.tight_layout()
            fig.savefig(
                examples_dir / f"amg_triplet_mesh_{mesh_idx:03d}.png",
                dpi=200,
            )
            plt.close(fig)

    # Dataset-level summary per modality
    num_meshes_eval = num_to_eval
    avg_gt_instances_per_mesh = (
        float(total_gt_instances) / float(num_meshes_eval)
        if num_meshes_eval > 0
        else 0.0
    )

    avg_raw_pred_masks_per_modality: Dict[str, float] = {}
    avg_mesh_runtime = float(np.mean(mesh_runtimes)) if len(mesh_runtimes) > 0 else 0.0

    for modality in modalities:
        means_iou = np.array(mesh_mean_ious[modality], dtype=float)
        means_dice = np.array(mesh_mean_dices[modality], dtype=float)

        if means_iou.size == 0:
            mean_iou_dataset = 0.0
            std_iou_dataset = 0.0
            mean_dice_dataset = 0.0
            std_dice_dataset = 0.0
        else:
            mean_iou_dataset = float(means_iou.mean())
            std_iou_dataset = float(means_iou.std())
            mean_dice_dataset = float(means_dice.mean())
            std_dice_dataset = float(means_dice.std())

        avg_raw_pred_masks = (
            float(total_pred_masks_raw[modality]) / float(num_meshes_eval)
            if num_meshes_eval > 0
            else 0.0
        )
        avg_raw_pred_masks_per_modality[modality] = avg_raw_pred_masks

        summary_per_modality[modality] = {
            "Modality": modality,
            "Mean_IoU": mean_iou_dataset,
            "Std_IoU": std_iou_dataset,
            "Mean_Dice": mean_dice_dataset,
            "Std_Dice": std_dice_dataset,
            "Num_Meshes": int(num_meshes_eval),
            "Avg_GT_Instances": avg_gt_instances_per_mesh,
            "Avg_Predicted_Masks": avg_raw_pred_masks,
            "Avg_Mesh_Runtime": avg_mesh_runtime,
        }

    summary_rows = list(summary_per_modality.values())

    # Write CSVs
    write_csv(
        output_dir / "amg_modality_raw_results.csv",
        fieldnames=[
            "Mesh_ID",
            "Mesh_Index",
            "Modality",
            "GT_Instance_ID",
            "IoU",
            "Dice",
            "GT_Area",
            "Pred_Area",
            "Num_Predicted_Masks",
            "Best_Pred_Mask_ID",
        ],
        rows=raw_rows,
    )
    write_csv(
        output_dir / "amg_mesh_level_results.csv",
        fieldnames=[
            "Mesh_ID",
            "Mesh_Index",
            "Modality",
            "Mean_IoU",
            "Mean_Dice",
            "Std_IoU",
            "Std_Dice",
            "Num_Instances",
            "Num_Predicted_Masks",
            "Mesh_Runtime",
        ],
        rows=mesh_rows,
    )
    write_csv(
        output_dir / "amg_modality_summary.csv",
        fieldnames=[
            "Modality",
            "Mean_IoU",
            "Std_IoU",
            "Mean_Dice",
            "Std_Dice",
            "Num_Meshes",
            "Avg_GT_Instances",
            "Avg_Predicted_Masks",
            "Avg_Mesh_Runtime",
        ],
        rows=summary_rows,
    )

    # Human-readable log
    end_time = datetime.now()
    exp_total_runtime_s = time.time() - exp_start_time_s
    best_by_iou = max(
        modalities, key=lambda m: summary_per_modality[m]["Mean_IoU"]
    )
    best_by_dice = max(
        modalities, key=lambda m: summary_per_modality[m]["Mean_Dice"]
    )

    log_lines: List[str] = []
    log_lines.append("Experiment: AMG Modality Ablation")
    log_lines.append(f"Dataset: {project_root / 'training_dataset'}")
    log_lines.append(
        f"Checkpoint: {amg_cfg_dict['checkpoint']}"
    )
    log_lines.append(
        f"Model config: {amg_cfg_dict['model_config']}"
    )
    log_lines.append(
        f"Points per side: {amg_cfg_dict['points_per_side']}"
    )
    log_lines.append(
        f"IoU threshold (pred_iou_thresh): {amg_cfg_dict['pred_iou_thresh']}"
    )
    log_lines.append(
        f"Stability threshold: {amg_cfg_dict['stability_score_thresh']}"
    )
    log_lines.append(
        f"Stability offset: {amg_cfg_dict['stability_score_offset']}"
    )
    log_lines.append(
        f"Min mask region area: {amg_cfg_dict['min_mask_region_area']}"
    )
    log_lines.append(f"Box NMS thresh: {amg_cfg_dict['box_nms_thresh']}")
    log_lines.append(f"Crop NMS thresh: {amg_cfg_dict['crop_nms_thresh']}")
    log_lines.append(f"Crop n layers: {amg_cfg_dict['crop_n_layers']}")
    log_lines.append(f"use_m2m: {amg_cfg_dict['use_m2m']}")
    log_lines.append(f"multimask_output: {amg_cfg_dict['multimask_output']}")
    log_lines.append("")
    log_lines.append(f"Evaluation start time: {start_time.isoformat()}")
    log_lines.append(f"Evaluation end time:   {end_time.isoformat()}")
    log_lines.append(f"Total runtime (s): {exp_total_runtime_s:.2f}")
    log_lines.append(f"Average runtime per mesh (s): {avg_mesh_runtime:.4f}")
    log_lines.append(f"Total meshes evaluated: {num_meshes_eval}")
    log_lines.append(
        f"Average GT instances per mesh: {avg_gt_instances_per_mesh:.3f}"
    )
    for m in modalities:
        log_lines.append(
            f"Average raw predicted masks ({m}): {avg_raw_pred_masks_per_modality[m]:.3f}"
        )
    log_lines.append("")
    log_lines.append("Final Results (dataset-level means over meshes)")
    for m in modalities:
        s = summary_per_modality[m]
        log_lines.append(
            f"{m.capitalize()}: "
            f"Mean IoU = {s['Mean_IoU']:.4f}, "
            f"Mean Dice = {s['Mean_Dice']:.4f}, "
            f"Std IoU = {s['Std_IoU']:.4f}, "
            f"Std Dice = {s['Std_Dice']:.4f}"
        )
    log_lines.append("")
    log_lines.append(f"Best modality by IoU: {best_by_iou}")
    log_lines.append(f"Best modality by Dice: {best_by_dice}")

    log_path = output_dir / "amg_modality_ablation.log"
    log_path.write_text("\n".join(log_lines))

    # Save experiment configuration snapshot as JSON
    config_snapshot = {
        "experiment_name": "AMG Modality Ablation",
        "dataset_path": str(project_root / "training_dataset"),
        "num_meshes": int(num_meshes_eval),
        "num_views": 12,
        "sam_checkpoint": amg_cfg_dict["checkpoint"],
        "model_config": amg_cfg_dict["model_config"],
        "amg_parameters": {k: v for k, v in amg_cfg_dict.items() if k not in ["checkpoint", "model_config"]},
        "evaluation_start_time": start_time.isoformat(),
        "evaluation_end_time": end_time.isoformat(),
        "total_runtime_seconds": exp_total_runtime_s,
        "average_runtime_per_mesh_seconds": avg_mesh_runtime,
        "summary_metrics": summary_per_modality,
        "created_at": datetime.now().isoformat(),
    }
    config_path = output_dir / "amg_experiment_config.json"
    config_path.write_text(json.dumps(config_snapshot, indent=2))

    # Save per-instance IoU values for future analysis
    write_csv(
        output_dir / "amg_instance_iou_values.csv",
        fieldnames=["Modality", "IoU", "Mesh_ID", "Instance_ID"],
        rows=instance_iou_rows,
    )

    # Plots from raw stats
    make_plots(
        output_dir=output_dir,
        modalities=modalities,
        modality_summary=summary_per_modality,
        instance_ious_all=instance_ious_all,
        avg_raw_pred_masks_per_modality=avg_raw_pred_masks_per_modality,
        mesh_num_pred_masks_raw=mesh_num_pred_masks_raw,
    )

    # Final console output
    print(f"Total meshes evaluated: {num_meshes_eval}")
    print(f"Average GT instances per mesh: {avg_gt_instances_per_mesh:.3f}")
    for m in modalities:
        print(
            f"Average raw predicted masks ({m}): "
            f"{avg_raw_pred_masks_per_modality[m]:.3f}"
        )
    print(f"Best modality by IoU: {best_by_iou}")
    print(f"Best modality by Dice: {best_by_dice}")


if __name__ == "__main__":
    run_ablation()

