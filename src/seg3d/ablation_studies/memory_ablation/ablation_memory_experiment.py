"""
Memory ablation experiment script (Python version of `ablation_memory.ipynb`).

This script:
- Builds 8 SAM2 predictors:
  - 3 with original memory (normal / point / matte)
  - 3 with geo-aware memory (normal / point / matte)
  - 2 dual-modality (normal+point) finetuned models (orig mem vs geo-aware mem)
- Evaluates them on the same 40-mesh eval split used by the modality ablation
  (`eval_dataset_tinyobjaverse/eval_indices.txt`) using GT instance masks from view 0
  as prompts.
- Computes:
  - Per-mesh mean IoU / Dice
  - Overall mean / std IoU & Dice per model
  - Per-view mean / std IoU & Dice across views for each model
- Saves:
  - `ablation_memory_metrics.png` (overall bar plots)
  - `ablation_memory_per_view_iou.png` (per-view IoU trends)
  - `ablation_memory_log.csv` (per-mesh detailed log)
  - `ablation_memory_summary.csv` (overall summary table)
  - `ablation_memory_per_view.csv` (per-view aggregates)
under:
  `/home/mengnan/seg3d/src/seg3d/ablation_studies/memory_ablation/{figures,logging}`
"""

from __future__ import annotations

import csv
import logging
import sys
from pathlib import Path
from typing import Dict, List, Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


def setup_paths_and_logging() -> Dict[str, Any]:
    """Set up project paths, sys.path, and logging."""
    # This file lives at: <PROJECT_ROOT>/src/seg3d/ablation_studies/ablation_memory_experiment.py
    # So PROJECT_ROOT is three levels up from this file's directory.
    project_root = Path("/home/mengnan/seg3d").resolve()
    sam2_root = project_root / "sam2"

    if str(project_root / "src") not in sys.path:
        sys.path.insert(0, str(project_root / "src"))
    if str(sam2_root) not in sys.path:
        sys.path.insert(0, str(sam2_root))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    logger = logging.getLogger("ablation_memory")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Output roots (absolute, to match modality ablation)
    ablation_root = Path("/home/mengnan/seg3d/src/seg3d/ablation_studies/memory_ablation")
    figures_dir = ablation_root / "figures"
    logging_dir = ablation_root / "logging"
    figures_dir.mkdir(parents=True, exist_ok=True)
    logging_dir.mkdir(parents=True, exist_ok=True)

    return {
        "PROJECT_ROOT": project_root,
        "SAM2_ROOT": sam2_root,
        "logger": logger,
        "device": device,
        "FIGURES_DIR": figures_dir,
        "LOGGING_DIR": logging_dir,
    }


def build_dataset(project_root: Path, num_views: int = 12):
    from seg3d.dataset.multiview_dataset import MultiViewSAM2Dataset

    dataset_root = project_root / "training_dataset"
    dataset = MultiViewSAM2Dataset(dataset_root=str(dataset_root), num_views=num_views)
    return dataset


def build_predictors(
    device: torch.device,
    project_root: Path,
    logger: logging.Logger,
) -> Dict[str, Any]:
    """
    Build the predictors used in the ablation:
      - orig_model_normal / point / matte
      - geo_aware_mem_model_normal / point / matte
      - dual_modality_orig_mem (finetuned, dual-modality, original memory)
      - dual_modality_geo_aware_mem (finetuned, dual-modality, geo-aware memory)
    """
    from seg3d.ablation_studies.models.orig_mem_predictor import (
        SAM2VideoPredictor as OrigMemPredictor,
    )
    from seg3d.ablation_studies.models.single_modality_predictor import (
        SAM2AngularMemSingleModalityPredictor as AngularMemPredictor,
    )
    from seg3d.ablation_studies.models.sam2_dual_predictor import (
        SAM2DualOrigMemPredictor,
    )
    from seg3d.models.sam2_mesh_predictor import SAM2MeshPredictor

    # Hydra config file and checkpoint should match the notebook
    config_file_orig = "configs/sam2.1/sam2.1_hiera_l.yaml"
    ckpt_path_orig = project_root / "sam2" / "checkpoints" / "sam2.1_hiera_large.pt"

    logger.info("Building original-memory predictors...")
    orig_model_normal = OrigMemPredictor.from_pretrained(
        config_file=config_file_orig,
        ckpt_path=str(ckpt_path_orig),
        device=device,
    )
    orig_model_point = OrigMemPredictor.from_pretrained(
        config_file=config_file_orig,
        ckpt_path=str(ckpt_path_orig),
        device=device,
    )
    orig_model_matte = OrigMemPredictor.from_pretrained(
        config_file=config_file_orig,
        ckpt_path=str(ckpt_path_orig),
        device=device,
    )
    logger.info("Built 3 original-memory models (normal / point / matte)")

    logger.info("Building geo-aware memory predictors...")
    ang_mem_model_normal = AngularMemPredictor.from_pretrained(
        config_file=config_file_orig,
        ckpt_path=str(ckpt_path_orig),
        device=device,
    )
    ang_mem_model_point = AngularMemPredictor.from_pretrained(
        config_file=config_file_orig,
        ckpt_path=str(ckpt_path_orig),
        device=device,
    )
    ang_mem_model_matte = AngularMemPredictor.from_pretrained(
        config_file=config_file_orig,
        ckpt_path=str(ckpt_path_orig),
        device=device,
    )
    logger.info("Built 3 geo-aware memory models (normal / point / matte)")

    # Finetuned multimodal models (dual-modality: normal+point)
    finetuned_checkpoint = Path("/home/mengnan/seg3d/checkpoints/best.pt")
    finetuned_model_config = Path("/home/mengnan/seg3d/configs/mm_sam2.1_hiera_l.yaml")

    logger.info("Building finetuned dual-modality (orig mem) predictor...")
    dual_orig_mem = SAM2DualOrigMemPredictor.from_pretrained(
        config_file=str(finetuned_model_config),
        ckpt_path=str(finetuned_checkpoint),
        device=device,
    )
    logger.info("Built finetuned dual-modality (orig mem) predictor")

    logger.info("Building finetuned dual-modality (geo-aware mem) predictor...")
    dual_ang_mem = SAM2MeshPredictor.from_pretrained(
        config_file=str(finetuned_model_config),
        ckpt_path=str(finetuned_checkpoint),
        device=device,
    )
    logger.info("Built finetuned dual-modality (geo-aware mem) predictor")

    return {
        "orig_normal": orig_model_normal,
        "orig_point": orig_model_point,
        "orig_matte": orig_model_matte,
        "angular_normal": ang_mem_model_normal,
        "angular_point": ang_mem_model_point,
        "angular_matte": ang_mem_model_matte,
        "dual_orig_mem": dual_orig_mem,
        "dual_ang_mem": dual_ang_mem,
    }


def evaluate_predictions_fast(
    pred_masks_list: List[torch.Tensor | None],
    gt_instance_masks: torch.Tensor,
    threshold: float = 0.0,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, Any]:
    from seg3d.ablation_studies.utils import (
        evaluate_predictions_fast as _eval_fast,
    )

    return _eval_fast(
        pred_masks_list=pred_masks_list,
        gt_instance_masks=gt_instance_masks,
        threshold=threshold,
        device=device,
    )


def _predict_from_view0_all_frames(
    predictor,
    images: torch.Tensor,
    gt_instance_masks: torch.Tensor,
    modality: str = "generic",
    view_dirs: torch.Tensor | None = None,
    use_view_dirs: bool = False,
) -> List[torch.Tensor | None]:
    """
    Use all GT instance masks from view 0 (frame 0) as prompts,
    then propagate to all frames.
    """
    # Initialize predictor state
    if use_view_dirs:
        state = predictor.init_state(
            images=images, view_dirs=view_dirs, modality=modality
        )
    else:
        state = predictor.init_state(images=images, modality=modality)

    # Collect instance ids from view 0 only (exclude background 0)
    gt_view0 = gt_instance_masks[0]
    gt_ids = torch.unique(gt_view0)
    gt_ids = gt_ids[gt_ids != 0]

    # Add a mask prompt per object at frame 0
    for obj_id in gt_ids.tolist():
        mask0 = gt_view0 == obj_id
        if mask0.any():
            predictor.add_new_mask(
                state, frame_idx=0, obj_id=int(obj_id), mask=mask0
            )

    # Propagate through video (always full sequence)
    num_frames = state["num_frames"]
    pred_masks_list: List[torch.Tensor | None] = [None] * num_frames
    for frame_idx, obj_ids_step, video_res_masks in predictor.propagate_in_video(state):
        pred_masks_list[frame_idx] = video_res_masks[:, 0, :, :].detach()

    return pred_masks_list


def _predict_from_view0_all_frames_dual_orig_mem(
    predictor,
    normal_imgs: torch.Tensor,
    point_imgs: torch.Tensor,
    gt_instance_masks: torch.Tensor,
) -> List[torch.Tensor | None]:
    """
    Dual-modality model with original memory (no view_dirs).
    Prompts with all GT instance masks from view 0, then propagates to all frames.
    """
    state = predictor.init_state(normal_imgs=normal_imgs, point_imgs=point_imgs)

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


def _predict_from_view0_all_frames_dual_ang_mem(
    predictor,
    normal_imgs: torch.Tensor,
    point_imgs: torch.Tensor,
    gt_instance_masks: torch.Tensor,
    view_dirs: torch.Tensor,
) -> List[torch.Tensor | None]:
    """
    Dual-modality model with angular/geo-aware memory (requires view_dirs).
    Prompts with all GT instance masks from view 0, then propagates to all frames.
    """
    state = predictor.init_state(
        normal_imgs=normal_imgs, point_imgs=point_imgs, view_dirs=view_dirs
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


def run_ablation() -> None:
    ctx = setup_paths_and_logging()
    project_root: Path = ctx["PROJECT_ROOT"]
    logger: logging.Logger = ctx["logger"]
    device: torch.device = ctx["device"]
    figures_dir: Path = ctx["FIGURES_DIR"]
    logging_dir: Path = ctx["LOGGING_DIR"]

    # Dataset
    dataset = build_dataset(project_root, num_views=12)
    logger.info(f"Dataset size: {len(dataset)} meshes")

    # Predictors
    predictors = build_predictors(device=device, project_root=project_root, logger=logger)

    model_keys = [
        "orig_normal",
        "orig_point",
        "orig_matte",
        "angular_normal",
        "angular_point",
        "angular_matte",
        "dual_orig_mem",
        "dual_ang_mem",
    ]

    all_results: Dict[str, Dict[str, List[Any]]] = {
        k: {"ious": [], "dices": [], "iou_per_view_list": [], "dice_per_view_list": []}
        for k in model_keys
    }

    name_map = {
        "orig_normal": "Orig mem + Normal",
        "orig_point": "Orig mem + Point",
        "orig_matte": "Orig mem + Matte",
        "angular_normal": "Geo-aware mem + Normal",
        "angular_point": "Geo-aware mem + Point",
        "angular_matte": "Geo-aware mem + Matte",
        "dual_orig_mem": "Dual (Normal+Point) + Orig mem",
        "dual_ang_mem": "Dual (Normal+Point) + Geo-aware mem",
    }

    # CSV log for per-mesh metrics
    log_path = logging_dir / "ablation_memory_log.csv"

    header = [
        "mesh_idx",
        "mesh_id",
        "model_key",
        "model_name",
        "modality",
        "mean_iou",
        "mean_dice",
    ] + [f"iou_view_{v}" for v in range(12)] + [f"dice_view_{v}" for v in range(12)]

    # If re-running, overwrite log
    with log_path.open("w", newline="") as f_log:
        writer = csv.writer(f_log)
        writer.writerow(header)

    # Use the same fixed 40-mesh eval split as modality ablation (seed=42 80/20 split)
    eval_idx_path = Path(
        "/home/mengnan/seg3d/src/seg3d/ablation_studies/eval_dataset_tinyobjaverse/eval_indices.txt"
    )
    if not eval_idx_path.exists():
        raise FileNotFoundError(f"Missing eval_indices.txt at {eval_idx_path}")
    eval_mesh_indices = [
        int(line.strip())
        for line in eval_idx_path.read_text().splitlines()
        if line.strip()
    ]
    logger.info(f"Evaluating ablation memory on {len(eval_mesh_indices)} validation meshes.")

    for mesh_idx in tqdm(eval_mesh_indices, desc="Evaluating meshes"):
        try:
            sample = dataset[mesh_idx]
            mesh_id = sample.get("mesh_id", None)
            gt_instance_masks = sample["gt_instance_masks"].to(
                device, non_blocking=True
            )  # (T,H,W)
            view_dirs = sample["view_dirs"].to(device, non_blocking=True)  # (T,3)

            has_matte = "matte_imgs" in sample and sample["matte_imgs"] is not None

            # Normal
            normal_imgs = sample["normal_imgs"].to(device, non_blocking=True)
            preds_orig_normal = _predict_from_view0_all_frames(
                predictors["orig_normal"],
                images=normal_imgs,
                gt_instance_masks=gt_instance_masks,
                modality="normal",
                use_view_dirs=False,
            )
            metrics = evaluate_predictions_fast(
                preds_orig_normal, gt_instance_masks, device=device
            )
            all_results["orig_normal"]["ious"].append(metrics["mean_iou"])
            all_results["orig_normal"]["dices"].append(metrics["mean_dice"])
            all_results["orig_normal"]["iou_per_view_list"].append(
                metrics["iou_per_view"]
            )
            all_results["orig_normal"]["dice_per_view_list"].append(
                metrics["dice_per_view"]
            )

            preds_ang_normal = _predict_from_view0_all_frames(
                predictors["angular_normal"],
                images=normal_imgs,
                gt_instance_masks=gt_instance_masks,
                modality="normal",
                view_dirs=view_dirs,
                use_view_dirs=True,
            )
            metrics_ang = evaluate_predictions_fast(
                preds_ang_normal, gt_instance_masks, device=device
            )
            all_results["angular_normal"]["ious"].append(metrics_ang["mean_iou"])
            all_results["angular_normal"]["dices"].append(metrics_ang["mean_dice"])
            all_results["angular_normal"]["iou_per_view_list"].append(
                metrics_ang["iou_per_view"]
            )
            all_results["angular_normal"]["dice_per_view_list"].append(
                metrics_ang["dice_per_view"]
            )

            # Point map
            point_imgs = sample["point_imgs"].to(device, non_blocking=True)
            preds_orig_point = _predict_from_view0_all_frames(
                predictors["orig_point"],
                images=point_imgs,
                gt_instance_masks=gt_instance_masks,
                modality="point",
                use_view_dirs=False,
            )
            metrics = evaluate_predictions_fast(
                preds_orig_point, gt_instance_masks, device=device
            )
            all_results["orig_point"]["ious"].append(metrics["mean_iou"])
            all_results["orig_point"]["dices"].append(metrics["mean_dice"])
            all_results["orig_point"]["iou_per_view_list"].append(
                metrics["iou_per_view"]
            )
            all_results["orig_point"]["dice_per_view_list"].append(
                metrics["dice_per_view"]
            )

            preds_ang_point = _predict_from_view0_all_frames(
                predictors["angular_point"],
                images=point_imgs,
                gt_instance_masks=gt_instance_masks,
                modality="point",
                view_dirs=view_dirs,
                use_view_dirs=True,
            )
            metrics_ang = evaluate_predictions_fast(
                preds_ang_point, gt_instance_masks, device=device
            )
            all_results["angular_point"]["ious"].append(metrics_ang["mean_iou"])
            all_results["angular_point"]["dices"].append(metrics_ang["mean_dice"])
            all_results["angular_point"]["iou_per_view_list"].append(
                metrics_ang["iou_per_view"]
            )
            all_results["angular_point"]["dice_per_view_list"].append(
                metrics_ang["dice_per_view"]
            )

            # Matte (if available)
            if has_matte:
                matte_imgs = sample["matte_imgs"].to(device, non_blocking=True)

                preds_orig_matte = _predict_from_view0_all_frames(
                    predictors["orig_matte"],
                    images=matte_imgs,
                    gt_instance_masks=gt_instance_masks,
                    modality="matte",
                    use_view_dirs=False,
                )
                metrics = evaluate_predictions_fast(
                    preds_orig_matte, gt_instance_masks, device=device
                )
                all_results["orig_matte"]["ious"].append(metrics["mean_iou"])
                all_results["orig_matte"]["dices"].append(metrics["mean_dice"])
                all_results["orig_matte"]["iou_per_view_list"].append(
                    metrics["iou_per_view"]
                )
                all_results["orig_matte"]["dice_per_view_list"].append(
                    metrics["dice_per_view"]
                )

                preds_ang_matte = _predict_from_view0_all_frames(
                    predictors["angular_matte"],
                    images=matte_imgs,
                    gt_instance_masks=gt_instance_masks,
                    modality="matte",
                    view_dirs=view_dirs,
                    use_view_dirs=True,
                )
                metrics_ang = evaluate_predictions_fast(
                    preds_ang_matte, gt_instance_masks, device=device
                )
                all_results["angular_matte"]["ious"].append(metrics_ang["mean_iou"])
                all_results["angular_matte"]["dices"].append(metrics_ang["mean_dice"])
                all_results["angular_matte"]["iou_per_view_list"].append(
                    metrics_ang["iou_per_view"]
                )
                all_results["angular_matte"]["dice_per_view_list"].append(
                    metrics_ang["dice_per_view"]
                )

            # Dual-modality (normal + point)
            preds_dual_orig = _predict_from_view0_all_frames_dual_orig_mem(
                predictors["dual_orig_mem"],
                normal_imgs=normal_imgs,
                point_imgs=point_imgs,
                gt_instance_masks=gt_instance_masks,
            )
            metrics_dual_orig = evaluate_predictions_fast(
                preds_dual_orig, gt_instance_masks, device=device
            )
            all_results["dual_orig_mem"]["ious"].append(metrics_dual_orig["mean_iou"])
            all_results["dual_orig_mem"]["dices"].append(metrics_dual_orig["mean_dice"])
            all_results["dual_orig_mem"]["iou_per_view_list"].append(
                metrics_dual_orig["iou_per_view"]
            )
            all_results["dual_orig_mem"]["dice_per_view_list"].append(
                metrics_dual_orig["dice_per_view"]
            )

            preds_dual_ang = _predict_from_view0_all_frames_dual_ang_mem(
                predictors["dual_ang_mem"],
                normal_imgs=normal_imgs,
                point_imgs=point_imgs,
                gt_instance_masks=gt_instance_masks,
                view_dirs=view_dirs,
            )
            metrics_dual_ang = evaluate_predictions_fast(
                preds_dual_ang, gt_instance_masks, device=device
            )
            all_results["dual_ang_mem"]["ious"].append(metrics_dual_ang["mean_iou"])
            all_results["dual_ang_mem"]["dices"].append(metrics_dual_ang["mean_dice"])
            all_results["dual_ang_mem"]["iou_per_view_list"].append(
                metrics_dual_ang["iou_per_view"]
            )
            all_results["dual_ang_mem"]["dice_per_view_list"].append(
                metrics_dual_ang["dice_per_view"]
            )

            # Append per-mesh rows to CSV log
            with log_path.open("a", newline="") as f_log:
                writer = csv.writer(f_log)
                for key in model_keys:
                    # Skip matte models if this mesh has no matte
                    if not has_matte and key in ("orig_matte", "angular_matte"):
                        continue
                    metrics_list = all_results[key]
                    # metrics just computed for this mesh are last entries
                    mean_iou = metrics_list["ious"][-1]
                    mean_dice = metrics_list["dices"][-1]
                    iou_per_view = metrics_list["iou_per_view_list"][-1]
                    dice_per_view = metrics_list["dice_per_view_list"][-1]
                    # Modalities from key suffix
                    if key in ("dual_orig_mem", "dual_ang_mem"):
                        modality = "dual"
                    elif "normal" in key:
                        modality = "normal"
                    elif "point" in key:
                        modality = "point"
                    else:
                        modality = "matte"
                    row = [
                        mesh_idx,
                        mesh_id,
                        key,
                        name_map[key],
                        modality,
                        mean_iou,
                        mean_dice,
                    ]
                    # pad/truncate per-view lists to 12 for CSV
                    max_views = 12
                    iou_pad = list(iou_per_view)[:max_views]
                    dice_pad = list(dice_per_view)[:max_views]
                    if len(iou_pad) < max_views:
                        iou_pad += [0.0] * (max_views - len(iou_pad))
                    if len(dice_pad) < max_views:
                        dice_pad += [0.0] * (max_views - len(dice_pad))
                    writer.writerow(row + iou_pad + dice_pad)

        except Exception as e:
            logger.error(f"Error evaluating mesh {mesh_idx}: {str(e)}")
            continue

    # -------------------------
    # Summarize and plot
    # -------------------------
    summary_rows: List[Dict[str, Any]] = []
    per_view_rows: List[Dict[str, Any]] = []

    for key in model_keys:
        ious = all_results[key]["ious"]
        dices = all_results[key]["dices"]
        if len(ious) == 0:
            continue
        summary_rows.append(
            {
                "Model": name_map[key],
                "Mean IoU": float(np.mean(ious) * 100.0),
                "Std IoU": float(np.std(ious) * 100.0),
                "Mean Dice": float(np.mean(dices) * 100.0),
                "Std Dice": float(np.std(dices) * 100.0),
                "Num Meshes": len(ious),
            }
        )

    import pandas as pd

    df_summary = pd.DataFrame(summary_rows)

    # Aggregate per-view metrics (mean/std over meshes for each view)
    for key in model_keys:
        iou_view_lists = all_results[key].get("iou_per_view_list", [])
        dice_view_lists = all_results[key].get("dice_per_view_list", [])
        if not iou_view_lists:
            continue
        iou_mat = np.array(iou_view_lists)  # (num_meshes, T)
        dice_mat = np.array(dice_view_lists)
        mean_iou_per_view = iou_mat.mean(axis=0) * 100.0
        std_iou_per_view = iou_mat.std(axis=0) * 100.0
        mean_dice_per_view = dice_mat.mean(axis=0) * 100.0
        std_dice_per_view = dice_mat.std(axis=0) * 100.0
        T_views = iou_mat.shape[1]
        for v in range(T_views):
            per_view_rows.append(
                {
                    "Model": name_map[key],
                    "View": v,
                    "Mean IoU": mean_iou_per_view[v],
                    "Std IoU": std_iou_per_view[v],
                    "Mean Dice": mean_dice_per_view[v],
                    "Std Dice": std_dice_per_view[v],
                }
            )

    df_per_view = pd.DataFrame(per_view_rows)

    if not df_summary.empty:
        df_summary = df_summary.sort_values("Mean IoU", ascending=False)

        logger.info("\n" + "=" * 80)
        logger.info("ABLATION MEMORY: QUANTITATIVE RESULTS")
        logger.info("=" * 80)
        logger.info(f"\nEvaluated on {len(eval_mesh_indices)} meshes")
        logger.info("\nSummary Statistics:\n" + df_summary.to_string(index=False))

        summary_csv_path = logging_dir / "ablation_memory_summary.csv"
        df_summary.to_csv(summary_csv_path, index=False)
        logger.info(f"Saved summary CSV to: {summary_csv_path}")

        if not df_per_view.empty:
            per_view_csv_path = logging_dir / "ablation_memory_per_view.csv"
            df_per_view.to_csv(per_view_csv_path, index=False)
            logger.info(f"Saved per-view CSV to: {per_view_csv_path}")

        # Overall IoU / Dice bar plots
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        model_names = df_summary["Model"].values
        mean_ious = df_summary["Mean IoU"].values
        std_ious = df_summary["Std IoU"].values
        mean_dices = df_summary["Mean Dice"].values
        std_dices = df_summary["Std Dice"].values

        axes[0].barh(model_names, mean_ious, xerr=std_ious, capsize=5)
        axes[0].set_xlabel("Mean IoU (%)", fontsize=12)
        axes[0].set_title(
            "Ablation: Intersection over Union (IoU)", fontsize=14, fontweight="bold"
        )
        axes[0].grid(axis="x", alpha=0.3)
        axes[0].set_xlim(0, 100)

        axes[1].barh(model_names, mean_dices, xerr=std_dices, capsize=5)
        axes[1].set_xlabel("Mean Dice Coefficient (%)", fontsize=12)
        axes[1].set_title(
            "Ablation: Dice Coefficient", fontsize=14, fontweight="bold"
        )
        axes[1].grid(axis="x", alpha=0.3)
        axes[1].set_xlim(0, 100)

        plt.tight_layout()

        plot_path = figures_dir / "ablation_memory_metrics.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved ablation comparison plot to: {plot_path}")
        plt.show()

        # Per-view IoU trends across views (memory-focused figure)
        if not df_per_view.empty:
            fig, axes = plt.subplots(1, 3, figsize=(18, 4), sharey=True)
            modality_pairs = [
                ("Normal", "Orig mem + Normal", "Geo-aware mem + Normal"),
                ("Point", "Orig mem + Point", "Geo-aware mem + Point"),
                ("Matte", "Orig mem + Matte", "Geo-aware mem + Matte"),
            ]
            for ax, (mod_name, orig_label, ang_label) in zip(axes, modality_pairs):
                df_mod = df_per_view[df_per_view["Model"].isin([orig_label, ang_label])]
                if df_mod.empty:
                    ax.set_visible(False)
                    continue
                for model_name, style in [
                    (orig_label, {"color": "tab:blue", "linestyle": "-"}),
                    (ang_label, {"color": "tab:orange", "linestyle": "--"}),
                ]:
                    df_m = df_mod[df_mod["Model"] == model_name].sort_values("View")
                    ax.plot(
                        df_m["View"].values,
                        df_m["Mean IoU"].values,
                        label=model_name,
                        **style,
                    )
                    ax.fill_between(
                        df_m["View"].values,
                        df_m["Mean IoU"].values - df_m["Std IoU"].values,
                        df_m["Mean IoU"].values + df_m["Std IoU"].values,
                        alpha=0.2,
                        color=style["color"],
                    )
                ax.set_xlabel("View index")
                ax.set_title(f"Per-view IoU ({mod_name})")
                ax.grid(alpha=0.3)
            axes[0].set_ylabel("Mean IoU (%)")
            handles, labels = axes[0].get_legend_handles_labels()
            fig.legend(handles, labels, loc="upper center", ncol=2)
            plt.tight_layout(rect=(0, 0, 1, 0.9))

            per_view_plot_path = figures_dir / "ablation_memory_per_view_iou.png"
            plt.savefig(per_view_plot_path, dpi=150, bbox_inches="tight")
            logger.info(f"Saved per-view IoU plot to: {per_view_plot_path}")
            plt.show()

        logger.info("\n" + "=" * 80)
        logger.info(f"Best (by Mean IoU): {df_summary.iloc[0]['Model']}")
        logger.info(
            f"  IoU: {df_summary.iloc[0]['Mean IoU']:.2f}% ± {df_summary.iloc[0]['Std IoU']:.2f}%"
        )
        logger.info(
            f"  Dice: {df_summary.iloc[0]['Mean Dice']:.2f}% ± {df_summary.iloc[0]['Std Dice']:.2f}%"
        )
        logger.info("=" * 80)
    else:
        logger.info("No results to summarize (df_summary is empty).")


if __name__ == "__main__":
    run_ablation()

