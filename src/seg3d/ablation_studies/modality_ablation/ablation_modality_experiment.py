"""
Modality ablation experiment script (Python version of `ablation_modality.ipynb`).

This script:
- Builds 4 predictors that all use your angular memory:
  - 3 single-modality angular-memory models (normal / point / matte)
  - 1 finetuned multimodal model (normal + point, with angular memory)
- Evaluates them on the first 40 meshes in `MultiViewSAM2Dataset`
  using GT instance masks from view 0 as mask prompts.
- Computes:
  - Per-mesh mean IoU / Dice (+ std over views)
  - Overall mean / std IoU & Dice per model
  - Per-view mean / std IoU across meshes for each model
- Saves:
  - `ablation_modality_summary.csv`   (overall statistics)
  - `ablation_modality_detailed.csv`  (per-mesh statistics)
  - `ablation_modality_metrics.png`   (overall IoU/Dice bar plots)
  - `ablation_modality_per_view_iou.png` (per-view IoU curves)
under:
  `<PROJECT_ROOT>/visualization_results/evaluation_metrics/`
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import logging
import colorsys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


def setup_paths_and_logging() -> Dict[str, Any]:
    """
    Set up project paths, sys.path, and logging.

    This file lives at:
      <PROJECT_ROOT>/src/seg3d/ablation_studies/ablation_modality_experiment.py
    so PROJECT_ROOT is three levels up from this file's directory.
    """
    import sys

    # Use an absolute repo root to avoid any ambiguity in where outputs land.
    # (This machine's repo lives at /home/mengnan/seg3d.)
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
    logger = logging.getLogger("ablation_modality")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Modality-ablation output roots (absolute paths)
    modality_root = Path(
        "/home/mengnan/seg3d/src/seg3d/ablation_studies/modality_ablation"
    )
    figures_dir = modality_root / "figures"
    logging_dir = modality_root / "logging"
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
    Build the 4 predictors used in the modality ablation:
      - ang_mem_model_normal / point / matte
      - finetuned multimodal (normal + point, angular memory)
    """
    from seg3d.ablation_studies.models.single_modality_predictor import (
        SAM2AngularMemSingleModalityPredictor as AngularMemPredictor,
    )
    from seg3d.models.sam2_mesh_predictor import SAM2MeshPredictor

    # Matches the notebook CONFIG_FILE_ORIG / CHECKPOINT_PATH_ORIG
    config_file_orig = "configs/sam2.1/sam2.1_hiera_l.yaml"
    ckpt_path_orig = project_root / "sam2" / "checkpoints" / "sam2.1_hiera_large.pt"

    logger.info("Building angular-memory single-modality predictors...")
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
    logger.info("Built 3 angular-memory models (normal / point / matte)")

    # Finetuned multimodal model (normal + point + angular memory)
    finetuned_checkpoint = project_root / "checkpoints" / "best.pt"
    finetuned_model_config = project_root / "configs" / "mm_sam2.1_hiera_l.yaml"

    logger.info("Building finetuned multimodal predictor...")
    finetuned_model = SAM2MeshPredictor.from_pretrained(
        config_file=str(finetuned_model_config),
        ckpt_path=str(finetuned_checkpoint),
        device=device,
    )
    logger.info("Built finetuned multimodal predictor")

    return {
        "angular_normal": ang_mem_model_normal,
        "angular_point": ang_mem_model_point,
        "angular_matte": ang_mem_model_matte,
        "finetuned_mm": finetuned_model,
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


def get_color_palette(num_colors: int) -> List[tuple[float, float, float]]:
    """Generate distinct colors for mask overlays (matches visualization notebook style)."""
    colors: List[tuple[float, float, float]] = []
    for i in range(max(num_colors, 1)):
        hue = i / max(num_colors, 1)
        saturation = 0.7 + (i % 3) * 0.1
        lightness = 0.5 + (i % 2) * 0.2
        rgb = colorsys.hls_to_rgb(hue, lightness, saturation)
        colors.append(rgb)
    return colors


def overlay_masks_on_matte(
    matte: torch.Tensor | np.ndarray,
    mask: torch.Tensor | np.ndarray,
    colors: List[tuple[float, float, float]] | None = None,
    alpha: float = 0.5,
    threshold: float = 0.0,
) -> np.ndarray:
    """
    Overlay mask(s) on matte image with different colors for each object.

    This closely mirrors the helper used in `compare_models_visualization.ipynb`.
    """
    # Convert matte to numpy if needed
    if isinstance(matte, torch.Tensor):
        matte_np = matte.permute(1, 2, 0).detach().cpu().numpy()
    else:
        matte_np = matte.copy()

    matte_np = np.clip(matte_np, 0.0, 1.0)

    # Convert mask to numpy and handle shape
    if isinstance(mask, torch.Tensor):
        mask_np = mask.detach().cpu().numpy()
    else:
        mask_np = mask.copy()

    # Handle mask shape: (N, H, W) or (H, W)
    if mask_np.ndim == 2:
        mask_np = mask_np[np.newaxis, :, :]
    elif mask_np.ndim == 3:
        pass  # Already (N, H, W)
    else:
        raise ValueError(f"Unexpected mask shape for overlay: {mask_np.shape}")

    # Apply sigmoid and threshold (stable)
    mask_np = 1.0 / (1.0 + np.exp(-np.clip(mask_np, -10.0, 10.0)))
    mask_np = (mask_np > threshold).astype(np.float32)

    num_objects = mask_np.shape[0]

    # Generate colors if not provided
    if colors is None:
        colors = get_color_palette(max(num_objects, 1))

    overlay = matte_np.copy()

    # Overlay each object's mask with a different color
    for obj_idx in range(num_objects):
        obj_mask = mask_np[obj_idx]  # (H, W)
        obj_mask_binary = obj_mask > 0.5

        if np.any(obj_mask_binary):
            color = np.array(colors[obj_idx % len(colors)])
            overlay[obj_mask_binary] = (
                (1.0 - alpha) * overlay[obj_mask_binary] + alpha * color
            )

    return overlay


def visualize_all_views_grid_modality(
    mattes: torch.Tensor,
    preds_ang_matte: List[torch.Tensor | None] | None,
    preds_ang_point: List[torch.Tensor | None] | None,
    preds_ang_normal: List[torch.Tensor | None] | None,
    preds_finetuned: List[torch.Tensor | None] | None,
    save_path: Path,
    mesh_id: str | None,
    logger: logging.Logger,
    threshold: float = 0.5,
) -> None:
    """
    Visualize all views in a grid:
      rows   = views
      columns= 4 models (Angular + Matte / Point / Normal, Finetuned MM)

    This mirrors the layout from `compare_models_visualization.ipynb`,
    but using your angular-memory models.
    """
    if isinstance(mattes, torch.Tensor):
        num_views = int(mattes.shape[0])
    else:
        num_views = len(mattes)

    num_models = 4

    model_data: List[tuple[List[torch.Tensor | None] | None, str]] = [
        (preds_ang_matte, "Geo-aware mem + Matte"),
        (preds_ang_point, "Geo-aware mem + Point"),
        (preds_ang_normal, "Geo-aware mem + Normal"),
        (preds_finetuned, "Finetuned Multimodal (MM + angular mem)"),
    ]

    fig, axes = plt.subplots(
        num_views, num_models, figsize=(5 * num_models, 5 * num_views)
    )
    if num_views == 1:
        axes = axes.reshape(1, -1)
    elif num_models == 1:
        axes = axes.reshape(-1, 1)

    colors = get_color_palette(32)

    for view_idx in range(num_views):
        if isinstance(mattes, torch.Tensor):
            matte = mattes[view_idx]
        else:
            matte = mattes[view_idx]

        for model_idx, (pred_masks, model_title) in enumerate(model_data):
            ax = axes[view_idx, model_idx]

            masks = None
            if pred_masks is not None and view_idx < len(pred_masks):
                masks = pred_masks[view_idx]

            if masks is not None:
                overlay = overlay_masks_on_matte(
                    matte, masks, colors=colors, alpha=0.5, threshold=threshold
                )
            else:
                # Show matte only if no prediction
                if isinstance(matte, torch.Tensor):
                    overlay = (
                        matte.permute(1, 2, 0).detach().cpu().numpy()
                    )
                else:
                    overlay = matte.copy()
                overlay = np.clip(overlay, 0.0, 1.0)

            ax.imshow(overlay)
            ax.axis("off")

            # Column titles on top row
            if view_idx == 0:
                ax.set_title(model_title, fontsize=10, fontweight="bold")
            # Row labels on first column (horizontal text)
            if model_idx == 0:
                ax.text(
                    -0.05,
                    0.5,
                    f"View {view_idx}",
                    transform=ax.transAxes,
                    rotation=0,
                    va="center",
                    ha="right",
                    fontsize=10,
                    fontweight="bold",
                )

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    logger.info(
        f"Saved modality comparison visualization for mesh "
        f"{mesh_id if mesh_id is not None else '?'} to: {save_path}"
    )
    plt.close(fig)


def _predict_from_view0_all_frames_single_modality(
    predictor,
    images: torch.Tensor,
    gt_instance_masks: torch.Tensor,
    modality: str,
    view_dirs: torch.Tensor | None,
    use_view_dirs: bool,
) -> List[torch.Tensor | None]:
    """
    Wrapper for angular-memory single-modality predictors (normal / point / matte).

    Uses all GT instance masks from view 0 as prompts, then propagates to all frames.
    """
    if use_view_dirs and view_dirs is not None:
        state = predictor.init_state(images=images, view_dirs=view_dirs, modality=modality)
    else:
        state = predictor.init_state(images=images, modality=modality)

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


def _predict_from_view0_all_frames_finetuned(
    predictor,
    normal_imgs: torch.Tensor,
    point_imgs: torch.Tensor,
    gt_instance_masks: torch.Tensor,
    view_dirs: torch.Tensor,
) -> List[torch.Tensor | None]:
    """
    Same prompting strategy, but for the multimodal finetuned predictor.
    """
    from seg3d.ablation_studies.utils import (
        predict_from_view0_all_frames_finetuned_gt,
    )

    return predict_from_view0_all_frames_finetuned_gt(
        predictor=predictor,
        normal_imgs=normal_imgs,
        point_imgs=point_imgs,
        gt_instance_masks=gt_instance_masks,
        view_dirs=view_dirs,
    )


def run_modality_ablation() -> None:
    ctx = setup_paths_and_logging()
    project_root: Path = ctx["PROJECT_ROOT"]
    logger: logging.Logger = ctx["logger"]
    device: torch.device = ctx["device"]
    figures_dir: Path = ctx["FIGURES_DIR"]
    logging_dir: Path = ctx["LOGGING_DIR"]

    # Dataset
    num_views = 12
    dataset = build_dataset(project_root, num_views=num_views)
    logger.info(f"Dataset size: {len(dataset)} meshes")

    # Predictors
    predictors = build_predictors(device=device, project_root=project_root, logger=logger)

    model_keys = [
        "angular_normal",
        "angular_point",
        "angular_matte",
        "finetuned_mm",
    ]
    name_map = {
        "angular_normal": "Geo-aware mem + Normal",
        "angular_point": "Geo-aware mem + Point",
        "angular_matte": "Geo-aware mem + Matte",
        "finetuned_mm": "Finetuned Multimodal (MM + Geo-aware mem)",
    }

    all_results: Dict[str, Dict[str, Any]] = {
        k: {"ious": [], "dices": [], "per_mesh": []} for k in model_keys
    }

    # ------------------------------------------------------------------
    # Use the SAME 40-mesh eval split as training (80/20 with seed=42)
    # ------------------------------------------------------------------
    import numpy as np

    # Load fixed eval indices (matching training val split)
    eval_idx_path = (
        project_root
        / "src"
        / "seg3d"
        / "ablation_studies"
        / "eval_dataset_tinyobjaverse"
        / "eval_indices.txt"
    )
    if not eval_idx_path.exists():
        raise FileNotFoundError(f"Missing eval_indices.txt at {eval_idx_path}")

    eval_mesh_indices = [
        int(line.strip())
        for line in eval_idx_path.read_text().splitlines()
        if line.strip()
    ]

    logger.info(
        f"Evaluating modality ablation on {len(eval_mesh_indices)} validation meshes "
        f"(seed=42 80/20 split)."
    )

    # Directory to cache finetuned GT-prompt predictions for reuse by
    # prompt-source ablation (avoids running finetuned model twice).
    prompt_cache_dir = Path("/home/mengnan/seg3d/src/seg3d/ablation_studies/ablation_prompt_cache")
    prompt_cache_dir.mkdir(parents=True, exist_ok=True)
    save_qualitative_overlays = False

    for mesh_idx in tqdm(eval_mesh_indices, desc="Evaluating meshes"):
        try:
            sample = dataset[mesh_idx]

            gt_instance_masks = sample["gt_instance_masks"].to(device, non_blocking=True)
            view_dirs = sample["view_dirs"].to(device, non_blocking=True)

            normal_imgs = sample.get("normal_imgs", None)
            point_imgs = sample.get("point_imgs", None)
            matte_imgs = sample.get("matte_imgs", None)

            if normal_imgs is not None:
                normal_imgs = normal_imgs.to(device, non_blocking=True)
            if point_imgs is not None:
                point_imgs = point_imgs.to(device, non_blocking=True)
            if matte_imgs is not None:
                matte_imgs = matte_imgs.to(device, non_blocking=True)

            # Placeholders for predicted masks so we can reuse them for visualization
            preds_ang_normal: List[torch.Tensor | None] | None = None
            preds_ang_point: List[torch.Tensor | None] | None = None
            preds_ang_matte: List[torch.Tensor | None] | None = None
            preds_finetuned: List[torch.Tensor | None] | None = None

            # Angular memory models (single-modality)
            if normal_imgs is not None:
                preds_ang_normal = _predict_from_view0_all_frames_single_modality(
                    predictors["angular_normal"],
                    images=normal_imgs,
                    gt_instance_masks=gt_instance_masks,
                    modality="normal",
                    view_dirs=view_dirs,
                    use_view_dirs=True,
                )
                metrics = evaluate_predictions_fast(
                    preds_ang_normal, gt_instance_masks, device=device
                )
                all_results["angular_normal"]["ious"].append(metrics["mean_iou"])
                all_results["angular_normal"]["dices"].append(metrics["mean_dice"])
                all_results["angular_normal"]["per_mesh"].append(
                    {
                        "mesh_idx": mesh_idx,
                        "mesh_id": sample.get("mesh_id", None),
                        "metrics": metrics,
                    }
                )

            if point_imgs is not None:
                preds_ang_point = _predict_from_view0_all_frames_single_modality(
                    predictors["angular_point"],
                    images=point_imgs,
                    gt_instance_masks=gt_instance_masks,
                    modality="point",
                    view_dirs=view_dirs,
                    use_view_dirs=True,
                )
                metrics = evaluate_predictions_fast(
                    preds_ang_point, gt_instance_masks, device=device
                )
                all_results["angular_point"]["ious"].append(metrics["mean_iou"])
                all_results["angular_point"]["dices"].append(metrics["mean_dice"])
                all_results["angular_point"]["per_mesh"].append(
                    {
                        "mesh_idx": mesh_idx,
                        "mesh_id": sample.get("mesh_id", None),
                        "metrics": metrics,
                    }
                )

            if matte_imgs is not None:
                preds_ang_matte = _predict_from_view0_all_frames_single_modality(
                    predictors["angular_matte"],
                    images=matte_imgs,
                    gt_instance_masks=gt_instance_masks,
                    modality="matte",
                    view_dirs=view_dirs,
                    use_view_dirs=True,
                )
                metrics = evaluate_predictions_fast(
                    preds_ang_matte, gt_instance_masks, device=device
                )
                all_results["angular_matte"]["ious"].append(metrics["mean_iou"])
                all_results["angular_matte"]["dices"].append(metrics["mean_dice"])
                all_results["angular_matte"]["per_mesh"].append(
                    {
                        "mesh_idx": mesh_idx,
                        "mesh_id": sample.get("mesh_id", None),
                        "metrics": metrics,
                    }
                )

            # Finetuned multimodal model (normal + point)
            if (normal_imgs is not None) and (point_imgs is not None):
                preds_finetuned = _predict_from_view0_all_frames_finetuned(
                    predictors["finetuned_mm"],
                    normal_imgs=normal_imgs,
                    point_imgs=point_imgs,
                    gt_instance_masks=gt_instance_masks,
                    view_dirs=view_dirs,
                )
                metrics = evaluate_predictions_fast(
                    preds_finetuned, gt_instance_masks, device=device
                )
                all_results["finetuned_mm"]["ious"].append(metrics["mean_iou"])
                all_results["finetuned_mm"]["dices"].append(metrics["mean_dice"])
                all_results["finetuned_mm"]["per_mesh"].append(
                    {
                        "mesh_idx": mesh_idx,
                        "mesh_id": sample.get("mesh_id", None),
                        "metrics": metrics,
                    }
                )

                # Cache GT-prompt finetuned predictions + mattes for reuse
                cache_path = prompt_cache_dir / f"mesh_{mesh_idx:03d}.pt"
                try:
                    torch.save(
                        {
                            "mesh_idx": mesh_idx,
                            "mesh_id": sample.get("mesh_id", None),
                            "preds_finetuned": [
                                m.cpu() if m is not None else None
                                for m in preds_finetuned
                            ],
                            "matte_imgs": matte_imgs.cpu()
                            if matte_imgs is not None
                            else None,
                        },
                        cache_path,
                    )
                except Exception as e:  # pragma: no cover - defensive
                    logger.warning(
                        f"Failed to cache finetuned predictions for mesh {mesh_idx}: {e}"
                    )
            if save_qualitative_overlays:
                raise RuntimeError("Unreachable: qualitative overlays disabled.")

        except Exception as e:  # pragma: no cover - defensive logging
            logger.error(f"Error evaluating mesh {mesh_idx}: {str(e)}")
            continue

    # -------------------------
    # Summarize and plot
    # -------------------------
    import pandas as pd

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

    if not summary_rows:
        logger.info("No results to summarize (no successful meshes).")
        return

    df_summary = pd.DataFrame(summary_rows)
    df_summary = df_summary.sort_values("Mean IoU", ascending=False)

    logger.info("\n" + "=" * 80)
    logger.info("MODALITY ABLATION (ANGULAR MEMORY + FINETUNED): QUANTITATIVE RESULTS")
    logger.info("=" * 80)
    logger.info(f"\nEvaluated on {len(eval_mesh_indices)} meshes")
    logger.info("\nSummary Statistics:\n" + df_summary.to_string(index=False))

    # Save summary CSV into local logging directory
    summary_path = logging_dir / "ablation_modality_summary.csv"
    df_summary.to_csv(summary_path, index=False)
    logger.info(f"Saved summary to: {summary_path}")

    # Save detailed per-mesh CSV
    detailed_rows: List[Dict[str, Any]] = []
    for key in model_keys:
        title = name_map[key]
        for mesh_data in all_results[key]["per_mesh"]:
            detailed_rows.append(
                {
                    "Model": title,
                    "Mesh ID": mesh_data["mesh_id"],
                    "Mesh Index": mesh_data["mesh_idx"],
                    "Mean IoU": mesh_data["metrics"]["mean_iou"] * 100.0,
                    "Mean Dice": mesh_data["metrics"]["mean_dice"] * 100.0,
                    "Std IoU": mesh_data["metrics"]["std_iou"] * 100.0,
                    "Std Dice": mesh_data["metrics"]["std_dice"] * 100.0,
                }
            )

    if detailed_rows:
        df_detailed = pd.DataFrame(detailed_rows)
        detailed_path = logging_dir / "ablation_modality_detailed.csv"
        df_detailed.to_csv(detailed_path, index=False)
        logger.info(f"Saved detailed results to: {detailed_path}")

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
        "Modality Ablation: Intersection over Union (IoU)",
        fontsize=14,
        fontweight="bold",
    )
    axes[0].grid(axis="x", alpha=0.3)
    axes[0].set_xlim(0, 100)

    axes[1].barh(model_names, mean_dices, xerr=std_dices, capsize=5)
    axes[1].set_xlabel("Mean Dice Coefficient (%)", fontsize=12)
    axes[1].set_title(
        "Modality Ablation: Dice Coefficient",
        fontsize=14,
        fontweight="bold",
    )
    axes[1].grid(axis="x", alpha=0.3)
    axes[1].set_xlim(0, 100)

    plt.tight_layout()
    plot_path = figures_dir / "ablation_modality_metrics.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    logger.info(f"Saved modality ablation plot to: {plot_path}")
    plt.close(fig)

    # Per-view IoU curves (propagation stability)
    for key in model_keys:
        title = name_map[key]
        per_mesh_list = all_results[key]["per_mesh"]
        if not per_mesh_list:
            continue
        iou_view_lists = [m["metrics"]["iou_per_view"] for m in per_mesh_list]
        if not iou_view_lists:
            continue
        iou_mat = np.array(iou_view_lists) * 100.0  # (num_meshes, T) in percent
        mean_iou_per_view = iou_mat.mean(axis=0)
        std_iou_per_view = iou_mat.std(axis=0)
        T_views = iou_mat.shape[1]
        for v in range(T_views):
            per_view_rows.append(
                {
                    "Model": title,
                    "View": v,
                    "Mean IoU": mean_iou_per_view[v],
                    "Std IoU": std_iou_per_view[v],
                }
            )

    if per_view_rows:
        df_per_view = pd.DataFrame(per_view_rows)

        # Save per-view aggregates so we can regenerate/modify the plot from CSV later
        per_view_csv_path = logging_dir / "ablation_modality_per_view.csv"
        df_per_view.to_csv(per_view_csv_path, index=False)
        logger.info(f"Saved per-view metrics to: {per_view_csv_path}")

        fig, ax = plt.subplots(figsize=(10, 4))

        color_map = {
            "Geo-aware mem + Normal": "tab:blue",
            "Geo-aware mem + Point": "tab:orange",
            "Geo-aware mem + Matte": "tab:green",
            "Finetuned Multimodal (MM + Geo-aware mem)": "tab:red",
        }

        # Use same ordering as summary bar chart
        for model_name in df_summary["Model"].values:
            df_m = df_per_view[df_per_view["Model"] == model_name].sort_values("View")
            if df_m.empty:
                continue
            color = color_map.get(model_name, None)
            ax.plot(
                df_m["View"].values,
                df_m["Mean IoU"].values,
                label=model_name,
                linewidth=2,
                color=color,
            )
            ax.fill_between(
                df_m["View"].values,
                df_m["Mean IoU"].values - df_m["Std IoU"].values,
                df_m["Mean IoU"].values + df_m["Std IoU"].values,
                alpha=0.2,
                color=color,
            )

        ax.set_xlabel("View index")
        ax.set_ylabel("Mean IoU (%)")
        ax.grid(alpha=0.3)
        ax.set_xlim(0, num_views - 1)

        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(loc="lower left")

        plt.tight_layout()
        per_view_plot_path = figures_dir / "ablation_modality_per_view_iou.png"
        plt.savefig(per_view_plot_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved per-view IoU plot to: {per_view_plot_path}")
        plt.close(fig)

    logger.info("\n" + "=" * 80)
    logger.info(f"Best Model (by Mean IoU): {df_summary.iloc[0]['Model']}")
    logger.info(
        f"  IoU: {df_summary.iloc[0]['Mean IoU']:.2f}% ± {df_summary.iloc[0]['Std IoU']:.2f}%"
    )
    logger.info(
        f"  Dice: {df_summary.iloc[0]['Mean Dice']:.2f}% ± {df_summary.iloc[0]['Std Dice']:.2f}%"
    )
    logger.info("=" * 80)


if __name__ == "__main__":
    run_modality_ablation()

