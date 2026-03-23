"""
Prompt source ablation for the finetuned multimodal SAM2 model.

We compare:
  - The finetuned dual-modality model with geo-aware memory, prompted with GT
    instance masks from view 0
  - The SAME model when prompts come from an automatic mask generator (AMG)
    run on the first-view normal image

This script:
  - Reuses the same `MultiViewSAM2Dataset` and finetuned predictor as
    `ablation_modality_experiment.py`.
  - Builds a `MaskProposalGenerator` with the AMG config taken from
    `configs/mesh_seg_coseg_new.yaml` (`sam.amg_sam` block).
  - For each mesh in the fixed 40-mesh eval split (`eval_indices.txt`), runs:
      * GT-prompted propagation (view-0 GT masks)
      * AMG on view-0 normal image to get a stack of proposal masks
      * Adds each proposal as a prompt at frame 0 to the finetuned predictor
      * Propagates to all frames and evaluates IoU / Dice vs GT instance masks
  - Saves summary + per-mesh + per-view CSVs and plots:
      * `ablation_prompt_source_summary.csv`
      * `ablation_prompt_source_detailed.csv`
      * `ablation_prompt_source_metrics.png`
      * `ablation_prompt_source_per_view.csv`
      * `ablation_prompt_source_per_view_iou.png`

Usage (from repo root):

  python src/seg3d/ablation_studies/ablation_prompt_source_experiment.py

You can then compare these AMG-prompt results to the GT-prompt results
from `ablation_modality_experiment.py` in your thesis.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import logging

import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm
import matplotlib.pyplot as plt


def setup_paths_and_logging() -> Dict[str, Any]:
    """
    Same path / logging setup as `ablation_modality_experiment`.
    """
    import sys

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
    logger = logging.getLogger("ablation_prompt_source")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    ablation_root = Path(
        "/home/mengnan/seg3d/src/seg3d/ablation_studies/amg_vs_gt_prompt_ablation"
    )
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


def build_finetuned_predictor(
    device: torch.device,
    project_root: Path,
    logger: logging.Logger,
):
    """
    Build only the finetuned multimodal predictor (normal + point + angular memory).
    """
    from seg3d.models.sam2_mesh_predictor import SAM2MeshPredictor

    finetuned_checkpoint = project_root / "checkpoints" / "best.pt"
    finetuned_model_config = project_root / "configs" / "mm_sam2.1_hiera_l.yaml"

    logger.info(
        "Building finetuned dual-modality predictor (geo-aware mem) for prompt-source ablation..."
    )
    finetuned_model = SAM2MeshPredictor.from_pretrained(
        config_file=str(finetuned_model_config),
        ckpt_path=str(finetuned_checkpoint),
        device=device,
    )
    logger.info("Built finetuned multimodal predictor")
    return finetuned_model


def build_amg_from_config(project_root: Path, device: torch.device, logger: logging.Logger):
    """
    Build a `MaskProposalGenerator` whose config is taken from
    `configs/mesh_seg_coseg_new.yaml` → `sam.amg_sam`.
    """
    from seg3d.models.mask_proposal_generator import MaskProposalGenerator

    cfg_path = project_root / "configs" / "mesh_seg_coseg_new.yaml"
    cfg = OmegaConf.load(str(cfg_path))
    if "sam" not in cfg or "amg_sam" not in cfg.sam:
        raise RuntimeError(
            f"Expected `sam.amg_sam` block in {cfg_path}, but did not find it."
        )
    amg_cfg = cfg.sam.amg_sam

    logger.info(
        f"Building MaskProposalGenerator from config.sam.amg_sam in {cfg_path} "
        f"(checkpoint={amg_cfg.checkpoint}, model_config={amg_cfg.model_config})"
    )
    amg = MaskProposalGenerator(config=amg_cfg, device=str(device))
    logger.info("Built MaskProposalGenerator")
    return amg


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


def _predict_from_view0_all_frames_finetuned_amg(
    predictor,
    normal_imgs: torch.Tensor,
    point_imgs: torch.Tensor,
    gt_instance_masks: torch.Tensor,
    view_dirs: torch.Tensor,
    amg,
    max_prompts: int = 16,
) -> List[torch.Tensor | None]:
    """
    Finetuned multimodal predictor, but prompts come from AMG on view-0 normal image
    instead of GT instance masks.

    - Runs AMG on the first-view normal image to get proposal masks (N, H, W).
    - Uses top `max_prompts` proposals as initial masks at frame 0 via `add_new_mask`.
    - Propagates to all frames via `propagate_in_video`.
    """
    # Initialize tracking state
    state = predictor.init_state(
        normal_imgs=normal_imgs,
        point_imgs=point_imgs,
        view_dirs=view_dirs,
    )

    # Convert view-0 normal image to uint8 HxWx3 for AMG
    normal0 = normal_imgs[0]  # (C, H, W)
    normal0_np = (
        normal0.detach()
        .cpu()
        .permute(1, 2, 0)
        .numpy()
    )
    # Assume in [0, 1] or roughly normalized; rescale/clamp to [0, 255]
    normal0_np = np.clip(normal0_np, 0.0, 1.0)
    normal0_uint8 = (normal0_np * 255.0).astype(np.uint8)

    # Run AMG on view 0
    bmasks = amg(image=normal0_uint8, view_idx=0)  # (N, H, W) bool
    if bmasks.size == 0:
        # No proposals; nothing to prompt with
        num_frames = state["num_frames"]
        return [None] * num_frames

    # Limit number of prompts to keep compute manageable
    bmasks = bmasks[:max_prompts]

    # Add each AMG proposal as an object mask at frame 0
    device = normal_imgs.device
    for obj_id, mask_np in enumerate(bmasks, start=1):
        mask_t = torch.from_numpy(mask_np.astype(bool)).to(device)
        if mask_t.any():
            predictor.add_new_mask(
                state, frame_idx=0, obj_id=int(obj_id), mask=mask_t
            )

    # Propagate through all frames
    num_frames = state["num_frames"]
    pred_masks_list: List[torch.Tensor | None] = [None] * num_frames
    for frame_idx, obj_ids_step, video_res_masks in predictor.propagate_in_video(state):
        # video_res_masks: (num_objects, 1, H, W) — take logit channel 0
        pred_masks_list[frame_idx] = video_res_masks[:, 0, :, :].detach()

    return pred_masks_list


def run_prompt_source_ablation() -> None:
    ctx = setup_paths_and_logging()
    project_root: Path = ctx["PROJECT_ROOT"]
    logger: logging.Logger = ctx["logger"]
    device: torch.device = ctx["device"]
    figures_dir: Path = ctx["FIGURES_DIR"]
    logging_dir: Path = ctx["LOGGING_DIR"]

    # Dataset
    num_views = 12
    dataset = build_dataset(project_root, num_views=num_views)
    logger.info(f"Dataset size: {len(dataset)} meshes (train + val)")

    # Finetuned predictor and AMG
    finetuned_model = build_finetuned_predictor(
        device=device, project_root=project_root, logger=logger
    )
    amg = build_amg_from_config(project_root=project_root, device=device, logger=logger)

    model_keys = ["finetuned_mm_gt", "finetuned_mm_amg"]
    name_map = {
        "finetuned_mm_gt": "Dual (Normal+Point) + Geo-aware mem (GT prompts)",
        "finetuned_mm_amg": "Dual (Normal+Point) + Geo-aware mem (AMG prompts)",
    }
    all_results: Dict[str, Dict[str, Any]] = {
        k: {"ious": [], "dices": [], "per_mesh": []} for k in model_keys
    }

    # ------------------------------------------------------------------
    # Use the SAME fixed 40-mesh eval split as memory/modality ablations
    # ------------------------------------------------------------------
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

    logger.info(
        f"Evaluating prompt-source ablation (GT vs AMG prompts) on {len(eval_mesh_indices)} "
        f"validation meshes."
    )

    for mesh_idx in tqdm(eval_mesh_indices, desc="Evaluating meshes (AMG prompts)"):
        try:
            sample = dataset[mesh_idx]

            gt_instance_masks = sample["gt_instance_masks"].to(device, non_blocking=True)
            view_dirs = sample["view_dirs"].to(device, non_blocking=True)

            normal_imgs = sample.get("normal_imgs", None)
            point_imgs = sample.get("point_imgs", None)

            if normal_imgs is None or point_imgs is None:
                logger.warning(
                    f"Mesh {mesh_idx}: missing normal or point images; skipping."
                )
                continue

            normal_imgs = normal_imgs.to(device, non_blocking=True)
            point_imgs = point_imgs.to(device, non_blocking=True)

            # GT-prompted predictions for the finetuned dual-modality model
            from seg3d.ablation_studies.utils import (
                predict_from_view0_all_frames_finetuned_gt,
            )

            preds_finetuned_gt = predict_from_view0_all_frames_finetuned_gt(
                predictor=finetuned_model,
                normal_imgs=normal_imgs,
                point_imgs=point_imgs,
                gt_instance_masks=gt_instance_masks,
                view_dirs=view_dirs,
            )
            metrics_gt = evaluate_predictions_fast(
                preds_finetuned_gt, gt_instance_masks, device=device
            )
            all_results["finetuned_mm_gt"]["ious"].append(metrics_gt["mean_iou"])
            all_results["finetuned_mm_gt"]["dices"].append(metrics_gt["mean_dice"])
            all_results["finetuned_mm_gt"]["per_mesh"].append(
                {
                    "mesh_idx": mesh_idx,
                    "mesh_id": sample.get("mesh_id", None),
                    "metrics": metrics_gt,
                }
            )

            # AMG-prompted predictions for finetuned MM model
            preds_finetuned_amg = _predict_from_view0_all_frames_finetuned_amg(
                predictor=finetuned_model,
                normal_imgs=normal_imgs,
                point_imgs=point_imgs,
                gt_instance_masks=gt_instance_masks,
                view_dirs=view_dirs,
                amg=amg,
                max_prompts=16,
            )

            metrics_amg = evaluate_predictions_fast(
                preds_finetuned_amg, gt_instance_masks, device=device
            )
            all_results["finetuned_mm_amg"]["ious"].append(metrics_amg["mean_iou"])
            all_results["finetuned_mm_amg"]["dices"].append(metrics_amg["mean_dice"])
            all_results["finetuned_mm_amg"]["per_mesh"].append(
                {
                    "mesh_idx": mesh_idx,
                    "mesh_id": sample.get("mesh_id", None),
                    "metrics": metrics_amg,
                }
            )

        except Exception as e:  # pragma: no cover - defensive logging
            logger.error(f"Error evaluating mesh {mesh_idx}: {str(e)}")
            continue

    # -------------------------
    # Summarize and plot
    # -------------------------
    import pandas as pd
    import matplotlib.pyplot as plt

    summary_rows: List[Dict[str, Any]] = []
    per_view_rows: List[Dict[str, Any]] = []

    if len(all_results["finetuned_mm_gt"]["ious"]) == 0 and len(all_results["finetuned_mm_amg"]["ious"]) == 0:
        logger.info("No results to summarize (no successful meshes).")
        return

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

        iou_view_lists = [m["metrics"]["iou_per_view"] for m in all_results[key]["per_mesh"]]
        if iou_view_lists:
            iou_mat = np.array(iou_view_lists) * 100.0
            mean_iou_per_view = iou_mat.mean(axis=0)
            std_iou_per_view = iou_mat.std(axis=0)
            for v in range(iou_mat.shape[1]):
                per_view_rows.append(
                    {
                        "Model": name_map[key],
                        "View": v,
                        "Mean IoU": mean_iou_per_view[v],
                        "Std IoU": std_iou_per_view[v],
                    }
                )

    df_summary = pd.DataFrame(summary_rows)
    df_per_view = pd.DataFrame(per_view_rows)

    logger.info("\n" + "=" * 80)
    logger.info("PROMPT SOURCE ABLATION (DUAL + GEO-AWARE MEM, GT vs AMG PROMPTS): RESULTS")
    logger.info("=" * 80)
    logger.info(f"\nEvaluated on {len(eval_mesh_indices)} meshes")
    logger.info("\nSummary Statistics:\n" + df_summary.to_string(index=False))

    # Save summary CSV
    summary_path = logging_dir / "ablation_prompt_source_summary.csv"
    df_summary.to_csv(summary_path, index=False)
    logger.info(f"Saved summary to: {summary_path}")

    # Save detailed per-mesh CSV
    detailed_rows: List[Dict[str, Any]] = []
    for key in model_keys:
        for mesh_data in all_results[key]["per_mesh"]:
            row = {
                "Model": name_map[key],
                "Mesh ID": mesh_data["mesh_id"],
                "Mesh Index": mesh_data["mesh_idx"],
                "Mean IoU": mesh_data["metrics"]["mean_iou"] * 100.0,
                "Mean Dice": mesh_data["metrics"]["mean_dice"] * 100.0,
                "Std IoU": mesh_data["metrics"]["std_iou"] * 100.0,
                "Std Dice": mesh_data["metrics"]["std_dice"] * 100.0,
            }
            # store per-view IoU for regen plots
            for v, val in enumerate(mesh_data["metrics"]["iou_per_view"]):
                row[f"iou_view_{v}"] = val * 100.0
            for v, val in enumerate(mesh_data["metrics"]["dice_per_view"]):
                row[f"dice_view_{v}"] = val * 100.0
            detailed_rows.append(row)

    if detailed_rows:
        df_detailed = pd.DataFrame(detailed_rows)
        detailed_path = logging_dir / "ablation_prompt_source_detailed.csv"
        df_detailed.to_csv(detailed_path, index=False)
        logger.info(f"Saved detailed results to: {detailed_path}")

    if not df_per_view.empty:
        per_view_path = logging_dir / "ablation_prompt_source_per_view.csv"
        df_per_view.to_csv(per_view_path, index=False)
        logger.info(f"Saved per-view results to: {per_view_path}")

    # Bar plot: IoU and Dice for GT vs AMG
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    model_names = df_summary["Model"].values
    mean_ious = df_summary["Mean IoU"].values
    std_ious = df_summary["Std IoU"].values
    mean_dices = df_summary["Mean Dice"].values
    std_dices = df_summary["Std Dice"].values

    axes[0].barh(model_names, mean_ious, xerr=std_ious, capsize=5)
    axes[0].set_xlabel("Mean IoU (%)", fontsize=12)
    axes[0].set_title("Prompt source ablation (GT vs AMG) – IoU", fontsize=12)
    axes[0].grid(axis="x", alpha=0.3)
    axes[0].set_xlim(0, 100)

    axes[1].barh(model_names, mean_dices, xerr=std_dices, capsize=5)
    axes[1].set_xlabel("Mean Dice (%)", fontsize=12)
    axes[1].set_title("Prompt source ablation (GT vs AMG) – Dice", fontsize=12)
    axes[1].grid(axis="x", alpha=0.3)
    axes[1].set_xlim(0, 100)

    plt.tight_layout()
    plot_path = figures_dir / "ablation_prompt_source_metrics.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    logger.info(f"Saved prompt source ablation plot to: {plot_path}")
    plt.close(fig)

    # Per-view IoU plot (regenerable from CSV)
    if not df_per_view.empty:
        fig, ax = plt.subplots(figsize=(10, 4))
        for model_name in df_summary["Model"].values:
            df_m = df_per_view[df_per_view["Model"] == model_name].sort_values("View")
            if df_m.empty:
                continue
            ax.plot(df_m["View"], df_m["Mean IoU"], label=model_name, linewidth=2)
            ax.fill_between(
                df_m["View"],
                df_m["Mean IoU"] - df_m["Std IoU"],
                df_m["Mean IoU"] + df_m["Std IoU"],
                alpha=0.2,
            )
        ax.set_xlabel("View index")
        ax.set_ylabel("Mean IoU (%)")
        ax.grid(alpha=0.3)
        ax.set_xlim(0, num_views - 1)
        ax.legend(loc="lower left")
        plt.tight_layout()
        per_view_plot_path = figures_dir / "ablation_prompt_source_per_view_iou.png"
        plt.savefig(per_view_plot_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved per-view IoU plot to: {per_view_plot_path}")
        plt.close(fig)


if __name__ == "__main__":
    run_prompt_source_ablation()

