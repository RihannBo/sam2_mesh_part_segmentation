
# import re
# from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from omegaconf import OmegaConf
# from transformers import AutoProcessor, AutoModel

from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

from seg3d.data.common import NumpyTensor

def remove_artifacts(mask: NumpyTensor['h w'], mode: str, min_area=128) -> NumpyTensor['h w']:
    """
    Removes small islands/fill holes from a mask.
    """
    assert mode in ['holes', 'islands']
    mode_holes = (mode == 'holes')

    def remove_helper(bmask):
        # opencv connected components operates on binary masks only
        bmask = (mode_holes ^ bmask).astype(np.uint8)
        nregions, regions, stats, _ = cv2.connectedComponentsWithStats(bmask, 8)
        sizes = stats[:, -1][1:]  # Row 0 corresponds to 0 pixels
        fill = [i + 1 for i, s in enumerate(sizes) if s < min_area] + [0]
        if not mode_holes:
            fill = [i for i in range(nregions) if i not in fill]
        return np.isin(regions, fill)

    mask_combined = np.zeros_like(mask)
    for label in np.unique(mask): # also process background
        mask_combined[remove_helper(mask == label)] = label
    return mask_combined

def deduplicate_masks(masks, iou_threshold=0.95):
    keep = []
    for m in masks:
        is_duplicate = False
        for k in keep:
            inter = np.logical_and(m, k).sum()
            union = np.logical_or(m, k).sum()
            iou = inter / (union + 1e-8)
            if iou > iou_threshold:
                is_duplicate = True
                break
        if not is_duplicate:
            keep.append(m)
    return keep

def combine_bmasks(masks: NumpyTensor['n h w'], sort=False) -> NumpyTensor['h w']:
    """
    Combine multiple binary masks into a single label map.

    Each input mask represents one object instance (boolean 2D array). 
    The function assigns a unique integer label to each mask and merges 
    them into a single 2D array of integer IDs.

    Args:
        masks (NumpyTensor['n h w']): 
            A stack or list of N binary masks, where each mask has shape (H, W)
            and contains boolean or {0,1} values indicating the object region.
        sort (bool, optional): 
            If True, masks are sorted in descending order by area (pixel count) 
            before merging. Larger regions get lower labels, which helps 
            prioritize large objects in overlapping areas. Default is False.

    Returns:
        NumpyTensor['h w']: 
            A 2D integer label map of shape (H, W), where:
                - 0 represents background,
                - 1...N represent individual mask IDs.

    Example:
        >>> m1 = np.array([[1,0],[0,0]], dtype=bool)
        >>> m2 = np.array([[0,0],[1,1]], dtype=bool)
        >>> combine_bmasks([m1, m2])
        array([[1,0],
               [2,2]])
    """
    mask_combined = np.zeros_like(masks[0], dtype=int)
    if sort:
        masks = sorted(masks, key=lambda x: x.sum(), reverse=True)
    for i, mask in enumerate(masks):
        mask_combined[mask] = i + 1
    return mask_combined

def colormap_mask(
    mask : NumpyTensor['h w'], 
    image: NumpyTensor['h w 3']=None, background=np.array([255, 255, 255]), foreground=None, blend=0.25
) -> Image.Image:
    """
    """
    palette = np.random.randint(0, 255, (np.max(mask) + 1, 3))
    palette[0] = background
    if foreground is not None:
        for i in range(1, len(palette)):
            palette[i] = foreground
    image_mask = palette[mask.astype(int)] # type conversion for boolean masks
    image_blend = image_mask if image is None else image_mask * (1 - blend) + image * blend
    image_blend = np.clip(image_blend, 0, 255).astype(np.uint8)
    return Image.fromarray(image_blend)

def colormap_bmasks(
    masks: NumpyTensor['n h w'], 
    image: NumpyTensor['h w 3']=None, background=np.array([255, 255, 255]), blend=0.25
) -> Image.Image:
    """
    """
    mask = combine_bmasks(masks)
    return colormap_mask(mask, image, background=background, blend=blend)

def point_grid_from_mask(mask: NumpyTensor['h w'], n: int) -> NumpyTensor['n 2']:
    """
    Sample up to N valid pixels from a binary mask and normalize them to [0, 1]².

    Args:
        mask: (H, W) boolean array of valid pixels.
        n: Maximum number of points to sample.

    Returns:
        (N, 2) float array of normalized (x, y) coordinates.
    """
    valid = np.argwhere(mask)   ## (M, 2) indices of True pixels (y, x)
    if len(valid) == 0:
        raise ValueError('No valid points in mask')
    
    h, w = mask.shape
    n = min(n, len(valid))
    
    samples_idx = np.random.choice(len(valid), n, replace=False)
    samples = valid[samples_idx].astype(float)
    samples[:, 0] /= h - 1      ## normalize y to [0, 1]
    samples[:, 1] /= w - 1      ### normalize x to [0, 1]
    samples = samples[:, [1, 0]] ## (y, x) -> (x, y)
    samples = samples[np.lexsort((samples[:, 1], samples[:, 0]))]
    return samples
    
    
class MaskProposalGenerator(nn.Module):
    """
    """
    def __init__(self, config: OmegaConf, device='cuda:0'):
        """
        """
        super().__init__()
        self.config = config
        self.device = device
        self.setup_sam()
    
    def setup_sam(self):
        #sam_cfg = self.config.sam   # ← already equal to config.sam

        self.sam_model = build_sam2(
            self.config.model_config,
            self.config.checkpoint,
            device=self.device,
            apply_postprocessing=True
        )
        self.sam_model.eval()

        # Extract AMG parameters from config, excluding model loading parameters
        # Convert OmegaConf to dict and filter out non-AMG parameters
        config_dict = OmegaConf.to_container(self.config, resolve=True)
        amg_params = {k: v for k, v in config_dict.items() 
                     if k not in ['checkpoint', 'model_config']}
        
        self.engine = SAM2AutomaticMaskGenerator(
            self.sam_model,
            **amg_params
        )          
        

    
    def process_image(self, image, view_idx: int, fg_mask=None) -> NumpyTensor['n h w']:
        """
        Run SAM2 automatic segmentation on an image and return a [N, H, W] boolean mask stack,
        sorted by area (largest first).
        
        Args:
            image: PIL Image or numpy array (H, W, 3) in uint8 format
            view_idx: View index for logging
            fg_mask: Optional foreground mask
        """
        # 1) Ensure 3-channel RGB uint8 (SAM2 expects HxWx3)
        if isinstance(image, Image.Image):
            image = np.array(image.convert('RGB'), dtype=np.uint8)
        else:
            # Already a numpy array
            image = np.array(image, dtype=np.uint8)
            # Ensure 3 channels
            if image.ndim == 2:
                image = np.stack([image] * 3, axis=-1)
            elif image.shape[-1] == 4:  # RGBA
                image = image[..., :3]
        
        H, W = image.shape[:2]
        
        # 2) Generate raw mask proposals (list of dicts, one dict per mask)
        with torch.no_grad():
            annotations = self.engine.generate(image)

        if len(annotations) == 0:
            print(f"[AMG] view {view_idx}: no masks.")
            return np.zeros((0, H, W), dtype=bool)

        # 3) Sort proposals by area (descending)
        annotations = sorted(annotations, key=lambda x: x['area'], reverse=True)
        
        # --- DEBUG LOGGING ---
        areas = [a["area"] for a in annotations]
        print(f"[AMG] view {view_idx}: #masks={len(annotations)}, "
            f"max_area={max(areas)}, min_area={min(areas)}, "
            f"median_area={np.median(areas):.1f}")
        
        # 4) Convert to boolean stack
        raw_masks = [ann["segmentation"].astype(bool) for ann in annotations]
        bmasks = np.stack(raw_masks, axis=0)  # [N, H, W]

        return bmasks
        
    
    def forward(self, image: Image, view_idx:int, fg_mask=None) -> NumpyTensor['n h w']:
        """
        """
        return self.process_image(image=image, view_idx=view_idx, fg_mask=fg_mask)
        
