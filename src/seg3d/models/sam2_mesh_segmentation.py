"""
End-to-end mesh segmentation pipeline (multiview SAM2 + lifting).

Primary entry point: :class:`MeshSegmentation`.

"""
# --- standard library ---
import os
from pathlib import Path
from collections import defaultdict, Counter
from contextlib import nullcontext

# --- third-party ---
import numpy as np
import torch
import torch.nn as nn
import cv2
import scipy.ndimage as ndi
import trimesh
from trimesh import Trimesh, Scene
from sklearn.neighbors import NearestNeighbors
from omegaconf import OmegaConf
from tqdm import tqdm
import multiprocessing as mp

# --- project ---
from seg3d.data.common import NumpyTensor
from seg3d.renderer.renderer import (
    Renderer, render_multiview,
    colormap_faces, colormap_norms, colormap_points
)

from seg3d.models.mask_proposal_generator import MaskProposalGenerator
from seg3d.models.sam2_mesh_predictor import SAM2MeshPredictor
from seg3d.models.shape_diameter_function import repartition
from seg3d.utils.mesh import duplicate_verts
from seg3d.utils.sam2_geometry_utils import project_world_to_cam_pixels
from seg3d.utils.view_sampling import compute_spatial_traversal_order


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


def combine_bmasks(bmasks: NumpyTensor['n h w'], sort: bool = False) -> NumpyTensor['h w']:
    """
    bmasks: (num_objects, H, W) binary masks
    sort: if True, process masks in descending area order (larger first); affects overlap resolution.
    returns: (H, W) label map with consistent object IDs.
    """
    num_objects, H, W = bmasks.shape
    cmask = np.zeros((H, W), dtype=np.int32)

    order = range(num_objects)
    if sort:
        order = sorted(order, key=lambda i: np.sum(bmasks[i] > 0), reverse=True)
    for idx, obj_id in enumerate(order):
        cmask[bmasks[obj_id] > 0] = idx + 1
    return cmask


def remove_artifacts(mask, mode="islands", min_area=128):
    """
    Multi-label safe artifact removal.

    mask: (H, W) integer mask (0=background, 1..N labels)
    mode: "islands" (remove small disconnected label islands)
          "holes"   (fill small empty holes inside each label)
    """

    assert mode in ("islands", "holes")
    H, W = mask.shape

    output = np.zeros_like(mask)

    for label in np.unique(mask):

        region = (mask == label).astype(np.uint8)

        # Connected components
        n_cc, cc_map, stats, _ = cv2.connectedComponentsWithStats(region, connectivity=8)

        # Skip if just background or only one component
        if n_cc <= 1:
            # copy background or single connected component
            output[region == 1] = label
            continue

        if mode == "islands":
            # keep components with area >= min_area
            for comp_id in range(1, n_cc):
                area = stats[comp_id, cv2.CC_STAT_AREA]
                if area >= min_area:
                    output[cc_map == comp_id] = label

        elif mode == "holes":
            # fill small holes (i.e., keep ALL and then fill small empty regions inside)
            # invert the mask to find holes
            region_inv = 1 - region
            n_inv, inv_map, inv_stats, _ = cv2.connectedComponentsWithStats(region_inv, connectivity=8)

            # start with original region
            output[region == 1] = label

            # Only process if there are components beyond the background (n_inv > 1)
            if n_inv > 1:
                # keep the large "background" component; fill small holes
                # largest component = actual outside region
                largest_comp = np.argmax(inv_stats[1:, cv2.CC_STAT_AREA]) + 1

                # fill small hole components
                for comp_id in range(1, n_inv):
                    if comp_id == largest_comp:
                        continue
                    area = inv_stats[comp_id, cv2.CC_STAT_AREA]
                    if area < min_area:
                        output[inv_map == comp_id] = label

    return output


def fill_unlabeled_foreground(label_map: NumpyTensor['h w'], foreground_mask: NumpyTensor['h w']) -> NumpyTensor['h w']:
    """
    Assign unlabeled foreground pixels (label 0 in foreground) to nearest non-zero label.
    Only fills pixels where foreground_mask is True.
    
    This function uses distance transform to find the nearest labeled pixel for each
    unlabeled foreground pixel, similar to the approach in rendering.ipynb.
    
    Args:
        label_map: (H, W) integer label map (0=unlabeled/background, 1..N=labels)
        foreground_mask: (H, W) boolean mask where True indicates foreground pixels
        
    Returns:
        filled_label_map: (H, W) label map with unlabeled foreground pixels filled
    """
    # Create a mask of labeled foreground pixels (foreground AND has label > 0)
    labeled_mask = foreground_mask & (label_map > 0)
    
    # Find unlabeled foreground pixels (foreground but label == 0)
    unlabeled_fg = foreground_mask & (label_map == 0)
    
    if not np.any(unlabeled_fg):
        return label_map  # Nothing to fill
    
    # Use distance transform to find nearest labeled pixel
    # distance_transform_edt computes distance from each pixel to nearest True pixel
    distance, indices = ndi.distance_transform_edt(
        ~labeled_mask,  # Invert: True where we want to find distance FROM
        return_indices=True
    )
    
    # Fill unlabeled foreground pixels with labels from nearest labeled pixel
    filled = label_map.copy()
    filled[unlabeled_fg] = label_map[
        indices[0][unlabeled_fg],
        indices[1][unlabeled_fg]
    ]
    
    return filled

def colormap_faces_mesh(mesh: Trimesh, face2label: dict[int, int], background=np.array([0, 0, 0])) -> Trimesh:
    """
    Color mesh faces by label using distinct colors generated in HSV color space.
    
    Uses HSV color space to generate evenly spaced, visually distinct colors.
    This produces much more distinguishable colors than random RGB values.
    """
    import colorsys
    
    label_max = max(face2label.values())
    num_colors = label_max + 1  # +1 for unlabeled faces
    
    # Generate distinct colors using HSV color space
    # Use golden ratio to maximize color separation
    golden_ratio = 0.618033988749895
    palette = np.zeros((num_colors, 3), dtype=np.uint8)
    palette[0] = background  # Background color
    
    for i in range(1, num_colors):
        # Use golden ratio spacing in hue for maximum distinctness
        hue = (i * golden_ratio) % 1.0
        # Use high saturation and value for vibrant, distinct colors
        saturation = 0.8 + (i % 3) * 0.05  # Vary between 0.8-0.9
        value = 0.9 + (i % 2) * 0.05      # Vary between 0.9-0.95
        
        # Convert HSV to RGB
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        palette[i] = np.array([int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)], dtype=np.uint8)
    
    mesh = duplicate_verts(mesh) # needed to prevent face color interpolation
    faces_colored = set()
    for face, label in face2label.items():
        mesh.visual.face_colors[face, :3] = palette[label]
        faces_colored.add(face)
    #print(np.unique(mesh.visual.face_colors, axis=0, return_counts=True))
    '''
    for face in range(len(mesh.faces)):
        if face not in faces_colored:
            mesh.visual.face_colors[face, :3] = background
            print('Unlabeled face ', face)
    '''
    return mesh

def compute_front_facing_mask(
    norms: NumpyTensor['h w 3'],
    cam2world: NumpyTensor['4 4'],
    threshold: float = 0.0,
) -> NumpyTensor['h w']:
    """ 
    Compute a boolean mask where surface normals are *truly* front-facing.

    Args:
        norms: (H, W, 3) array of surface normals in world coordinates.
        cam2world: (4, 4) camera-to-world transformation matrix.
        threshold: cosine threshold for front-facing visibility. A value of
            0.0 keeps all normals with a positive dot-product w.r.t. the
            view direction; higher values keep only strongly front-facing
            normals.

    Returns:
        (H, W) boolean mask: True where normals face the camera.
    """
    # Camera looks along +Z in its local space; map that to world space.
    view_dir_world = cam2world[:3, :3] @ np.array([0, 0, 1], dtype=np.float32)
    # Front-facing ⇔ dot(n, view_dir_world) > threshold (no abs: back-faces are excluded).
    visible_mask = np.dot(norms, view_dir_world) > threshold
    return visible_mask

def visualize_items(items: dict, path: str | Path) -> None:
    """
    Save visualization images for each key in `items` into structured subfolders.
    `path` is already the mesh-specific folder inside rendering_results.
    """
    print("Save rendering results")
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    # ---- MATTE ----
    if 'matte' in items:
        d = path / "matte"
        d.mkdir(exist_ok=True)
        for i, img in enumerate(items['matte']):
            img.save(d / f"{i}.jpg")

    # ---- SDF ----
    if 'sdf' in items:
        d = path / "sdf"
        d.mkdir(exist_ok=True)
        for i, img in enumerate(items['sdf']):
            img.save(d / f"{i}.jpg")

    # ---- NORMALS ----
    if 'norms' in items:
        d = path / "norms"
        d.mkdir(exist_ok=True)
        for i, norms in enumerate(items['norms']):
            colormap_norms(norms).save(d / f"{i}.jpg")

    # ---- MASKED NORMALS ----
    if 'norms_masked' in items:
        d = path / "norms_masked"
        d.mkdir(exist_ok=True)
        for i, norms in enumerate(items['norms_masked']):
            colormap_norms(norms).save(d / f"{i}.jpg")

    # ---- FACES ----
    if 'faces' in items:
        d = path / "faces"
        d.mkdir(exist_ok=True)
        for i, faces in enumerate(items['faces']):
            colormap_faces(faces).save(d / f"{i}.jpg")

    # ---- POINT MAP ----
    if 'point_map' in items:
        d = path / "pointmap"
        d.mkdir(exist_ok=True)
        for i, pmap in enumerate(items['point_map']):
            colormap_points(pmap).save(d / f"{i}.jpg")

    # ---- CLEAN MASKS FOR VIEW 0 ----
    # if 'view0_clean_masks' in items:
    #     d = path / "view0_clean_masks"
    #     d.mkdir(exist_ok=True)
    #     cmasks = items['view0_clean_masks']
    #     for i, m in enumerate(cmasks):
    #         colormap_mask(m.astype(int)).save(d / f"{i}.png")


def _enforce_instance_identity_overlap(
    bmasks_v: np.ndarray,
    overlap_thresh: int,
) -> None:
    """
    Enforce mutual exclusion between object pairs: if two objects overlap
    by more than overlap_thresh pixels, assign the overlap to the larger
    object (remove from the smaller). Mutates bmasks_v in place.

    bmasks_v: (N, H, W) bool
    overlap_thresh: max allowed overlap in pixels; above this we resolve.
    """
    n = bmasks_v.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            overlap_mask = bmasks_v[i] & bmasks_v[j]
            if overlap_mask.sum() <= overlap_thresh:
                continue
            # Remove overlap from the smaller object (assign overlap to larger)
            if bmasks_v[i].sum() < bmasks_v[j].sum():
                bmasks_v[i][overlap_mask] = False
            else:
                bmasks_v[j][overlap_mask] = False


class MeshSegmentation(nn.Module):
    """
    End-to-end mesh segmentation pipeline using SAM2 for multiview segmentation
    and graph-based 3D label refinement. 
    """
    def __init__(self, config: OmegaConf, device: str | torch.device = 'cuda:0'):
        """
        Args:
            config: OmegaConf configuration object
            device: Device to run on
            mm_sam_config_path: Optional override for mm_sam model_config path (if config composition removes it)
            mm_sam_checkpoint_path: Optional override for mm_sam checkpoint path
        """
        super().__init__()
        self.config = config

        # Resolve desired device: prefer the requested CUDA device when available,
        # but fall back to CPU gracefully if CUDA is not present.
        if isinstance(device, str) and device.startswith("cuda"):
            if not torch.cuda.is_available():
                print(
                    f"[MeshSegmentation] Requested device '{device}' but CUDA is not "
                    "available; falling back to CPU."
                )
                device = "cpu"
        self.device = device
        
        # initialize renderer
        self.renderer = Renderer(config=config.renderer)
    
        
        self.mask_proposer = MaskProposalGenerator(
            config=config.sam.amg_sam,
            device=self.device,
        )
        
        # Flexibly pass any extra predictor kwargs defined under sam.mm_sam.
        mm_cfg = config.sam.mm_sam
        # Start from all keys except the ones used to locate the checkpoint/config.
        predictor_kwargs = {
            k: v
            for k, v in mm_cfg.items()
            if k not in ("checkpoint", "model_config")
        }
        # non_overlap_masks_for_mem_enc is a flag on the underlying SAM2 base,
        # not a predictor kwarg; handle it explicitly after construction.
        non_overlap_masks_for_mem_enc = predictor_kwargs.pop(
            "non_overlap_masks_for_mem_enc", None
        )
        
        self.predictor = SAM2MeshPredictor.from_pretrained(
            config_file=mm_cfg.model_config,  # Path to finetuning config (e.g. mm_sam2.1_hiera_l.yaml)
            ckpt_path=mm_cfg.checkpoint,      # Path to finetuned checkpoint (e.g. /path/to/checkpoints/best.pt)
            device=self.device,
            **predictor_kwargs,
        )
        if non_overlap_masks_for_mem_enc is not None:
            # This attribute is defined on MultiViewSAM2Base / sam2_base.
            self.predictor.non_overlap_masks_for_mem_enc = bool(
                non_overlap_masks_for_mem_enc
            )
        # Cache for residual point grids; cleared whenever we prepare a new scene.
        self._residual_grid_cache: dict[tuple, np.ndarray] = {}
    # ------------------------------------------------------------
    # Step 1 — Scene preparation
    # ------------------------------------------------------------
    def prepare_scene(self, source:Scene | Trimesh): 
        """ 
        Set renderer object, cameras, and adjacency graph.
        """ 
        # New scene: clear any cached SAM2 per-object masks or residual grids.
        self._residual_grid_cache.clear()

        self.renderer.set_object(source)
        self.renderer.set_camera()
        
        ## build adjaceny graph (which faces are neighbors)
        self.mesh_edges = trimesh.graph.face_adjacency(mesh=self.renderer.tmesh) ## returns np.array(#edges, 2)
        self.mesh_graph = defaultdict(set)  
        for face1, face2 in self.mesh_edges:
            self.mesh_graph[face1].add(face2)
            self.mesh_graph[face2].add(face1)
        ## self.mesh_graph: dict[int, set[int]]  →  key = face ID, value = set of adjacent face IDs
     
    def render_raw_views(self, source: Scene | Trimesh):

        if not hasattr(self.renderer, "tmesh"):
            raise RuntimeError(
                "render_raw_views() requires prepare_scene() to be called first"
            )
        
        renderer_args = self.config.renderer.renderer_args
        sampling_args = self.config.renderer.sampling_args
        
        return render_multiview(
                renderer= self.renderer,
                camera_generation_method= self.config.renderer.camera_generation_method,
                renderer_args= renderer_args,
                sampling_args= sampling_args,
                lighting_args= self.config.renderer.lighting_args
            )

    def select_view_order(self, renders, manual_view0=None):
        """
        Reorder views so that consecutive indices are geometrically similar
        (smooth path on the sphere, like video frames). Front view (index 0)
        is chosen manually or by visibility; remaining order is a spatial
        traversal (nearest-neighbor + 2-opt) from that front.
        """
        num_views = len(renders["view_dirs"])
        if manual_view0 is not None:
            front_idx = int(manual_view0)
            if not (0 <= front_idx < num_views):
                raise ValueError(
                    f"manual_view0={front_idx} out of range [0, {num_views})"
                )
        else:
            front_idx = int(np.argmax([np.mean(d > 0) for d in renders["depth_raw"]]))

        view_dirs = renders["view_dirs"]
        directions = np.stack(
            [np.asarray(d, dtype=np.float32) for d in view_dirs], axis=0
        )
        # Front direction for traversal: we want start_idx = front_idx, so pass
        # front = -view_dirs[front_idx] (convention in compute_spatial_traversal_order).
        front_dir = np.asarray(view_dirs[front_idx], dtype=np.float32)
        front_dir = front_dir / (np.linalg.norm(front_dir) + 1e-8)
        order, _ = compute_spatial_traversal_order(
            directions, front=-front_dir
        )
        # order[0] should be front_idx; rotate so front_idx is first if not
        if order[0] != front_idx:
            k = order.index(front_idx)
            order = order[k:] + order[:k]

        for key, value in renders.items():
            if isinstance(value, list):
                renders[key] = [value[i] for i in order]

        return renders

    def prepare_sam_inputs(self, renders: dict):
        """
        Prepare SAM prompt inputs.

        Assumes `renders` comes from render_raw_views() and contains:
        norms, point_map, faces, matte, poses, view_dirs, depth_raw.

        If config sam_mesh.bootstrap is True, duplicates the first view so
        propagation sees frame 0 twice (stronger anchor), then propagates.
        """

        # --------------------------------------------------
        # 0) Optional: bootstrap = duplicate first view then propagate
        # --------------------------------------------------
        bootstrap = self.config.sam_mesh.get("bootstrap", False)
        if bootstrap:
            for key in (
                "norms", "point_map", "faces", "matte",
                "view_dirs", "depth_raw", "poses", "intrinsics",
            ):
                if key not in renders:
                    continue
                val = renders[key]
                if isinstance(val, list):
                    first = val[0]
                    first_copy = first.copy() if hasattr(first, "copy") else first
                    renders[key] = [first_copy] + list(val)
                elif isinstance(val, np.ndarray):
                    first = val[0:1]
                    renders[key] = np.concatenate([first, val], axis=0)
            print("[prepare_sam_inputs] bootstrap: duplicated first view (T+1 frames)")

        # --------------------------------------------------
        # Inner helpers (NumPy-only, stack internally)
        # --------------------------------------------------
        def normalize_normals(norms_list: list[np.ndarray]) -> np.ndarray:
            """
            Normalize normal maps to [0, 1] range for SAM2 ImageNet normalization.
            
            Handles two input formats:
            1. From renderer: float32 in [-1, 1] (already decoded)
            2. From dataset: uint8 encoded as (n + 1) * 127.5 (needs decoding)

            Input:
                norms_list: list of (H, W, 3)
                    - If from renderer: float32 in [-1, 1]
                    - If from dataset: uint8 in [0, 255]
            Output:
                (T, H, W, 3), float32 in [0, 1]
            """
            normals = np.stack(norms_list, axis=0)
            
            # Detect format: uint8 means dataset encoding, float in [-1,1] means renderer
            if normals.dtype == np.uint8:
                # Dataset format: uint8 encoded as (n + 1) * 127.5
                # Decode: (n_u8 / 127.5) - 1.0 gives [-1, 1]
                normals = normals.astype(np.float32)
                normals = (normals / 127.5) - 1.0
            else:
                # Renderer format: already float32 in [-1, 1]
                normals = normals.astype(np.float32)
                # Clamp to [-1, 1] in case of any outliers
                normals = np.clip(normals, -1.0, 1.0)
            
            # Convert from [-1, 1] to [0, 1]
            normals = (normals + 1.0) * 0.5
            return np.clip(normals, 0.0, 1.0)

        def normalize_pointmaps(
            pointmaps_list: list[np.ndarray],
            faces_list: list[np.ndarray],
        ) -> np.ndarray:
            """
            Min–max normalize point maps per mesh.

            Input:
                pointmaps_list: list of (H, W, 3)
                faces_list:     list of (H, W), -1 = background
            Output:
                (T, H, W, 3), float32 in [0,1]
            """
            pm    = np.stack(pointmaps_list, axis=0).astype(np.float32)
            faces = np.stack(faces_list, axis=0)

            pm[faces == -1] = 0.0
            pm = np.nan_to_num(pm, nan=0.0, posinf=0.0, neginf=0.0)

            flat = pm.reshape(-1, 3)
            pmin = flat.min(axis=0)
            pmax = flat.max(axis=0)
            prng = np.maximum(pmax - pmin, 1e-6)

            pm = (pm - pmin) / prng
            return np.clip(pm, 0.0, 1.0)

        # --------------------------------------------------
        # 1) Normalize modalities (SAM inputs)
        # --------------------------------------------------
        renders["normal_imgs"] = normalize_normals(renders["norms"])
        renders["point_imgs"]  = normalize_pointmaps(
            renders["point_map"],
            renders["faces"],
        )

        # --------------------------------------------------
        # 2) Prepare prompt image (view 0 only)
        # --------------------------------------------------
        foreground_mask0 = renders["faces"][0] != -1
        use_mode = self.config.sam_mesh.use_mode

        if use_mode == "norms":
            image0 = colormap_norms(renders["norms"][0])
        elif use_mode == "matte":
            image0 = renders["matte"][0]

        else:
            raise ValueError(f"Unknown prompt mode: {use_mode}")

        # --------------------------------------------------
        # 3) AMG sampling
        # --------------------------------------------------
        print("Generating prompt masks for view 0")
        points_per_side = self.config.sam.amg_sam.points_per_side
        grid0 = point_grid_from_mask(foreground_mask0, points_per_side ** 2)
        # AMG uses one point grid per crop layer (layer 0 = full image, then crop_n_layers layers)
        n_grids = self.mask_proposer.engine.crop_n_layers + 1
        self.mask_proposer.engine.point_grids = [grid0] * n_grids
        masks = self.mask_proposer(image0, view_idx=0)

        print("Prompt masks generated for view 0")
        # --------------------------------------------------
        # 4) Prompt mask postprocess
        # --------------------------------------------------
        masks = self._clean_prompt_masks_2d(masks, foreground_mask0, fill_unlabeled=False)
        renders["prompt_masks"] = masks
        renders["prompt_frame_idxs"] = [0] * len(masks)
        # Temporary propagation anchors (can drift), kept separate from identity anchors.
        renders["propagation_frame_idxs"] = renders["prompt_frame_idxs"].copy()
        # Keep immutable view-0 seed prompts for residual ownership scoring.
        renders["prompt_masks_view0"] = [m.copy() for m in masks]
        renders["num_seed_objects"] = len(masks)

        return renders

    def _clean_prompt_masks_2d(self, masks, foreground_mask0, *, fill_unlabeled: bool = True):
        """
        2D prompt-mask cleanup on the image plane (view 0 only).

        Input domain: pixels (H × W) in the prompt image.
        Input masks are 2D binary masks (AMG proposals) and a foreground mask
        derived from rendered faces (faces != -1).

        Typical operations:
          - fill unlabeled foreground pixels,
          - remove tiny islands,
          - resolve overlaps,
          - ensure prompt masks roughly cover the visible object.

        Args:
            masks: np.ndarray[N, H, W] or list of (H, W) boolean / {0,1} masks.
            foreground_mask0: (H, W) boolean mask where True = visible foreground.

        Returns:
            cleaned_masks: list of (H, W) boolean masks, one per surviving object.
        """
        # Convert inputs to numpy arrays with explicit shapes
        masks_np = np.asarray(masks)
        if masks_np.ndim == 2:
            masks_np = masks_np[None, ...]  # (1, H, W)

        if masks_np.size == 0 or masks_np.shape[0] == 0:
            # No proposals – nothing to clean
            return []

        num_objects, H, W = masks_np.shape

        # Ensure boolean masks and restrict them to the rendered foreground
        fg = np.asarray(foreground_mask0, dtype=bool)
        if fg.shape != (H, W):
            raise ValueError(
                f"foreground_mask0 shape {fg.shape} does not match masks shape {(H, W)}"
            )

        bmasks = masks_np.astype(bool)
        bmasks = bmasks & fg[None, ...]

        # Combine per-object masks into a single integer label map (0 = background)
        # We keep the original order (no sorting) so object indices remain stable.
        cmask = combine_bmasks(bmasks, sort=False)

        # Ensure background outside visible region
        cmask[~fg] = 0

        # Optionally fill unlabeled foreground pixels by nearest labeled neighbor
        if fill_unlabeled:
            cmask = fill_unlabeled_foreground(cmask, fg)

    
        min_area = int(
            self.config.sam_mesh.get("min_area_2d", 1024)
        )

        # Remove tiny islands and fill small holes per label
        cmask = remove_artifacts(cmask, mode="islands", min_area=min_area)
        cmask = remove_artifacts(cmask, mode="holes",   min_area=min_area)

        # Rebuild per-object binary masks in the original index space.
        cleaned_masks = []
        for obj_id in range(1, num_objects + 1):
            m = (cmask == obj_id)
            if np.any(m):
                cleaned_masks.append(m)

        return cleaned_masks
    # ------------------------------------------------------------
    def run_propagation(self, renders: dict, start_obj_id: int = 0):
        """
        Run SAM2 propagation using prepared SAM inputs.

        Requires:
        - prepare_sam_inputs() has been called
        - renders contains:
            normal_imgs : (T,H,W,3) np.float32 in [0,1]
            point_imgs  : (T,H,W,3) np.float32 in [0,1]
            prompt_masks: list of (H,W)
            view_dirs   : list[(3,)] or (T,3)
        Optional:
            start_obj_id: if > 0, reuse cached masks for objects [0, start_obj_id)
                from a previous call and only propagate objects in
                [start_obj_id, num_objects). This is used by residual discovery
                to avoid re-running SAM2 for already-tracked objects.
        """

        # ------------------------------------------------------------
        # 0) Sanity checks
        # ------------------------------------------------------------
        assert "normal_imgs" in renders and "point_imgs" in renders, \
            "run_propagation() requires prepare_sam_inputs() to be called first"

        prompt_masks = renders["prompt_masks"]
        prompt_frame_idxs = renders.get("prompt_frame_idxs")
        if prompt_frame_idxs is None:
            prompt_frame_idxs = [0] * len(prompt_masks)
        elif len(prompt_frame_idxs) != len(prompt_masks):
            raise ValueError(
                "prompt_frame_idxs length must match prompt_masks length."
            )
        # Store back the (possibly newly created) frame index list
        renders["prompt_frame_idxs"] = prompt_frame_idxs

        # Propagation anchors may differ from identity anchors.
        propagation_frame_idxs = renders.get("propagation_frame_idxs")
        if propagation_frame_idxs is None:
            propagation_frame_idxs = prompt_frame_idxs.copy()
        elif len(propagation_frame_idxs) != len(prompt_masks):
            raise ValueError(
                "propagation_frame_idxs length must match prompt_masks length."
            )
        renders["propagation_frame_idxs"] = propagation_frame_idxs
        num_objects = len(prompt_masks)

        # Clamp and sanitize start_obj_id
        start_obj_id = int(start_obj_id)
        if start_obj_id < 0 or start_obj_id > num_objects:
            raise ValueError(
                f"start_obj_id={start_obj_id} is out of range for "
                f"num_objects={num_objects}"
            )

        if num_objects == 0:
            raise RuntimeError("[run_propagation] No objects to track.")

        # ------------------------------------------------------------
        # 1) Prepare tensors for SAM2 (NO normalization here)
        # ------------------------------------------------------------
        # normals / points: NumPy → torch, channel-last → channel-first
        normals = torch.from_numpy(
            renders["normal_imgs"]
        ).permute(0, 3, 1, 2).contiguous()   # (T,3,H,W)

        point_imgs = torch.from_numpy(
            renders["point_imgs"]
        ).permute(0, 3, 1, 2).contiguous()   # (T,3,H,W)

        # view directions
        view_dirs_np = renders["view_dirs"]
        if isinstance(view_dirs_np, list):
            view_dirs_np = np.stack(view_dirs_np, axis=0)
        view_dirs = torch.from_numpy(
            np.asarray(view_dirs_np, dtype=np.float32)
        )  # (T,3)

        num_frames = normals.shape[0]

        print(
            f"[run_propagation] Tracking {num_objects} objects across {num_frames} views "
            f"(recomputing from obj_id={start_obj_id})"
        )

        # ------------------------------------------------------------
        # 2) Initialize SAM2 inference state
        # ------------------------------------------------------------
        inference_state = self.predictor.init_state(
            normal_imgs=normals,
            point_imgs=point_imgs,
            view_dirs=view_dirs,
            # IMPORTANT: reset_state() below is expected to clear all
            # object-specific state (prompts, tracks, memory) while
            # reusing cached image features / embeddings. If the SAM2
            # implementation changes this contract, per-object
            # propagation may produce incorrect results.
            offload_inputs_to_cpu=self.config.sam_mesh.get(
                "offload_inputs_to_cpu", False
            ),
            offload_state_to_cpu=self.config.sam_mesh.get(
                "offload_state_to_cpu", False
            ),
        )

        # ------------------------------------------------------------
        # 3) Prepare storage
        # ------------------------------------------------------------
        renders["bmasks"] = None
        renders["cmasks"] = None

        use_batching = self.config.sam_mesh.get("use_batching", False)
        batch_size = (
            self.config.sam_mesh.batch_size if use_batching else num_objects
        )

        # Try to reuse cached masks for existing objects when start_obj_id > 0.
        # This avoids re-running SAM2 for already-tracked objects across views.
        cached_masks = renders.get("_masks_per_object")
        if start_obj_id == 0 or cached_masks is None:
            # Full recomputation: initialize fresh storage.
            masks_per_object = {
                obj_id: [None] * num_frames for obj_id in range(num_objects)
            }
        else:
            # Reuse cached masks for objects [0, start_obj_id), and allocate
            # empty slots for new objects in [start_obj_id, num_objects).
            masks_per_object = cached_masks
            # Sanity check cache shape; if mismatch, fall back to full recompute.
            try:
                any_obj_id = next(iter(masks_per_object))
                if len(masks_per_object[any_obj_id]) != num_frames:
                    raise RuntimeError
            except Exception:
                masks_per_object = {
                    obj_id: [None] * num_frames for obj_id in range(num_objects)
                }
                start_obj_id = 0
            else:
                # Ensure we have entries for all objects, including new ones.
                for obj_id in range(num_objects):
                    if obj_id not in masks_per_object:
                        masks_per_object[obj_id] = [None] * num_frames
                # When we are recomputing a suffix [start_obj_id, num_objects),
                # clear any stale masks for those objects to avoid mixing
                # previous runs with the current propagation.
                if start_obj_id > 0:
                    for obj_id in range(start_obj_id, num_objects):
                        masks_per_object[obj_id] = [None] * num_frames

        if use_batching:
            print(f"[run_propagation] Batching enabled (batch size = {batch_size})")
        else:
            print("[run_propagation] Batching disabled (single batch)")

        # ------------------------------------------------------------
        # 4) Batch-wise propagation
        # ------------------------------------------------------------
        # Only propagate objects in [start_obj_id, num_objects); others reuse cache.
        for batch_start in range(start_obj_id, num_objects, batch_size):
            batch_end = min(batch_start + batch_size, num_objects)
            batch_indices = list(range(batch_start, batch_end))

            self.predictor.reset_state(inference_state)

            # ---- add prompt masks for this batch ----
            for obj_id in batch_indices:
                mask_2d = prompt_masks[obj_id].astype(bool)
                frame_idx = int(propagation_frame_idxs[obj_id])

                if frame_idx < 0 or frame_idx >= num_frames:
                    raise ValueError(
                        f"propagation_frame_idxs[{obj_id}]={frame_idx} is out of "
                        f"range for num_frames={num_frames}"
                    )

                self.predictor.add_new_mask(
                    inference_state=inference_state,
                    frame_idx=frame_idx,
                    obj_id=obj_id,
                    mask=mask_2d.astype(np.uint8),
                )

            # ---- propagate across views ----
            for frame_idx, out_obj_ids, out_mask_logits in (
                self.predictor.propagate_in_video(
                    inference_state,
                    start_frame_idx=0,
                    max_frame_num_to_track=num_frames,
                )
            ):
                video_masks = (out_mask_logits > 0).cpu().numpy()[:, 0]

                for j, returned_id in enumerate(out_obj_ids):
                    returned_id = int(returned_id)
                    masks_per_object[returned_id][frame_idx] = video_masks[j]

        # ------------------------------------------------------------
        # 5) Build bmasks per view (canonical object order)
        # ------------------------------------------------------------
        # Always use a canonical ordering 0..num_objects-1 so that:
        # - Outputs are stable across runs / batchings.
        # - We do not depend on inference_state["obj_ids"], which may only
        #   reflect the last propagated batch after reset_state().
        obj_ids_ordered = list(range(num_objects))

        # Canonical empty mask shape: fall back to prompt mask resolution.
        # All propagated masks should share this (H, W).
        ref_mask_shape = prompt_masks[0].shape
        empty_mask = np.zeros(ref_mask_shape, dtype=bool)
        bmasks_per_view = []
        for v in range(num_frames):
            view_masks = []
            for obj_id in obj_ids_ordered:
                mask = masks_per_object[obj_id][v]

                if mask is None:
                    # Prefer using an existing mask for this object (any frame)
                    # to infer shape; if none exist (e.g. object never visible),
                    # fall back to the canonical prompt mask shape.
                    ref = next(
                        (m for m in masks_per_object[obj_id] if m is not None),
                        None,
                    )
                    if ref is not None:
                        mask = np.zeros_like(ref, dtype=bool)
                    else:
                        mask = empty_mask.copy()

                view_masks.append(mask)

            bmasks_per_view.append(np.stack(view_masks, axis=0))

        # Invariants: we must have contiguous object IDs [0, num_objects)
        # and every view must contain exactly num_objects instance masks.
        expected_ids = set(range(num_objects))
        if set(masks_per_object.keys()) != expected_ids:
            raise RuntimeError(
                f"masks_per_object keys {sorted(masks_per_object.keys())} "
                f"do not match expected 0..{num_objects - 1}"
            )
        for v, bmasks in enumerate(bmasks_per_view):
            if bmasks.shape[0] != num_objects:
                raise RuntimeError(
                    f"bmasks_per_view[{v}] has {bmasks.shape[0]} objects, "
                    f"expected {num_objects}"
                )

        # ------------------------------------------------------------
        # 5b) Enforce instance identity: no overlap beyond threshold
        #     (objects separate in view 0 stay separate in later views)
        # ------------------------------------------------------------
        overlap_thresh = int(
            self.config.sam_mesh.get("propagation_overlap_thresh", 0)
        )
        if overlap_thresh > 0:
            for v in range(num_frames):
                _enforce_instance_identity_overlap(
                    bmasks_per_view[v], overlap_thresh
                )

        # ------------------------------------------------------------
        # 6) Build cmasks per view (2D cleanup only)
        # ------------------------------------------------------------
        min_area = int(self.config.sam_mesh.get("min_area_2d", 1024))
        cmasks_per_view = []

        for v in range(num_frames):
            bmasks = bmasks_per_view[v]
            faces = renders["faces"][v]

            cmask = combine_bmasks(bmasks)
            cmask[faces == -1] = 0

            cmasks_per_view.append(cmask)

        renders["bmasks"] = bmasks_per_view
        renders["cmasks"] = cmasks_per_view
        # Cache per-object masks so residual discovery can avoid
        # re-propagating existing objects in later iterations.
        renders["_masks_per_object"] = masks_per_object
        # Store inference state for residual discovery: add_new_mask + single propagate.
        renders["_inference_state"] = inference_state

        return renders
   
    def lift_geosam2(
        self,
        mesh,
        renders,
        rng_seed=0,
    ):
        """
        Exact GeoSAM2 first lifting stage (Supplement 8.1.1).

        Returns:
            face2label : dict[int, int]
        """
        rng = np.random.RandomState(rng_seed)

        # Hyperparameters from config (sam_mesh)
        depth_tol = float(self.config.sam_mesh.get("lift_depth_tolerance", 0.001))
        points_per_face = int(self.config.sam_mesh.get("lift_points_per_face", 5))
        lift_view_weight_min = float(self.config.sam_mesh.get("lift_view_weight_min", 0.1))
        lift_view_weight_max = float(self.config.sam_mesh.get("lift_view_weight_max", 1.0))
        lift_view_confidence_gamma = float(self.config.sam_mesh.get("lift_view_confidence_gamma", 1.5))
        # Exponential anchor-distance decay scale: larger tau = slower decay, tau→∞ ≈ no anchor weighting.
        lift_anchor_tau = float(self.config.sam_mesh.get("lift_anchor_tau", 3.0))
        # Optional extra multiplicative boost for anchor frames themselves.
        lift_anchor_boost = float(self.config.sam_mesh.get("lift_anchor_boost", 1.8))
        lift_face_confidence_thresh = float(self.config.sam_mesh.get("lift_face_confidence_thresh", 0.6))
        lift_face_min_votes = float(self.config.sam_mesh.get("lift_face_min_votes", 2.0))

        vertices = mesh.vertices
        faces = mesh.faces
        num_faces = faces.shape[0]
        num_views = len(renders["cmasks"])

        # ------------------------------------------------------------
        # 0) Early face pruning: keep only faces that are ever labeled
        # ------------------------------------------------------------
        # A face can only receive non-zero votes if at least one view has a
        # non-zero cmask label on some pixel that belongs to that face.
        # Faces that are never labeled would end up with zero votes and be
        # assigned background (0) anyway, so we can skip them entirely in
        # the expensive geometric lifting loops.
        active_face_mask = np.zeros(num_faces, dtype=bool)
        for v in range(num_views):
            faces_v = renders["faces"][v]  # per-pixel face indices, -1 = background
            cmask_v = renders["cmasks"][v]
            # Pixels where some object label is present on a valid face.
            active_pixels = (faces_v >= 0) & (cmask_v > 0)
            if not np.any(active_pixels):
                continue
            active_ids = np.unique(faces_v[active_pixels])
            active_ids = active_ids[active_ids >= 0]
            if len(active_ids) > 0:
                active_face_mask[active_ids] = True

        active_face_indices = np.nonzero(active_face_mask)[0]

        def _compute_anchor_distances(num_views_: int, propagation_frame_idxs) -> np.ndarray:
            """
            For each view index v in [0, num_views_), compute its integer distance
            to the nearest *anchor view* used during SAM2 propagation.

            Anchor views are frames where at least one object received an explicit
            SAM/AMG prompt (possibly via residual completion). All other views are
            obtained purely by SAM2 propagation along the view path, so uncertainty
            should grow monotonically with propagation distance from these anchors.

            This function is order-invariant w.r.t. object IDs: it only depends on
            the set of anchor frame indices, not on how prompts are grouped into objects.
            """
            if propagation_frame_idxs is None:
                return np.zeros(num_views_, dtype=np.float32)

            # Unique, sorted anchor frame indices within [0, num_views_).
            anchors = sorted(
                {
                    int(a)
                    for a in propagation_frame_idxs
                    if 0 <= int(a) < num_views_
                }
            )
            if not anchors:
                # No explicit anchors recorded (should not happen in practice) –
                # fall back to uniform reliability across views.
                return np.zeros(num_views_, dtype=np.float32)

            view_indices = np.arange(num_views_, dtype=np.float32)
            # Naive but tiny: O(num_views * num_anchors), num_views is typically small (≤ 32).
            distances = np.full(num_views_, np.inf, dtype=np.float32)
            for a in anchors:
                distances = np.minimum(distances, np.abs(view_indices - float(a)))
            return distances

        def sample_triangle_points(v0, v1, v2, k=5, rng=None):
            """
            Uniformly sample k points inside a triangle.
            """
            if rng is None:
                rng = np.random

            u = rng.rand(k, 1)
            v = rng.rand(k, 1)
            mask = (u + v) > 1.0
            u[mask] = 1.0 - u[mask]
            v[mask] = 1.0 - v[mask]
            w = 1.0 - u - v

            return u * v0 + v * v1 + w * v2

        # Pre-sample points per face (ONCE) — only for active faces.
        face_points = [None] * num_faces
        for f in active_face_indices:
            v0, v1, v2 = vertices[faces[f]]
            pts = sample_triangle_points(
                v0, v1, v2, k=points_per_face, rng=rng
            )
            face_points[f] = pts

        # View confidence: (labeled fg pixels) / (visible fg pixels). Gamma emphasizes clean views.
        # We further down-weight views that are far (in propagation steps) from any
        # semantic anchor frame, so uncertainty accumulation is modeled explicitly
        # instead of relying on a hand-coded "early view" heuristic.
        propagation_frame_idxs = renders.get("propagation_frame_idxs")
        anchor_distances = _compute_anchor_distances(num_views, propagation_frame_idxs)
        if lift_anchor_tau > 0.0:
            # Anchor-based reliability: distance 0 → weight 1, larger distance → exponential decay.
            w_anchor_all = np.exp(-anchor_distances / lift_anchor_tau)
        else:
            # tau <= 0 disables anchor weighting (keeps pure coverage-based confidence).
            w_anchor_all = np.ones_like(anchor_distances, dtype=np.float32)

        # For optional anchor amplification: treat anchor frames themselves as slightly
        # more reliable than equally distant non-anchor frames (cleaner prompts).
        anchor_set = set(
            int(a)
            for a in (propagation_frame_idxs or [])
            if 0 <= int(a) < num_views
        )

        view_weight = []
        for v in range(num_views):
            faces_v = renders["faces"][v]  # -1 = background
            cmask_v = renders["cmasks"][v]
            foreground = faces_v != -1
            labeled_fg = foreground & (cmask_v > 0)
            confidence = labeled_fg.sum() / max(foreground.sum(), 1)
            # Coverage-based confidence (how cleanly this view covers visible foreground)
            # combined multiplicatively with anchor-based reliability (how far we are
            # from the nearest prompted view along the propagation chain).
            w_coverage = confidence ** lift_view_confidence_gamma
            w_anchor = float(w_anchor_all[v])
            # Optional: amplify anchor frames themselves (on top of distance-based decay).
            if lift_anchor_boost > 1.0 and v in anchor_set:
                w_anchor *= lift_anchor_boost
            w = float(np.clip(
                w_coverage * w_anchor,
                lift_view_weight_min,
                lift_view_weight_max,
            ))
            view_weight.append(w)

        # Storage: per-face weighted vote sums per label (weighted voting instead of plain majority).
        face_point_votes = [defaultdict(float) for _ in range(num_faces)]

        # Use vectorized lifting by default. Config can override with
        # sam_mesh.lift_vectorized: true/false.
        use_vectorized_lift = bool(
            self.config.sam_mesh.get("lift_vectorized", True)
        )

        # Loop over views
        for v in range(num_views):
            cmask = renders["cmasks"][v]
            depth = renders["depth_raw"][v]
            pose = renders["poses"][v]
            norms_v = renders["norms"][v]
            H, W = cmask.shape

            K = renders["intrinsics"][v]
            # Restrict lifting to front-facing pixels (huge win for legs, wings, stems)
            cam2world = np.linalg.inv(pose) if np.linalg.det(pose) > 0 else pose
            front_facing = compute_front_facing_mask(
                np.asarray(norms_v, dtype=np.float64),
                np.asarray(cam2world, dtype=np.float64),
                threshold=0.0,
            )

            if not use_vectorized_lift:
                # Original per-face lifting (with early face pruning).
                for f in active_face_indices:
                    pts = face_points[f]

                    u, v_pix, valid, depth_proj = project_world_to_cam_pixels(
                        pts, pose, K, H, W
                    )

                    if len(u) == 0:
                        continue

                    valid = np.asarray(valid, dtype=bool)
                    u = np.asarray(u)[valid].astype(int)
                    v_pix = np.asarray(v_pix)[valid].astype(int)
                    depth_proj = np.asarray(depth_proj)[valid]

                    depth_err = np.abs(depth_proj - depth[v_pix, u])
                    visible = (
                        (depth_err / (depth[v_pix, u] + 1e-8) < depth_tol)
                        & front_facing[v_pix, u]
                    )

                    if not np.any(visible):
                        continue

                    labels = cmask[v_pix[visible], u[visible]]

                    for lbl in labels:
                        if lbl != 0:
                            # Weighted vote: confident views (high view_weight[v]) dominate.
                            face_point_votes[f][int(lbl)] += view_weight[v]
            else:
                # Vectorized lifting: project all active-face points for this view in one call
                # and aggregate votes in a batched manner.
                if len(active_face_indices) == 0:
                    continue

                # Stack all sampled points for active faces: shape (F_active * K, 3)
                all_points = np.concatenate(
                    [face_points[f] for f in active_face_indices],
                    axis=0,
                )
                # Track which face each point belongs to.
                face_ids_for_points = np.repeat(
                    active_face_indices, points_per_face
                )

                u_all, v_all, valid_all, depth_proj_all = project_world_to_cam_pixels(
                    all_points, pose, K, H, W
                )

                if len(u_all) == 0:
                    continue

                valid_all = np.asarray(valid_all, dtype=bool)
                idx_valid = np.nonzero(valid_all)[0]
                if len(idx_valid) == 0:
                    continue

                u_all = np.asarray(u_all).astype(int)
                v_all = np.asarray(v_all).astype(int)
                depth_proj_all = np.asarray(depth_proj_all)
                face_ids_valid = face_ids_for_points[idx_valid]

                depth_target = depth[v_all, u_all]
                depth_err = np.abs(depth_proj_all - depth_target)
                visible = (
                    (depth_err / (depth_target + 1e-8) < depth_tol)
                    & front_facing[v_all, u_all]
                )

                if not np.any(visible):
                    continue

                visible_idx = np.nonzero(visible)[0]
                faces_visible = face_ids_valid[visible_idx]
                labels_visible = cmask[
                    v_all[visible_idx],
                    u_all[visible_idx],
                ]

                for f_id, lbl in zip(faces_visible, labels_visible):
                    if lbl != 0:
                        face_point_votes[int(f_id)][int(lbl)] += view_weight[v]

        # Final face labels: weighted max (not plain majority) so clean views dominate
        # and thin structures (legs, stems) are not merged by noisy propagation views.
        face2label = {}
        eps = 1e-10
        face_confidence = [0.0] * num_faces

        for f in range(num_faces):
            votes = face_point_votes[f]
            if votes:
                best_votes = max(votes.values())
                total_votes = sum(votes.values())
                face_confidence[f] = best_votes / max(total_votes, eps)
                face2label[f] = max(votes.items(), key=lambda x: x[1])[0]
            else:
                face2label[f] = 0

        # Prune low-confidence / low-support faces to unlabeled (noise → 0); do not assign alternative.
        for f in range(num_faces):
            total_votes = sum(face_point_votes[f].values()) if face_point_votes[f] else 0.0
            if face_confidence[f] < lift_face_confidence_thresh or total_votes < lift_face_min_votes:
                face2label[f] = 0

        # Recover topology: fill only fully surrounded unlabeled components (preserves thin parts).
        face2label = self.fill_unlabeled_components_by_boundary(face2label)

        # Store for smooth_repartition_faces (repartition unary cost softening).
        self._face_confidence = face_confidence

        return face2label

    def smooth_face_labels(
        self,
        face2label: dict[int, int],
        mesh_graph: dict[int, set[int]],
        num_iters: int = 1,
        ignore_label: int = 0,
    ):
        """
        Majority-vote smoothing on mesh face labels.
        """
        labels = face2label.copy()

        for _ in range(num_iters):
            new_labels = labels.copy()

            for f, lbl in labels.items():
                if lbl == ignore_label:
                    continue

                neighbors = mesh_graph.get(f, [])
                neigh_labels = [
                    labels[n]
                    for n in neighbors
                    if labels[n] != ignore_label
                ]

                if len(neigh_labels) >= 3:
                    majority = Counter(neigh_labels).most_common(1)[0][0]
                    new_labels[f] = majority

            labels = new_labels

        return labels


    def remove_small_components_relative(
        self,
        face2label,
        mesh_graph,
        mesh,
        size_ratio=0.05,
        area_ratio=0.05,
    ):
        """
        Remove tiny connected components per label: for each label, remove only
        components that are small relative to that label's largest component.
        Preserves small-but-semantic parts (e.g. wings) when they are the only
        component of their label.
        """
        components = self.label_components(face2label)

        # Group components by label (skip unlabeled)
        label_to_comps = defaultdict(list)
        for comp in components:
            if not comp:
                continue
            lbl = face2label[next(iter(comp))]
            if lbl == 0:
                continue
            label_to_comps[lbl].append(comp)

        new = face2label.copy()

        for label, comps in label_to_comps.items():
            areas = [sum(mesh.area_faces[f] for f in c) for c in comps]
            sizes = [len(c) for c in comps]
            max_area = max(areas)
            max_size = max(sizes)

            for comp, area, size in zip(comps, areas, sizes):
                if size < max_size * size_ratio and area < max_area * area_ratio:
                    for f in comp:
                        new[f] = 0

        return new

    def label_components(self, face2label: dict) -> list[set]:
        """
        """
        components = []
        visited = set()

        def dfs(source: int):
            stack = [source]
            components.append({source})
            visited.add(source)
            
            while stack:
                node = stack.pop()
                for adj in self.mesh_graph[node]:
                    if adj not in visited and adj in face2label and face2label[adj] == face2label[node]:
                        stack.append(adj)
                        components[-1].add(adj)
                        visited.add(adj)

        for face in range(len(self.renderer.tmesh.faces)):
            if face not in visited and face in face2label:
                dfs(face)
        return components

    def fill_unlabeled_components_by_boundary(self, face2label):
        """
        Fill unlabeled (0) face components ONLY if they are fully enclosed
        by exactly one neighboring label.
        """

        labels = face2label.copy()
        visited = set()
        num_faces = len(self.renderer.tmesh.faces)

        for f in range(num_faces):
            if labels.get(f, 0) != 0 or f in visited:
                continue

            # BFS to extract one unlabeled connected component
            stack = [f]
            component = set()
            boundary_labels = Counter()

            while stack:
                cur = stack.pop()
                if cur in visited:
                    continue

                visited.add(cur)
                component.add(cur)

                for nb in self.mesh_graph[cur]:
                    nb_label = labels.get(nb, 0)
                    if nb_label == 0:
                        stack.append(nb)
                    else:
                        boundary_labels[nb_label] += 1

            # ✅ SAFE RULE:
            # fill ONLY if surrounded by exactly one label
            if len(boundary_labels) == 1:
                target = boundary_labels.most_common(1)[0][0]
                for cf in component:
                    labels[cf] = target

        return labels

    def merge_labels_by_cross_view_overlap(
        self,
        face2label: dict[int, int],
        renders: dict,
    ) -> dict[int, int]:
        """
        Fast and correct cross-view label merge.
        Reads cmasks, faces from renders; uses self.mesh_graph and config.sam_mesh for thresholds.
        """
        cmasks = renders["cmasks"]
        faces = renders["faces"]
        mesh_graph = self.mesh_graph
        cfg = self.config.sam_mesh
        min_shared_faces = int(cfg.get("merge_min_shared_faces", 150))
        overlap_ratio_thresh = float(cfg.get("merge_overlap_ratio_thresh", 0.6))
        min_supporting_views = int(cfg.get("merge_min_support_views", 4))
        min_adjacent_shared_faces = int(cfg.get("merge_min_adjacent_shared_faces", 5))
        min_compactness = float(cfg.get("merge_min_compactness", 2.0))
        min_pixels_per_face = int(cfg.get("merge_min_pixels_per_face", 10))
        debug_merge_groups = bool(cfg.get("merge_debug_groups", False))

        # ------------------------------------------------------------------
        # 1) Build:
        #   label_faces[label] -> set(face_id)
        #   label_face_views[(label, face)] -> set(view_idx)
        # ------------------------------------------------------------------
        label_faces = defaultdict(set)
        label_face_views = defaultdict(set)

        for v in range(len(cmasks)):
            cm = np.asarray(cmasks[v], dtype=np.int32)
            fv = np.asarray(faces[v], dtype=np.int32)
            if cm.shape != fv.shape:
                continue

            valid = (fv >= 0) & (cm > 0)
            if not np.any(valid):
                continue

            pairs = np.stack([fv[valid], cm[valid]], axis=1)

            # count pixels per (face,label)
            uniq, counts = np.unique(pairs, axis=0, return_counts=True)
            for (face_id, label_id), cnt in zip(uniq, counts):
                if cnt < min_pixels_per_face:
                    continue
                face_id = int(face_id)
                label_id = int(label_id)
                label_faces[label_id].add(face_id)
                label_face_views[(label_id, face_id)].add(v)

        if not label_faces:
            return dict(face2label)

        # ------------------------------------------------------------------
        # 2) Candidate label pairs via shared faces
        # ------------------------------------------------------------------
        face_to_labels = defaultdict(list)
        for lab, fset in label_faces.items():
            for f in fset:
                face_to_labels[f].append(lab)

        candidate_pairs = defaultdict(set)
        for f, labs in face_to_labels.items():
            if len(labs) < 2:
                continue
            for i in range(len(labs)):
                for j in range(i + 1, len(labs)):
                    l1, l2 = sorted((labs[i], labs[j]))
                    candidate_pairs[(l1, l2)].add(f)

        # ------------------------------------------------------------------
        # 3) Union-find (deterministic)
        # ------------------------------------------------------------------
        parent = {l: l for l in label_faces}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)

        # ------------------------------------------------------------------
        # 4) Evaluate candidate merges
        # ------------------------------------------------------------------
        for (l1, l2), shared_faces in candidate_pairs.items():
            if len(shared_faces) < min_shared_faces:
                continue

            ratio = len(shared_faces) / min(len(label_faces[l1]), len(label_faces[l2]))
            if ratio < overlap_ratio_thresh:
                continue

            # ---- multi-view support: same face seen in same view ----
            supporting_views = set()
            for f in shared_faces:
                views1 = label_face_views.get((l1, f), set())
                views2 = label_face_views.get((l2, f), set())
                supporting_views |= (views1 & views2)
                if len(supporting_views) >= min_supporting_views:
                    break

            if len(supporting_views) < min_supporting_views:
                continue

            # ---- compactness ----
            boundary = 0
            for f in shared_faces:
                for nb in mesh_graph.get(f, ()):
                    if nb not in shared_faces:
                        boundary += 1
            compactness = len(shared_faces) / (boundary ** 0.5 + 1e-6)
            if compactness < min_compactness:
                continue

            # ---- locality ----
            adjacent = 0
            for f in shared_faces:
                if any(nb in shared_faces for nb in mesh_graph.get(f, ())):
                    adjacent += 1
                    if adjacent >= min_adjacent_shared_faces:
                        union(l1, l2)
                        break

        # ------------------------------------------------------------------
        # 5) Relabel faces
        # ------------------------------------------------------------------
        new_face2label = {}
        for f, lab in face2label.items():
            new_face2label[f] = 0 if lab == 0 else find(lab)

        # ------------------------------------------------------------------
        # 6) Debug
        # ------------------------------------------------------------------
        if debug_merge_groups:
            groups = defaultdict(list)
            for l in label_faces:
                groups[find(l)].append(l)
            print("Merged label groups:")
            for rep, members in sorted(groups.items()):
                if len(members) > 1:
                    print(f"  {members} → {rep}")

        return new_face2label

    def _extract_residual_components(
        self,
        renders: dict,
        *,
        min_missing_ratio: float,
        min_area: int,
    ) -> list[dict]:
        """
        Extract per-view residual connected components and map each one to
        mesh faces visible in that view.
        """
        cmasks = renders.get("cmasks")
        faces = renders.get("faces")
        if cmasks is None or faces is None:
            return []

        residual_components = []

        for v, (cmask, face) in enumerate(zip(cmasks, faces)):
            foreground = (face != -1)
            missing = foreground & (cmask == 0)

            fg_pixels = int(foreground.sum())
            if fg_pixels == 0:
                continue

            missing_ratio = float(missing.sum()) / max(fg_pixels, 1)
            if missing_ratio < min_missing_ratio:
                continue

            labeled, n_cc = ndi.label(missing)
            if n_cc == 0:
                continue

            for cc_id in range(1, n_cc + 1):
                comp_mask = (labeled == cc_id)
                area = int(comp_mask.sum())
                if area < min_area:
                    continue

                comp_faces = face[comp_mask]
                comp_faces = comp_faces[comp_faces >= 0]
                if comp_faces.size == 0:
                    continue

                face_set = set(int(f) for f in np.unique(comp_faces))
                residual_components.append(
                    {
                        "view": int(v),
                        "mask": comp_mask,
                        "area": area,
                        "faces": face_set,
                    }
                )

        return residual_components

    def _cluster_residual_components_by_faces(
        self,
        components: list[dict],
        *,
        min_support_views: int,
        min_shared_faces: int,
        min_face_overlap_ratio: float,
    ) -> list[dict]:
        """
        Cluster residual components from different views using face-set overlap.
        """
        if not components:
            return []

        n = len(components)
        parent = list(range(n))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)

        for i in range(n):
            faces_i = components[i]["faces"]
            if not faces_i:
                continue
            for j in range(i + 1, n):
                # Components from the same view are separate residuals by construction.
                if components[i]["view"] == components[j]["view"]:
                    continue

                faces_j = components[j]["faces"]
                if not faces_j:
                    continue

                inter = len(faces_i & faces_j)
                if inter < min_shared_faces:
                    continue

                overlap = inter / max(min(len(faces_i), len(faces_j)), 1)
                if overlap >= min_face_overlap_ratio:
                    union(i, j)

        groups = defaultdict(list)
        for idx in range(n):
            groups[find(idx)].append(idx)

        clusters = []
        for group_indices in groups.values():
            per_view_masks = {}
            per_view_areas = defaultdict(int)
            faces_union = set()
            total_area = 0

            for idx in group_indices:
                comp = components[idx]
                view = int(comp["view"])
                mask = comp["mask"]
                area = int(comp["area"])

                faces_union |= comp["faces"]
                total_area += area
                per_view_areas[view] += area

                if view in per_view_masks:
                    per_view_masks[view] = np.logical_or(per_view_masks[view], mask)
                else:
                    per_view_masks[view] = mask.copy()

            support_views = len(per_view_masks)
            if support_views < min_support_views:
                continue

            anchor_view = max(
                per_view_areas.keys(),
                key=lambda vv: (per_view_areas[vv], -vv),
            )
            anchor_area = int(per_view_areas[anchor_view])

            clusters.append(
                {
                    "support_views": support_views,
                    "views": sorted(per_view_masks.keys()),
                    "faces": faces_union,
                    "anchor_view": int(anchor_view),
                    "anchor_mask": per_view_masks[anchor_view],
                    "anchor_area": anchor_area,
                    "total_area": int(total_area),
                    "num_components": len(group_indices),
                }
            )

        clusters.sort(
            key=lambda c: (c["support_views"], c["anchor_area"], c["total_area"]),
            reverse=True,
        )
        return clusters

    def _build_view0_prompt_face_sets(self, renders: dict) -> list[set[int]]:
        """
        Build face sets for original AMG prompts in view 0 only.
        """
        faces = renders.get("faces")
        if not faces:
            return []

        faces0 = np.asarray(faces[0], dtype=np.int32)
        prompt_masks_view0 = renders.get("prompt_masks_view0")
        if prompt_masks_view0 is None:
            prompt_masks_view0 = renders.get("prompt_masks", [])

        num_seed_objects = int(
            renders.get("num_seed_objects", len(prompt_masks_view0))
        )
        num_seed_objects = min(num_seed_objects, len(prompt_masks_view0))

        prompt_face_sets = []
        for obj_id in range(num_seed_objects):
            mask = np.asarray(prompt_masks_view0[obj_id], dtype=bool)
            if mask.shape != faces0.shape:
                prompt_face_sets.append(set())
                continue

            obj_faces = faces0[mask]
            obj_faces = obj_faces[obj_faces >= 0]
            if obj_faces.size == 0:
                prompt_face_sets.append(set())
                continue

            prompt_face_sets.append(set(int(f) for f in np.unique(obj_faces)))

        return prompt_face_sets

    def _infer_residual_owner_from_view0_faces(
        self,
        residual_faces: set[int],
        prompt_face_sets: list[set[int]],
        *,
        min_overlap_ratio: float,
        min_margin: float,
    ) -> tuple[int | None, float, float]:
        """
        Owner inference from face overlap with immutable view-0 seed prompts.
        """
        if not residual_faces or not prompt_face_sets:
            return None, 0.0, 0.0

        denom = max(len(residual_faces), 1)
        scores = []

        for obj_id, obj_faces in enumerate(prompt_face_sets):
            if not obj_faces:
                scores.append((obj_id, 0.0))
                continue
            inter = len(residual_faces & obj_faces)
            scores.append((obj_id, inter / denom))

        scores.sort(key=lambda x: x[1], reverse=True)
        best_id, best_score = scores[0]
        second_score = scores[1][1] if len(scores) > 1 else 0.0

        if best_score >= min_overlap_ratio and (best_score - second_score) >= min_margin:
            return int(best_id), float(best_score), float(second_score)

        return None, float(best_score), float(second_score)

    def discover_residual_prompts(
        self,
        renders: dict,
        *,
        min_missing_ratio: float = 0.03,
        min_area: int = 800,
    ) -> list[dict]:
        """
        Discover and cluster missing residual components across views using face
        overlap. Returns residual clusters with anchor view + anchor mask.
        """
        cmasks = renders.get("cmasks")
        faces = renders.get("faces")
        if cmasks is None or faces is None:
            print(
                "[discover_residual_prompts] cmasks or faces is None; "
                "skipping residual discovery."
            )
            return []

        min_support_views = int(
            self.config.sam_mesh.get("residual_min_support_views", 2)
        )
        min_shared_faces = int(
            self.config.sam_mesh.get("residual_cluster_min_shared_faces", 10)
        )
        min_face_overlap_ratio = float(
            self.config.sam_mesh.get("residual_cluster_face_overlap_ratio", 0.2)
        )

        residual_components = self._extract_residual_components(
            renders,
            min_missing_ratio=min_missing_ratio,
            min_area=min_area,
        )
        clusters = self._cluster_residual_components_by_faces(
            residual_components,
            min_support_views=min_support_views,
            min_shared_faces=min_shared_faces,
            min_face_overlap_ratio=min_face_overlap_ratio,
        )

        print(
            "[discover_residual_prompts] "
            f"components={len(residual_components)}, clusters={len(clusters)}, "
            f"min_area={min_area}, min_missing_ratio={min_missing_ratio}, "
            f"min_support_views={min_support_views}"
        )
        return clusters

    def run_residual_propagation(self, renders: dict):
        """
        Propagate ONLY the newly discovered residual object
        using a FRESH semantic memory but SHARED visual features.

        ONLY for NEW objects. Residuals that are assigned to an existing
        seed object in residual_completion must use
        run_propagation(renders, start_obj_id=owner) instead.

        Strategy:
        - Use the best residual prompt (last added)
        - Backward propagation → fills earlier views
        - Reset semantic memory
        - Forward propagation → fills later views
        - Earlier views keep backward result
        - Later views use forward result
        """

        # --------------------------------------------------
        # 1) Residual prompt (last added)
        # --------------------------------------------------
        obj_id = len(renders["prompt_masks"]) - 1
        propagation_frame_idxs = renders.get(
            "propagation_frame_idxs",
            renders["prompt_frame_idxs"],
        )
        frame_idx = int(propagation_frame_idxs[obj_id])
        mask_2d = renders["prompt_masks"][obj_id].astype(np.uint8)

        num_frames = len(renders["normal_imgs"])

        # --------------------------------------------------
        # 2) Reuse cached inference state (visual features)
        # --------------------------------------------------
        inference_state = renders.get("_inference_state")
        if inference_state is None:
            raise RuntimeError(
                "run_residual_propagation requires an existing inference_state. "
                "Call run_propagation() before residual discovery."
            )

        # storage for final residual masks
        masks_per_view = [None] * num_frames

        # ==================================================
        # BACKWARD PASS
        # ==================================================
        self.predictor.reset_state(inference_state)

        # residual becomes object 0
        self.predictor.add_new_mask(
            inference_state=inference_state,
            frame_idx=frame_idx,
            obj_id=0,
            mask=mask_2d,
        )

        for fidx, _, out_mask_logits in self.predictor.propagate_in_video(
            inference_state,
            start_frame_idx=frame_idx,
            max_frame_num_to_track=frame_idx + 1,
            reverse=True,
        ):
            masks_per_view[fidx] = (
                out_mask_logits > 0
            ).cpu().numpy()[0, 0]

        # ==================================================
        # FORWARD PASS
        # ==================================================
        self.predictor.reset_state(inference_state)

        self.predictor.add_new_mask(
            inference_state=inference_state,
            frame_idx=frame_idx,
            obj_id=0,
            mask=mask_2d,
        )

        for fidx, _, out_mask_logits in self.predictor.propagate_in_video(
            inference_state,
            start_frame_idx=frame_idx,
            max_frame_num_to_track=num_frames - frame_idx,
            reverse=False,
        ):
            masks_per_view[fidx] = (
                out_mask_logits > 0
            ).cpu().numpy()[0, 0]

        return masks_per_view
    
    def inject_residual_object(
        self,
        renders: dict,
        residual_masks_per_view: list[np.ndarray],
    ):
        """
        Inject residual object as a NEW semantic label = max_label + 1.
        Residual overwrites previous labels wherever present.
        """

        bmasks = renders["bmasks"]
        #cmasks = renders["cmasks"]

        num_views = len(bmasks)
        assert len(residual_masks_per_view) == num_views

        # --------------------------------------------------
        # 1) Find next available global label
        # --------------------------------------------------
        # max_label = 0
        # for cm in cmasks:
        #     if cm.size > 0:
        #         max_label = max(max_label, int(cm.max()))

        # new_label = max_label + 1

        # --------------------------------------------------
        # 2) Inject residual per view
        # --------------------------------------------------
        for v in range(num_views):
            res_mask = residual_masks_per_view[v]
            if res_mask is None or res_mask.sum() == 0:
                continue

            # --- overwrite cmask ---
            #cmasks[v][res_mask] = new_label

            # --- remove residual pixels from existing objects ---
            # for obj_id in range(bmasks[v].shape[0]):
            #     bmasks[v][obj_id][res_mask] = False

            # --- append residual as new object ---
            bmasks[v] = np.concatenate(
                [bmasks[v], res_mask[None]],
                axis=0,
            )

        renders["bmasks"] = bmasks
        #renders["cmasks"] = cmasks
        #renders["_last_added_label"] = new_label  # optional debug/info

        return renders

    def residual_completion(
        self,
        renders: dict,
        
    ):
        """
        Iteratively discover clustered residuals, assign them to a seed object
        by view-0 face overlap when confident, otherwise introduce a new object.
        """

        # Read residual completion hyperparameters from config.
        max_rounds = int(
            self.config.sam_mesh.get("residual_max_rounds", 3)
        )

        min_missing_ratio = float(
            self.config.sam_mesh.get("residual_min_uncovered_ratio", 0.08)
        )

        min_area = int(
            self.config.sam_mesh.get("residual_min_area_2d", 2000)
        )
        owner_overlap_thresh = float(
            self.config.sam_mesh.get("residual_owner_overlap_ratio", 0.2)
        )
        owner_margin = float(
            self.config.sam_mesh.get("residual_owner_margin", 0.05)
        )

        total_residual_objects = renders.get("_num_residuals", 0)
        num_assigned = 0
        num_new = 0
        renders.setdefault(
            "propagation_frame_idxs",
            renders["prompt_frame_idxs"].copy(),
        )
        prompt_face_sets_view0 = self._build_view0_prompt_face_sets(renders)

        for r in range(max_rounds):
            residual_clusters = self.discover_residual_prompts(
                renders,
                min_missing_ratio=min_missing_ratio,
                min_area=min_area,
            )

            if not residual_clusters:
                print(f"[residual_completion] stop at round {r}: no residuals")
                break

            # Process one best-supported cluster per round.
            cluster = residual_clusters[0]
            view = int(cluster["anchor_view"])
            res_mask = cluster["anchor_mask"]

            print(
                f"[residual_completion] round {r}: "
                f"anchor_view={view} area={cluster['anchor_area']} "
                f"support_views={cluster['support_views']}"
            )

            owner, best_score, second_score = self._infer_residual_owner_from_view0_faces(
                cluster["faces"],
                prompt_face_sets_view0,
                min_overlap_ratio=owner_overlap_thresh,
                min_margin=owner_margin,
            )

            if owner is not None:
                num_assigned += 1
                print(
                    f"[residual_completion] assigned to seed object {owner} "
                    f"(best={best_score:.3f}, second={second_score:.3f})"
                )

                # Re-propagate existing object from anchor view.
                bm_view = renders["bmasks"][view]
                if (
                    bm_view is not None
                    and owner < bm_view.shape[0]
                    and bm_view[owner].shape == res_mask.shape
                ):
                    new_prompt = (bm_view[owner] | res_mask).astype(bool)
                else:
                    new_prompt = res_mask.copy()
                renders["prompt_masks"][owner] = new_prompt
                renders["propagation_frame_idxs"][owner] = view

                # SAM2 stores maskmem_features in bfloat16; when running on CUDA,
                # use autocast so model and memory dtypes match. On CPU, fall
                # back to a no-op context.
                if str(self.device).startswith("cuda") and torch.cuda.is_available():
                    autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                else:
                    autocast_ctx = nullcontext()

                with autocast_ctx:
                    renders = self.run_propagation(
                        renders,
                        start_obj_id=owner,
                    )
            else:
                num_new += 1
                print(
                    f"[residual_completion] new object "
                    f"(best={best_score:.3f}, second={second_score:.3f})"
                )

                renders["prompt_masks"].append(res_mask)
                renders["prompt_frame_idxs"].append(view)
                renders["propagation_frame_idxs"].append(view)

                residual_masks_per_view = self.run_residual_propagation(renders)
                self.inject_residual_object(
                    renders,
                    residual_masks_per_view,
                )
                renders = self.rebuild_cmasks_from_bmasks(renders)
                total_residual_objects += 1

        renders["_num_residuals"] = total_residual_objects

        print(
            f"[residual_completion] residuals assigned to existing: {num_assigned}, "
            f"new objects: {num_new}, total new objects: {total_residual_objects}"
        )
        return renders

    def rebuild_cmasks_from_bmasks(self, renders):
        cmasks = []
        for v in range(len(renders["bmasks"])):
            cmask = combine_bmasks(renders["bmasks"][v])
            cmask[renders["faces"][v] == -1] = 0
            cmasks.append(cmask)
        renders["cmasks"] = cmasks
        return renders

    def resolve_residual_overwrite_by_votes(
        self,
        face2label,
        renders,
        alpha=1.3,
    ):
        """
        Overwrite face labels ONLY if residual has stronger multi-view support.
        Votes are accumulated per-object from bmasks, with overlap pixels
        split fractionally across all overlapping objects to avoid ordering bias.
        """
        bmasks = renders["bmasks"]
        faces  = renders["faces"]

        face_votes = defaultdict(Counter)

        for v, (bm, fv) in enumerate(zip(bmasks, faces)):
            if bm is None:
                continue
            fv = np.asarray(fv)
            valid_faces = fv >= 0
            if not np.any(valid_faces):
                continue

            # Compute per-pixel overlap count for fractional voting
            bm_bool = np.asarray(bm, dtype=bool)
            overlap_counts = bm_bool.sum(axis=0)
            inv_overlap = np.zeros_like(overlap_counts, dtype=np.float32)
            inv_overlap[overlap_counts > 0] = 1.0 / overlap_counts[overlap_counts > 0]

            # Vote per object mask directly (label_id = obj_idx + 1),
            # with overlap pixels split across all active objects.
            for obj_idx in range(bm.shape[0]):
                mask = bm_bool[obj_idx]
                valid = valid_faces & mask
                if not np.any(valid):
                    continue

                face_ids = fv[valid].astype(int)
                weights = inv_overlap[valid]
                uniq, inv = np.unique(face_ids, return_inverse=True)
                counts = np.bincount(inv, weights=weights)
                label_id = obj_idx + 1
                for f, c in zip(uniq, counts):
                    face_votes[int(f)][label_id] += float(c)

        labels = face2label.copy()

        for face, votes in face_votes.items():
            if face not in labels:
                continue

            current = labels[face]
            best_label, best_votes = votes.most_common(1)[0]

            cur_votes = votes.get(current, 0)

            # ✅ overwrite ONLY if residual is clearly stronger
            if best_votes >= alpha * max(cur_votes, 1):
                labels[face] = best_label

        return labels

    def smooth_repartition_faces(self, face2label_consistent: dict, target_labels=None) -> dict:
        """
        """
        tmesh = self.renderer.tmesh

        partition = np.array([face2label_consistent[face] for face in range(len(tmesh.faces))])

        cost_data = np.zeros((len(tmesh.faces), np.max(partition) + 1))
        for f in range(len(tmesh.faces)):
            for l in np.unique(partition):
                cost_data[f, l] = 0 if partition[f] == l else 1

        # Softer unary cost for low-confidence faces so repartition does not over-commit.
        face_confidence = getattr(self, "_face_confidence", None)
        conf_thresh = float(self.config.sam_mesh.get("lift_face_confidence_thresh", 0.6))
        if face_confidence is not None and len(face_confidence) == len(tmesh.faces):
            for f in range(len(tmesh.faces)):
                if face_confidence[f] < conf_thresh:
                    cost_data[f, :] *= 0.3
                    cost_data[f, partition[f]] = 0.1  # don't bias "stay"; keep boundary flexible

        cost_smoothness = -np.log(tmesh.face_adjacency_angles / np.pi + 1e-20)
        
        lambda_seed = self.config.sam_mesh.repartition_lambda
        if target_labels is None:
            refined_partition = repartition(tmesh, partition, cost_data, cost_smoothness, self.config.sam_mesh.repartition_iterations, lambda_seed)
            return {
                face: refined_partition[face] for face in range(len(tmesh.faces))
            }
    
        lambda_range=(
            self.config.sam_mesh.repartition_lambda_lb, 
            self.config.sam_mesh.repartition_lambda_ub
        )
        lambdas = np.linspace(*lambda_range, num=mp.cpu_count())
        chunks = [
            (tmesh, partition, cost_data, cost_smoothness, self.config.sam_mesh.repartition_iterations, _lambda) 
            for _lambda in lambdas
        ]
        with mp.Pool(mp.cpu_count() // 2) as pool:
            refined_partitions = pool.starmap(repartition, chunks)

        def compute_cur_labels(part, noise_threshold=10):
            """
            """
            values, counts = np.unique(part, return_counts=True)
            return values[counts > noise_threshold]

        # lambda crawling algorithm when target_labels is specified i.e. Princeton Mesh Segmentation Benchmark
        max_iteration = 8
        cur_iteration = 0
        cur_lambda_index = np.searchsorted(lambdas, lambda_seed)
        cur_labels = len(compute_cur_labels(refined_partitions[cur_lambda_index]))
        while not (
            target_labels - self.config.sam_mesh.repartition_lambda_tolerance <= cur_labels and
            target_labels + self.config.sam_mesh.repartition_lambda_tolerance >= cur_labels
        ) and cur_iteration < max_iteration:
            
            if cur_labels < target_labels and cur_lambda_index > 0:
                # want more labels so decrease lambda
                cur_lambda_index -= 1
            if cur_labels > target_labels and cur_lambda_index < len(refined_partitions) - 1:
                # want less labels so increase lambda
                cur_lambda_index += 1

            cur_labels = len(compute_cur_labels(refined_partitions[cur_lambda_index]))
            cur_iteration += 1

        print('Repartitioned with ', cur_labels, ' labels aiming for ', target_labels, 'target labels using lambda ', lambdas[cur_lambda_index], ' in ', cur_iteration, ' iterations')
        
        refined_partition = refined_partitions[cur_lambda_index]
        return {
            face: refined_partition[face] for face in range(len(tmesh.faces))
        }

    def split_by_label_connectivity(self, face2label_consistent: dict[int, int]) -> dict[int, int]:
        """
        Ensure that each semantic label corresponds to exactly ONE
        connected component in the mesh graph.

        If a label appears in multiple disconnected components,
        all but the first component are assigned new labels.
        """
        if face2label_consistent is None:
            raise ValueError(
                "split_by_label_connectivity received None. "
                "Ensure the previous step (e.g. fill_unlabeled_components_by_boundary or forward) produced face2label and run the pipeline/cells in order."
            )
        labels = face2label_consistent.copy()
        visited = set()
        mesh_graph = self.mesh_graph

        # 🔴 next available label (residual-safe)
        next_label = max(labels.values()) + 1

        # Iterate over faces in deterministic order
        for face in sorted(labels.keys()):
            lbl = labels[face]

            if lbl == 0 or face in visited:
                continue

            # --------------------------------------------------
            # BFS: connected component for THIS label
            # --------------------------------------------------
            stack = [face]
            component = {face}
            visited.add(face)

            while stack:
                cur = stack.pop()
                for nb in mesh_graph[cur]:
                    if nb not in visited and labels.get(nb) == lbl:
                        visited.add(nb)
                        component.add(nb)
                        stack.append(nb)

            # --------------------------------------------------
            # All faces with this label
            # --------------------------------------------------
            all_faces_with_label = {
                f for f, l in labels.items() if l == lbl
            }

            # --------------------------------------------------
            # If disconnected → split
            # --------------------------------------------------
            remaining = all_faces_with_label - component
            if remaining:
                for f in remaining:
                    labels[f] = next_label
                next_label += 1

        return labels

    def forward(
        self,
        source: Scene | Trimesh,
        visualize_path=None,
        target_labels=None,
        manual_view0=None,
    ) -> tuple[dict, Trimesh]:
        """
        End-to-end mesh segmentation pipeline:
        render → view ordering → SAM2 propagation → residual completion →
        GeoSAM2 lifting → cross-view merging → smoothing → repartition.
        """
        # 1) Scene and multiview renders
        self.prepare_scene(source=source)
        renders = self.render_raw_views(source=source)

        # Front-view selection precedence:
        # runtime arg > config value > automatic
        if manual_view0 is None:
            manual_view0 = self.config.sam_mesh.get("manual_view0", None)
        renders = self.select_view_order(renders, manual_view0=manual_view0)

        # 2) SAM2 inputs + propagation
        renders = self.prepare_sam_inputs(renders)
        renders = self.run_propagation(renders=renders)
        renders = self.residual_completion(renders)

        # 3) GeoSAM2 lifting to mesh labels
        face2label = self.lift_geosam2(mesh=source, renders=renders)
        
        # 4) Cross-view label merging (thresholds from config.sam_mesh)
        #face2label = self.merge_labels_by_cross_view_overlap(face2label, renders)
        # face2label = self.resolve_residual_overwrite_by_votes(
        #     face2label,
        #     renders,
        #     alpha=1.6,  # residuals should almost never overwrite clean prompt objects
        # )
        
        # 5) Optional mesh-graph smoothing
        smooth_iters = int(self.config.sam_mesh.get("mesh_smoothing_iters", 0))
        if smooth_iters > 0:
            face2label = self.smooth_face_labels(
                face2label,
                self.mesh_graph,
                num_iters=smooth_iters,
            )

        face2label = self.resolve_residual_overwrite_by_votes(
            face2label,
            renders,
            alpha=1.5,
        )
        
        # 6) Optional repartitioning (Princeton-style α-expansion)
        face2label = self.smooth_repartition_faces(
            face2label,
            target_labels=target_labels,
        )

        # 7) Remove very small components relative to dominant parts
        size_ratio = float(
            self.config.sam_mesh.get(
                "smoothing_threshold_percentage_size", 0.03
            )
        )
        area_ratio = float(
            self.config.sam_mesh.get(
                "smoothing_threshold_percentage_area", 0.03
            )
        )
        face2label = self.remove_small_components_relative(
            face2label,
            self.mesh_graph,
            self.renderer.tmesh,
            size_ratio=size_ratio,
            area_ratio=area_ratio,
        )

        # 8) Fill remaining unlabeled components via boundary labels
        face2label = self.fill_unlabeled_components_by_boundary(face2label)
        face2label = self.split_by_label_connectivity(face2label)

        return face2label, self.renderer.tmesh
