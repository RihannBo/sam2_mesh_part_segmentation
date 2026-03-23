import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset

from seg3d.dataset.transforms import ViewConsistentRotation


class MultiViewSAM2Dataset(Dataset):
    """
    Loads per-mesh multi-view data from:
      mesh_dir/mesh_data.npz

    NPZ schema (new):
      normals_u8:   (T,H,W,3) uint8   (normals in [-1,1] encoded)
      view_dirs:    (T,3)   float32 (view directions)
      pointmap_f16: (T,H,W,3) float16 (world xyz)
      gt_masks:     (T,H,W)   uint16  (instance ids)
      mattes_u8:    (T,H,W,3) uint8   (optional)
    """
    def __init__(self, dataset_root, num_views=12, enable_rotation_aug=False):
        self.dataset_root = Path(dataset_root)
        self.mesh_ids = sorted([p.name for p in self.dataset_root.iterdir() if p.is_dir()])
        self.num_views = num_views

        self.enable_rotation_aug = enable_rotation_aug
        if self.enable_rotation_aug:
            self.rotation_aug = ViewConsistentRotation(
                max_deg=5.0,  # start conservative
                p=0.5
            )

    def __len__(self):
        return len(self.mesh_ids)

    def __getitem__(self, idx):
        mesh_id = self.mesh_ids[idx]
        mesh_dir = self.dataset_root / mesh_id

        with np.load(mesh_dir / "mesh_data.npz") as data:

            T_total = data["normals_u8"].shape[0]
            T = min(self.num_views, T_total) if self.num_views is not None else T_total

            normals_u8   = data["normals_u8"][: T]       # (T,H,W,3)
            pointmap_f16 = data["pointmap_f16"][: T]     # (T,H,W,3)
            gt_masks_np  = data["gt_masks"][: T]         # (T,H,W)
            view_dirs_np = data["view_dirs"][: T]         # (T,3)
            

            mattes = None
            if "mattes_u8" in data:
                mattes_u8 = data["mattes_u8"][: T]       # (T,H,W,3)
                mattes = torch.from_numpy(mattes_u8.astype(np.float32) / 255.0) \
                            .permute(0, 3, 1, 2).contiguous()         # (T,3,H,W)

        # normals: uint8 -> float [0,1] for SAM2 ImageNet normalization, (T,3,H,W)
        # Note: normals_u8 stores normals in [-1,1] encoded as uint8: (n+1)*127.5
        # So to decode: (n_u8 / 127.5) - 1.0 gives [-1,1], then normalize to [0,1]
        normals = torch.from_numpy((normals_u8.astype(np.float32) / 127.5) - 1.0) \
                    .permute(0, 3, 1, 2).contiguous()
        normals = (normals + 1.0) / 2.0  # Convert from [-1, 1] to [0, 1]
        normals = torch.clamp(normals, 0.0, 1.0)


        # pointmap: f16 -> f32, (T,3,H,W)
        point_imgs = torch.from_numpy(pointmap_f16.astype(np.float32)) \
                        .permute(0, 3, 1, 2).contiguous()
        
        # Normalize point maps to [0, 1] range for SAM2 ImageNet normalization
        # Point maps are world coordinates, normalize per-channel across all frames for consistency
        pts_flat = point_imgs.permute(0, 2, 3, 1).reshape(-1, 3)  # (T*H*W, 3)
        pts_min = pts_flat.min(dim=0, keepdim=True)[0]  # (1, 3)
        pts_max = pts_flat.max(dim=0, keepdim=True)[0]  # (1, 3)
        pts_range = pts_max - pts_min
        pts_range = torch.clamp(pts_range, min=1e-6)  # Avoid division by zero
        # Reshape back and normalize
        pts_min = pts_min.view(1, 3, 1, 1)  # (1, 3, 1, 1)
        pts_range = pts_range.view(1, 3, 1, 1)  # (1, 3, 1, 1)
        point_imgs = (point_imgs - pts_min) / pts_range  # Normalize to [0, 1]
        point_imgs = torch.clamp(point_imgs, 0.0, 1.0)

        # masks
        gt_masks = torch.from_numpy(gt_masks_np.astype(np.int64))          # (T,H,W) int64 for training

        view_dirs = torch.from_numpy(view_dirs_np.astype(np.float32))
        
        # --- GeoSAM2-safe augmentation ---
        if self.enable_rotation_aug:
            normals, point_imgs, gt_masks, mattes = self.rotation_aug(
                normals,
                point_imgs,
                gt_masks,
                mattes
            )
        
        result = {
            "normal_imgs": normals,                 # (T,3,H,W)
            "point_imgs": point_imgs,               # (T,3,H,W)
            "gt_instance_masks": gt_masks,          # (T,H,W) int64
            "view_dirs": view_dirs,                 # (T,3) float32
            "num_frames": T,
            "mesh_id": mesh_id,
        }
        if mattes is not None:
            result["matte_imgs"] = mattes           # (T,3,H,W)

        return result