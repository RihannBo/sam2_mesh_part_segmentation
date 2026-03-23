import math
import torch
import torch.nn.functional as F
import random


class ViewConsistentRotation:
    """
    View-consistent in-plane rotation for GeoSAM2.

    - Same random angle for all views
    - Rotates normals, point maps, masks, mattes
    - Angle in degrees
    """
    def __init__(self, max_deg=5.0, p=0.5):
        self.max_deg = max_deg
        self.p = p
        # For debugging/verification (set when augmentation is applied)
        self.last_angle_deg = None

    def __call__(self, normals, point_imgs, gt_masks, mattes=None):
        """
        normals:    (T,3,H,W) float
        point_imgs:(T,3,H,W) float
        gt_masks:  (T,H,W)   int64
        mattes:    (T,3,H,W) float or None
        """

        if random.random() > self.p:
            self.last_angle_deg = None
            return normals, point_imgs, gt_masks, mattes

        T, _, H, W = normals.shape
        device = normals.device

        # Sample a single rotation angle for all views
        angle_deg = random.uniform(-self.max_deg, self.max_deg)
        self.last_angle_deg = float(angle_deg)
        angle_rad = angle_deg * math.pi / 180.0

        # Build affine rotation matrix (same for all views)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)

        theta = torch.tensor(
            [[cos_a, -sin_a, 0.0],
             [sin_a,  cos_a, 0.0]],
            dtype=torch.float32,
            device=device
        )  # (2,3)

        theta = theta.unsqueeze(0).repeat(T, 1, 1)  # (T,2,3)

        # Create sampling grid
        grid = F.affine_grid(theta, size=normals.size(), align_corners=False)

        # Rotate normals
        normals = F.grid_sample(
            normals, grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False
        )

        # Rotate point maps
        point_imgs = F.grid_sample(
            point_imgs, grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False
        )

        # Rotate masks (nearest!)
        gt_masks = gt_masks.unsqueeze(1).float()  # (T,1,H,W)
        gt_masks = F.grid_sample(
            gt_masks, grid,
            mode="nearest",
            padding_mode="zeros",
            align_corners=False
        )
        gt_masks = gt_masks.squeeze(1).long()

        # Rotate mattes if present
        if mattes is not None:
            mattes = F.grid_sample(
                mattes, grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False
            )

        return normals, point_imgs, gt_masks, mattes
