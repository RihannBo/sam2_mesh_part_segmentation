# mesh_dataset_loader.py

import math
import random
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader, Dataset, Subset

from seg3d.dataset.multiview_dataset import MultiViewSAM2Dataset


# ============================================================
# Wrapper: GeoSAM2-style first-frame bootstrapping
# ============================================================
class BootstrapFirstFrameDataset(Dataset):
    """
    Wraps a MultiViewSAM2Dataset and duplicates frame 0 if enabled:
        F0, F0, F1, F2, ...

    This duplication is COMPLETE:
    - images
    - masks
    - geometry metas
    """

    def __init__(self, base_dataset: Dataset, repeat: int = 1):
        """
        Args:
            base_dataset: MultiViewSAM2Dataset
            repeat: number of times to repeat frame 0 (GeoSAM2 uses 1)
        """
        assert repeat >= 1
        self.base = base_dataset
        self.repeat = repeat

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        sample = self.base[idx]

        if self.repeat <= 0:
            return sample

        # ---- duplicate frame 0 ----
        def dup_first(x):
            # x: Tensor with time dimension first
            first = x[0:1]
            return torch.cat([first.repeat(self.repeat, *([1] * (x.ndim - 1))), x], dim=0)

        # tensors
        sample["normal_imgs"] = dup_first(sample["normal_imgs"])
        sample["point_imgs"]  = dup_first(sample["point_imgs"])
        sample["gt_instance_masks"] = dup_first(sample["gt_instance_masks"])
        
        # Optional fields (may not exist in simplified dataset)
        if "ff_masks" in sample:
            sample["ff_masks"] = dup_first(sample["ff_masks"])
        if "depth_raw" in sample:
            sample["depth_raw"] = dup_first(sample["depth_raw"])
        if "poses" in sample:
            sample["poses"] = dup_first(sample["poses"])

        # view_dirs (numpy array or torch tensor) - for angular memory
        if "view_dirs" in sample:
            view_dirs = sample["view_dirs"]
            if isinstance(view_dirs, torch.Tensor):
                first_view_dir = view_dirs[0:1]  # (1, 3)
                sample["view_dirs"] = torch.cat([
                    first_view_dir.repeat(self.repeat, 1), 
                    view_dirs
                ], dim=0)
            else:
                # numpy array
                import numpy as np
                first_view_dir = view_dirs[0:1]  # (1, 3)
                sample["view_dirs"] = np.concatenate([
                    np.repeat(first_view_dir, self.repeat, axis=0),
                    view_dirs
                ], axis=0)
        
        # metas (list of dicts) - for geometric memory (backward compatibility)
        if "metas" in sample:
            first_meta = sample["metas"][0]
            sample["metas"] = (
                [first_meta.copy() for _ in range(self.repeat)] + sample["metas"]
            )

        sample["num_frames"] = sample["normal_imgs"].shape[0]

        if "matte_imgs" in sample:
            sample["matte_imgs"] = dup_first(sample["matte_imgs"])

        return sample


# ============================================================
# Dataset + Loader wrapper (train / val split)
# ============================================================
class MeshDatasetWithLoader:
    """
    Handles:
      - dataset instantiation
      - deterministic train/val split
      - optional bootstrapping
      - DataLoader creation
    """

    def __init__(
        self,
        dataset_root: str,
        split: str = "train",
        train_ratio: float = 0.8,
        split_seed: int = 42,
        num_views: int = 12,
        batch_size: int = 1,
        num_workers: int = 4,
        shuffle: bool = True,
        pin_memory: bool = True,
        drop_last: bool = True,
        bootstrap_first_frame: bool = False,
        bootstrap_repeat: int = 1,
        collate_fn=None,
        dict_key: str = "all",
        enable_rotation_aug: bool = False,
    ):
        assert split in ["train", "val"]

        self.dataset_root = Path(dataset_root)
        self.split = split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle if split == "train" else False
        self.pin_memory = pin_memory
        self.drop_last = drop_last if split == "train" else False

        # --------------------------------------------------
        # Base dataset (NO bootstrapping here)
        # --------------------------------------------------
        base_dataset = MultiViewSAM2Dataset(
            dataset_root=self.dataset_root,
            num_views=num_views,
            enable_rotation_aug=enable_rotation_aug,
        )

        # --------------------------------------------------
        # Deterministic split
        # --------------------------------------------------
        indices = list(range(len(base_dataset)))
        rng = random.Random(split_seed)
        rng.shuffle(indices)

        split_idx = int(math.floor(train_ratio * len(indices)))

        if split == "train":
            indices = indices[:split_idx]
        else:
            indices = indices[split_idx:]

        dataset = Subset(base_dataset, indices)

        # --------------------------------------------------
        # Optional GeoSAM2-style bootstrapping
        # --------------------------------------------------
        if bootstrap_first_frame:
            dataset = BootstrapFirstFrameDataset(
                dataset,
                repeat=bootstrap_repeat,
            )

        self.dataset = dataset
        self.collate_fn = collate_fn
        self.dict_key = dict_key

    def get_loader(self, epoch=None, collate_fn=None, dict_key=None):
        """
        Returns a PyTorch DataLoader.
        
        Args:
            epoch: Epoch number (for compatibility with trainer, can be ignored)
            collate_fn: Optional override for collate function
            dict_key: Optional override for dict_key
        """
        # Use provided overrides or fall back to stored values
        collate_fn = collate_fn or self.collate_fn
        dict_key = dict_key or self.dict_key
        
        if collate_fn is None:
            # Import here to avoid circular imports
            from seg3d.training.utils.data_utils import collate_fn as default_collate_fn
            collate_fn = default_collate_fn
        
        if dict_key is None:
            dict_key = "all"

        def wrapped_collate(batch):
            return collate_fn(batch, dict_key=dict_key)

        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            collate_fn=wrapped_collate,
            persistent_workers=self.num_workers > 0,
        )
