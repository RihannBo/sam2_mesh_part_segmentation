# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Misc functions, including distributed helpers.

Mostly copy-paste from torchvision references.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import numpy as np
import random

from tensordict import tensorclass


@tensorclass
class BatchedMeshMetaData:
    """
    This class represents metadata about a batch of videos.
    Attributes:
        unique_objects_identifier: A tensor of shape Bx3 containing unique identifiers for each object in the batch. Index consists of (video_id, obj_id, frame_id)
        frame_orig_size: A tensor of shape Bx2 containing the original size of each frame in the batch.
    """

    unique_objects_identifier: torch.LongTensor
    frame_orig_size: torch.LongTensor


@tensorclass
class BatchedMeshDatapoint:
    """
    This class represents a batch of videos with associated annotations and metadata.
    Attributes:
        normal_batch: A [TxBxCxHxW] tensor containing the normal data for each frame in the batch, where T is the number of frames per video, and B is the number of videos in the batch.
        point_batch: A [TxBxCxHxW] tensor containing the point map data for each frame in the batch, where T is the number of frames per video, and B is the number of videos in the batch.
        obj_to_frame_idx: A [TxOx2] tensor containing the image_batch index which the object belongs to. O is the number of objects in the batch.
        masks: A [TxOxHxW] tensor containing binary masks for each object in the batch.
        metadata: An instance of BatchedVideoMetaData containing metadata about the batch.
        geometry_metas: per-frame camera & geometry metadata
        dict_key: A string key used to identify the batch.
    """

    normal_batch: torch.FloatTensor    # [T,B,3,H,W]
    point_batch: torch.FloatTensor     # [T,B,3,H,W]
    
    obj_to_frame_idx: torch.IntTensor  # [T,O,2]
    masks: torch.BoolTensor            # [T,O,H,W]

    metadata: BatchedMeshMetaData
    geometry_metas: list               ## length T, each is a dict: pose, intrinsics, depth_raw, ff_mask, pointmap

    dict_key: str

    def pin_memory(self, device=None):
        return self.apply(torch.Tensor.pin_memory, device=device)

    @property
    def num_frames(self) -> int:
        """
        Returns the number of frames per video.
        """
        return self.batch_size[0]

    @property
    def num_videos(self) -> int:
        """
        Returns the number of videos in the batch.
        """
        return self.normal_batch.shape[1]

    @property
    def flat_obj_to_img_idx(self) -> torch.IntTensor:
        """
        Returns a flattened tensor containing the object to img index.
        The flat index can be used to access a flattened img_batch of shape [(T*B)xCxHxW]
        """
        frame_idx, video_idx = self.obj_to_frame_idx.unbind(dim=-1)
        flat_idx = video_idx * self.num_frames + frame_idx
        return flat_idx

    @property
    def flat_normal_batch(self) -> torch.FloatTensor:
        """
        Returns a flattened img_batch_tensor of shape [(B*T)xCxHxW]
        """

        return self.normal_batch.transpose(0, 1).flatten(0, 1)

    @property
    def flat_point_batch(self) -> torch.FloatTensor:
        """
        Returns a flattened img_batch_tensor of shape [(B*T)xCxHxW]
        """

        return self.point_batch.transpose(0, 1).flatten(0, 1)



def collate_fn(
    batch,
    dict_key,
    max_objects_per_mesh: int | None = None,
):
    """
    Collate function producing a BatchedMeshDatapoint with:
      - dual modalities (normal / point)
      - SAM2-style object batching
      - objects defined ONLY from view 0 (prompt-based)
      - optional random subsampling of up to K objects
      - if GT objects <= K, all objects are kept
      - per-frame geometry metadata

    Args:
        batch: list of samples from MultiViewSAM2Dataset
        dict_key: string identifier
        max_objects_per_mesh: maximum number of objects (K) to use per mesh.
                              If None, all objects are used.

    Returns:
        BatchedMeshDatapoint
    """

    # -------------------------------------------------
    # Basic dimensions
    # -------------------------------------------------
    B = len(batch)
    T, _, H, W = batch[0]["normal_imgs"].shape

    # -------------------------------------------------
    # Stack dual-modality image batches
    # -------------------------------------------------
    normal_batch = torch.stack(
        [sample["normal_imgs"] for sample in batch], dim=1
    )  # [T, B, 3, H, W]

    point_batch = torch.stack(
        [sample["point_imgs"] for sample in batch], dim=1
    )  # [T, B, 3, H, W]

    # -------------------------------------------------
    # Per-frame containers (SAM2-style, fixed O)
    # -------------------------------------------------
    step_t_masks = [[] for _ in range(T)]
    step_t_obj_to_frame_idx = [[] for _ in range(T)]
    step_t_object_ids = [[] for _ in range(T)]
    step_t_frame_sizes = [[] for _ in range(T)]

    # -------------------------------------------------
    # Build objects (define from view 0 + optional subsampling)
    # -------------------------------------------------
    for video_idx, sample in enumerate(batch):
        gt_masks = sample["gt_instance_masks"]  # (T, H, W)

        # --- define objects ONLY from view 0 ---
        instance_ids = torch.unique(gt_masks[0])
        instance_ids = instance_ids[instance_ids != 0].tolist()
        num_objects = len(instance_ids)

        if num_objects == 0:
            raise RuntimeError(
                f"No GT objects found in view 0 for sample {video_idx}"
            )

        # --- subsample objects if requested ---
        if max_objects_per_mesh is not None:
            if num_objects <= max_objects_per_mesh:
                selected_instance_ids = instance_ids
            else:
                selected_instance_ids = random.sample(
                    instance_ids, max_objects_per_mesh
                )
        else:
            selected_instance_ids = instance_ids

        # -------------------------------------------------
        # Create binary masks for ALL frames for selected objects
        # -------------------------------------------------
        for inst_id in selected_instance_ids:
            for t in range(T):
                # binary mask (may be empty)
                bin_mask = (gt_masks[t] == inst_id).to(torch.bool)

                step_t_masks[t].append(bin_mask)

                # (frame_idx, video_idx)
                step_t_obj_to_frame_idx[t].append(
                    torch.tensor([t, video_idx], dtype=torch.int32)
                )

                # object introduced at frame 0 (prompt frame)
                step_t_object_ids[t].append(
                    torch.tensor([video_idx, inst_id, 0], dtype=torch.long)
                )

                # original frame size
                step_t_frame_sizes[t].append(
                    torch.tensor([H, W], dtype=torch.long)
                )

    # -------------------------------------------------
    # Stack per-frame tensors (safe: fixed O)
    # -------------------------------------------------
    obj_to_frame_idx = torch.stack(
        [torch.stack(x, dim=0) for x in step_t_obj_to_frame_idx],
        dim=0,
    )  # [T, O, 2]

    masks = torch.stack(
        [torch.stack(x, dim=0) for x in step_t_masks],
        dim=0,
    )  # [T, O, H, W] (bool)

    unique_objects_identifier = torch.stack(
        [torch.stack(x, dim=0) for x in step_t_object_ids],
        dim=0,
    )  # [T, O, 3]

    frame_orig_size = torch.stack(
        [torch.stack(x, dim=0) for x in step_t_frame_sizes],
        dim=0,
    )  # [T, O, 2]

    metadata = BatchedMeshMetaData(
        unique_objects_identifier=unique_objects_identifier,
        frame_orig_size=frame_orig_size,
    )

    # -------------------------------------------------
    # Per-frame geometry metadata (shared across batch)
    # For angular memory: geometry_metas is view_dirs directly
    # -------------------------------------------------
    if "metas" in batch[0]:
        # If dataset provides metas (list of dicts), use them
        geometry_metas = batch[0]["metas"]
    elif "view_dirs" in batch[0]:
        # For angular memory: geometry_metas is view_dirs (T, 3) as numpy array/list
        view_dirs = batch[0]["view_dirs"]  # (T, 3) torch.Tensor
        # Convert to numpy array for compatibility (can be indexed by frame_idx)
        geometry_metas = view_dirs.numpy()  # (T, 3) numpy array
    else:
        raise ValueError(
            "Batch must contain either 'metas' or 'view_dirs'. "
            f"Available keys: {list(batch[0].keys())}"
        )

    # -------------------------------------------------
    # Return final batched datapoint
    # -------------------------------------------------
    return BatchedMeshDatapoint(
        normal_batch=normal_batch,
        point_batch=point_batch,
        obj_to_frame_idx=obj_to_frame_idx,
        masks=masks,
        metadata=metadata,
        geometry_metas=geometry_metas,
        dict_key=dict_key,
        batch_size=[T],
    )