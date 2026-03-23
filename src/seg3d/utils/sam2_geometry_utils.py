import numpy as np
from seg3d.utils.cameras import depth_to_point_map

def project_world_to_cam_pixels(points_world, pose_cam_to_world, K, H, W):
    """
    Project 3D world points into a camera view.
    OpenGL convention: forward is -Z.
    Returns pixel coordinates (u,v), valid mask, and projected depth z.
    """
    world2cam = np.linalg.inv(pose_cam_to_world) # (4, 4)
    pts_w_h = np.concatenate([points_world, np.ones((points_world.shape[0], 1), dtype=points_world.dtype)], axis=1) # (N, 4)
    pts_cam = (world2cam @ pts_w_h.T).T[:, :3]  # (N, 3)

    # Forward depth is -Z
    z = -pts_cam[:, 2]
    in_front = z > 0

    # extract intrinsics
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # Y is flipped (matches depth_to_point_map)
    u = fx * (pts_cam[:, 0] / z) + cx
    v = fy * (-pts_cam[:, 1] / z) + cy

    inside = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    valid = in_front & inside
    return u[valid].astype(int), v[valid].astype(int), valid, z[valid]


def compute_view_overlap(mem, curr, sample_ratio=0.01, depth_tolerance=0.01):
    """
    Compute geometric overlap between one memory view and the current frame.

    The overlap is defined as the fraction of sampled 3D points from the memory
    view that are visible and depth-consistent in the current view.
    """
    # --- unpack data ---
    mask_mem = mem["ff_mask"].astype(bool)
    pointmap_mem = mem["pointmap"]
    pose_curr = curr["pose"]
    mask_curr = curr["ff_mask"].astype(bool)
    depth_curr = curr["depth_raw"]
    K_curr = curr["intrinsics"]
    H, W = mask_curr.shape

    # --- 1. sample some 3D points from the memory view ---
    ys, xs = np.where(mask_mem)
    if len(xs) == 0:
        return 0.0

    num_samples = max(1, int(sample_ratio * len(xs)))
    sample_idx = np.random.choice(len(xs), num_samples, replace=False)
    xs, ys = xs[sample_idx], ys[sample_idx]
    points_world = pointmap_mem[ys, xs]

    # --- 2. project those 3D points into the current camera ---
    u_pix, v_pix, valid_mask, depth_proj = project_world_to_cam_pixels(
        points_world, pose_cam_to_world=pose_curr, K=K_curr, H=H, W=W
    )
    if len(u_pix) == 0:
        return 0.0

    # --- 3. check visibility and depth consistency ---
    is_foreground = mask_curr[v_pix, u_pix]
    depth_target = depth_curr[v_pix, u_pix]
    is_depth_consistent = np.abs(depth_proj - depth_target) < depth_tolerance

    is_visible_and_consistent = is_foreground & is_depth_consistent
    overlap_ratio = is_visible_and_consistent.sum() / len(is_visible_and_consistent)

    return float(overlap_ratio)


def compute_geometric_overlap(prev_out, curr_meta, sample_ratio=0.01, depth_tolerance=0.01):
    """
    Wrapper that checks validity and calls compute_view_overlap safely.
    """
    required = ("pose", "intrinsics", "depth_raw", "ff_mask")
    if prev_out is None or curr_meta is None:
        return 0.0
    if not all(k in prev_out and prev_out[k] is not None for k in required):
        return 0.0
    if not all(k in curr_meta and curr_meta[k] is not None for k in required):
        return 0.0


    # --- Prefer precomputed pointmaps (from NPZ / renderer) ---
    if prev_out.get("pointmap", None) is None:
        prev_out["pointmap"] = depth_to_point_map(
            prev_out["depth_raw"], prev_out["pose"], prev_out["intrinsics"]
        ).astype(np.float32)

    if curr_meta.get("pointmap", None) is None:
        curr_meta["pointmap"] = depth_to_point_map(
            curr_meta["depth_raw"], curr_meta["pose"], curr_meta["intrinsics"]
        ).astype(np.float32)

    try:
        return compute_view_overlap(prev_out, curr_meta, sample_ratio, depth_tolerance)
    except Exception as e:
        print(f"[WARN] overlap computation failed: {e}")
        return 0.0

