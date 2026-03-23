#!/usr/bin/env python3
import os
import json
import gc
import multiprocessing as mp
from functools import partial
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm
from omegaconf import OmegaConf



def save_mesh_npz(
    mesh_out_dir: Path,
    normals_list,
    faces_list,
    gt_masks_list,
    matte_list,
    pointmap_list,
    view_dirs,  
):
    mesh_out_dir.mkdir(parents=True, exist_ok=True)

    # normals: (T,H,W,3) uint8 from [-1,1]
    normals_u8 = np.stack(
        [((n + 1.0) * 127.5).clip(0, 255).astype(np.uint8) for n in normals_list],
        axis=0
    )

    # masks: (T,H,W) uint16 if possible
    gt_masks_i32 = np.stack(gt_masks_list, axis=0).astype(np.int32)
    max_id = int(gt_masks_i32.max())
    min_id = int(gt_masks_i32.min())
    if min_id < 0:
        raise ValueError(f"[{mesh_out_dir}] gt_masks has negative ids (min={min_id}).")

    if max_id > 65535:
        raise ValueError(f"[{mesh_out_dir}] gt_masks max id {max_id} > 65535; cannot store as uint16.")

    gt_masks_u16 = gt_masks_i32.astype(np.uint16)

    # matte: (T,H,W,3) uint8
    mattes_u8 = np.stack(
        [np.array(m.convert("RGB"), dtype=np.uint8) for m in matte_list],
        axis=0
    )

    # Clean pointmaps: set background pixels to 0
    pointmap_list_clean = []
    for pm, f in zip(pointmap_list, faces_list):
        pm = np.asarray(pm, np.float32)
        pm[f == -1] = 0.0
        pointmap_list_clean.append(pm)

    pointmap_f16 = np.stack([pm.astype(np.float16) for pm in pointmap_list_clean], axis=0)

    # view directions: (T,3) float32
    view_dirs_f32 = np.asarray(view_dirs, dtype=np.float32)

    tmp = mesh_out_dir / "mesh_data.tmp.npz"
    out = mesh_out_dir / "mesh_data.npz"

    np.savez_compressed(
        str(tmp),
        normals_u8=normals_u8,
        gt_masks=gt_masks_u16,
        mattes_u8=mattes_u8,
        pointmap_f16=pointmap_f16, 
        view_dirs=view_dirs_f32,
    )
    if not tmp.exists():
        raise RuntimeError(f"np.savez_compressed did not create: {tmp}")

    os.replace(tmp, out)
    
# ============================================================
# FACE MAP → INSTANCE MASK (robust)
# ============================================================
def face_to_instance_mask(face_map: np.ndarray, face_labels: np.ndarray) -> np.ndarray:
    """
    face_map: (H,W) int32, -1 means background.
    face_labels: (num_faces,) int32 instance IDs.
    """
    mask = np.zeros_like(face_map, dtype=np.int32)
    num_faces = len(face_labels)

    valid = (face_map >= 0) & (face_map < num_faces)
    if np.any(valid):
        mask[valid] = face_labels[face_map[valid]]

    return mask

# ============================================================
# MESH NORMALIZATION (Center + scale)
# ============================================================
def normalize_mesh(mesh):
    """Center mesh at origin and scale to max radius = 1.0."""
    verts = mesh.vertices
    center = verts.mean(axis=0)
    verts = verts - center

    radius = np.max(np.linalg.norm(verts, axis=1))
    if radius > 0:
        verts = verts / radius

    mesh.vertices = verts
    return mesh

# ============================================================
# WORKER (runs in its own process)
# ============================================================
def process_single_mesh(mesh_path, gt_dir, out_dir, config):
    from trimesh import load as trimesh_load  # safe import
    from seg3d.renderer.renderer import Renderer, render_multiview

    mesh_path = Path(mesh_path)
    mesh_id = mesh_path.stem

    gt_path = Path(gt_dir) / f"{mesh_id}.npy"
    if not gt_path.exists():
        return f"⚠ Missing GT for {mesh_id}"

    # ---------------- Load mesh exactly like GT labels expect ----------------
    try:
        mesh = trimesh_load(mesh_path, force="mesh")
        mesh = normalize_mesh(mesh)   # 👍 safe normalization
        mesh.visual = mesh.visual.to_color()  # removes textures WITHOUT touching geometry
    except Exception as e:
        return f"❌ Mesh load error {mesh_id}: {e}"

    # ---------------- Load GT labels ----------------
    try:
        face_labels = np.load(gt_path)
    except Exception as e:
        return f"❌ GT load error {mesh_id}: {e}"

    if len(face_labels) != len(mesh.faces):
        return f"❌ Face count mismatch {mesh_id}: mesh={len(mesh.faces)}, gt={len(face_labels)}"

    # ---------------- Initialize renderer ----------------
    renderer_cfg = config.renderer
    try:
        renderer = Renderer(renderer_cfg)
        renderer.set_object(mesh, smooth=False)
        renderer.set_camera()
    except Exception as e:
        return f"❌ Renderer init failed {mesh_id}: {e}"

    # ---------------- Render all views ----------------
    try:
        result = render_multiview(
            renderer,
            camera_generation_method=renderer_cfg.camera_generation_method,
            renderer_args=renderer_cfg.renderer_args,
            sampling_args=renderer_cfg.sampling_args,
            lighting_args=renderer_cfg.lighting_args,
            verbose=False,
        )
    except Exception as e:
        try:
            renderer.renderer.delete()
        except:
            pass
        return f"❌ Render failed {mesh_id}: {e}"

    normals_list   = result["norms"]
    faces_list     = result["faces"]
    matte_list = result["matte"]  # PIL Images (because renderer returns outputs['matte']=Image.fromarray(...))
    pointmap_list = result["point_map"]
    view_dirs      = result["view_dirs"] 


    gt_masks_list = []
    for face_map in faces_list:
        mask = face_to_instance_mask(face_map.astype(np.int64), face_labels)
        gt_masks_list.append(mask)

    mesh_out_dir = Path(out_dir) / mesh_id
    save_mesh_npz(
    mesh_out_dir,
    normals_list=normals_list,
    faces_list=faces_list,
    gt_masks_list=gt_masks_list,
    matte_list=matte_list,
    pointmap_list=pointmap_list,
    view_dirs=view_dirs,
    )

    # meta.json (just once)
    meta = {
        "mesh_id": mesh_id,
        "camera": {
            "width":  int(renderer_cfg.target_dim[0]),
            "height": int(renderer_cfg.target_dim[1]),
            "yfov":   float(renderer.camera_params["yfov"]),
        },
    }
    with open(mesh_out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    # Cleanup
    try:
        renderer.renderer.delete()
    except:
        pass
    del renderer
    gc.collect()

    return f"✅ Done {mesh_id}"


# ============================================================
# MULTIPROCESS DATASET BUILDER
# ============================================================
def build_geosam2_dataset_mp(mesh_dir, gt_dir, out_dir, config, num_workers=4):
    mesh_paths = sorted(Path(mesh_dir).glob("*.glb"))
    print(f"Found {len(mesh_paths)} meshes.\n")

    ctx = mp.get_context("spawn")
    worker = partial(process_single_mesh, gt_dir=gt_dir, out_dir=out_dir, config=config)

    with ctx.Pool(num_workers) as pool:
        results = list(
            tqdm(pool.imap_unordered(worker, mesh_paths),
                 total=len(mesh_paths),
                 desc="Dataset")
        )

    print("\n".join(results))
    print("\n🎉 Dataset generation complete!\n")


# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    CONFIG_PATH = "/home/mengnan/seg3d/configs/mesh_segmentation.yaml"
    MESH_DIR    = "/home/mengnan/seg3d/PartObjaverse/PartObjaverse-Tiny_mesh"
    GT_DIR      = "/home/mengnan/seg3d/PartObjaverse/PartObjaverse-Tiny_instance_gt"
    OUT_DIR     = "/home/mengnan/seg3d/training_dataset"

    config = OmegaConf.load(CONFIG_PATH)

    build_geosam2_dataset_mp(
        mesh_dir=MESH_DIR,
        gt_dir=GT_DIR,
        out_dir=OUT_DIR,
        config=config,
        num_workers=4,
    )