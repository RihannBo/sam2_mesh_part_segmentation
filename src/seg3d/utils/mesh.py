# Code adapted from Segment Any Mesh (samesh)
# https://github.com/gtangg12/samesh

import numpy as np
import trimesh
from trimesh.base import Trimesh, Scene

from seg3d.data.common import NumpyTensor, TorchTensor


# def duplicate_verts(mesh: Trimesh) -> Trimesh:
#     """
#     Call before coloring mesh to avoid face interpolation since openGL stores color attributes per vertex.

#         ...
#         mesh = duplicate_verts(mesh)
#         mesh.visual.face_colors = colors
#         ...

#     NOTE: removes visuals for verticies, but preserves for faces.
#     """
#     verts = mesh.vertices[mesh.faces.reshape(-1), :]
#     faces = np.arange(0, verts.shape[0])
#     faces = faces.reshape(-1, 3)
#     return Trimesh(vertices=verts, faces=faces, face_colors=mesh.visual.face_colors, process=False)


def duplicate_verts(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """
    Duplicate vertices so each face has its own unique vertices (unwelded mesh).
    Supports:
        • TextureVisuals (UV / texture)
        • ColorVisuals (face colors)
        • vertex_colors meshes
    """

    faces_orig = mesh.faces
    flat_idx = faces_orig.reshape(-1)

    # Duplicate vertices
    verts_new = mesh.vertices[flat_idx]

    # Duplicate normals if present
    normals_new = None
    if mesh.vertex_normals is not None:
        normals_new = mesh.vertex_normals[flat_idx]

    # -------------------------------------------------------
    # Handle visualization (face colors / texture)
    # -------------------------------------------------------
    face_colors = None

    # Case 1 — has face colors already
    if hasattr(mesh.visual, "face_colors") and mesh.visual.face_colors is not None:
        face_colors = mesh.visual.face_colors.copy()

    # Case 2 — TextureVisuals → discard UV, generate stable synthetic colors
    elif isinstance(mesh.visual, trimesh.visual.texture.TextureVisuals):
        rng = np.random.default_rng(42)
        face_colors = rng.integers(0, 255, size=(len(faces_orig), 4), dtype=np.uint8)
        face_colors[:, 3] = 255

    # Case 3 — vertex-colored mesh
    elif hasattr(mesh.visual, "vertex_colors") and mesh.visual.vertex_colors is not None:
        # We rely on duplicated vertex colors implicitly,
        # no need to create face colors unless required.
        pass

    # -------------------------------------------------------
    # Build new faces
    # -------------------------------------------------------
    faces_new = np.arange(len(verts_new)).reshape(-1, 3)

    # -------------------------------------------------------
    # Construct unwelded mesh
    # -------------------------------------------------------
    return trimesh.Trimesh(
        vertices=verts_new,
        faces=faces_new,
        vertex_normals=normals_new,
        face_colors=face_colors,
        process=False
    )

def handle_pose(pose: NumpyTensor['4 4']) -> NumpyTensor['4 4']:
    """
    Handles common case that results in numerical instability in rendering faceids:

        ...
        pose, _ = scene.graph[name]
        pose = handle_pose(pose)
        ...
    """
    identity = np.eye(4)
    if np.allclose(pose, identity, atol=1e-6):
        return identity
    return pose


def transform(pose: NumpyTensor['4 4'], vertices: NumpyTensor['nv 3']) -> NumpyTensor['nv 3']:
    """
    """
    homogeneous = np.concatenate([vertices, np.ones((vertices.shape[0], 1))], axis=1)
    return (pose @ homogeneous.T).T[:, :3]


def concat_scene_vertices(scene: Scene) -> NumpyTensor['nv 3']:
    """
    """
    verts = []
    for name, geom in scene.geometry.items():
        if name in scene.graph:
            pose, _ = scene.graph[name]
            pose = handle_pose(pose)
            geom.vertices = transform(pose, geom.vertices)
        verts.append(geom.vertices)
    return np.concatenate(verts)


def bounding_box(vertices: NumpyTensor['n 3']) -> NumpyTensor['2 3']:
    """
    Compute bounding box from vertices.
    """
    return np.array([vertices.min(axis=0), vertices.max(axis=0)])


def bounding_box_centroid(vertices: NumpyTensor['n 3']) -> NumpyTensor['3']:
    """
    Compute bounding box centroid from vertices.
    """
    return bounding_box(vertices).mean(axis=0)


def norm_mesh(mesh: Trimesh) -> Trimesh:
    """
    Normalize mesh vertices to bounding box [-1, 1]. 
    
    NOTE:: In place operation that consumes mesh.
    """
    centroid = bounding_box_centroid(mesh.vertices)
    mesh.vertices -= centroid
    mesh.vertices /= np.abs(mesh.vertices).max()
    mesh.vertices *= (1 - 1e-3)
    return mesh


def norm_scene(scene: Scene) -> Scene:
    """
    Normalize scene vertices to bounding box [-1, 1]. 
    
    NOTE:: In place operation that consumes scene.
    """
    centroid = bounding_box_centroid(concat_scene_vertices(scene))
    for geom in scene.geometry.values():
        geom.vertices -= centroid
    extent = np.abs(concat_scene_vertices(scene)).max()
    for geom in scene.geometry.values():
        geom.vertices /= extent
        geom.vertices *= (1 - 1e-3)
    return scene


