import os
# Visualization helpers: OpenGL/pyrender offscreen renderer + `colormap_faces` / `colormap_norms`
# adapted from Segment Any Mesh (samesh).
# https://github.com/gtangg12/samesh
# -----------------------------------------------------------------------------------
# OpenGL platform setup
# -----------------------------------------------------------------------------------
# On Linux servers (e.g., cluster or headless rendering), EGL is required for offscreen GPU rendering.
# On macOS, EGL is NOT supported — leave these lines commented out to avoid import errors.
#
# 👉 If you move to a Linux GPU system, uncomment the following lines:
#
os.environ['PYOPENGL_PLATFORM'] = 'egl'
os.environ['EGL_DEVICE_ID'] = '-1'   # avoids GPU contention on multi-user systems
# -----------------------------------------------------------------------------------

### START VOODOO ###
# Dark encantation for disabling anti-aliasing in pyrender (if needed)
import OpenGL.GL
antialias_active = False
old_gl_enable = OpenGL.GL.glEnable
def new_gl_enable(value):
    if not antialias_active and value == OpenGL.GL.GL_MULTISAMPLE:
        OpenGL.GL.glDisable(value)
    else:
        old_gl_enable(value)
OpenGL.GL.glEnable = new_gl_enable
import pyrender
### END VOODOO ###

import cv2
import numpy as np
import torch
from numpy.random import RandomState
from PIL import Image
from pyrender.shader_program import ShaderProgramCache as DefaultShaderCache
from trimesh import Trimesh, Scene
from omegaconf import OmegaConf
from tqdm import tqdm

from seg3d.data.common import NumpyTensor
from seg3d.data.loaders import scene2mesh
from seg3d.utils.view_sampling import generate_ordered_views
from seg3d.utils.cameras import (
    HomogeneousTransform, 
    sample_view_matrices_polyhedra, 
    sample_view_matrices_orbit, 
    fov_to_intrinsics, 
    depth_to_point_map)

from seg3d.utils.math import range_norm
from seg3d.utils.mesh import duplicate_verts
from seg3d.renderer.shader_programs import *


def colormap_faces(faces: NumpyTensor['h w'], background=np.array([255, 255, 255])) -> Image.Image:
    """
    Given a face id map, color each face with a random color.
    """
    #print(np.unique(faces, return_counts=True))
    palette = RandomState(0).randint(0, 255, (np.max(faces + 2), 3)) # must init every time to get same colors
    #print(palette)
    palette[0] = background
    image = palette[faces + 1, :].astype(np.uint8) # shift -1 to 0
    return Image.fromarray(image)


def colormap_norms(norms: NumpyTensor['h w'], background=np.array([255, 255, 255])) -> Image.Image:
    """
    Given a normal map, color each normal with a color.
    """
    norms = (norms + 1) / 2
    norms = (norms * 255).astype(np.uint8)
    return Image.fromarray(norms)


def colormap_points(points: NumpyTensor['h w 3'], background=np.array([255, 255, 255])) -> Image.Image:
    """
    Visualize a point map (world-space coordinates) as a color image.
    Normalizes XYZ to [0,1] and uses white background for invalid points.
    """
    valid = np.isfinite(points).all(axis=-1)
    Pv = points.copy()
    lo = np.nanmin(Pv, axis=(0, 1))
    hi = np.nanmax(Pv, axis=(0, 1))
    rng = np.maximum(hi - lo, 1e-8)
    Pn = (Pv - lo) / rng
    Pn[~valid] = 1.0  # white background
    image = (np.clip(Pn, 0, 1) * 255).astype(np.uint8)
    return Image.fromarray(image)

DEFAULT_CAMERA_PARAMS = {'fov': 60, 'znear': 0.01, 'zfar': 16}


class Renderer:
    """
    """
    def __init__(self, config: OmegaConf):
        """
        """
        self.config = config
        self.renderer = pyrender.OffscreenRenderer(*config.target_dim)

        self.shaders = {
            'default': DefaultShaderCache(),
            'normals': NormalShaderCache(),
            'faceids': FaceidShaderCache(),
            'barycnt': BarycentricShaderCache(),
        }

    def set_object(self, source: Trimesh | Scene, smooth=False):
        """
        """
        if isinstance(source, Scene):
            self.tmesh = scene2mesh(source)
            
            # center + scale normalization
            self.tmesh = self.tmesh.copy()
            center = self.tmesh.vertices.mean(axis=0)
            self.tmesh.vertices -= center

            # Optional scale normalization (recommended)
            scale = np.max(np.linalg.norm(self.tmesh.vertices, axis=1))
            if scale > 0:
                self.tmesh.vertices /= scale
            
            # use centered mesh for pyrender scene
            self.scene = pyrender.Scene(ambient_light=[1.0, 1.0, 1.0]) # RGB no direction
            for name, geom in source.geometry.items():
                if name in source.graph:
                    pose, _ = source.graph[name]
                else:
                    pose = None
                self.scene.add(pyrender.Mesh.from_trimesh(geom, smooth=smooth), pose=pose)
        
        elif isinstance(source, Trimesh):
            self.tmesh = source
            
            # 🔵 CENTER + SCALE NORMALIZE TRIMESH
            # -------------------------------
            center = self.tmesh.vertices.mean(axis=0)
            self.tmesh.vertices -= center

            scale = np.max(np.linalg.norm(self.tmesh.vertices, axis=1))
            if scale > 0:
                self.tmesh.vertices /= scale
                
            self.scene = pyrender.Scene(ambient_light=[1.0, 1.0, 1.0])
            self.scene.add(pyrender.Mesh.from_trimesh(source, smooth=smooth))

        else:
            raise ValueError(f'Invalid source type {type(source)}')
        
        # rearrange mesh for faceid rendering
        self.tmesh_faceid = duplicate_verts(self.tmesh)
        self.scene_faceid = pyrender.Scene(ambient_light=[1.0, 1.0, 1.0])
        self.scene_faceid.add(
            pyrender.Mesh.from_trimesh(self.tmesh_faceid, smooth=smooth)
        )

    def set_camera(self, camera_params: dict = None):
        """
        """
        self.camera_params = camera_params or dict(DEFAULT_CAMERA_PARAMS)
        self.camera_params['yfov'] = self.camera_params.get('yfov', self.camera_params.pop('fov'))
        self.camera_params['yfov'] = self.camera_params['yfov'] * np.pi / 180.0
        self.camera = pyrender.PerspectiveCamera(**self.camera_params)
        
        self.camera_node        = self.scene       .add(self.camera)
        self.camera_node_faceid = self.scene_faceid.add(self.camera)
        
    def render(
        self, 
        pose: HomogeneousTransform, 
        lightdir=np.array([0.0, 0.0, 1.0]), uv_map=False, interpolate_norms=True, blur_matte=False
    ) -> dict:
        """
        """
        self.scene       .set_pose(self.camera_node       , pose)
        self.scene_faceid.set_pose(self.camera_node_faceid, pose)

        def render(shader: str, scene):
            """
            """
            self.renderer._renderer._program_cache = self.shaders[shader]
            return self.renderer.render(scene)
        
        if uv_map:
            raw_color, raw_depth = render('default', self.scene)
        raw_norms, raw_depth = render('normals', self.scene)
        raw_faces, raw_depth = render('faceids', self.scene_faceid)
        raw_bcent, raw_depth = render('barycnt', self.scene_faceid)

        def render_norms(norms: NumpyTensor['h w 3']) -> NumpyTensor['h w 3']:
            """
            """
            return np.clip((norms / 255.0 - 0.5) * 2, -1, 1)

        def render_depth(depth: NumpyTensor['h w'], offset=2.8, alpha=0.8) -> NumpyTensor['h w']:
            """
            """
            return np.where(depth > 0, alpha * (1.0 - range_norm(depth, offset=offset)), 1)

        def render_faces(faces: NumpyTensor['h w 3']) -> NumpyTensor['h w']:
            """
            """
            faces = faces.astype(np.int32)
            faces = faces[:, :, 0] * 65536 + faces[:, :, 1] * 256 + faces[:, :, 2]
            faces[faces == (256 ** 3 - 1)] = -1 # set background to -1
            return faces

        def render_bcent(bcent: NumpyTensor['h w 3']) -> NumpyTensor['h w 3']:
            """
            """
            return np.clip(bcent / 255.0, 0, 1)

        def render_matte(
            norms: NumpyTensor['h w 3'],
            depth: NumpyTensor['h w'],
            faces: NumpyTensor['h w'],
            bcent: NumpyTensor['h w 3'],
            alpha=0.5, beta=0.25, gaussian_kernel_width=5, gaussian_sigma=1,
        ) -> NumpyTensor['h w 3']:
            """
            """
            if interpolate_norms: # NOTE requires process=True
                valid_mask = (faces >= 0) & (faces < len(self.tmesh.faces))
                flat_faces = np.where(valid_mask.reshape(-1), faces.reshape(-1), 0)
                verts_index = self.tmesh.faces[flat_faces]           # (n, 3)
                verts_norms = self.tmesh.vertex_normals[verts_index] # (n, 3, 3)
                norms = np.sum(verts_norms * bcent.reshape(-1, 3, 1), axis=1)
                norms = norms.reshape(bcent.shape)

            diffuse = np.sum(norms * lightdir, axis=2)
            diffuse = np.clip(diffuse, -1, 1)
            matte = 255 * (diffuse[:, :, None] * alpha + beta)
            matte = np.where(depth[:, :, None] > 0, matte, 255)
            matte = np.clip(matte, 0, 255).astype(np.uint8)
            matte = np.repeat(matte, 3, axis=2)
            
            if blur_matte:
                matte = (faces == -1)[:, :, None] * matte + \
                        (faces != -1)[:, :, None] * cv2.GaussianBlur(matte, (gaussian_kernel_width, gaussian_kernel_width), gaussian_sigma)
            return matte 

        norms = render_norms(raw_norms)
        depth = render_depth(raw_depth)
        faces = render_faces(raw_faces)
        bcent = render_bcent(raw_bcent)
        matte = raw_color if uv_map else render_matte(norms, raw_depth, faces, bcent) # use original depth for matte
        
        # ---- NEW: compute 3D point map (world coords) ----
        W, H = self.config.target_dim
        K = fov_to_intrinsics(W, H, self.camera_params["yfov"])
        point_map = depth_to_point_map(raw_depth, pose, K)

        return {
            'norms': norms, 
            'depth': depth, 
            'depth_raw': raw_depth, 
            'point_map': point_map, 
            'matte': matte, 
            'faces': faces, 
            'intrinsics': K
        }


def render_multiview(
    renderer: Renderer,
    camera_generation_method='sphere',
    renderer_args: dict=None,
    sampling_args: dict=None,
    lighting_args: dict=None, 
    lookat_position=np.array([0, 0, 0]),
    verbose=False,
) -> list[Image.Image]:
    """
    """
    lookat_position_torch = torch.from_numpy(lookat_position)
    # if camera_generation_method == 'sphere':
    #     views = sample_view_matrices(lookat_position=lookat_position_torch, **sampling_args).numpy()
    if camera_generation_method == 'sphere':

        num_views = sampling_args["num_views"]
        radius = sampling_args.get("radius", 2.0)
        front_direction = sampling_args.get("front_dir") if sampling_args else None

        # NEW unified sampling+ordering+debug
        views, directions, order, D, neighbor = generate_ordered_views(
            num_views=num_views,
            radius=radius,
            lookat_position_torch=lookat_position_torch,
            front_direction=front_direction,
            verbose=verbose
        )
    elif camera_generation_method == 'orbit':
        radius = (sampling_args or {}).get("radius", 2.0)
        views = sample_view_matrices_orbit(radius=radius, lookat_position=lookat_position_torch).numpy()
        # Compute view directions from camera positions (pointing toward lookat)
        directions = []
        for pose in views:
            camera_pos = pose[:3, 3]
            view_dir = lookat_position - camera_pos
            view_dir = view_dir / (np.linalg.norm(view_dir) + 1e-8)
            directions.append(view_dir)
        directions = np.array(directions)
    else:
        views = sample_view_matrices_polyhedra(camera_generation_method, lookat_position=lookat_position_torch, **sampling_args).numpy()
        # Compute view directions for polyhedra method
        directions = []
        for pose in views:
            camera_pos = pose[:3, 3]
            view_dir = lookat_position - camera_pos
            view_dir = view_dir / (np.linalg.norm(view_dir) + 1e-8)
            directions.append(view_dir)
        directions = np.array(directions)
    
    def compute_lightdir(pose: HomogeneousTransform) -> NumpyTensor[3]:
        """
        """
        lightdir = pose[:3, 3] - (lookat_position)
        return lightdir / np.linalg.norm(lightdir)

    renders = []
    if verbose:
        views = tqdm(views, 'Rendering Multiviews...')
    # When use_texture_prompt=True, matte is the textured render (UV/materials); else shaded matte from normals
    uv_map = bool(renderer.config.get("use_texture_prompt", False))
    for pose, view_dir in zip(views, directions):
        outputs = renderer.render(
            pose,
            lightdir=compute_lightdir(pose),
            uv_map=uv_map,
            **renderer_args,
        )

        outputs['matte'] = Image.fromarray(outputs['matte'])
        outputs['poses'] = pose
        outputs['view_dirs'] = view_dir
        renders.append(outputs)
    return {
        name: [render[name] for render in renders] for name in renders[0].keys()
    }
