import numpy as np
import torch
import torch.nn.functional as F

from seg3d.data.common import NumpyTensor, TorchTensor
from seg3d.utils.polyhedra import *

import numpy as np
import torch


HomogeneousTransform = NumpyTensor['b... 4 4'] | TorchTensor['b... 4 4']


# Adapted from Segment Any Mesh (samesh) for camera pose/view-matrix computation.
def matrix3x4_to_4x4(matrix3x4: HomogeneousTransform) -> HomogeneousTransform:
    """
    Convert a 3x4 transformation matrix to a 4x4 transformation matrix.
    """
    bottom = torch.zeros_like(matrix3x4[:, 0, :].unsqueeze(-2))
    bottom[..., -1] = 1
    return torch.cat([matrix3x4, bottom], dim=-2)

# Adapted from Segment Any Mesh (samesh).
def view_matrix(
    camera_position: TorchTensor['n... 3'],
    lookat_position: TorchTensor['n... 3'] = torch.tensor([0, 0, 0]),
    up             : TorchTensor['3']      = torch.tensor([0, 1, 0]),
) -> HomogeneousTransform:
    """
    Given lookat position, camera position, and up vector, compute cam2world poses.
    """
    if camera_position.ndim == 1:
        camera_position = camera_position.unsqueeze(0)
    if lookat_position.ndim == 1:
        lookat_position = lookat_position.unsqueeze(0)
    camera_position = camera_position.float()
    lookat_position = lookat_position.float()

    cam_u = up.unsqueeze(0).repeat(len(lookat_position), 1).float().to(camera_position.device)

    # handle degenerate cases
    crossp = torch.abs(torch.cross(lookat_position - camera_position, cam_u, dim=-1)).max(dim=-1).values
    camera_position[crossp < 1e-6] += 1e-6

    cam_z = F.normalize((lookat_position - camera_position), dim=-1)
    cam_x = F.normalize(torch.cross(cam_z, cam_u, dim=-1), dim=-1)
    cam_y = F.normalize(torch.cross(cam_x, cam_z, dim=-1), dim=-1)
    poses = torch.stack([cam_x, cam_y, -cam_z, camera_position], dim=-1) # same as nerfstudio convention [right, up, -lookat]
    poses = matrix3x4_to_4x4(poses)
    return poses

# Adapted from Segment Any Mesh (samesh).
def sample_view_matrices(n: int, radius: float, lookat_position: TorchTensor = torch.tensor([0, 0, 0])) -> HomogeneousTransform:
    """
    Sample n uniformly distributed view matrices spherically with given radius.
    """
    tht = torch.rand(n) * torch.pi * 2
    phi = torch.rand(n) * torch.pi
    world_x = radius * torch.sin(phi) * torch.cos(tht)
    world_y = radius * torch.sin(phi) * torch.sin(tht)
    world_z = radius * torch.cos(phi)
    camera_position = torch.stack([world_x, world_y, world_z], dim=-1)
    lookat_position = lookat_position.unsqueeze(0).repeat(n, 1)
    return view_matrix(
        camera_position.to(lookat_position.device),
        lookat_position,
        up=torch.tensor([0, 1, 0], device=lookat_position.device)
    )

# Adapted from Segment Any Mesh (samesh).
def sample_view_matrices_polyhedra(polygon: str, radius: float, lookat_position: TorchTensor['3']=torch.tensor([0, 0, 0]), **kwargs) -> HomogeneousTransform:
    """
    Sample view matrices according to a polygon with given radius.
    """
    camera_position = torch.from_numpy(eval(polygon)(**kwargs)) * radius
    return view_matrix(
        camera_position.to(lookat_position.device) + lookat_position,
        lookat_position,
        up=torch.tensor([0, 1, 0], device=lookat_position.device)
    )

# Adapted from Segment Any Mesh (samesh).
def cam2world_opengl2pytorch3d(cam2world: HomogeneousTransform) -> HomogeneousTransform:
    """
    Convert OpenGL camera matrix to PyTorch3D camera matrix. Compare view_matrix function with

    https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/renderer/cameras.py#L1637
    
    for details regarding convention PyTorch3D uses.
    """
    if isinstance(cam2world, np.ndarray):
        cam2world = torch.from_numpy(cam2world).float()

    world2cam = torch.zeros_like(cam2world)
    world2cam[:3, :3] = cam2world[:3, :3]
    world2cam[:3, 0] = -world2cam[:3, 0]
    world2cam[:3, 2] = -world2cam[:3, 2]
    world2cam[:3, 3] = -world2cam[:3, :3].T @ cam2world[:3, 3]
    return world2cam

def sample_view_matrices_orbit(
    radius: float = 2.0,
    lookat_position: torch.Tensor = torch.tensor([0, 0, 0]),
) -> torch.Tensor:
    """
    Exact 12-view GeoSAM2 camera layout:
    - 4 azimuths: 0°, 90°, 180°, 270°
    - 3 elevations: +25°, 0°, -25°
    - CCW ordering (inverse-clockwise sequence)
    """

    azimuths = [0, 90, 180, 270]         # CCW
    elevations = [25, 0, -25]            # 3 rings

    poses = []
    for elev in elevations:
        for azim in azimuths:

            az = np.deg2rad(azim)
            el = np.deg2rad(elev)

            # Spherical to Cartesian
            x = radius * np.cos(el) * np.sin(az)
            y = radius * np.sin(el)
            z = radius * np.cos(el) * np.cos(az)

            camera_pos = torch.tensor([x, y, z], dtype=torch.float32)

            # Always look at object center
            pose = view_matrix(
                camera_position=camera_pos.unsqueeze(0),
                lookat_position=lookat_position.unsqueeze(0),
                up=torch.tensor([0, 1, 0]),
            )[0]

            poses.append(pose)

    return torch.stack(poses, dim=0)

def fov_to_intrinsics(width: int, height: int, fov_rad: float) -> NumpyTensor['3 3']:
    """
    Compute the intrinsic camera matrix from the field of view (FOV) and image dimensions.
    Args:
        width: Image width in pixels.
        height: Image height in pixels.
        fov: Horizontal field of view in radians.
    """
    fy = (0.5 * height) / np.tan(0.5 * fov_rad)
    fx = fy * (width / height)
    cx, cy = width / 2, height / 2
    
    return np.array([[fx, 0, cx],
                     [0, fy, cy],
                     [0, 0, 1]], dtype=np.float32)
    
def depth_to_point_map(depth_raw: np.ndarray,
                       pose_cam_to_world: np.ndarray,
                       K: np.ndarray) -> np.ndarray:
    H, W = depth_raw.shape
    fx, fy = K[0,0], K[1,1]
    cx, cy = K[0,2], K[1,2]
    xs = np.arange(W, dtype=np.float32)
    ys = np.arange(H, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xs, ys)

    z = depth_raw.astype(np.float32)
    valid = z > 0

    X = (grid_x - cx) * z / fx
    Y = (grid_y - cy) * z / fy
    Z = -z  # OpenGL camera looks along −Z

    points_cam = np.stack([X, -Y, Z, np.ones_like(Z)], axis=-1)
    points_world = points_cam @ pose_cam_to_world.T
    Pw = points_world[..., :3] / points_world[..., 3:]
    Pw[~valid] = np.nan
    return Pw  # (H, W, 3) 

def build_ordered_view_matrices(directions: np.ndarray,
                                order: list[int],
                                radius: float,
                                lookat_position: torch.Tensor):
    """
    Convert ordered sphere directions into ordered cam2world matrices.
    """
    # reorder directions
    directions = directions[order]    # (N,3)

    # convert to 3D positions
    positions = torch.from_numpy(directions).float() * radius

    # camera looks at center
    lookat = lookat_position.unsqueeze(0).repeat(len(positions), 1)

    # use your existing view_matrix function
    poses = view_matrix(
        camera_position=positions,
        lookat_position=lookat,
        up=torch.tensor([0,1,0])
    )

    return poses