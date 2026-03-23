import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from seg3d.utils.cameras import build_ordered_view_matrices 

# SAMPLING
def sample_fibonacci_directions(num_views: int):
    indices = torch.arange(num_views, dtype=torch.float32)
    phi = np.pi * (3.0 - np.sqrt(5.0))
    y = 1 - (2 * indices / (num_views - 1))
    r = torch.sqrt(1 - y * y)
    theta = phi * indices
    x = torch.cos(theta) * r
    z = torch.sin(theta) * r
    return torch.stack([x, y, z], dim=-1)   # (N, 3) unit vectors

# ANGULAR GEOMETRY
def compute_angular_distance(u, v):
    cross = np.linalg.norm(np.cross(u, v))
    dot = np.clip(np.dot(u, v), -1.0, 1.0)
    return np.arctan2(cross, dot)

def compute_angular_distance_matrix(directions):
    N = directions.shape[0]
    D = np.zeros((N, N), dtype=np.float32)

    for i in range(N):
        for j in range(N):
            if i != j:
                D[i, j] = np.degrees(compute_angular_distance(directions[i], directions[j]))

    return D

# PATH SOLVING
def compute_neighbor_distances(D, order):
    dists = []
    for t in range(len(order)-1):
        i = order[t]
        j = order[t+1]
        dists.append(D[i, j])
    return dists

def two_opt(order, D, max_iter=50):
    """
    2-OPT optimization: smooth the traversal path by removing large jumps.
    order: list of indices (initial traversal)
    D: angular distance matrix (degrees)
    max_iter: number of passes

    Returns:
        improved_order
    """
    best = order.copy()
    improved = True
    N = len(order)

    def path_length(order):
        return sum(D[order[i], order[i+1]] for i in range(N-1))

    best_dist = path_length(best)

    for _ in range(max_iter):
        improved = False
        # try all segment swaps (i,j)
        for i in range(1, N-2):
            for j in range(i+1, N-1):
                if j - i == 1:
                    continue  # segments adjacent, skip
                
                # candidate swap
                new_order = best[:i] + best[i:j+1][::-1] + best[j+1:]
                new_dist = path_length(new_order)
                
                if new_dist < best_dist:
                    best = new_order
                    best_dist = new_dist
                    improved = True
                    break
            if improved:
                break

        if not improved:
            break

    return best

def rotate_order_to_front(order, front_idx):
    """
    Rotate traversal order so that front_idx becomes order[0].
    Preserves path continuity while ensuring semantic start position.
    """
    if front_idx not in order:
        raise ValueError(f"front_idx {front_idx} not found in order")
    k = order.index(front_idx)
    return order[k:] + order[:k]

def compute_spatial_traversal_order(directions, front=None):
    """
    directions: (N,3) numpy array of unit vectors
    front: (3,) direction that should become view 0 (camera position direction)
    returns (order, D)
    """
    N = directions.shape[0]
    D = compute_angular_distance_matrix(directions)

    # define front direction (view 0)
    if front is None:
        front = np.array([0, 0, 1], dtype=np.float32)
    front = np.asarray(front, dtype=np.float32)
    norm = np.linalg.norm(front)
    if norm == 0:
        raise ValueError("front direction must be non-zero.")
    front = front / norm

    camera_front = -front  # 🔑 camera must face the model front

    start_idx = int(np.argmin([
        compute_angular_distance(d, camera_front) for d in directions
    ]))

    visited = {start_idx}
    order = [start_idx]
    current = start_idx

    # greedy nearest neighbor
    for _ in range(N - 1):
        drow = D[current].copy()
        drow[list(visited)] = np.inf
        nxt = int(np.argmin(drow))
        order.append(nxt)
        visited.add(nxt)
        current = nxt

    # apply 2-OPT smoothing
    order = two_opt(order, D)

    # 🔥 CRITICAL FIX: rotate so front view is first
    # 2-OPT optimizes path continuity but may break the start position
    # Since traversal on a sphere is cyclic, we rotate to restore semantic start
    order = rotate_order_to_front(order, start_idx)

    return order, D

# HIGH-LEVEL GENERATION WRAPPER
def generate_ordered_views(
    num_views: int,
    radius: float,
    lookat_position_torch: torch.Tensor,
    front_direction=None,
    verbose=False
):
    """
    High-level wrapper:
    - samples directions
    - computes traversal order (with 2-opt smoothing)
    - computes angular distances
    - prints debug info
    - returns ordered view matrices
    """
    # Step 1: uniform sphere directions -> camera direction vectors
    directions = sample_fibonacci_directions(num_views).numpy()

    # Step 2: traversal ordering (with 2-opt smoothing)
    order, D = compute_spatial_traversal_order(directions, front=front_direction)

    # Step 3: neighbor angular distances
    neighbor = compute_neighbor_distances(D, order)
    maxdist = float(np.max(neighbor))
    mindist = float(np.min(neighbor))
    meandist = float(np.mean(neighbor))

    if verbose:
        print(f"\n[SphereSampling] num_views = {num_views}")
        print(f"[SphereSampling] Angular neighbor distances:")
        print(f"       min:  {mindist:.2f}°")
        print(f"       mean: {meandist:.2f}°")
        print(f"       max:  {maxdist:.2f}°")

    # Step 4: actual cam2world matrices in order
    views = build_ordered_view_matrices(
        directions,
        order,
        radius,
        lookat_position_torch
    ).numpy()

    ordered_directions = directions[order]

    return views, ordered_directions, order, D, neighbor

#AUTOMATIC SEARCH
def find_min_views_for_angle(target_angle_degrees=50, start=8, end=200):
    """
    Find the smallest num_views such that the max neighbor distance < target_angle_degrees.
    """
    best_data = None

    for n in range(start, end+1):
        directions = sample_fibonacci_directions(n).numpy()
        order, D = compute_spatial_traversal_order(directions)
        neighbor = compute_neighbor_distances(D, order)
        maxdist = max(neighbor)

        if maxdist < target_angle_degrees:
            best_data = (n, maxdist, directions, order, D)
            break

    return best_data  # returns (num_views, maxdist, directions, order, D)

# VISUALIZATION / DEBUGGING
def visualize_angular_matrix(D):
    plt.figure(figsize=(6,5))
    plt.imshow(D, cmap='hot', interpolation='nearest')
    plt.colorbar(label='Angular Distance (degrees)')
    plt.title("Angular Distance Matrix Between Views")
    plt.xlabel("View index")
    plt.ylabel("View index")
    plt.show()
    
def draw_unit_sphere(ax, alpha=0.15):
    u, v = np.mgrid[0:2*np.pi:50j, 0:np.pi:25j]
    x = np.cos(u) * np.sin(v)
    y = np.sin(u) * np.sin(v)
    z = np.cos(v)
    ax.plot_surface(x, y, z, color='gray', alpha=alpha, linewidth=0)

def visualize_camera_views(directions, order=None, radius=1.0):
    pts = directions * radius

    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(111, projection='3d')
    draw_unit_sphere(ax)

    ax.scatter(pts[:, 0], pts[:,1], pts[:,2], color='blue', s=50)

    # index labels
    for i, p in enumerate(pts):
        ax.text(p[0], p[1], p[2], str(i), color='black')

    # traversal arrows
    if order is not None:
        for t in range(len(order)-1):
            a = pts[order[t]]
            b = pts[order[t+1]]
            ax.plot([a[0],b[0]], [a[1],b[1]], [a[2],b[2]], color='green', linewidth=2)

    ax.set_xlim([-radius, radius])
    ax.set_ylim([-radius, radius])
    ax.set_zlim([-radius, radius])
    ax.set_title("Camera View Directions on Sphere")
    plt.show()
    
def analyze_views(num_views, radius=1.0, visualize=False):
    directions = sample_fibonacci_directions(num_views).numpy()

    # IMPORTANT: use the SAME traversal function
    order, D = compute_spatial_traversal_order(directions)

    neighbor = compute_neighbor_distances(D, order)

    maxdist = float(np.max(neighbor))
    mindist = float(np.min(neighbor))
    meandist = float(np.mean(neighbor))

    print(f"\n=== Angular Neighbor Statistics for {num_views} Views ===")
    print(f"Max angular neighbor distance : {maxdist:.2f}°")
    print(f"Min angular neighbor distance : {mindist:.2f}°")
    print(f"Mean angular neighbor distance: {meandist:.2f}°")

    if visualize:
        visualize_angular_matrix(D)
        visualize_camera_views(directions, order, radius)

    return {
        "num_views": num_views,
        "max": maxdist,
        "min": mindist,
        "mean": meandist,
        "order": order,
        "D": D,
        "directions": directions
    }

# ==============================================================================
# TWO-STAGE FRONT SELECTION
# ==============================================================================

def auto_front_by_visibility(depth_raw):
    """
    Pick the view with maximum visible surface area (Stage A: automatic).
    
    Args:
        depth_raw: list of (H, W) depth maps
    
    Returns:
        int: index of view with most visible surface
    """
    scores = [np.mean(d > 0) for d in depth_raw]
    return int(np.argmax(scores))

def select_front_index(renders, manual_view0=None):
    """
    Decide which view should become View 0 (Stage B: auto or manual).
    
    Args:
        renders: dict with render outputs including "depth_raw" and "view_dirs"
        manual_view0: int or None. If set, use this view as front (manual override).
    
    Returns:
        int: index of view to become View 0
    """
    N = len(renders["view_dirs"])

    if manual_view0 is not None:
        if not (0 <= manual_view0 < N):
            raise ValueError(f"manual_view0={manual_view0} out of range [0, {N})")
        print(f"[FrontSelection] MANUAL View {manual_view0}")
        return manual_view0

    idx = auto_front_by_visibility(renders["depth_raw"])
    print(f"[FrontSelection] AUTO (visibility) View {idx}")
    return idx


def rotate_renders_to_front(renders, front_idx):
    """
    Rotate all render outputs so front_idx becomes View 0.
    """
    N = len(renders["view_dirs"])

    if front_idx == 0:
        return renders

    order = list(range(front_idx, N)) + list(range(0, front_idx))

    for key, value in renders.items():
        if isinstance(value, list) and len(value) == N:
            renders[key] = [value[i] for i in order]

    return renders

    
