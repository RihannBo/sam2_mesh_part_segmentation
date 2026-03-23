# Code adapted from Segment Any Mesh (samesh)
# https://github.com/gtangg12/samesh


import glob
import json
import os
import copy
from pathlib import Path
from collections import defaultdict

import numpy as np
import pymeshlab
import trimesh
import networkx as nx
import igraph
from numpy.random import RandomState
from trimesh.base import Trimesh, Scene
from sklearn.mixture import GaussianMixture
from tqdm import tqdm
from omegaconf import OmegaConf

from seg3d.data.common import NumpyTensor
from seg3d.data.loaders import scene2mesh, read_mesh
from seg3d.utils.mesh import duplicate_verts
from matplotlib.pyplot import get_cmap

# Define the cmap object
jet_cmap = get_cmap('jet')

EPSILON = 1e-20
SCALE = 1e6


def partition_cost(      ## :))))
    mesh           : Trimesh,
    partition      : NumpyTensor['f'],
    cost_data      : NumpyTensor['f num_components'],
    cost_smoothness: NumpyTensor['e']
) -> float:
    """
    """
    cost = 0
    for f in range(len(partition)): ## loop through all faces
        cost += cost_data[f, partition[f]]      ## this term encourages the solution to stay close to the original SAM segmentation.
    for i, edge in enumerate(mesh.face_adjacency):  ## for every pair of adjacent faces (f1, f2)
        f1, f2 = int(edge[0]), int(edge[1])
        if partition[f1] != partition[f2]:         
            cost += cost_smoothness[i]          ## this penalizes label boundaries that cross smooth surfaces.
    return cost

## This function builds a graph whose min-cut gives the optimal label assignment
## for one expansion step (“should each face stay in its label or switch to α?”).
def construct_expansion_graph(      ## :)))
    label          : int,
    mesh           : Trimesh,
    partition      : NumpyTensor['f'],
    cost_data      : NumpyTensor['f num_components'],
    cost_smoothness: NumpyTensor['e']
) -> nx.Graph:
    """
    """
    ## Step 1: Create the graph and special terminals 
    G = nx.Graph() # undirected graph
    A = 'alpha'    ## source = stay with current label
    B = 'alpha_complement'  ## sink = switch to α

    node2index = {} ## mapping to integer indices
    G.add_node(A)   ## add the terminal nodes to the graph
    G.add_node(B)
    node2index[A] = 0
    node2index[B] = 1
    
    ## Step 2: Add one node per face
    for i in range(len(mesh.faces)):
        G.add_node(i)
        node2index[i] = 2 + i

    ## Step 3: Add auxiliary nodes for currently different-label boundaries
    aux_count = 0
    for i, edge in enumerate(mesh.face_adjacency): # auxillary nodes
        f1, f2 = int(edge[0]), int(edge[1])
        if partition[f1] != partition[f2]:
            a = (f1, f2)
            if a in node2index: # duplicate edge
                continue
            G.add_node(a)
            node2index[a] = len(mesh.faces) + 2 + aux_count
            aux_count += 1
            
    ## Step 4: Add data-term edges (A↔face and B↔face)

    for f in range(len(mesh.faces)):
        G.add_edge(A, f, capacity=cost_data[f, label])
        G.add_edge(B, f, capacity=float('inf') if partition[f] == label else cost_data[f, partition[f]])

    ## Step 5 — Add smoothness-term edges (face–face or via auxiliary)
    for i, edge in enumerate(mesh.face_adjacency):
        f1, f2 = int(edge[0]), int(edge[1])
        a = (f1, f2)
        if partition[f1] == partition[f2]: ## Neighbors currently have the same label
            if partition[f1] != label:
                G.add_edge(f1, f2, capacity=cost_smoothness[i])
        else:       ## Neighbors currently have different labels
            G.add_edge(a, B, capacity=cost_smoothness[i])
            if partition[f1] != label:
                G.add_edge(f1, a, capacity=cost_smoothness[i])
            if partition[f2] != label:
                G.add_edge(a, f2, capacity=cost_smoothness[i])
    
    return G, node2index


def repartition(        ## :)))
    mesh: trimesh.Trimesh,
    partition      : NumpyTensor['f'],
    cost_data      : NumpyTensor['f num_components'],
    cost_smoothness: NumpyTensor['e'],
    smoothing_iterations: int,
    _lambda=1.0,
    verbose: bool = True,
):
    A = 'alpha'
    B = 'alpha_complement'
    labels = np.unique(partition)

    cost_smoothness = cost_smoothness * _lambda

    # networkx broken for float capacities
    #cost_data       = np.round(cost_data       * SCALE).astype(int)
    #cost_smoothness = np.round(cost_smoothness * SCALE).astype(int)

    cost_min = partition_cost(mesh, partition, cost_data, cost_smoothness)  ## compute the current total energy (data + smoothness)

    for i in range(smoothing_iterations):
        label_iter = tqdm(labels, desc="repartition") if verbose else labels
        for label in label_iter:
            ## Build the α-expansion graph & index maps
            G, node2index = construct_expansion_graph(label, mesh, partition, cost_data, cost_smoothness)
            index2node = {v: k for k, v in node2index.items()}

            '''
            _, (S, T) = nx.minimum_cut(G, A, B)
            assert A in S and B in T
            S = np.array([v for v in S if isinstance(v, int)]).astype(int)
            T = np.array([v for v in T if isinstance(v, int)]).astype(int)
            '''
            ## Run min s–t cut (iGraph) and get the two partitions
            G = igraph.Graph.from_networkx(G) ## convert the grapg into iGraph
            outputs = G.st_mincut(source=node2index[A], target=node2index[B], capacity='capacity') ## st_mincut computes the minimum s-t cut using capacities
            ## it returns two disjoint sets:
            S = outputs.partition[0]  ## nodes on the source side (A’s side)
            T = outputs.partition[1]  ## nodes on the sink side (B’s side)
            assert node2index[A] in S and node2index[B] in T  ## The asserts ensure terminals are on their correct sides.
            S = np.array([index2node[v] for v in S if isinstance(index2node[v], int)]).astype(int)
            T = np.array([index2node[v] for v in T if isinstance(index2node[v], int)]).astype(int)

            assert (partition[S] == label).sum() == 0 # T consists of those assigned 'alpha' and S 'alpha_complement' (see paper)
            partition[T] = label

            cost = partition_cost(mesh, partition, cost_data, cost_smoothness)
            if cost > cost_min:
                raise ValueError('Cost increased. This should not happen because the graph cut is optimal.')
            cost_min = cost
    
    return partition

def prep_mesh_shape_diameter_function(source: Trimesh | Scene) -> Trimesh:  ## :)))
    """
    Prepares a 3D mesh for computing the Shape Diameter Function (SDF)
    by ensuring it is a single, clean `Trimesh` object with merged vertices.
    """
    if isinstance(source, trimesh.Scene):
        source = scene2mesh(source)
    source.merge_vertices(merge_tex=True, merge_norm=True) # This line removes duplicate vertices
    return source

def colormap_shape_diameter_function(mesh: Trimesh, sdf_values: NumpyTensor['f']) -> Trimesh:
    """
    Apply a color map to a 3D mesh based on its Shape Diameter Function (SDF) values.
    """
    assert len(mesh.faces) == len(sdf_values)
    mesh = duplicate_verts(mesh) # needed to prevent face color interpolation
    mesh.visual.face_colors = trimesh.visual.interpolate(sdf_values, color_map=jet_cmap)
    return mesh

def shape_diameter_function(mesh: Trimesh, norm=True, alpha=4, rays=64, cone_amplitude=120) -> NumpyTensor['f']:   ###
    """
    Computes the Shape Diameter Function (SDF) for a mesh, optionally normalizing the result.
    """
    mesh = pymeshlab.Mesh(mesh.vertices, mesh.faces)
    meshset = pymeshlab.MeshSet() # MeshSet is a container for multiple Mesh objects and provides methods to manipulate and process these meshes.
    meshset.add_mesh(mesh)
    meshset.compute_scalar_by_shape_diameter_function_per_vertex(rays=rays, cone_amplitude=cone_amplitude)

    sdf_values = meshset.current_mesh().face_scalar_array()
    sdf_values[np.isnan(sdf_values)] = 0
    if norm:
        # normalize and smooth shape diameter function values
        min = sdf_values.min()
        max = sdf_values.max()
        sdf_values = (sdf_values - min) / (max - min)
        sdf_values = np.log(sdf_values * alpha + 1) / np.log(alpha + 1)
    return sdf_values


