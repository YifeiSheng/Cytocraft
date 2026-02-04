#!/usr/bin/env python
"""
Segmentation Leakage Robustness Analysis for Cytocraft Algorithm

This script models a realistic tissue scenario where:
- Multiple cell types exist with DIFFERENT chromosome structures (e.g., epithelial, 
  fibroblast, immune cells each with distinct chromatin organization)
- Only Type 0 cells are used for Cytocraft reconstruction
- Other cell types (1, 2, etc.) are neighboring "contaminating" cells
- Segmentation leakage causes transcripts from neighboring cells to leak INTO Type 0 cells
- Leaked transcripts carry positional information from the wrong cell's structure
  (especially problematic when leakage comes from different cell types)

Leakage Model:
- Leakage only occurs INTO Type 0 cells from their spatial neighbors
- At 0% leakage: No contamination, perfect reconstruction of Type 0 structure
- Cross-type leakage (from Type 1/2 to Type 0) introduces very different structural patterns
- Same-type leakage (from Type 0 to Type 0) introduces similar structural patterns

Analysis:
- Simulates "segmentation leakage" at rates from 5% to 30%
- Only Type 0 cells are used for Cytocraft reconstruction
- Ground truth is the Type 0 chromosome structure (simX)
- Evaluates robustness when contaminated by neighboring cells of different types

Usage:
    python simulation_segmentation_leakage.py <genome_structure.tsv> <output_dir> [options]
"""

import os
import sys
import csv
import time
import random
import argparse
import numpy as np
import pandas as pd
import scipy.stats
from pathlib import Path
from datetime import datetime
from scipy.stats import multivariate_normal
from scipy.spatial.distance import cdist
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('Agg')

# Import Cytocraft modules
from cytocraft.craft import *
from cytocraft.model import BasisShapeModel


def DeriveRotation(W, X, Mask):
    """Derive rotation matrices from W and X."""
    F = int(W.shape[0] / 2)
    Rotation = np.zeros((F, 3, 3))
    for i in range(F):
        Wi = W[:, Mask[i, :]]
        Wi_filter = Wi[~np.isnan(Wi).any(axis=1), :]
        while Wi_filter.shape[0] < 6:
            Mask[i, :] = change_last_true(Mask[i, :])
            Wi = W[:, Mask[i, :]]
            Wi_filter = Wi[~np.isnan(Wi).any(axis=1), :]
        idx = int(find_subarray(Wi_filter, Wi[i * 2]) / 2)
        Xi = X[Mask[i, :], :]
        model = factor(Wi_filter)
        R = kabsch_numpy(np.dot(model.Rs[idx], model.Ss[0]).T, Xi)[0]
        Rotation[i] = R
    return Rotation


def scale_X(X, size=1):
    """Scale coordinates to fit within specified size."""
    min_x, min_y, min_z = X.min(axis=0)
    max_x, max_y, max_z = X.max(axis=0)
    range_x = max_x - min_x
    range_y = max_y - min_y
    range_z = max_z - min_z
    scale = size / max(range_x, range_y, range_z)
    scaled_X = (X - [min_x, min_y, min_z]) * scale
    return scaled_X, scale


def euclidean_distance_3d_matrix(X):
    """Compute pairwise Euclidean distance matrix."""
    import math
    n = X.shape[0]
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1, z1 = X[i, :]
            x2, y2, z2 = X[j, :]
            d = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)
            matrix[i][j] = d
            matrix[j][i] = d
    return matrix


def distance_to_similarity(matrix):
    """Convert distance matrix to similarity matrix."""
    similarity = 1 / (1 + matrix)
    return similarity


def centerX(X):
    """Center coordinates around origin."""
    if len(X.shape) == 2 and X.shape[1] == 3:
        mean_x, mean_y, mean_z = X.mean(axis=0)
        centered_X = X - [mean_x, mean_y, mean_z]
        return centered_X, [mean_x, mean_y, mean_z]
    else:
        raise ValueError("X shape error!")


def generate_cell_positions_2d(Ncell, tissue_size=100, min_distance=5, seed=None):
    """
    Generate 2D spatial positions for cells in tissue space.
    Uses rejection sampling to ensure minimum distance between cells.
    
    Parameters:
    -----------
    Ncell : int
        Number of cells
    tissue_size : float
        Size of the tissue area (tissue_size x tissue_size)
    min_distance : float
        Minimum distance between cell centers
    seed : int, optional
        Random seed
    
    Returns:
    --------
    np.array : (Ncell, 2) array of cell positions
    """
    if seed is not None:
        np.random.seed(seed)
    
    positions = []
    max_attempts = 10000
    attempts = 0
    
    while len(positions) < Ncell and attempts < max_attempts:
        # Generate random position
        new_pos = np.random.uniform(0, tissue_size, 2)
        
        # Check distance to existing positions
        if len(positions) == 0:
            positions.append(new_pos)
        else:
            distances = np.linalg.norm(np.array(positions) - new_pos, axis=1)
            if np.all(distances >= min_distance):
                positions.append(new_pos)
        
        attempts += 1
    
    # If we couldn't place all cells with min_distance, use grid + jitter
    if len(positions) < Ncell:
        grid_size = int(np.ceil(np.sqrt(Ncell)))
        spacing = tissue_size / grid_size
        positions = []
        for i in range(grid_size):
            for j in range(grid_size):
                if len(positions) >= Ncell:
                    break
                pos = np.array([i * spacing + spacing/2, j * spacing + spacing/2])
                pos += np.random.uniform(-spacing/4, spacing/4, 2)
                positions.append(pos)
    
    return np.array(positions[:Ncell])


def generate_cell_type_conformations(base_conformation, n_cell_types, between_type_variation=0.5, seed=None):
    """
    Generate distinct chromosome conformations for different cell types.
    Each cell type has a fundamentally different 3D structure.
    
    Parameters:
    -----------
    base_conformation : np.array
        Reference 3D chromosome structure (Ngene, 3)
    n_cell_types : int
        Number of distinct cell types
    between_type_variation : float
        Level of structural difference between cell types (0.0-1.0)
        Higher values = more different structures between types
    seed : int, optional
        Random seed
    
    Returns:
    --------
    list : List of (Ngene, 3) arrays, one base conformation per cell type
    """
    if seed is not None:
        np.random.seed(seed)
    
    Ngene = base_conformation.shape[0]
    type_conformations = [base_conformation.copy()]  # First type uses base
    
    # Calculate the scale of the structure
    structure_scale = np.std(base_conformation)
    
    for t in range(1, n_cell_types):
        # Generate a significantly different conformation for each cell type
        # Use larger perturbation + random rotation to create distinct structures
        
        # Random rotation to create different orientation
        theta = np.random.uniform(0, 2 * np.pi)
        phi = np.random.uniform(0, np.pi)
        psi = np.random.uniform(0, 2 * np.pi)
        
        # Rotation matrices
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(theta), -np.sin(theta)],
                       [0, np.sin(theta), np.cos(theta)]])
        Ry = np.array([[np.cos(phi), 0, np.sin(phi)],
                       [0, 1, 0],
                       [-np.sin(phi), 0, np.cos(phi)]])
        Rz = np.array([[np.cos(psi), -np.sin(psi), 0],
                       [np.sin(psi), np.cos(psi), 0],
                       [0, 0, 1]])
        R = Rz @ Ry @ Rx
        
        # Apply rotation
        rotated = base_conformation @ R.T
        
        # Add large-scale structural variation (deformation)
        # Different cell types have fundamentally different chromatin organization
        noise = np.random.normal(0, structure_scale * between_type_variation, (Ngene, 3))
        
        # Also add some local rearrangements (shuffle nearby genes slightly)
        if between_type_variation > 0.3:
            # Swap some gene positions to simulate different folding patterns
            n_swaps = int(Ngene * between_type_variation * 0.1)
            for _ in range(n_swaps):
                i, j = np.random.choice(Ngene, 2, replace=False)
                rotated[[i, j]] = rotated[[j, i]]
        
        type_conformation = rotated + noise
        type_conformations.append(type_conformation)
    
    return type_conformations


def assign_cell_types(Ncell, n_cell_types, cell_positions=None, spatial_clustering=0.0, seed=None):
    """
    Assign cell types to cells, optionally with spatial clustering.
    
    Parameters:
    -----------
    Ncell : int
        Number of cells
    n_cell_types : int
        Number of distinct cell types
    cell_positions : np.array, optional
        (Ncell, 2) array of cell positions for spatial clustering
    spatial_clustering : float
        Degree of spatial clustering (0.0 = random, 1.0 = fully clustered)
    seed : int, optional
        Random seed
    
    Returns:
    --------
    np.array : Array of cell type assignments (0 to n_cell_types-1)
    """
    if seed is not None:
        np.random.seed(seed)
    
    if spatial_clustering == 0.0 or cell_positions is None:
        # Random assignment with roughly equal proportions
        cell_types = np.random.randint(0, n_cell_types, Ncell)
    else:
        # Spatially clustered assignment using k-means-like approach
        from scipy.cluster.vq import kmeans2
        
        # Use k-means to cluster cells spatially
        if Ncell >= n_cell_types:
            centroids, labels = kmeans2(cell_positions, n_cell_types, minit='points')
            
            # Mix random and clustered based on spatial_clustering parameter
            random_types = np.random.randint(0, n_cell_types, Ncell)
            use_clustered = np.random.random(Ncell) < spatial_clustering
            cell_types = np.where(use_clustered, labels, random_types)
        else:
            cell_types = np.random.randint(0, n_cell_types, Ncell)
    
    return cell_types


def generate_cell_specific_conformations(base_conformation, Ncell, variation_level=0.1, 
                                          n_cell_types=1, cell_types=None, 
                                          type_conformations=None, seed=None):
    """
    Generate cell-specific chromosome conformations with biological variation.
    Supports multiple cell types with different base structures.
    
    Parameters:
    -----------
    base_conformation : np.array
        Reference 3D chromosome structure (Ngene, 3)
    Ncell : int
        Number of cells
    variation_level : float
        Level of structural variation between cells of the SAME type (0.0-1.0)
    n_cell_types : int
        Number of distinct cell types (default: 1)
    cell_types : np.array, optional
        Array of cell type assignments (if None, all cells are same type)
    type_conformations : list, optional
        List of base conformations per cell type (if None, uses base_conformation)
    seed : int, optional
        Random seed
    
    Returns:
    --------
    list : List of (Ngene, 3) arrays, one per cell
    """
    if seed is not None:
        np.random.seed(seed)
    
    Ngene = base_conformation.shape[0]
    conformations = []
    
    # Default: all cells are the same type
    if cell_types is None:
        cell_types = np.zeros(Ncell, dtype=int)
    
    # Default: use base_conformation for all types
    if type_conformations is None:
        type_conformations = [base_conformation]
    
    # Calculate the scale of the structure for proportional noise
    structure_scale = np.std(base_conformation)
    
    for c in range(Ncell):
        # Get the base conformation for this cell's type
        cell_type = cell_types[c]
        type_base = type_conformations[min(cell_type, len(type_conformations) - 1)]
        
        # Add cell-specific random perturbation (within-type variation)
        # This models biological variation in chromosome structure between cells of same type
        noise = np.random.normal(0, structure_scale * variation_level, (Ngene, 3))
        cell_conformation = type_base + noise
        conformations.append(cell_conformation)
    
    return conformations


def get_cell_centroids_2d(gem):
    """
    Calculate 2D centroids for each cell based on transcript positions.
    
    Parameters:
    -----------
    gem : pd.DataFrame
        Gene expression matrix with columns: geneID, x, y, MIDCount, CellID
    
    Returns:
    --------
    dict : {CellID: (centroid_x, centroid_y)}
    """
    gem = gem.copy()
    gem['x'] = gem['x'].astype(float)
    gem['y'] = gem['y'].astype(float)
    gem['MIDCount'] = gem['MIDCount'].astype(float)
    
    centroids = {}
    for cell_id in gem['CellID'].unique():
        cell_data = gem[gem['CellID'] == cell_id]
        # Weighted centroid by MIDCount
        total_count = cell_data['MIDCount'].sum()
        if total_count > 0:
            cx = (cell_data['x'] * cell_data['MIDCount']).sum() / total_count
            cy = (cell_data['y'] * cell_data['MIDCount']).sum() / total_count
        else:
            cx = cell_data['x'].mean()
            cy = cell_data['y'].mean()
        centroids[cell_id] = (cx, cy)
    
    return centroids


def apply_segmentation_leakage(gem, leakage_rate, cell_positions, cell_types=None, seed=None):
    """
    Simulate segmentation leakage where transcripts leak INTO Type 0 cells
    from their spatial neighbors (including cells of other types).
    
    Model:
    - Only Type 0 cells can RECEIVE leaked transcripts
    - Transcripts from any neighboring cell can leak into nearby Type 0 cells
    - Cross-type leakage (from Type 1/2 to Type 0) introduces different structural patterns
    - Same-type leakage (from Type 0 to Type 0) introduces similar structural patterns
    - Leakage probability is weighted by distance (closer neighbors more likely)
    
    Parameters:
    -----------
    gem : pd.DataFrame
        Original gene expression matrix
    leakage_rate : float
        Fraction of transcripts from each cell that may leak (0.0 to 1.0)
    cell_positions : np.array
        (Ncell, 2) array of cell positions in tissue space
    cell_types : np.array, optional
        Array of cell type assignments. Type 0 = target cells for reconstruction.
    seed : int, optional
        Random seed for reproducibility
    
    Returns:
    --------
    tuple : (Modified gem with leakage applied, leakage_stats dict)
        leakage_stats contains: n_leaked, n_cross_type, n_same_type, cross_type_fraction
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    
    gem_leaked = gem.copy()
    gem_leaked['x'] = gem_leaked['x'].astype(float)
    gem_leaked['y'] = gem_leaked['y'].astype(float)
    
    # Get unique cell IDs
    cell_ids = sorted(gem['CellID'].unique(), key=lambda x: int(x))
    
    # Initialize leakage statistics
    leakage_stats = {
        'n_leaked': 0,
        'n_cross_type': 0,
        'n_same_type': 0,
        'cross_type_fraction': 0.0
    }
    
    if len(cell_ids) < 2:
        return gem_leaked, leakage_stats  # Cannot apply leakage with < 2 cells
    
    # Use provided cell positions for spatial neighbor calculation
    # Calculate pairwise distances between cells in tissue space
    dist_matrix = cdist(cell_positions, cell_positions)
    
    # For each cell, find k nearest neighbors (excluding self)
    k_neighbors = min(5, len(cell_ids) - 1)  # Use up to 5 nearest neighbors
    
    # Create mapping from cell_id to index
    cell_to_idx = {cid: i for i, cid in enumerate(cell_ids)}
    idx_to_cell = {i: cid for cid, i in cell_to_idx.items()}
    
    # Identify Type 0 cells (target cells for reconstruction)
    # Only Type 0 cells can RECEIVE leaked transcripts from their neighbors
    if cell_types is not None:
        type0_cell_indices = np.where(cell_types == 0)[0]
        type0_cell_set = set(type0_cell_indices)
    else:
        # If no cell types defined, all cells are Type 0
        type0_cell_indices = np.arange(len(cell_ids))
        type0_cell_set = set(range(len(cell_ids)))
    
    # Pre-compute for each cell: its Type 0 neighbors and their weights
    # This avoids repeated computation in the main loop
    cell_type0_neighbors = {}  # cell_idx -> list of (neighbor_idx, weight)
    
    for cell_idx in range(len(cell_ids)):
        distances = dist_matrix[cell_idx].copy()
        distances[cell_idx] = np.inf  # Exclude self
        
        # Find k nearest neighbors
        neighbor_indices = np.argsort(distances)[:k_neighbors]
        
        # Filter to Type 0 neighbors only
        type0_neighbors = [ni for ni in neighbor_indices if ni in type0_cell_set]
        
        if len(type0_neighbors) > 0:
            neighbor_distances = distances[type0_neighbors]
            neighbor_distances = np.maximum(neighbor_distances, 1e-6)
            weights = 1.0 / neighbor_distances
            weights = weights / weights.sum()
            cell_type0_neighbors[cell_idx] = (type0_neighbors, weights)
    
    # VECTORIZED APPROACH: Process by cell, not by transcript
    # For each cell, decide which transcripts leak and where
    
    rows_to_modify = []
    new_cell_ids = []
    original_cell_indices = []
    target_cell_indices = []
    
    for cell_id in cell_ids:
        cell_idx = cell_to_idx[cell_id]
        
        # Skip cells with no Type 0 neighbors
        if cell_idx not in cell_type0_neighbors:
            continue
        
        type0_neighbors, weights = cell_type0_neighbors[cell_idx]
        
        # Get all transcript indices for this cell
        cell_mask = gem_leaked['CellID'] == cell_id
        cell_transcript_indices = gem_leaked.index[cell_mask].values
        n_transcripts = len(cell_transcript_indices)
        
        if n_transcripts == 0:
            continue
        
        # Determine which transcripts leak (vectorized random)
        leak_mask = np.random.random(n_transcripts) < leakage_rate
        leaking_indices = cell_transcript_indices[leak_mask]
        
        if len(leaking_indices) == 0:
            continue
        
        # For each leaking transcript, choose a Type 0 neighbor
        chosen_neighbors = np.random.choice(type0_neighbors, size=len(leaking_indices), p=weights)
        
        for idx, neighbor_idx in zip(leaking_indices, chosen_neighbors):
            rows_to_modify.append(idx)
            new_cell_ids.append(idx_to_cell[neighbor_idx])
            original_cell_indices.append(cell_idx)
            target_cell_indices.append(neighbor_idx)
    
    # Apply modifications using loc for efficiency
    if len(rows_to_modify) > 0:
        gem_leaked.loc[rows_to_modify, 'CellID'] = new_cell_ids
    
    # Calculate leakage statistics
    leakage_stats['n_leaked'] = len(rows_to_modify)
    
    if cell_types is not None and len(rows_to_modify) > 0:
        # Vectorized cross-type counting
        orig_types = cell_types[original_cell_indices]
        target_types = cell_types[target_cell_indices]
        leakage_stats['n_cross_type'] = int(np.sum(orig_types != target_types))
        leakage_stats['n_same_type'] = int(np.sum(orig_types == target_types))
        
        leakage_stats['cross_type_fraction'] = (
            leakage_stats['n_cross_type'] / leakage_stats['n_leaked'] 
            if leakage_stats['n_leaked'] > 0 else 0.0
        )
    
    return gem_leaked, leakage_stats


def generate_simulation_data(Conformation, Ncell, noise, resolution, rateCap, rateDrop, seed,
                             structure_variation=0.1, tissue_size=100,
                             n_cell_types=1, between_type_variation=0.5, spatial_clustering=0.0):
    """
    Generate simulation data for leakage analysis with cell-specific chromosome structures.
    
    This models a realistic tissue scenario where:
    - Multiple cell types exist with DIFFERENT chromosome structures
    - Cells of the SAME type have similar structures (with small variation)
    - Cells are spatially distributed in 2D tissue space
    - Each cell views the chromosome from a different angle (rotation)
    
    Parameters:
    -----------
    Conformation : np.array
        Reference 3D chromosome structure (Ngene, 3)
    Ncell : int
        Number of cells
    noise : float
        Noise level for transcript localization
    resolution : int
        Resolution for scaling
    rateCap : float
        Capture rate
    rateDrop : float
        Gene dropout rate
    seed : int
        Random seed for reproducibility
    structure_variation : float
        Level of structural variation between cells of SAME type (0.0-1.0)
    tissue_size : float
        Size of the tissue area for cell placement
    n_cell_types : int
        Number of distinct cell types with different chromosome structures
    between_type_variation : float
        Level of structural difference between cell types (0.0-1.0)
    spatial_clustering : float
        Degree of spatial clustering of cell types (0.0 = random, 1.0 = clustered)
    
    Returns:
    --------
    gem : pd.DataFrame
        Simulated gene expression matrix
    simX : np.array
        Ground truth reference coordinates (consensus structure)
    cell_positions : np.array
        (Ncell, 2) array of cell positions in tissue space
    cell_conformations : list
        List of cell-specific chromosome conformations
    cell_types : np.array
        Array of cell type assignments for each cell
    type_conformations : list
        List of base conformations for each cell type
    """
    random.seed(seed)
    np.random.seed(seed)
    
    Ngene = Conformation.shape[0]
    
    # Scale and center the reference coordinates
    ScaledCoords, _ = scale_X(Conformation, resolution)
    simX, _ = centerX(ScaledCoords)
    
    # Generate cell positions in 2D tissue space
    cell_positions = generate_cell_positions_2d(Ncell, tissue_size=tissue_size, 
                                                 min_distance=tissue_size/np.sqrt(Ncell)/2,
                                                 seed=seed)
    
    # Generate distinct base conformations for each cell type
    type_conformations = generate_cell_type_conformations(
        simX, n_cell_types, 
        between_type_variation=between_type_variation,
        seed=seed + 500
    )
    
    # Assign cell types (optionally with spatial clustering)
    cell_types = assign_cell_types(
        Ncell, n_cell_types, 
        cell_positions=cell_positions,
        spatial_clustering=spatial_clustering,
        seed=seed + 600
    )
    
    # Generate cell-specific chromosome conformations with biological variation
    # Each cell gets variation around its cell type's base conformation
    cell_conformations = generate_cell_specific_conformations(
        simX, Ncell, 
        variation_level=structure_variation,
        n_cell_types=n_cell_types,
        cell_types=cell_types,
        type_conformations=type_conformations,
        seed=seed + 1000
    )
    
    # Generate random rotation matrices for each cell (viewing angle)
    randRM = generate_random_rotation_matrices(Ncell)
    
    # Generate W (2D projections) for each cell using its specific conformation
    simW = np.zeros((Ncell * 2, Ngene))
    for c in range(Ncell):
        # Use cell-specific conformation
        cell_X = cell_conformations[c]
        # Apply cell-specific rotation (viewing angle)
        XY = np.dot(cell_X, randRM[c, :, :])[:, 0:2]
        simW[c * 2] = XY[:, 0]
        simW[c * 2 + 1] = XY[:, 1]
    
    # Add noise to locations
    if noise > 0:
        simW = np.random.normal(simW, noise)
    
    # Calculate bounds for expression matrix
    UBx = int(np.max(simW[::2], axis=1).max())
    LBx = int(np.min(simW[::2], axis=1).min())
    UBy = int(np.max(simW[1::2], axis=1).max())
    LBy = int(np.min(simW[1::2], axis=1).min())
    
    Matrix = np.empty((Ncell, Ngene, UBx - LBx, UBy - LBy), dtype="i1")
    x = np.arange(LBx, UBx)
    y = np.arange(LBy, UBy)
    xx, yy = np.meshgrid(x, y)
    xy = np.stack((xx, yy), axis=-1)
    
    # Set covariance based on noise
    if noise == 0:
        sigma = np.eye(2) * 8
    else:
        A = np.random.rand(2, 2)
        sigma = np.dot(A, A.transpose()) * 8
    
    # Generate expression values
    for c in range(Ncell):
        for g in range(Ngene):
            loc = simW[c * 2 : (c + 1) * 2, g]
            exp = multivariate_normal.pdf(xy, mean=loc, cov=sigma, allow_singular=True) * 128
            Matrix[c, g] = exp.T.round().clip(min=0, max=127).astype("i1")
    
    # Apply capture rate
    CapMatrix = np.random.binomial(Matrix, rateCap)
    
    # Apply dropout
    mask = np.random.random((Ncell, Ngene)) < rateDrop
    CapMatrix[mask] = 0
    
    # Create gem DataFrame
    columns = ["geneID", "x", "y", "MIDCount", "ExonCount", "CellID"]
    rows = []
    for index, n in np.ndenumerate(CapMatrix):
        if n > 0:
            row = [
                "gene" + str(index[1]),
                str(index[2]),
                str(index[3]),
                str(n),
                str(n),
                str(index[0]),
            ]
            rows.append(row)
    
    gem = pd.DataFrame(rows, columns=columns)
    return gem, simX, cell_positions, cell_conformations, cell_types, type_conformations


def run_cytocraft_reconstruction(gem, simX, Ngene_for_rotation, max_iterations=30):
    """
    Run Cytocraft reconstruction and evaluate against ground truth.
    
    Returns:
    --------
    dict : Results containing RMSD, Spearman correlation, etc.
    """
    start_time = time.time()
    
    # Get gene list
    X_genes = pd.Series(
        sorted(gem.geneID.drop_duplicates(), key=lambda fname: int(fname.strip("gene")))
    )
    Ngene = len(X_genes)
    
    if Ngene == 0:
        return {
            'success': False,
            'error': 'No genes in data',
            'rmsd': np.nan,
            'relative_error': np.nan,
            'spearman': np.nan
        }
    
    # Get centers
    try:
        W = get_centers(gem, gem.CellID.drop_duplicates().values, X_genes)
        W = normalizeZ(W)
    except Exception as e:
        return {
            'success': False,
            'error': f'Error in get_centers: {str(e)}',
            'rmsd': np.nan,
            'relative_error': np.nan,
            'spearman': np.nan
        }
    
    # Initialize with random rotation matrices
    RM = generate_random_rotation_matrices(int(W.shape[0] / 2))
    
    CellUIDs = list(gem.CellID.drop_duplicates())
    Mask = MASK(gem, GeneIDs=X_genes, CellIDs=CellUIDs, Ngene=Ngene_for_rotation)
    
    # Initial F-step
    try:
        X, _, _ = UpdateF(RM, W, X_genes)
    except np.linalg.LinAlgError:
        return {
            'success': False,
            'error': 'LinAlgError at initialization',
            'rmsd': np.nan,
            'relative_error': np.nan,
            'spearman': np.nan
        }
    
    # Iterative optimization
    try:
        for loop in range(max_iterations):
            # R-step
            RM = DeriveRotation(W, X, Mask)
            
            # F-step
            try:
                X, _, _ = UpdateF(RM, W, X_genes)
            except np.linalg.LinAlgError:
                break
            
            # Evaluate RMSD
            rmsd1 = kabsch_numpy(
                normalizeF(simX, method="mean"), normalizeF(X, method="mean")
            )[2]
            X_mirror = np.copy(X)
            X_mirror[:, 2] = -X_mirror[:, 2]
            rmsd2 = kabsch_numpy(
                normalizeF(simX, method="mean"), normalizeF(X_mirror, method="mean")
            )[2]
            minrmsd = min(rmsd1, rmsd2)
            
            # Check convergence
            if minrmsd < 0.005:
                break
    
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'rmsd': np.nan,
            'relative_error': np.nan,
            'spearman': np.nan
        }
    
    end_time = time.time()
    
    # Final evaluation
    rmsd1 = kabsch_numpy(
        normalizeF(simX, method="mean"), normalizeF(X, method="mean")
    )[2]
    X_mirror = np.copy(X)
    X_mirror[:, 2] = -X_mirror[:, 2]
    rmsd2 = kabsch_numpy(
        normalizeF(simX, method="mean"), normalizeF(X_mirror, method="mean")
    )[2]
    final_rmsd = min(rmsd1, rmsd2)
    mirror = 1 if final_rmsd == rmsd2 else 0
    
    # Calculate relative error: RMSD / RMS of ground truth
    simX_normalized = normalizeF(simX, method="mean")
    rms_ground_truth = np.sqrt(np.mean(np.sum(simX_normalized ** 2, axis=1)))
    relative_error = final_rmsd / rms_ground_truth
    
    # Calculate Spearman correlation on distance/similarity matrices
    D = euclidean_distance_3d_matrix(X)
    S = distance_to_similarity(D)
    D_ = euclidean_distance_3d_matrix(simX)
    S_ = distance_to_similarity(D_)
    spearman = scipy.stats.spearmanr(S_.flatten(), S.flatten())[0]
    
    return {
        'success': True,
        'error': None,
        'rmsd': final_rmsd,
        'relative_error': relative_error,
        'spearman': spearman,
        'mirror': mirror,
        'iterations': loop + 1,
        'time': end_time - start_time,
        'final_X': X
    }


def run_leakage_analysis(
    genome_structure_file,
    output_dir,
    leakage_rates=None,
    n_repeats=10,
    Ngene=100,
    Ncell=50,
    rateCap=0.8,
    rateDrop=0.2,
    resolution=10,
    Ngene_for_rotation=30,
    noise=1,
    mode="random",
    structure_variation=0.1,
    tissue_size=100,
    n_cell_types=1,
    between_type_variation=0.5,
    spatial_clustering=0.0
):
    """
    Run comprehensive segmentation leakage analysis with multi-cell-type tissue model.
    
    This models a realistic tissue scenario where:
    - Multiple cell types exist with DIFFERENT chromosome structures
      (e.g., epithelial, fibroblast, immune cells with distinct chromatin organization)
    - Cells of the SAME type have similar structures (with small within-type variation)
    - Cells are spatially arranged in 2D tissue space (optionally clustered by type)
    - Segmentation leakage causes transcripts to be misassigned to nearby cells
    - Leakage between DIFFERENT cell types introduces very different structural patterns
    
    Parameters:
    -----------
    genome_structure_file : str
        Path to genome structure TSV file
    output_dir : str
        Output directory
    leakage_rates : list
        List of leakage rates to test (default: [0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30])
    n_repeats : int
        Number of repeats per leakage rate (default: 10)
    structure_variation : float
        Level of structural variation between cells of SAME type (0.0-1.0, default: 0.1)
    tissue_size : float
        Size of tissue area for cell placement (default: 100)
    n_cell_types : int
        Number of distinct cell types with different chromosome structures (default: 1)
    between_type_variation : float
        Level of structural difference between cell types (0.0-1.0, default: 0.5)
    spatial_clustering : float
        Degree of spatial clustering of cell types (0.0=random, 1.0=clustered, default: 0.0)
    
    Returns:
    --------
    pd.DataFrame : Results DataFrame
    """
    if leakage_rates is None:
        leakage_rates = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    analysis_dir = os.path.join(output_dir, f"leakage_analysis_{timestamp}")
    Path(analysis_dir).mkdir(parents=True, exist_ok=True)
    
    # Load genome structure
    GenomeStructure = pd.read_csv(genome_structure_file, delimiter="\t")
    
    print("=" * 70)
    print("CYTOCRAFT SEGMENTATION LEAKAGE ROBUSTNESS ANALYSIS")
    print("(with multi-cell-type tissue model)")
    print("=" * 70)
    print(f"Start time: {datetime.now()}")
    print(f"Output directory: {analysis_dir}")
    print(f"Leakage rates to test: {leakage_rates}")
    print(f"Repeats per rate: {n_repeats}")
    print(f"Parameters: Ngene={Ngene}, Ncell={Ncell}, rateCap={rateCap}, "
          f"rateDrop={rateDrop}, noise={noise}")
    print(f"Within-type structure variation: {structure_variation*100:.0f}%")
    print(f"Number of cell types: {n_cell_types}")
    print(f"Between-type structure variation: {between_type_variation*100:.0f}%")
    print(f"Spatial clustering of cell types: {spatial_clustering*100:.0f}%")
    print(f"Tissue size: {tissue_size}")
    print("=" * 70)
    
    # Generate base simulation data
    data_seed = 42
    if mode == "continous":
        Conformation = np.array(GenomeStructure[["x", "y", "z"]].head(Ngene))
    elif mode == "random":
        np.random.seed(data_seed)
        sampled_indices = sorted(
            np.random.choice(len(GenomeStructure), size=Ngene, replace=False)
        )
        Conformation = np.array(GenomeStructure[["x", "y", "z"]].iloc[sampled_indices])
    
    # Generate clean simulation data with multi-cell-type tissue model
    gem_clean, simX, cell_positions, cell_conformations, cell_types, type_conformations = generate_simulation_data(
        Conformation, Ncell, noise, resolution, rateCap, rateDrop, data_seed,
        structure_variation=structure_variation, tissue_size=tissue_size,
        n_cell_types=n_cell_types, between_type_variation=between_type_variation,
        spatial_clustering=spatial_clustering
    )
    
    print(f"Generated simulation data: {len(gem_clean)} transcript entries")
    print(f"Ground truth (reference) shape: {simX.shape}")
    print(f"Number of cells: {Ncell}")
    print(f"Number of cell types: {n_cell_types}")
    # Count cells per type
    type_counts = {t: np.sum(cell_types == t) for t in range(n_cell_types)}
    print(f"Cells per type: {type_counts}")
    print(f"Cell positions in tissue: {cell_positions.shape}")
    
    # Identify Type 0 cells (target cells for reconstruction)
    type0_cell_ids = [str(i) for i, ct in enumerate(cell_types) if ct == 0]
    n_type0_cells = len(type0_cell_ids)
    print(f"Type 0 cells (used for reconstruction): {n_type0_cells}")
    print("-" * 70)
    
    # Save type conformations for visualization/analysis
    if n_cell_types > 1:
        type_conf_file = os.path.join(analysis_dir, "type_conformations.npz")
        np.savez(type_conf_file, 
                 type_conformations=np.array(type_conformations),
                 cell_types=cell_types,
                 cell_positions=cell_positions,
                 simX=simX)
        print(f"Saved {n_cell_types} type conformations to: {type_conf_file}")
    
    # Run analysis for each leakage rate
    results = []
    
    for leakage_rate in leakage_rates:
        print(f"\n[Testing leakage rate: {leakage_rate*100:.0f}%]")
        
        for repeat in range(n_repeats):
            leakage_seed = int(leakage_rate * 1000) + repeat * 100 + 7
            
            # Apply segmentation leakage: transcripts from neighbors leak INTO Type 0 cells
            if leakage_rate > 0:
                gem_leaked, leakage_stats = apply_segmentation_leakage(
                    gem_clean, leakage_rate, cell_positions, 
                    cell_types=cell_types, seed=leakage_seed
                )
            else:
                gem_leaked = gem_clean.copy()
                leakage_stats = {'n_leaked': 0, 'n_cross_type': 0, 
                                 'n_same_type': 0, 'cross_type_fraction': 0.0}
            
            # Filter to Type 0 cells only for reconstruction
            # Only Type 0 cells are used for Cytocraft - they may now contain
            # leaked transcripts from neighboring cells (including other types)
            gem_type0 = gem_leaked[gem_leaked['CellID'].isin(type0_cell_ids)].copy()
            
            # Run reconstruction on Type 0 cells only
            # simX is the ground truth for Type 0 (the reference conformation)
            result = run_cytocraft_reconstruction(gem_type0, simX, Ngene_for_rotation)
            
            result['leakage_rate'] = leakage_rate
            result['repeat'] = repeat
            result['leakage_seed'] = leakage_seed
            
            # Add leakage statistics
            result['n_leaked'] = leakage_stats['n_leaked']
            result['n_cross_type'] = leakage_stats['n_cross_type']
            result['n_same_type'] = leakage_stats['n_same_type']
            result['cross_type_fraction'] = leakage_stats['cross_type_fraction']
            
            results.append(result)
            
            if result['success']:
                cross_type_info = f", CrossType={result['cross_type_fraction']*100:.1f}%" if n_cell_types > 1 else ""
                print(f"  Repeat {repeat + 1}/{n_repeats}: "
                      f"Spearman={result['spearman']:.4f}, "
                      f"RMSD={result['rmsd']:.4f}, "
                      f"RelErr={result['relative_error']:.4f}{cross_type_info}")
            else:
                print(f"  Repeat {repeat + 1}/{n_repeats}: FAILED - {result['error']}")
    
    # Convert to DataFrame
    results_df = pd.DataFrame([
        {
            'leakage_rate': r['leakage_rate'],
            'leakage_percent': r['leakage_rate'] * 100,
            'repeat': r['repeat'],
            'success': r['success'],
            'rmsd': r['rmsd'],
            'relative_error': r['relative_error'],
            'spearman': r['spearman'],
            'mirror': r.get('mirror', np.nan),
            'iterations': r.get('iterations', np.nan),
            'time_seconds': r.get('time', np.nan),
            'n_leaked': r.get('n_leaked', 0),
            'n_cross_type': r.get('n_cross_type', 0),
            'n_same_type': r.get('n_same_type', 0),
            'cross_type_fraction': r.get('cross_type_fraction', 0.0),
            'error': r.get('error', None)
        }
        for r in results
    ])
    
    # Save results
    results_csv = os.path.join(analysis_dir, "leakage_analysis_results.csv")
    results_df.to_csv(results_csv, index=False)
    print(f"\nResults saved to: {results_csv}")
    
    # Calculate and print summary statistics
    print("\n" + "=" * 70)
    print("ANALYSIS SUMMARY")
    print("=" * 70)
    
    print("\n{:>12} {:>10} {:>12} {:>12} {:>12} {:>10}".format(
        "Leakage %", "N Success", "Med Spearman", "Mean Spearman", "Med RMSD", "Med RelErr"))
    print("-" * 70)
    
    summary_data = []
    for leakage_rate in leakage_rates:
        rate_data = results_df[(results_df['leakage_rate'] == leakage_rate) & 
                                (results_df['success'] == True)]
        n_success = len(rate_data)
        
        if n_success > 0:
            med_spearman = rate_data['spearman'].median()
            mean_spearman = rate_data['spearman'].mean()
            std_spearman = rate_data['spearman'].std()
            med_rmsd = rate_data['rmsd'].median()
            med_rel_err = rate_data['relative_error'].median()
            
            # Cross-type leakage statistics
            mean_cross_type_frac = rate_data['cross_type_fraction'].mean()
            
            print(f"{leakage_rate*100:>10.0f}% {n_success:>10} "
                  f"{med_spearman:>12.4f} {mean_spearman:>12.4f} "
                  f"{med_rmsd:>12.4f} {med_rel_err:>12.4f}")
            
            summary_data.append({
                'leakage_rate': leakage_rate,
                'leakage_percent': leakage_rate * 100,
                'n_success': n_success,
                'n_total': n_repeats,
                'median_spearman': med_spearman,
                'mean_spearman': mean_spearman,
                'std_spearman': std_spearman,
                'median_rmsd': med_rmsd,
                'mean_rmsd': rate_data['rmsd'].mean(),
                'std_rmsd': rate_data['rmsd'].std(),
                'median_relative_error': med_rel_err,
                'mean_relative_error': rate_data['relative_error'].mean(),
                'std_relative_error': rate_data['relative_error'].std(),
                'mean_n_leaked': rate_data['n_leaked'].mean(),
                'mean_cross_type_fraction': mean_cross_type_frac
            })
        else:
            print(f"{leakage_rate*100:>10.0f}% {n_success:>10} {'N/A':>12} {'N/A':>12} {'N/A':>12} {'N/A':>12}")
    
    # Save summary statistics
    summary_df = pd.DataFrame(summary_data)
    summary_csv = os.path.join(analysis_dir, "leakage_summary_statistics.csv")
    summary_df.to_csv(summary_csv, index=False)
    
    # Generate summary text file
    summary_file = os.path.join(analysis_dir, "summary_statistics.txt")
    with open(summary_file, 'w') as f:
        f.write("CYTOCRAFT SEGMENTATION LEAKAGE ROBUSTNESS ANALYSIS\n")
        f.write("(with multi-cell-type tissue model)\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Analysis Date: {datetime.now()}\n")
        f.write(f"Genome Structure: {genome_structure_file}\n\n")
        
        f.write("Simulation Parameters:\n")
        f.write(f"  Number of genes: {Ngene}\n")
        f.write(f"  Number of cells: {Ncell}\n")
        f.write(f"  Capture rate: {rateCap}\n")
        f.write(f"  Dropout rate: {rateDrop}\n")
        f.write(f"  Noise level: {noise}\n")
        f.write(f"  Resolution: {resolution}\n")
        f.write(f"  Genes for rotation: {Ngene_for_rotation}\n")
        f.write(f"  Repeats per leakage rate: {n_repeats}\n")
        f.write(f"  Tissue size: {tissue_size}\n\n")
        
        f.write("Multi-Cell-Type Model Parameters:\n")
        f.write(f"  Number of cell types: {n_cell_types}\n")
        f.write(f"  Type 0 cells (used for reconstruction): {n_type0_cells}\n")
        f.write(f"  Within-type structure variation: {structure_variation*100:.0f}%\n")
        f.write(f"  Between-type structure variation: {between_type_variation*100:.0f}%\n")
        f.write(f"  Spatial clustering of cell types: {spatial_clustering*100:.0f}%\n")
        f.write(f"  Cells per type: {type_counts}\n\n")
        
        f.write("Leakage Model:\n")
        f.write("  - Only Type 0 cells are used for Cytocraft reconstruction\n")
        f.write("  - Other cell types (1, 2, etc.) are neighboring 'contaminating' cells\n")
        f.write("  - Transcripts leak INTO Type 0 cells from their spatial neighbors\n")
        f.write("  - Cross-type leakage (from Type 1/2 -> Type 0): different structure patterns\n")
        f.write("  - Same-type leakage (from Type 0 -> Type 0): similar structure patterns\n")
        f.write("  - At 0% leakage: no contamination, pure Type 0 reconstruction\n\n")
        
        f.write("Results Summary:\n")
        f.write("-" * 70 + "\n")
        
        for row in summary_data:
            f.write(f"\nLeakage Rate: {row['leakage_percent']:.0f}%\n")
            f.write(f"  Success rate: {row['n_success']}/{row['n_total']}\n")
            f.write(f"  Spearman Correlation: {row['median_spearman']:.4f} (median), "
                    f"{row['mean_spearman']:.4f} ± {row['std_spearman']:.4f} (mean ± std)\n")
            f.write(f"  RMSD: {row['median_rmsd']:.4f} (median), "
                    f"{row['mean_rmsd']:.4f} ± {row['std_rmsd']:.4f} (mean ± std)\n")
            f.write(f"  Relative Error: {row['median_relative_error']:.4f} (median)\n")
            if n_cell_types > 1 and row['leakage_percent'] > 0:
                f.write(f"  Cross-type leakage: {row['mean_cross_type_fraction']*100:.1f}% of leaked transcripts\n")
                f.write(f"  Mean transcripts leaked: {row['mean_n_leaked']:.0f}\n")
        
        # Find threshold for correlation > 0.9
        high_corr_rates = [d for d in summary_data if d['median_spearman'] >= 0.9]
        if high_corr_rates:
            max_robust_rate = max(d['leakage_percent'] for d in high_corr_rates)
            f.write(f"\n\nConclusion:\n")
            f.write(f"  Cytocraft maintains robust reconstruction accuracy (median Spearman\n")
            f.write(f"  correlation > 0.90) even under {max_robust_rate:.0f}% transcript misassignment\n")
            f.write(f"  due to segmentation leakage. This demonstrates that the weighted\n")
            f.write(f"  centroid approach effectively dampens the impact of segmentation errors.\n")
    
    print(f"\nSummary saved to: {summary_file}")
    
    # Generate plots
    generate_leakage_plots(results_df, summary_df, analysis_dir, leakage_rates)
    
    return results_df, summary_df


def generate_leakage_plots(results_df, summary_df, output_dir, leakage_rates):
    """Generate visualization plots for leakage analysis."""
    
    # Figure 1: Main results - Spearman correlation vs leakage rate
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1a: Spearman correlation vs leakage rate (box plot)
    ax1 = axes[0, 0]
    successful = results_df[results_df['success'] == True]
    
    box_data = [successful[successful['leakage_rate'] == rate]['spearman'].values 
                for rate in leakage_rates]
    box_labels = [f"{rate*100:.0f}%" for rate in leakage_rates]
    
    bp = ax1.boxplot(box_data, labels=box_labels, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('steelblue')
        patch.set_alpha(0.7)
    
    ax1.axhline(0.9, color='red', linestyle='--', alpha=0.7, label='Threshold: 0.9')
    ax1.set_xlabel('Segmentation Leakage Rate')
    ax1.set_ylabel('Spearman Correlation')
    ax1.set_title('Reconstruction Accuracy vs Segmentation Leakage')
    ax1.legend()
    ax1.set_ylim(0, 1.05)
    
    # 1b: Mean Spearman with error bars
    ax2 = axes[0, 1]
    if len(summary_df) > 0:
        ax2.errorbar(summary_df['leakage_percent'], summary_df['mean_spearman'],
                     yerr=summary_df['std_spearman'], marker='o', capsize=5,
                     color='steelblue', linewidth=2, markersize=8)
        ax2.axhline(0.9, color='red', linestyle='--', alpha=0.7, label='Threshold: 0.9')
        ax2.fill_between(summary_df['leakage_percent'], 0.9, 1.0, alpha=0.2, color='green',
                         label='Robust region (>0.9)')
    ax2.set_xlabel('Segmentation Leakage Rate (%)')
    ax2.set_ylabel('Spearman Correlation')
    ax2.set_title('Mean Reconstruction Accuracy (± std)')
    ax2.legend()
    ax2.set_ylim(0, 1.05)
    ax2.set_xlim(-1, max(leakage_rates) * 100 + 1)
    
    # 1c: RMSD vs leakage rate
    ax3 = axes[1, 0]
    box_data_rmsd = [successful[successful['leakage_rate'] == rate]['rmsd'].values 
                     for rate in leakage_rates]
    
    bp3 = ax3.boxplot(box_data_rmsd, labels=box_labels, patch_artist=True)
    for patch in bp3['boxes']:
        patch.set_facecolor('coral')
        patch.set_alpha(0.7)
    
    ax3.set_xlabel('Segmentation Leakage Rate')
    ax3.set_ylabel('RMSD')
    ax3.set_title('RMSD vs Segmentation Leakage')
    
    # 1d: Relative Error vs leakage rate
    ax4 = axes[1, 1]
    box_data_relerr = [successful[successful['leakage_rate'] == rate]['relative_error'].values 
                       for rate in leakage_rates]
    
    bp4 = ax4.boxplot(box_data_relerr, labels=box_labels, patch_artist=True)
    for patch in bp4['boxes']:
        patch.set_facecolor('mediumseagreen')
        patch.set_alpha(0.7)
    
    ax4.set_xlabel('Segmentation Leakage Rate')
    ax4.set_ylabel('Relative Error')
    ax4.set_title('Relative Error vs Segmentation Leakage')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'leakage_analysis_summary.png'), dpi=150)
    plt.savefig(os.path.join(output_dir, 'leakage_analysis_summary.pdf'))
    plt.close()
    
    # Figure 2: Publication-quality main figure
    fig, ax = plt.subplots(figsize=(8, 6))
    
    if len(summary_df) > 0:
        ax.errorbar(summary_df['leakage_percent'], summary_df['median_spearman'],
                    yerr=[summary_df['median_spearman'] - successful.groupby('leakage_rate')['spearman'].quantile(0.25).values,
                          successful.groupby('leakage_rate')['spearman'].quantile(0.75).values - summary_df['median_spearman']],
                    marker='o', capsize=5, color='#2E86AB', linewidth=2.5, markersize=10,
                    label='Median (IQR)')
        
        ax.axhline(0.9, color='#E94F37', linestyle='--', linewidth=2, 
                   label='Robust threshold (0.9)')
        ax.fill_between([0, 35], 0.9, 1.0, alpha=0.15, color='#A3BE8C',
                        label='Robust performance region')
    
    ax.set_xlabel('Segmentation Leakage Rate (%)', fontsize=12)
    ax.set_ylabel('Spearman Correlation', fontsize=12)
    ax.set_title('Cytocraft Robustness to Segmentation Leakage', fontsize=14)
    ax.legend(loc='lower left', fontsize=10)
    ax.set_ylim(0.5, 1.02)
    ax.set_xlim(-1, max(leakage_rates) * 100 + 2)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'leakage_robustness_main.png'), dpi=300)
    plt.savefig(os.path.join(output_dir, 'leakage_robustness_main.pdf'))
    plt.close()
    
    print(f"\nPlots saved to: {output_dir}")


def main():
    """Main function to run segmentation leakage analysis."""
    parser = argparse.ArgumentParser(
        description='Cytocraft Segmentation Leakage Robustness Analysis (Multi-Cell-Type Model)'
    )
    parser.add_argument('genome_structure', type=str,
                        help='Path to genome structure TSV file')
    parser.add_argument('output_dir', type=str,
                        help='Output directory for results')
    parser.add_argument('--leakage_rates', type=float, nargs='+',
                        default=[0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
                        help='Leakage rates to test (default: 0 0.05 0.10 0.15 0.20 0.25 0.30)')
    parser.add_argument('--n_repeats', type=int, default=10,
                        help='Number of repeats per leakage rate (default: 10)')
    parser.add_argument('--ngene', type=int, default=100,
                        help='Number of genes (default: 100)')
    parser.add_argument('--ncell', type=int, default=50,
                        help='Number of cells (default: 50)')
    parser.add_argument('--rate_cap', type=float, default=0.8,
                        help='Capture rate (default: 0.8)')
    parser.add_argument('--rate_drop', type=float, default=0.2,
                        help='Dropout rate (default: 0.2)')
    parser.add_argument('--resolution', type=int, default=10,
                        help='Resolution (default: 10)')
    parser.add_argument('--ngene_rotation', type=int, default=30,
                        help='Number of genes for rotation derivation (default: 30)')
    parser.add_argument('--noise', type=float, default=1,
                        help='Noise level (default: 1)')
    parser.add_argument('--mode', type=str, default='random',
                        choices=['random', 'continous'],
                        help='Gene selection mode (default: random)')
    parser.add_argument('--structure_variation', type=float, default=0.1,
                        help='Level of structural variation between cells of SAME type (0.0-1.0, default: 0.1). '
                             'This is the within-type variation.')
    parser.add_argument('--tissue_size', type=float, default=100,
                        help='Size of 2D tissue space for cell positioning (default: 100).')
    
    # Multi-cell-type model arguments
    parser.add_argument('--n_cell_types', type=int, default=1,
                        help='Number of distinct cell types with different chromosome structures (default: 1). '
                             'Set >1 to model tissues with multiple cell types (e.g., epithelial, fibroblast, immune).')
    parser.add_argument('--between_type_variation', type=float, default=0.5,
                        help='Level of structural difference BETWEEN cell types (0.0-1.0, default: 0.5). '
                             'Higher values = more different chromosome structures between types.')
    parser.add_argument('--spatial_clustering', type=float, default=0.0,
                        help='Degree of spatial clustering of cell types (0.0-1.0, default: 0.0). '
                             '0.0 = random mixing, 1.0 = fully spatially clustered by type.')
    
    args = parser.parse_args()
    
    # Run analysis
    results_df, summary_df = run_leakage_analysis(
        genome_structure_file=args.genome_structure,
        output_dir=args.output_dir,
        leakage_rates=args.leakage_rates,
        n_repeats=args.n_repeats,
        Ngene=args.ngene,
        Ncell=args.ncell,
        rateCap=args.rate_cap,
        rateDrop=args.rate_drop,
        resolution=args.resolution,
        Ngene_for_rotation=args.ngene_rotation,
        noise=args.noise,
        mode=args.mode,
        structure_variation=args.structure_variation,
        tissue_size=args.tissue_size,
        n_cell_types=args.n_cell_types,
        between_type_variation=args.between_type_variation,
        spatial_clustering=args.spatial_clustering
    )
    
    print("\n" + "=" * 70)
    print("SEGMENTATION LEAKAGE ANALYSIS COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()
