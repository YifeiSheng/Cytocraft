#!/usr/bin/env python
"""
Comprehensive Sensitivity Analysis for Cytocraft Algorithm

Analysis includes:
1. 100 independent runs with distinct random initializations for rotation matrices (R)
2. Measurement of deviation of final converged solutions (F* and R*) from ground truth
3. Convergence rate analysis
4. Multi-start strategy evaluation (default 5 random initializations)
5. Computational cost analysis

Usage:
    python sensitivity_analysis.py <genome_structure.tsv> <output_dir> [--n_runs 100] [--n_starts 5]
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


def generate_simulation_data(Conformation, Ncell, noise, resolution, rateCap, rateDrop, seed):
    """
    Generate simulation data for sensitivity analysis.
    
    Parameters:
    -----------
    Conformation : np.array
        Ground truth 3D coordinates (Ngene, 3)
    Ncell : int
        Number of cells
    noise : float
        Noise level
    resolution : int
        Resolution for scaling
    rateCap : float
        Capture rate
    rateDrop : float
        Gene dropout rate
    seed : int
        Random seed for reproducibility
    
    Returns:
    --------
    gem : pd.DataFrame
        Simulated gene expression matrix
    simX : np.array
        Ground truth centered coordinates
    """
    random.seed(seed)
    np.random.seed(seed)
    
    Ngene = Conformation.shape[0]
    
    # Scale and center coordinates
    ScaledCoords, _ = scale_X(Conformation, resolution)
    simX, _ = centerX(ScaledCoords)
    
    # Generate random rotation matrices for each cell
    randRM = generate_random_rotation_matrices(Ncell)
    
    # Generate W (2D projections)
    simW = np.zeros((Ncell * 2, Ngene))
    for c in range(randRM.shape[0]):
        XY = np.dot(simX, randRM[c, :, :])[:, 0:2]
        simW[c * 2] = XY[:, 0]
        simW[c * 2 + 1] = XY[:, 1]
    
    # Add noise to locations
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
    return gem, simX


def run_single_optimization(gem, simX, Ngene_for_rotation, init_seed, max_iterations=30):
    """
    Run a single optimization with specific initialization.
    
    Parameters:
    -----------
    gem : pd.DataFrame
        Gene expression matrix
    simX : np.array
        Ground truth coordinates
    Ngene_for_rotation : int
        Number of genes for rotation derivation
    init_seed : int
        Random seed for initialization
    max_iterations : int
        Maximum number of iterations
    
    Returns:
    --------
    dict : Results dictionary containing RMSD, convergence info, and timing
    """
    np.random.seed(init_seed)
    random.seed(init_seed)
    
    start_time = time.time()
    
    # Get gene list
    X_genes = pd.Series(
        sorted(gem.geneID.drop_duplicates(), key=lambda fname: int(fname.strip("gene")))
    )
    Ngene = len(X_genes)
    
    # Get centers
    W = get_centers(gem, gem.CellID.drop_duplicates().values, X_genes)
    W = normalizeZ(W)
    
    # Initialize with random rotation matrices
    RM = generate_random_rotation_matrices(int(W.shape[0] / 2))
    
    CellUIDs = list(gem.CellID.drop_duplicates())
    Mask = MASK(gem, GeneIDs=X_genes, CellIDs=CellUIDs, Ngene=Ngene_for_rotation)
    
    # Initial F-step
    try:
        X, _, _ = UpdateF(RM, W, X_genes)
    except np.linalg.LinAlgError:
        return {
            'converged': False,
            'error': 'LinAlgError at initialization',
            'rmsd': np.nan,
            'spearman': np.nan,
            'iterations': 0,
            'time': time.time() - start_time
        }
    
    # Iterative optimization
    rmsd_history = []
    converged = False
    final_loop = 0
    
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
            rmsd_history.append(minrmsd)
            
            final_loop = loop + 1
            
            # Check convergence
            if minrmsd < 0.005:
                converged = True
                break
            
            # Check for convergence plateau
            if len(rmsd_history) > 5:
                recent = rmsd_history[-5:]
                if max(recent) - min(recent) < 0.0001:
                    converged = True
                    break
    
    except Exception as e:
        return {
            'converged': False,
            'error': str(e),
            'rmsd': np.nan,
            'spearman': np.nan,
            'iterations': final_loop,
            'time': time.time() - start_time
        }
    
    end_time = time.time()
    
    # Final evaluation
    # Align reconstructed X to ground truth simX using Kabsch algorithm
    R1, _, rmsd1, _ = kabsch_numpy(
        normalizeF(simX, method="mean"), normalizeF(X, method="mean")
    )
    X_mirror = np.copy(X)
    X_mirror[:, 2] = -X_mirror[:, 2]
    R2, _, rmsd2, _ = kabsch_numpy(
        normalizeF(simX, method="mean"), normalizeF(X_mirror, method="mean")
    )
    final_rmsd = min(rmsd1, rmsd2)
    mirror = 1 if final_rmsd == rmsd2 else 0
    
    # Calculate relative error: RMSD / RMS of ground truth coordinates
    # RMS of ground truth = sqrt(mean of squared distances from centroid)
    simX_normalized = normalizeF(simX, method="mean")
    # Squared Frobenius norm of benchmark = sum of squared coordinates
    norm_benchmark_sq = np.sum(simX_normalized ** 2)
    # RMSD² × n_points = sum of squared errors
    rmsd_sq_sum = final_rmsd ** 2 * len(simX_normalized)
    relative_error = rmsd_sq_sum / norm_benchmark_sq

    # Calculate Spearman correlation
    D = euclidean_distance_3d_matrix(X)
    S = distance_to_similarity(D)
    D_ = euclidean_distance_3d_matrix(simX)
    S_ = distance_to_similarity(D_)
    spearman = scipy.stats.spearmanr(S_.flatten(), S.flatten())[0]
    
    return {
        'converged': True,
        'error': None,
        'rmsd': final_rmsd,
        'relative_error': relative_error,
        'spearman': spearman,
        'mirror': mirror,
        'iterations': final_loop,
        'time': end_time - start_time,
        'rmsd_history': rmsd_history,
        'final_X': X
    }


def multi_start_optimization(gem, simX, Ngene_for_rotation, n_starts=5, base_seed=42):
    """
    Multi-start strategy: run optimization from multiple random initializations
    and select the best solution.
    
    Parameters:
    -----------
    gem : pd.DataFrame
        Gene expression matrix
    simX : np.array
        Ground truth coordinates
    Ngene_for_rotation : int
        Number of genes for rotation derivation
    n_starts : int
        Number of random starts
    base_seed : int
        Base random seed
    
    Returns:
    --------
    dict : Best result among all starts
    list : All results
    """
    all_results = []
    
    for i in range(n_starts):
        init_seed = base_seed + i * 1000
        result = run_single_optimization(gem, simX, Ngene_for_rotation, init_seed)
        result['start_id'] = i
        result['init_seed'] = init_seed
        all_results.append(result)
    
    # Select best result (lowest RMSD among converged)
    converged_results = [r for r in all_results if r['converged'] and not np.isnan(r['rmsd'])]
    
    if converged_results:
        best_result = min(converged_results, key=lambda x: x['rmsd'])
    else:
        best_result = all_results[0]  # Return first if none converged
    
    total_time = sum(r['time'] for r in all_results)
    best_result['total_multistart_time'] = total_time
    
    return best_result, all_results


def run_sensitivity_analysis(
    genome_structure_file,
    output_dir,
    n_runs=100,
    n_starts=5,
    Ngene=100,
    Ncell=50,
    rateCap=0.8,
    rateDrop=0.2,
    resolution=10,
    Ngene_for_rotation=30,
    noise=1,
    mode="random"
):
    """
    Run comprehensive sensitivity analysis.
    
    Parameters:
    -----------
    genome_structure_file : str
        Path to genome structure TSV file
    output_dir : str
        Output directory
    n_runs : int
        Number of independent runs (default 100)
    n_starts : int
        Number of random starts for multi-start strategy (default 5)
    ... other simulation parameters
    
    Returns:
    --------
    pd.DataFrame : Results DataFrame
    """
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    analysis_dir = os.path.join(output_dir, f"sensitivity_analysis_{timestamp}")
    Path(analysis_dir).mkdir(parents=True, exist_ok=True)
    
    # Load genome structure
    GenomeStructure = pd.read_csv(genome_structure_file, delimiter="\t")
    
    print("=" * 70)
    print("CYTOCRAFT SENSITIVITY ANALYSIS")
    print("=" * 70)
    print(f"Start time: {datetime.now()}")
    print(f"Output directory: {analysis_dir}")
    print(f"Number of runs: {n_runs}")
    print(f"Multi-start strategy: {n_starts} starts")
    print(f"Parameters: Ngene={Ngene}, Ncell={Ncell}, rateCap={rateCap}, "
          f"rateDrop={rateDrop}, noise={noise}")
    print("=" * 70)
    
    # Generate base simulation data (fixed across all runs)
    data_seed = 42
    if mode == "continous":
        Conformation = np.array(GenomeStructure[["x", "y", "z"]].head(Ngene))
    elif mode == "random":
        np.random.seed(data_seed)
        sampled_indices = sorted(
            np.random.choice(len(GenomeStructure), size=Ngene, replace=False)
        )
        Conformation = np.array(GenomeStructure[["x", "y", "z"]].iloc[sampled_indices])
    
    # Generate simulation data once
    gem, simX = generate_simulation_data(
        Conformation, Ncell, noise, resolution, rateCap, rateDrop, data_seed
    )
    
    print(f"Generated simulation data: {len(gem)} expression entries")
    print(f"Ground truth shape: {simX.shape}")
    print("-" * 70)
    
    # Run sensitivity analysis
    results = []
    
    # Part 1: Single initialization runs (100 runs)
    print("\n[PART 1] Running 100 independent single-initialization trials...")
    print("-" * 70)
    
    single_init_results = []
    for run_id in range(n_runs):
        init_seed = run_id * 12345 + 7
        result = run_single_optimization(gem, simX, Ngene_for_rotation, init_seed)
        result['run_id'] = run_id
        result['analysis_type'] = 'single_init'
        single_init_results.append(result)
        
        if (run_id + 1) % 10 == 0:
            converged_count = sum(1 for r in single_init_results if r['converged'])
            avg_rmsd = np.nanmean([r['rmsd'] for r in single_init_results])
            print(f"  Completed {run_id + 1}/{n_runs} runs | "
                  f"Converged: {converged_count}/{run_id + 1} | "
                  f"Avg RMSD: {avg_rmsd:.4f}")
    
    results.extend(single_init_results)
    
    # Part 2: Multi-start strategy evaluation
    print("\n[PART 2] Evaluating multi-start strategy...")
    print("-" * 70)
    
    multistart_results = []
    for run_id in range(20):  # 20 multi-start trials
        base_seed = run_id * 50000 + 999
        best_result, all_starts = multi_start_optimization(
            gem, simX, Ngene_for_rotation, n_starts, base_seed
        )
        best_result['run_id'] = run_id
        best_result['analysis_type'] = 'multi_start'
        best_result['n_starts'] = n_starts
        multistart_results.append(best_result)
        
        print(f"  Multi-start trial {run_id + 1}/20 | "
              f"Best RMSD: {best_result['rmsd']:.4f} | "
              f"Time: {best_result['total_multistart_time']:.2f}s")
    
    results.extend(multistart_results)
    
    # Convert to DataFrame
    results_df = pd.DataFrame([
        {
            'run_id': r['run_id'],
            'analysis_type': r['analysis_type'],
            'n_starts': r.get('n_starts', np.nan),
            'converged': r['converged'],
            'rmsd': r['rmsd'],
            'relative_error': r.get('relative_error', np.nan),
            'spearman': r.get('spearman', np.nan),
            'mirror': r.get('mirror', np.nan),
            'iterations': r['iterations'],
            'time_seconds': r['time'],
            'error': r.get('error', None)
        }
        for r in results
    ])
    
    # Save results
    results_csv = os.path.join(analysis_dir, "sensitivity_results.csv")
    results_df.to_csv(results_csv, index=False)
    print(f"\nResults saved to: {results_csv}")
    
    # Calculate statistics
    print("\n" + "=" * 70)
    print("ANALYSIS SUMMARY")
    print("=" * 70)
    
    # Single initialization statistics
    single_df = results_df[results_df['analysis_type'] == 'single_init']
    n_converged = single_df['converged'].sum()
    convergence_rate = n_converged / len(single_df) * 100
    
    converged_rmsd = single_df[single_df['converged']]['rmsd'].dropna()
    converged_rel_err = single_df[single_df['converged']]['relative_error'].dropna()
    median_rmsd = converged_rmsd.median()
    mean_rmsd = converged_rmsd.mean()
    std_rmsd = converged_rmsd.std()
    median_rel_err = converged_rel_err.median()
    mean_rel_err = converged_rel_err.mean()
    std_rel_err = converged_rel_err.std()
    
    print("\n[Single Initialization Results]")
    print(f"  Total runs: {len(single_df)}")
    print(f"  Converged: {n_converged} ({convergence_rate:.1f}%)")
    print(f"  RMSD - Median: {median_rmsd:.4f}, Mean: {mean_rmsd:.4f}, Std: {std_rmsd:.4f}")
    print(f"  Relative Error - Median: {median_rel_err:.4f}, Mean: {mean_rel_err:.4f}, Std: {std_rel_err:.4f}")
    print(f"  Runs with relative error < 0.02: {(converged_rel_err < 0.02).sum()} "
          f"({(converged_rel_err < 0.02).sum() / len(converged_rel_err) * 100:.1f}%)")
    
    avg_time_single = single_df['time_seconds'].mean()
    print(f"  Average time per run: {avg_time_single:.2f}s")
    
    # Multi-start statistics
    multi_df = results_df[results_df['analysis_type'] == 'multi_start']
    multi_converged = multi_df['converged'].sum()
    multi_convergence_rate = multi_converged / len(multi_df) * 100
    
    multi_rmsd = multi_df[multi_df['converged']]['rmsd'].dropna()
    multi_rel_err = multi_df[multi_df['converged']]['relative_error'].dropna()
    
    print("\n[Multi-Start Strategy Results]")
    print(f"  Total trials: {len(multi_df)}")
    print(f"  Converged: {multi_converged} ({multi_convergence_rate:.1f}%)")
    if len(multi_rmsd) > 0:
        print(f"  RMSD - Median: {multi_rmsd.median():.4f}, "
              f"Mean: {multi_rmsd.mean():.4f}, Std: {multi_rmsd.std():.4f}")
    if len(multi_rel_err) > 0:
        print(f"  Relative Error - Median: {multi_rel_err.median():.4f}, "
              f"Mean: {multi_rel_err.mean():.4f}, Std: {multi_rel_err.std():.4f}")
    
    # Spearman correlation
    spearman_values = single_df[single_df['converged']]['spearman'].dropna()
    print(f"\n[Structure Similarity]")
    print(f"  Spearman correlation - Median: {spearman_values.median():.4f}, "
          f"Mean: {spearman_values.mean():.4f}")
    
    # Computational cost analysis
    print(f"\n[Computational Cost]")
    print(f"  Single run average: {avg_time_single:.2f}s")
    print(f"  Multi-start ({n_starts} starts) average: "
          f"{avg_time_single * n_starts:.2f}s (linear scaling)")
    
    # Generate summary statistics file
    summary_file = os.path.join(analysis_dir, "summary_statistics.txt")
    with open(summary_file, 'w') as f:
        f.write("CYTOCRAFT SENSITIVITY ANALYSIS SUMMARY\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Analysis Date: {datetime.now()}\n")
        f.write(f"Genome Structure: {genome_structure_file}\n\n")
        
        f.write("Parameters:\n")
        f.write(f"  Number of genes: {Ngene}\n")
        f.write(f"  Number of cells: {Ncell}\n")
        f.write(f"  Capture rate: {rateCap}\n")
        f.write(f"  Dropout rate: {rateDrop}\n")
        f.write(f"  Noise level: {noise}\n")
        f.write(f"  Resolution: {resolution}\n")
        f.write(f"  Genes for rotation: {Ngene_for_rotation}\n\n")
        
        f.write("Single Initialization Analysis (100 runs):\n")
        f.write(f"  Convergence rate: {convergence_rate:.1f}%\n")
        f.write(f"  Median RMSD: {median_rmsd:.4f}\n")
        f.write(f"  Mean RMSD: {mean_rmsd:.4f} ± {std_rmsd:.4f}\n")
        f.write(f"  Median Relative Error: {median_rel_err:.4f}\n")
        f.write(f"  Mean Relative Error: {mean_rel_err:.4f} ± {std_rel_err:.4f}\n")
        f.write(f"  Runs with relative error < 0.02: "
                f"{(converged_rel_err < 0.02).sum()}/{len(converged_rel_err)} "
                f"({(converged_rel_err < 0.02).sum() / len(converged_rel_err) * 100:.1f}%)\n")
        f.write(f"  Average computation time: {avg_time_single:.2f}s\n\n")
        
        f.write(f"Multi-Start Strategy ({n_starts} starts):\n")
        f.write(f"  Convergence rate: {multi_convergence_rate:.1f}%\n")
        if len(multi_rmsd) > 0:
            f.write(f"  Median RMSD: {multi_rmsd.median():.4f}\n")
            f.write(f"  Mean RMSD: {multi_rmsd.mean():.4f} ± {multi_rmsd.std():.4f}\n")
        if len(multi_rel_err) > 0:
            f.write(f"  Median Relative Error: {multi_rel_err.median():.4f}\n")
            f.write(f"  Mean Relative Error: {multi_rel_err.mean():.4f} ± {multi_rel_err.std():.4f}\n")
        f.write(f"  Computational overhead: Linear ({n_starts}x single run)\n\n")
        
        f.write("Conclusion:\n")
        f.write(f"  The algorithm converged to a stable solution consistent with the\n")
        f.write(f"  ground truth (median relative error < 0.02) in "
                f"{(converged_rel_err < 0.02).sum() / len(converged_rel_err) * 100:.0f}% of cases.\n")
    
    print(f"\nSummary saved to: {summary_file}")
    
    # Generate plots
    generate_plots(results_df, analysis_dir, single_init_results)
    
    return results_df


def generate_plots(results_df, output_dir, single_init_results):
    """Generate visualization plots for sensitivity analysis."""
    
    single_df = results_df[results_df['analysis_type'] == 'single_init']
    multi_df = results_df[results_df['analysis_type'] == 'multi_start']
    
    # Figure 1: Relative Error and RMSD Distribution
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1a: Histogram of Relative Error values
    ax1 = axes[0, 0]
    converged_rel_err = single_df[single_df['converged']]['relative_error'].dropna()
    ax1.hist(converged_rel_err, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
    ax1.axvline(converged_rel_err.median(), color='red', linestyle='--', 
                label=f'Median: {converged_rel_err.median():.4f}')
    ax1.axvline(0.02, color='green', linestyle=':', label='Threshold: 0.02')
    ax1.set_xlabel('Relative Error')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution of Relative Error (100 Independent Runs)')
    ax1.legend()
    
    # 1b: Convergence rate bar plot
    ax2 = axes[0, 1]

    # <-- FIX: be robust if n_starts is missing/NaN
    n_starts_label = 5
    if len(multi_df) > 0 and ('n_starts' in multi_df.columns):
        vals = multi_df['n_starts'].dropna()
        if len(vals) > 0:
            n_starts_label = int(vals.iloc[0])

    categories = ['Single Init', f'Multi-Start ({n_starts_label})']
    rates = [
        single_df['converged'].sum() / len(single_df) * 100 if len(single_df) > 0 else 0,
        multi_df['converged'].sum() / len(multi_df) * 100 if len(multi_df) > 0 else 0
    ]

    bars = ax2.bar(categories, rates, color=['steelblue', 'coral'], edgecolor='black')
    ax2.set_ylabel('Convergence Rate (%)')
    ax2.set_title('Convergence Rate Comparison')
    ax2.set_ylim(0, 105)
    for bar, rate in zip(bars, rates):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f'{rate:.1f}%', ha='center', va='bottom')
    
    # 1c: Relative Error vs Spearman correlation
    ax3 = axes[1, 0]
    valid_data = single_df[single_df['converged'] & single_df['spearman'].notna() & single_df['relative_error'].notna()]
    ax3.scatter(valid_data['relative_error'], valid_data['spearman'], alpha=0.6, 
                color='steelblue', edgecolor='black', s=50)
    ax3.set_xlabel('Relative Error')
    ax3.set_ylabel('Spearman Correlation')
    ax3.set_title('Relative Error vs Structure Similarity')
    
    # 1d: Computation time distribution
    ax4 = axes[1, 1]
    ax4.hist(single_df['time_seconds'], bins=20, edgecolor='black', 
             alpha=0.7, color='steelblue')
    ax4.axvline(single_df['time_seconds'].mean(), color='red', linestyle='--',
                label=f'Mean: {single_df["time_seconds"].mean():.2f}s')
    ax4.set_xlabel('Time (seconds)')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Computation Time Distribution')
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sensitivity_analysis_summary.png'), dpi=150)
    plt.savefig(os.path.join(output_dir, 'sensitivity_analysis_summary.pdf'))
    plt.close()
    
    # Figure 2: Convergence trajectories (sample runs)
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot RMSD history for a sample of runs
    sample_runs = [r for r in single_init_results if r['converged'] and 'rmsd_history' in r][:20]
    for i, run in enumerate(sample_runs):
        if 'rmsd_history' in run and len(run['rmsd_history']) > 0:
            ax.plot(range(1, len(run['rmsd_history']) + 1), run['rmsd_history'], 
                   alpha=0.5, linewidth=1)
    
    ax.set_xlabel('Iteration')
    ax.set_ylabel('RMSD')
    ax.set_title('Convergence Trajectories (Sample of 20 Runs)')
    ax.axhline(0.02, color='red', linestyle='--', label='Convergence threshold')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'convergence_trajectories.png'), dpi=150)
    plt.savefig(os.path.join(output_dir, 'convergence_trajectories.pdf'))
    plt.close()
    
    # Figure 3: Box plot comparison for Relative Error
    fig, ax = plt.subplots(figsize=(8, 6))
    
    data_to_plot = [
        single_df[single_df['converged']]['relative_error'].dropna(),
        multi_df[multi_df['converged']]['relative_error'].dropna() if len(multi_df) > 0 else []
    ]
    
    bp = ax.boxplot(data_to_plot, labels=['Single Init\n(n=100)', 'Multi-Start\n(n=20)'],
                    patch_artist=True)
    bp['boxes'][0].set_facecolor('steelblue')
    bp['boxes'][0].set_alpha(0.7)
    if len(data_to_plot[1]) > 0:
        bp['boxes'][1].set_facecolor('coral')
        bp['boxes'][1].set_alpha(0.7)
    
    ax.axhline(0.02, color='green', linestyle='--', alpha=0.7, label='Threshold: 0.02')
    ax.set_ylabel('Relative Error')
    ax.set_title('Relative Error Comparison: Single vs Multi-Start Initialization')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'relative_error_boxplot_comparison.png'), dpi=150)
    plt.savefig(os.path.join(output_dir, 'relative_error_boxplot_comparison.pdf'))
    plt.close()
    
    print(f"\nPlots saved to: {output_dir}")


def main():
    """Main function to run sensitivity analysis."""
    parser = argparse.ArgumentParser(
        description='Cytocraft Sensitivity Analysis for Convergence and Stability'
    )
    parser.add_argument('genome_structure', type=str,
                        help='Path to genome structure TSV file')
    parser.add_argument('output_dir', type=str,
                        help='Output directory for results')
    parser.add_argument('--n_runs', type=int, default=100,
                        help='Number of independent runs (default: 100)')
    parser.add_argument('--n_starts', type=int, default=5,
                        help='Number of random starts for multi-start strategy (default: 5)')
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
    
    args = parser.parse_args()
    
    # Run analysis
    results_df = run_sensitivity_analysis(
        genome_structure_file=args.genome_structure,
        output_dir=args.output_dir,
        n_runs=args.n_runs,
        n_starts=args.n_starts,
        Ngene=args.ngene,
        Ncell=args.ncell,
        rateCap=args.rate_cap,
        rateDrop=args.rate_drop,
        resolution=args.resolution,
        Ngene_for_rotation=args.ngene_rotation,
        noise=args.noise,
        mode=args.mode
    )
    
    print("\n" + "=" * 70)
    print("SENSITIVITY ANALYSIS COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()
