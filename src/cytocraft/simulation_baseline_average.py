import sys, os, random, csv
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
from cytocraft.craft import generate_id, get_centers, normalizeZ, kabsch_numpy, normalizeF
from cytocraft.simulation import (
    scale_X,
    centerX,
    euclidean_distance_3d_matrix,
    distance_to_similarity,
    write_sim_pdb,
)
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")

def write_row_to_csv(filename, row):
    with open(filename, "a", newline="") as f:
        csv.writer(f).writerow(row)

def generate_random_rotation_matrices(n):
    # minimal local version (kept here to avoid importing heavy deps if reorganized later)
    R = np.zeros((n,3,3))
    for i in range(n):
        # random orthonormal via QR
        A = np.random.randn(3,3)
        Q,_ = np.linalg.qr(A)
        if np.linalg.det(Q) < 0:
            Q[:,0] *= -1
        R[i] = Q
    return R

def build_expression_and_W(simX, randRM, Ncell, Ngene, rateCap, rateDrop, noise):
    simW = np.zeros((Ncell * 2, Ngene))
    for c in range(Ncell):
        XY = (simX @ randRM[c])[:, :2]
        simW[2*c] = XY[:,0]
        simW[2*c+1] = XY[:,1]
    simW = np.random.normal(simW, noise)
    UBx = int(np.max(simW[::2], axis=1).max()); LBx = int(np.min(simW[::2], axis=1).min())
    UBy = int(np.max(simW[1::2], axis=1).max()); LBy = int(np.min(simW[1::2], axis=1).min())
    Matrix = np.empty((Ncell, Ngene, UBx-LBx, UBy-LBy), dtype="i1")
    x = np.arange(LBx, UBx); y = np.arange(LBy, UBy)
    xx, yy = np.meshgrid(x, y); xy = np.stack((xx, yy), axis=-1)
    # shared covariance per cell for speed
    if noise == 0:
        sigma = np.eye(2)*8
    else:
        A = np.random.rand(2,2); sigma = (A @ A.T)*8
    from scipy.stats import multivariate_normal
    for c in range(Ncell):
        for g in range(Ngene):
            loc = simW[2*c:2*c+2, g]
            exp = multivariate_normal.pdf(xy, mean=loc, cov=sigma, allow_singular=True)*128
            Matrix[c,g] = exp.T.round().clip(0,127).astype("i1")
    CapMatrix = np.random.binomial(Matrix, rateCap)
    mask = np.random.random((Ncell, Ngene)) < rateDrop
    CapMatrix[mask] = 0
    # build gem
    rows = []
    for (c,g,i,j), val in np.ndenumerate(CapMatrix):
        if val>0:
            rows.append([f"gene{g}", str(i+LBx), str(j+LBy), str(val), str(val), str(c)])
    gem = pd.DataFrame(rows, columns=["geneID","x","y","MIDCount","ExonCount","CellID"])
    Xgenes = pd.Series(sorted(gem.geneID.drop_duplicates(), key=lambda s:int(s[4:])))
    W = get_centers(gem, gem.CellID.drop_duplicates().values, Xgenes)
    W = normalizeZ(W)
    return gem, W, Xgenes, simW

def reconstruct_average(W):
    # W shape (2*Ncell, Ngene); may contain NaNs for missing gene detections
    xs = W[::2]      # (Ncell, Ngene)
    ys = W[1::2]
    mean_x = np.nanmean(xs, axis=0)
    mean_y = np.nanmean(ys, axis=0)
    # identify genes missing in every cell (all NaN)
    nan_cols = np.isnan(mean_x) | np.isnan(mean_y)
    if np.any(nan_cols):
        overall_x = np.nanmean(mean_x) if np.any(~np.isnan(mean_x)) else 0.0
        overall_y = np.nanmean(mean_y) if np.any(~np.isnan(mean_y)) else 0.0
        mean_x[nan_cols] = overall_x
        mean_y[nan_cols] = overall_y
    Xrec = np.column_stack([mean_x, mean_y, np.zeros_like(mean_x)])
    Xrec[~np.isfinite(Xrec)] = 0.0
    return Xrec

def safe_normalize(X):
    # Center
    Xc = X - np.nanmean(X, axis=0, keepdims=True)
    # Scale by mean radial distance (fallbacks if degenerate)
    r = np.linalg.norm(Xc, axis=1)
    scale = np.nanmean(r)
    if not np.isfinite(scale) or scale == 0:
        # fallback to std of flattened coords
        scale = np.nanstd(Xc)
    if not np.isfinite(scale) or scale == 0:
        scale = 1.0
    Xn = Xc / scale
    Xn[~np.isfinite(Xn)] = 0.0
    return Xn

def evaluate(simX, Xrec):
    # Use robust normalization to avoid NaNs in kabsch
    A = safe_normalize(simX)
    B = safe_normalize(Xrec)
    rmsd1 = kabsch_numpy(A, B)[2]
    Xmir = Xrec.copy()
    Xmir[:, 2] *= -1
    Bm = safe_normalize(Xmir)
    rmsd2 = kabsch_numpy(A, Bm)[2]
    minrmsd = min(rmsd1, rmsd2)
    mirror = 0 if minrmsd == rmsd1 else 1
    D = euclidean_distance_3d_matrix(Xrec); S = distance_to_similarity(D)
    Dg = euclidean_distance_3d_matrix(simX); Sg = distance_to_similarity(Dg)
    sp = spearmanr(Sg.flatten(), S.flatten(), nan_policy="omit")[0]
    if not np.isfinite(sp):
        sp = 0.0
    return minrmsd, mirror, sp

def main():
    if len(sys.argv) not in (12,13):
        print("Usage (same as cytocraft simulation): genome.tsv Ngene Ncell rateCap rateDrop resolution Ngene_for_rotation noise mode outdir csv [seed]")
        sys.exit(1)
    GenomeStructure = pd.read_csv(sys.argv[1], sep="\t")
    sample_name = os.path.basename(sys.argv[1]).split(".")[0]
    Ngene = int(sys.argv[2]); Ncell = int(sys.argv[3])
    rateCap = float(sys.argv[4]); rateDrop = float(sys.argv[5])
    resolution = int(sys.argv[6]); Ngene_rot = int(sys.argv[7])  # unused here, kept for interface consistency
    noise = int(sys.argv[8]); mode = sys.argv[9].strip()
    outpath_root = sys.argv[10]; csv_path = sys.argv[11]
    if len(sys.argv) == 13:
        seed = int(sys.argv[12])
    else:
        seed = random.randint(0,1_000_000)
    random.seed(seed); np.random.seed(seed)
    if mode == "continous":
        Conformation = GenomeStructure[["x","y","z"]].head(Ngene).to_numpy()
    elif mode == "random":
        idx = sorted(np.random.choice(len(GenomeStructure), size=Ngene, replace=False))
        Conformation = GenomeStructure.iloc[idx][["x","y","z"]].to_numpy()
    else:
        print("mode must be continous|random"); sys.exit(1)
    TID = generate_id()

    outdir = f"{outpath_root}/{sample_name}_NG{Ngene}_NC{Ncell}_RC{rateCap}_RD{rateDrop}_RS{resolution}_NGF{Ngene_rot}_NS{noise}_avg"
    Path(outdir).mkdir(parents=True, exist_ok=True)
    # start logging
    stdout = sys.stdout
    log_file = open(outdir + "/" + TID + ".log", "w")
    sys.stdout = log_file
    print(f"The seed is {seed}.")

    Scaled, _ = scale_X(Conformation, resolution)
    simX, _ = centerX(Scaled)
    randRM = generate_random_rotation_matrices(Ncell)
    gem, W, Xgenes, _ = build_expression_and_W(simX, randRM, Ncell, Ngene, rateCap, rateDrop, noise)
    # Subset ground truth to detected genes only (some genes may be fully dropped)
    detected_idx = [int(g[4:]) for g in Xgenes]
    simX_detected = simX[detected_idx, :]
    # safety: replace inf with NaN before averaging
    W[~np.isfinite(W)] = np.nan
    Xrec = reconstruct_average(W)
    # Ensure lengths match
    if Xrec.shape[0] != simX_detected.shape[0]:
        # simple alignment safeguard: truncate to min length
        m = min(Xrec.shape[0], simX_detected.shape[0])
        Xrec = Xrec[:m]
        simX_detected = simX_detected[:m]
    write_sim_pdb(scale_X(Xrec, 0.5)[0], prefix=TID+"_avg_initial", outpath=outdir)
    minrmsd, mirror, sp = evaluate(simX_detected, Xrec)
    print("Baseline Average Reconstruction")
    print("Seed:", seed)
    print("RMSD Distance\tMirror\tSpearman Correlation Coefficient")
    print(f"{minrmsd}\t{mirror}\t{sp}")

    # stop logging
    sys.stdout = stdout
    log_file.close()

    row = (TID,"HMEC",str(Ngene),str(Ncell),str(rateCap),str(rateDrop),str(resolution),
           str(Ngene_rot),str(noise),mode,str(minrmsd),str(mirror),str(sp),"average")
    if csv_path.lower() != "none":
        write_row_to_csv(csv_path, row)

if __name__ == "__main__":
    main()
