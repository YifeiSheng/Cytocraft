import sys, os, random, csv
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.manifold import Isomap
from cytocraft.craft import generate_id, get_centers, normalizeZ, kabsch_numpy, normalizeF
from cytocraft.simulation import (
    scale_X,
    centerX,
    euclidean_distance_3d_matrix,
    distance_to_similarity,
    write_sim_pdb,
)

def write_row_to_csv(filename, row):
    with open(filename, "a", newline="") as f:
        csv.writer(f).writerow(row)

def generate_random_rotation_matrices(n):
    R = np.zeros((n,3,3))
    for i in range(n):
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
    xx, yy = np.meshgrid(x,y); xy = np.stack((xx,yy), axis=-1)
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
    rows = []
    for (c,g,i,j), val in np.ndenumerate(CapMatrix):
        if val>0:
            rows.append([f"gene{g}", str(i+LBx), str(j+LBy), str(val), str(val), str(c)])
    gem = pd.DataFrame(rows, columns=["geneID","x","y","MIDCount","ExonCount","CellID"])
    Xgenes = pd.Series(sorted(gem.geneID.drop_duplicates(), key=lambda s:int(s[4:])))
    W = get_centers(gem, gem.CellID.drop_duplicates().values, Xgenes)
    W = normalizeZ(W)
    return gem, W, Xgenes, simW

def safe_normalize(X):
    Xc = X - np.nanmean(X, axis=0, keepdims=True)
    r = np.linalg.norm(Xc, axis=1)
    scale = np.nanmean(r)
    if not np.isfinite(scale) or scale == 0:
        scale = np.nanstd(Xc)
    if not np.isfinite(scale) or scale == 0:
        scale = 1.0
    Xn = Xc / scale
    Xn[~np.isfinite(Xn)] = 0.0
    return Xn

def reconstruct_isomap(W, n_neighbors):
    # genes as samples, features are concatenated per-cell (x,y)
    F = W.T  # (Ngene, 2*Ncell)
    # Impute NaNs column-wise
    col_means = np.nanmean(F, axis=0)
    # Replace NaN means (all-missing columns) with 0
    col_means[~np.isfinite(col_means)] = 0.0
    inds = np.where(~np.isfinite(F))
    F[inds] = np.take(col_means, inds[1])
    # Guard neighbors
    Ng = F.shape[0]
    k = max(2, min(n_neighbors, Ng - 1))
    iso = Isomap(n_neighbors=k, n_components=3)
    Xrec = iso.fit_transform(F)
    return Xrec

def evaluate(simX, Xrec):
    A = safe_normalize(simX)
    B = safe_normalize(Xrec)
    rmsd1 = kabsch_numpy(A, B)[2]
    Xm = Xrec.copy(); Xm[:,2] *= -1
    rmsd2 = kabsch_numpy(A, safe_normalize(Xm))[2]
    minrmsd = min(rmsd1, rmsd2)
    mirror = 0 if minrmsd == rmsd1 else 1
    D = euclidean_distance_3d_matrix(Xrec); S = distance_to_similarity(D)
    Dg = euclidean_distance_3d_matrix(simX); Sg = distance_to_similarity(Dg)
    sp = spearmanr(Sg.flatten(), S.flatten(), nan_policy="omit")[0]
    if not np.isfinite(sp):
        sp = 0.0
    return minrmsd, mirror, sp

def main():
    if len(sys.argv) not in (12,13,14):
        print("Usage: genome.tsv Ngene Ncell rateCap rateDrop resolution Ngene_for_rotation noise mode outdir csv [seed] [n_neighbors]")
        sys.exit(1)
    GenomeStructure = pd.read_csv(sys.argv[1], sep="\t")
    sample_name = os.path.basename(sys.argv[1]).split(".")[0]
    Ngene = int(sys.argv[2]); Ncell = int(sys.argv[3])
    rateCap = float(sys.argv[4]); rateDrop = float(sys.argv[5])
    resolution = int(sys.argv[6]); Ngene_rot = int(sys.argv[7])
    noise = int(sys.argv[8]); mode = sys.argv[9].strip()
    outpath_root = sys.argv[10]; csv_path = sys.argv[11]
    # Optional args: seed then n_neighbors
    if len(sys.argv) >= 13:
        seed = int(sys.argv[12])
    else:
        seed = random.randint(0,1_000_000)
    if len(sys.argv) == 14:
        n_neighbors = int(sys.argv[13])
    else:
        n_neighbors = 10
    random.seed(seed); np.random.seed(seed)
    if mode == "continous":
        Conformation = GenomeStructure[["x","y","z"]].head(Ngene).to_numpy()
    elif mode == "random":
        idx = sorted(np.random.choice(len(GenomeStructure), size=Ngene, replace=False))
        Conformation = GenomeStructure.iloc[idx][["x","y","z"]].to_numpy()
    else:
        print("mode must be continous|random"); sys.exit(1)
    TID = generate_id()
    
    outdir = f"{outpath_root}/{sample_name}_NG{Ngene}_NC{Ncell}_RC{rateCap}_RD{rateDrop}_RS{resolution}_NGF{Ngene_rot}_NS{noise}_isomap"
    Path(outdir).mkdir(parents=True, exist_ok=True)
    
    # start logging
    stdout = sys.stdout
    log_file = open(outdir + "/" + TID + ".log", "w")
    sys.stdout = log_file

    print(f"The seed is {seed}.")
    Scaled,_ = scale_X(Conformation, resolution)
    simX,_ = centerX(Scaled)
    randRM = generate_random_rotation_matrices(Ncell)
    gem, W, Xgenes, _ = build_expression_and_W(simX, randRM, Ncell, Ngene, rateCap, rateDrop, noise)
    # Subset simX to detected genes
    detected_idx = [int(g[4:]) for g in Xgenes]
    simX_detected = simX[detected_idx, :]
    # Replace non-finite in W with NaN (imputed later)
    W[~np.isfinite(W)] = np.nan
    Xrec = reconstruct_isomap(W, n_neighbors)
    # Align lengths (safeguard)
    if Xrec.shape[0] != simX_detected.shape[0]:
        m = min(Xrec.shape[0], simX_detected.shape[0])
        Xrec = Xrec[:m]
        simX_detected = simX_detected[:m]
    write_sim_pdb(scale_X(Xrec,0.5)[0], prefix=TID+f"_isomap_initial_k{n_neighbors}", outpath=outdir)
    minrmsd, mirror, sp = evaluate(simX_detected, Xrec)
    print("Baseline Isomap Reconstruction")
    print("Seed:", seed, "n_neighbors:", n_neighbors)
    print("RMSD Distance\tMirror\tSpearman Correlation Coefficient")
    print(f"{minrmsd}\t{mirror}\t{sp}")

    # stop logging
    sys.stdout = stdout
    log_file.close()

    row = (TID,"HMEC",str(Ngene),str(Ncell),str(rateCap),str(rateDrop),str(resolution),
           str(Ngene_rot),str(noise),mode,str(minrmsd),str(mirror),str(sp),f"isomap_k{n_neighbors}")
    if csv_path.lower() != "none":
        write_row_to_csv(csv_path, row)

if __name__ == "__main__":
    main()
