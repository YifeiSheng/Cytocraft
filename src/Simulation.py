import os, sys, string, random, pickle, copy, csv
from pathlib import Path
import numpy as np
import pandas as pd
import scanpy as sc
from tqdm import tqdm
from math import cos, sin
from matplotlib import pyplot as plt
import scipy.stats
from scipy.sparse import spmatrix, issparse, csr_matrix
from scipy import linalg as LA
from scipy.stats import multivariate_normal
from scipy.spatial.transform import Rotation as R
import util
from model import BasisShapeModel
from anndata import AnnData
from typing import Optional, Union
from shapely.geometry import Point, MultiPoint
from numpy.linalg import svd, solve, lstsq
from stereopy import *
from rigid import *


def genedistribution(gem, CellIDs, TopGenes):
    W = np.zeros((len(CellIDs) * 2, len(TopGenes)))
    i = 0
    for c in CellIDs:
        j = 0
        # subset the gem in terms of individual cells
        gem_cell = gem[gem.CellID == c]
        for n in TopGenes:
            if n not in gem_cell.geneID.values:
                x_median = np.nan
                y_median = np.nan
            else:
                gem_cell_gene = gem_cell[gem_cell.geneID == n]
                x_median = np.average(
                    gem_cell_gene.x.values, weights=gem_cell_gene.MIDCount.values
                )
                y_median = np.average(
                    gem_cell_gene.y.values, weights=gem_cell_gene.MIDCount.values
                )
            W[i * 2, j] = x_median
            W[i * 2 + 1, j] = y_median
            j += 1
        i += 1
    return W


def MASK(gem, GeneIDs, Ngene):
    CellUIDs = gem.CellID.drop_duplicates()
    mask = np.zeros((CellUIDs.size, GeneIDs.size))
    i = 0
    for c in CellUIDs:
        gem_cell = gem[gem.CellID == c]
        GeneUIDs_cell = (
            gem_cell.groupby(["geneID"])["MIDCount"]
            .count()
            .reset_index(name="Count")
            .sort_values(["Count"], ascending=False)
            .geneID[:Ngene]
        )

        mask[i] = np.isin(GeneIDs, GeneUIDs_cell)
        i += 1
    return np.bool_(mask)


def DeriveRotation_v3(W, X, Mask):
    F = int(W.shape[0] / 2)
    Rotation = np.zeros((F, 3, 3))
    # I = np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]])
    for i in range(F):
        # filter genes
        Wi = W[:, Mask[i, :]]
        # filter cells
        Wi_filter = Wi[~np.isnan(Wi).any(axis=1), :]
        while Wi_filter.shape[0] < 6:
            Mask[i, :] = change_last_true(Mask[i, :])
            Wi = W[:, Mask[i, :]]
            # filter cells
            Wi_filter = Wi[~np.isnan(Wi).any(axis=1), :]
        idx = int(find_subarray(Wi_filter, Wi[i * 2]) / 2)
        Xi = X[Mask[i, :], :]
        model = factor(Wi_filter)
        _, R, is_reflection = numpy_svd_rmsd_rot(
            np.dot(model.Rs[idx], model.Ss[0]).T, Xi
        )
        # Ss_mirror = np.copy(model.Ss[0].T)
        # Ss_mirror[:, 2] = np.flip(Ss_mirror[:, 2], axis=0)
        # rmsd2, R2 = numpy_svd_rmsd_rot(Xi, Ss_mirror)
        # if rmsd1 > rmsd2:
        #    R = R2
        # else:
        #    R = R1
        # print("R")
        # print(R1)
        # print(R2)
        # print(rmsd1)
        # print(rmsd2)
        Rotation[i] = R
        # print(i)
        # print(is_reflection)
        # if is_reflection:
        #     Rotation[i] = np.dot(np.dot(I, Rotation[i]), I)
    return Rotation


def change_last_true(arr):
    # Find the index of the last True value
    last_true_index = np.where(arr == True)[0][-1]

    # Set the value at that index to False
    arr[last_true_index] = False

    return arr


def find_subarray(arr1, arr2):
    n = arr1.shape[0]
    for i in range(n):
        if np.array_equal(arr1[i], arr2):
            return i
    return print("Error! Try to reduce Ngene in MASK function")


def UpdateX(RM, W):
    F = int(W.shape[0] / 2)
    for j in range(W.shape[1]):
        a1 = b1 = c1 = d1 = a2 = b2 = c2 = d2 = a3 = b3 = c3 = d3 = 0
        for i in range(F):
            if not (
                np.isnan(W[i * 2 : i * 2 + 2, j]).any() or np.isnan(RM[i, :, :]).any()
            ):
                a1 += RM[i, 0, 0] * RM[i, 0, 0] + RM[i, 0, 1] * RM[i, 0, 1]
                b1 += RM[i, 0, 0] * RM[i, 1, 0] + RM[i, 0, 1] * RM[i, 1, 1]
                c1 += RM[i, 0, 0] * RM[i, 2, 0] + RM[i, 0, 1] * RM[i, 2, 1]

                a2 += RM[i, 1, 0] * RM[i, 0, 0] + RM[i, 1, 1] * RM[i, 0, 1]
                b2 += RM[i, 1, 0] * RM[i, 1, 0] + RM[i, 1, 1] * RM[i, 1, 1]
                c2 += RM[i, 1, 0] * RM[i, 2, 0] + RM[i, 1, 1] * RM[i, 2, 1]

                a3 += RM[i, 2, 0] * RM[i, 0, 0] + RM[i, 2, 1] * RM[i, 0, 1]
                b3 += RM[i, 2, 0] * RM[i, 1, 0] + RM[i, 2, 1] * RM[i, 1, 1]
                c3 += RM[i, 2, 0] * RM[i, 2, 0] + RM[i, 2, 1] * RM[i, 2, 1]

                d1 += RM[i, 0, 0] * W[i * 2, j] + RM[i, 0, 1] * W[i * 2 + 1, j]
                d2 += RM[i, 1, 0] * W[i * 2, j] + RM[i, 1, 1] * W[i * 2 + 1, j]
                d3 += RM[i, 2, 0] * W[i * 2, j] + RM[i, 2, 1] * W[i * 2 + 1, j]
            else:
                # print("skip cell" + str(i) + "for gene" + str(j))
                pass

        args = np.array([[a1, b1, c1], [a2, b2, c2], [a3, b3, c3]])
        results = np.array([d1, d2, d3])
        newXi = LA.solve(args, results)
        # print(j)
        # print(args)
        # print(results)
        # print(newXi)
        try:
            newX = np.append(
                newX,
                [newXi],
                axis=0,
            )
        except NameError:
            newX = np.array([newXi])

    return newX

def numpy_svd_rmsd_rot(in_crds1, in_crds2):
    """
    Returns rmsd and optional rotation between 2 sets of [nx3] arrays.

    This requires numpy for svd decomposition.
    The transform direction: transform(m, ref_crd) => target_crd.
    """

    crds1 = np.array(in_crds1)
    crds2 = np.array(in_crds2)
    assert crds1.shape[1] == 3
    assert crds1.shape == crds2.shape

    n_vec = np.shape(crds1)[0]
    correlation_matrix = np.dot(np.transpose(crds1), crds2)
    v, s, w = np.linalg.svd(correlation_matrix)
    is_reflection = (np.linalg.det(v) * np.linalg.det(w)) < 0.0

    if is_reflection:
        s[-1] = -s[-1]
    E0 = sum(sum(crds1 * crds1)) + sum(sum(crds2 * crds2))
    rmsd_sq = (E0 - 2.0 * sum(s)) / float(n_vec)
    rmsd_sq = max([rmsd_sq, 0.0])
    rmsd = np.sqrt(rmsd_sq)

    if is_reflection:
        v[-1, :] = -v[-1, :]
    rot33 = np.dot(v, w).transpose()
    # print(is_reflection)
    return rmsd, rot33, is_reflection


def normalizeW(W):
    result = np.empty_like(W)
    for i in range(int(W.shape[0])):
        result[i] = W[i] - np.nanmean(W[i])
    return result


def write_sim_pdb(simX, prefix="simchain", outpath="./Results/"):
    show_shape = simX.T
    with open(outpath + "/" + prefix + ".pdb", "w") as out:
        out.write(
            "HEADER".ljust(10, " ")
            + "CHROMOSOMES".ljust(40, " ")
            + "21-FEB-23"
            + " "
            + "CMB1".ljust(4, " ")
            + "\n"
        )
        out.write(
            "TITLE".ljust(10, " ") + "CHROMOSOMES OF MICE BRAIN INFERED FROM sST DATA\n"
        )
        i = 1
        j = 1
        chain = 1
        Is = []
        atom_lines = []
        for idx in range(show_shape.shape[1]):
            atom_lines.append(
                "ATOM".ljust(6, " ")
                + str(i).rjust(5, " ")
                + " "
                + "O".ljust(3, " ")
                + " "
                + "GLY".ljust(3, " ")
                + " "
                + chr(64 + chain)
                + str(j).rjust(4, " ")
                + " "
                + str("%.3f" % (show_shape[0, idx])).rjust(8, " ")
                + str("%.3f" % (show_shape[1, idx])).rjust(8, " ")
                + str("%.3f" % (show_shape[2, idx])).rjust(8, " ")
                + "2.00".rjust(6, " ")
                + "10.00".rjust(6, " ")
                + " "
                + "O".rjust(2, " ")
                + "\n"
            )
            Is.append(i)
            i += 1
            j += 1
        atom_lines.append(
            "TER".ljust(6, " ")
            + str(i).rjust(5, " ")
            + " "
            + "".ljust(3, " ")
            + " "
            + "GLY".ljust(3, " ")
            + " "
            + chr(64 + chain)
            + str(j - 1).rjust(4, " ")
            + "\n"
        )
        chain = 1
        connect_lines = []
        for l in Is:
            if l - 1 in Is and l + 1 in Is:
                connect_lines.append(
                    "CONECT".ljust(6, " ")
                    + str(l).rjust(5, " ")
                    + str(l - 1).rjust(5, " ")
                    + str(l + 1).rjust(5, " ")
                    + "\n"
                )
            elif not l - 1 in Is:
                connect_lines.append(
                    "CONECT".ljust(6, " ")
                    + str(l).rjust(5, " ")
                    + str(l + 1).rjust(5, " ")
                    + "\n"
                )
            elif not l + 1 in Is:
                connect_lines.append(
                    "CONECT".ljust(6, " ")
                    + str(l).rjust(5, " ")
                    + str(l - 1).rjust(5, " ")
                    + "\n"
                )
        out.writelines(atom_lines)
        out.writelines(connect_lines)
        out.write("END\n")


def load_data(file):
    with open(file, "rb") as f:
        x = pickle.load(f)
    return x


def write_row_to_csv(filename, row):
    # open the file in append mode
    with open(filename, "a", newline="") as f:
        # create the csv writer
        writer = csv.writer(f)
        # write a row to the csv file
        writer.writerow(row)


def scale_X(X, size=1):
    # calculate the minimum and maximum values along each axis
    min_x, min_y, min_z = X.min(axis=0)
    max_x, max_y, max_z = X.max(axis=0)
    # calculate the range of values along each axis
    range_x = max_x - min_x
    range_y = max_y - min_y
    range_z = max_z - min_z
    # calculate the scale factor to fit the coordinates within the size
    scale = size / max(range_x, range_y, range_z)
    # normalize the coordinates by subtracting the minimum values and multiplying by the scale factor
    scaled_X = (X - [min_x, min_y, min_z]) * scale
    # return the scaled coordinates
    return scaled_X, scale


def euclidean_distance_3d_matrix(X):
    import math

    # coords is a list of tuples of the form (x, y, z)
    # returns a numpy array of shape (len(coords), len(coords)) where the element at (i, j) is the distance between coords[i] and coords[j]
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
    # matrix is a numpy array of shape (n, n) where the element at (i, j) is the distance between two points
    # returns a numpy array of shape (n, n) where the element at (i, j) is the similarity between two points
    # similarity is defined as 1 / (1 + distance)
    similarity = 1 / (1 + matrix)
    return similarity


def save_data(data, file):
    with open(file, "wb") as f:
        pickle.dump(data, f)


##### Functions
def normalizeX(X):
    """
    X = (nGene,3)
    """
    if len(X.shape) == 2 and X.shape[1] == 3:
        mean_x, mean_y, mean_z = X.mean(axis=0)
        centered_X = X - [mean_x, mean_y, mean_z]
        return centered_X, [mean_x, mean_y, mean_z]
    else:
        print("X shape error!")


def generate_random_rotation_matrices(n):
    # use the Rotation.random method to generate n random rotations
    rotations = R.random(n)
    # convert the rotations to matrices
    matrices = rotations.as_matrix()
    # return the matrices
    return matrices


def save_data(data, file):
    with open(file, "wb") as f:
        pickle.dump(data, f)


def generate_id():
    return "".join(random.choices(string.ascii_uppercase + string.digits, k=4))


def main():
    ##### SETTINGS
    seed = int(sys.argv[11])
    random.seed(seed)
    np.random.seed(seed)
    # HicCoords = np.loadtxt(sys.argv[1], delimiter="\t")
    GenomeStructure = pd.read_csv(sys.argv[1], delimiter="\t")
    sample_name = os.path.basename(sys.argv[1])
    Ngene = int(sys.argv[2])
    mode = sys.argv[12]
    if mode == "head":
        HicCoords = np.array(GenomeStructure[["x", "y", "z"]].head(Ngene))
    elif mode == "random":
        HicCoords = np.array(GenomeStructure[["x", "y", "z"]].sample(Ngene))
    Ncell = int(sys.argv[3])  ### simulated cell number
    rateCap = float(sys.argv[4])  ### set capture rate
    rateDrop = float(sys.argv[5])  ### set gene loss rate
    resolution = int(sys.argv[6])  ### difine the resolution
    Ngene_for_rotation_derivation = int(sys.argv[7])
    noise = int(sys.argv[8])
    outpath = sys.argv[9]
    # csv = "../Results/HEMC/SimulationResults.csv"
    csv = sys.argv[10]
    TID = generate_id()
    outpath = outpath + "/" + TID
    Path(outpath).mkdir(parents=True, exist_ok=True)

    # start logging
    stdout = sys.stdout
    log_file = open(outpath + "/" + TID + ".log", "w")
    sys.stdout = log_file
    print("seed: " + str(seed))

    ##### START RUNNING
    ScaledCoords, _ = scale_X(HicCoords, resolution)
    simX, _ = normalizeX(ScaledCoords)
    save_data(simX, outpath + "/HEMC_simX_resolution" + str(resolution) + ".pkl")

    #### generate random RM
    randRM = generate_random_rotation_matrices(Ncell)
    save_data(
        randRM,
        outpath
        + "/HEMC_randRM_nCell"
        + str(Ncell)
        + "_rateCap"
        + str(rateCap)
        + "_rateDrop"
        + str(rateDrop)
        + "_resolution"
        + str(resolution)
        + ".pkl",
    )

    #### generate W
    simW = np.zeros((Ncell * 2, Ngene))
    for c in range(randRM.shape[0]):
        XY = np.dot(simX, randRM[c, :, :])[:, 0:2]
        # XY = np.dot(randRM[c, :, :], simX.T).T[:, 0:2]
        simW[c * 2] = XY[:, 0]
        simW[c * 2 + 1] = XY[:, 1]

    save_data(simW, outpath + "/HEMC_simW_" + str(Ncell) + ".pkl")

    #### add noise
    simW = np.random.normal(simW, noise)
    print("noise: " + str(noise))

    #### expression matrix
    UBx = int(np.max(simW[::2], axis=1).max())
    LBx = int(np.min(simW[::2], axis=1).min())
    UBy = int(np.max(simW[1::2], axis=1).max())
    LBy = int(np.min(simW[1::2], axis=1).min())

    Matrix = np.empty((Ncell, Ngene, UBx - LBx, UBy - LBy), dtype="i1")
    x = np.arange(LBx, UBx)  # create a 1D array of x values
    y = np.arange(LBy, UBy)  # create a 1D array of y values
    xx, yy = np.meshgrid(x, y)  # create a 2D grid of x and y values
    xy = np.stack((xx, yy), axis=-1)  # create a 3D array of (x,y) pairs
    for c in tqdm(range(Ncell), desc="generate expression matrix for cell"):
        for g in range(Ngene):
            loc = simW[c * 2 : (c + 1) * 2, g]
            sigma = np.eye(2) * 8  # create a 2x2 identity matrix and multiply by 8
            exp = (
                multivariate_normal.pdf(xy, mean=loc, cov=sigma) * 128
            )  # compute the multivariate normal pdf on the grid
            Matrix[c, g] = exp.T.round().astype("i1")  # round and cast to integer

    #### drop counts based on capture rate
    CapMatrix = np.random.binomial(
        Matrix, rateCap
    )  # generate random numbers from a binomial distribution with Matrix as the number of trials and rateCap as the probability

    mask = (
        np.random.random((Ncell, Ngene)) < rateDrop
    )  # generate a boolean mask with rateDrop as the probability
    CapMatrix[
        mask
    ] = 0  # set the elements of CapMatrix that correspond to True in the mask to zero

    # ### write gem
    gem_write = (
        sample_name
        + "_nCell"
        + str(Ncell)
        + "_rateCap"
        + str(rateCap)
        + "_rateDrop"
        + str(rateDrop)
        + "_resolution"
        + str(resolution)
        + ".gem"
    )
    gout = open(outpath + "/" + gem_write, "wb", 100 * (2**20))
    gout.write(b"geneID\tx\ty\tMIDCount\tExonCount\tCellID\n")
    rows = []

    # ramdom cell loc center
    center = {}
    for c in range(int(Ncell)):
        x = random.randrange(0, 2000)
        y = random.randrange(0, 2000)
        center[c] = [x, y]

    for index, n in np.ndenumerate(CapMatrix):
        if n > 0:
            rows.append(
                "gene"
                + str(index[1])
                + "\t"
                + str(index[2])
                + "\t"
                + str(index[3])
                + "\t"
                + str(n)
                + "\t"
                + str(n)
                + "\t"
                + str(index[0])
                + "\n"
            )
    data = "".join(rows).encode()
    gout.write(data)
    gout.close()

    # Run
    gem_path = outpath + "/" + gem_write
    simX_path = outpath + "/HEMC_simX_resolution" + str(resolution) + ".pkl"
    # Hic Matrix = np.loadtxt(sys.argv[3], delimiter=" ")
    sample_name = "HEMC"

    # filter zero cols and rows
    # HicMatrix = HicMatrix[~np.all(HicMatrix == 0, axis=1)]
    # HicMatrix = HicMatrix[:, ~np.all(HicMatrix == 0, axis=0)]
    # make dir
    # Path(outpath).mkdir(parents=True, exist_ok=True)

    # read input gem
    models = []
    gem = pd.read_csv(gem_path, sep="\t", comment="#")
    GeneUIDs = pd.Series(
        sorted(gem.geneID.drop_duplicates(), key=lambda fname: int(fname.strip("gene")))
    )
    Ngene = len(GeneUIDs)
    print(
        "TASK ID: "
        + TID
        + "\n"
        + "Cell Number: "
        + str(Ncell)
        + "\n"
        + "Gene Number: "
        + str(Ngene)
        + "\n"
        + "Gene Number for Rotation Derivatives: "
        + str(Ngene_for_rotation_derivation)
        + "\n"
        + "rateCap: "
        + str(rateCap)
        + "\n"
        + "rateDrop: "
        + str(rateDrop)
        + "\n"
        + "resolution: "
        + str(resolution)
        + "\n"
        + "mode: "
        + mode
        + "\n"
    )

    ###### update X
    W = genedistribution(gem, gem.CellID.drop_duplicates().values, GeneUIDs)
    # save_data(W, outpath + "/" + TID + "_W.pkl")
    W = normalizeW(W)
    simX = load_data(simX_path)
    # save_data(Mask, outpath + "/" + TID + "_Mask.pkl")
    loop = 1
    # get rotation R through shared X and input W
    RM = generate_random_rotation_matrices(int(W.shape[0] / 2))
    Mask = MASK(gem, GeneIDs=GeneUIDs, Ngene=Ngene_for_rotation_derivation)

    while loop <= 20:
        newX = UpdateX(RM, W)
        rmsd1, _, _ = numpy_svd_rmsd_rot(
            simX / np.linalg.norm(simX), newX / np.linalg.norm(newX)
        )
        X_mirror = np.copy(newX)
        X_mirror[:, 2] = -X_mirror[:, 2]
        rmsd2, _, _ = numpy_svd_rmsd_rot(
            simX / np.linalg.norm(simX), X_mirror / np.linalg.norm(X_mirror)
        )
        # dist, _ = numpy_svd_rmsd_rot(simX, newX)
        minrmsd = min(rmsd1, rmsd2)
        if rmsd1 < rmsd2:
            X_scale = newX
        else:
            X_scale = X_mirror
        print(
            "Distance between ground truth and reconstructed structure for loop "
            + str(loop)
            + " is: "
            + str(rmsd1)
            + " and "
            + str(rmsd2)
        )
        write_sim_pdb(
            scale_X(X_scale, 0.5)[0],
            prefix=TID
            + "_top100_nCell"
            + str(Ncell)
            + "_rateCap"
            + str(rateCap)
            + "_rateDrop"
            + str(rateDrop)
            + "_resolution"
            + str(resolution)
            + "_updated"
            + str(loop)
            + "times",
            outpath=outpath,
        )
        loop += 1
        if minrmsd < 0.01:
            break
        else:
            RM = DeriveRotation_v3(W, X_scale, Mask)

    ####### evaluation
    D = euclidean_distance_3d_matrix(X_scale)
    S = distance_to_similarity(D)
    D_ = euclidean_distance_3d_matrix(simX)
    S_ = distance_to_similarity(D_)
    print(
        "Distance".ljust(18, " ") + "\t" + "Pearson".ljust(18, " ") + "\t" + "Spearman"
    )
    print(
        str(minrmsd)
        + "\t"
        + str(np.corrcoef(S_.flatten(), S.flatten())[0][1])
        + "\t"
        + str(scipy.stats.spearmanr(S_.flatten(), S.flatten())[0])
    )

    # stop logging
    sys.stdout = stdout
    log_file.close()

    row = (
        TID,
        sample_name,
        str(Ngene),
        str(Ncell),
        str(rateCap),
        str(rateDrop),
        str(resolution),
        str(Ngene_for_rotation_derivation),
        str(noise),
        mode,
        str(minrmsd),
        str(np.corrcoef(S_.flatten(), S.flatten())[0][1]),
        str(scipy.stats.spearmanr(S_.flatten(), S.flatten())[0]),
    )
    ## "../Results/HEMC/SimulationResults.csv"
    write_row_to_csv(csv, row)


if __name__ == "__main__":
    main()
