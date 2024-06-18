import sys, os, argparse, pickle, copy, random, string, csv, warnings
import numpy as np
import pandas as pd
import scanpy as sc
from tqdm import tqdm
from pathlib import Path
from scipy import linalg as LA
from scipy.linalg import LinAlgWarning
from scipy.spatial.transform import Rotation as R
from cytocraft import model
from cytocraft import util
from cytocraft.stereopy import *
from cytocraft.rigid import *
import warnings
import traceback

warnings.filterwarnings("ignore")

"""
This module implements main functions.
"""


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
                    gem_cell_gene.x.values.astype(int),
                    weights=gem_cell_gene.MIDCount.values.astype(float),
                )
                y_median = np.average(
                    gem_cell_gene.y.values.astype(int),
                    weights=gem_cell_gene.MIDCount.values.astype(float),
                )
            W[i * 2, j] = x_median
            W[i * 2 + 1, j] = y_median
            j += 1
        i += 1
    return W


def MASK(gem, GeneIDs, CellIDs, Ngene):
    mask = np.zeros((len(CellIDs), len(GeneIDs)))
    i = 0
    for c in CellIDs:
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


def DeriveRotation(W, X, Mask, CellUIDs, adata):
    F = int(W.shape[0] / 2)
    Rotation = np.zeros((F, 3, 3))
    pop_indices = []
    for i in range(F):
        try:
            # print("derive rotation for cell " + str(i))
            # select genes
            Wi = W[:, Mask[i, :]]
            # filter cells
            Wi_filter = Wi[~np.isnan(Wi).any(axis=1), :]
            while Wi_filter.shape[0] < 6:
                # reduce the numebr of gene in Mask if cell number less than 3
                Mask[i, :] = change_last_true(Mask[i, :])
                # print("reduce one gene for cell " + str(i))
                # filter genes
                Wi = W[:, Mask[i, :]]
                # filter cells
                Wi_filter = Wi[~np.isnan(Wi).any(axis=1), :]
            idx = int(find_subarray(Wi_filter, Wi[i * 2]) / 2)
            Xi = X[Mask[i, :], :]
            model = factor(Wi_filter)
            _, R, _ = numpy_svd_rmsd_rot(np.dot(model.Rs[idx], model.Ss[0]).T, Xi)
            Rotation[i] = R
        except Exception as err:
            print("Cell ID [" + str(CellUIDs[i]) + "] is removed due to: ", err)
            pop_indices.append(i)
    for i in sorted(pop_indices, reverse=True):
        Rotation = np.delete(Rotation, obj=i, axis=0)
        W = np.delete(W, obj=i * 2 + 1, axis=0)
        W = np.delete(W, obj=i * 2, axis=0)
        adata = adata[~(adata.obs.index.values == str(CellUIDs[i])), :]
        del CellUIDs[i]
    return Rotation, W, CellUIDs, adata


def UpdateX(RM, W, GeneUID, *X_old):
    F = int(W.shape[0] / 2)
    pop_indices = []
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
        try:
            newXi = LA.solve(args, results)
            try:
                newX = np.append(
                    newX,
                    [newXi],
                    axis=0,
                )
            except NameError:
                newX = np.array([newXi])
        except (np.linalg.LinAlgError, LinAlgWarning) as err:
            print("Gene No.[" + str(j + 1) + "] is removed due to: ", err)
            pop_indices.append(j)

    # Drop genes
    ## X_old also need to drop genes, so that Xs can be compared
    if X_old:
        X_old = X_old[0]
        for i in sorted(pop_indices, reverse=True):
            del GeneUID[i]
            W = np.delete(W, obj=i, axis=1)
            X_old = np.delete(X_old, obj=i, axis=0)
        return newX, W, GeneUID, X_old
    else:
        for i in sorted(pop_indices, reverse=True):
            del GeneUID[i]
            W = np.delete(W, obj=i, axis=1)
        return newX, W, GeneUID


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


def filterGenes(array, Genes, threshold):
    # threshold is a float between 0 and 1, indicating the maximum proportion of np.nans allowed in a column
    # returns a new array with only the columns that have less than threshold proportion of np.nans
    # if array is empty or threshold is invalid, returns None

    # check if array is empty
    if array.size == 0:
        return None

    # check if threshold is valid
    if not (0 <= threshold <= 1):
        return None

    # get the number of rows and columns in the array
    rows, cols = array.shape

    # create a list to store the indices of the columns to keep
    keep_cols = []

    # loop through each column
    for i in range(cols):
        # get the column as a 1D array
        col = array[:, i]

        # count the number of np.nans in the column
        nan_count = np.count_nonzero(np.isnan(col))

        # calculate the proportion of np.nans in the column
        nan_prop = nan_count / rows

        # if the proportion is less than the threshold, add the index to the list
        if nan_prop < threshold:
            keep_cols.append(i)

    # create a new array with only the columns in the list
    new_array = array[:, keep_cols]
    Genes = list(np.array(Genes)[keep_cols])

    # return the new array
    return new_array, Genes


def get_gene_chr(species):
    if species == "Mouse" or species == "Mice":
        gtf_file = (
            os.path.dirname(os.path.realpath(__file__))
            + "/gtf/gencode.vM34.annotation.gene.gtf"
        )
    elif species == "Axolotls" or species == "Axolotl":
        gtf_file = (
            os.path.dirname(os.path.realpath(__file__))
            + "/gtf/AmexT_v47-AmexG_v6.0-DD.gene.gtf"
        )
    elif species == "Human" or species == "Monkey":
        gtf_file = (
            os.path.dirname(os.path.realpath(__file__))
            + "/gtf/gencode.v44.basic.annotation.gene.gtf"
        )
    gene_chr = {}
    with open(gtf_file, "r") as f:
        for line in f:
            if not line.startswith("#"):
                inf = line.strip().split("\t")
                if inf[2] == "gene":
                    chrom = inf[0]
                    if chrom not in gene_chr.keys():
                        gene_chr[chrom] = []
                    gene_names = inf[8].split(";")
                    try:
                        if (
                            species == "Mouse"
                            or species == "Mice"
                            or species == "Human"
                            or species == "Monkey"
                        ):
                            gene_symbol = gene_names[2].split()[1].strip('"')
                        elif species == "Axolotls":
                            gene_symbol = gene_names[1].split()[1].strip('"')
                        gene_chr[gene_symbol] = chrom
                    except IndexError as e:
                        print(
                            f"{type(e).__name__} at line {e.__traceback__.tb_lineno} of {__file__}: {e}"
                        )
                        pass
    return gene_chr


def gene_gene_distance_matrix(X):
    GeneList = X.index
    N = len(GeneList)
    DM = np.zeros((N, N))
    for n, _ in tqdm(enumerate(GeneList)):
        for m, _ in enumerate(GeneList[: n + 1]):
            d = np.linalg.norm(X.iloc[n] - X.iloc[m])
            DM[n, m] = DM[m, n] = d
    return DM


def RMSD_distance_matrix(Xs, GeneLists, keys, ngene=100, method=None):
    N = len(keys)
    DM = np.zeros((N, N))

    for n, key_n in tqdm(enumerate(keys)):
        for m, key_m in enumerate(keys[: n + 1]):
            intersected_values = np.intersect1d(GeneLists[key_n], GeneLists[key_m])
            # print("number of common gene: " + str(len(intersected_values)))
            intersected_values = intersected_values[:ngene]
            boolean_arrays_n = np.in1d(GeneLists[key_n], intersected_values)
            boolean_arrays_m = np.in1d(GeneLists[key_m], intersected_values)
            X_n = Xs[key_n][boolean_arrays_n]
            X_m = Xs[key_m][boolean_arrays_m]
            if method:
                X_n = normalizeX(X_n, method=method)
                X_m = normalizeX(X_m, method=method)
            d1, _, _ = numpy_svd_rmsd_rot(X_n, X_m)
            d2, _, _ = numpy_svd_rmsd_rot(mirror(X_n), X_m)
            DM[n, m] = DM[m, n] = min(d1, d2)
    return DM


def mirror(X):
    mirrorX = np.copy(X)
    mirrorX[:, 2] = -mirrorX[:, 2]
    return mirrorX


def euclidean_distance(coord1, coord2):
    distance = math.sqrt(sum([(x1 - x2) ** 2 for x1, x2 in zip(coord1, coord2)]))
    return distance


def knn_neighbors(X, k):
    if k != 0:
        knn_m = np.zeros((X.shape[0], X.shape[0]))
        for i in range(X.shape[0]):
            ids = np.argpartition(X[i], -k)[-k:]
            top_set = set(X[i, ids])
            if len(top_set) == 1:
                b = X[i] == top_set.pop()
                ids = []
                offset = 1
                left = True
                while (len(ids) < k) and (offset < X.shape[0]):
                    # while len(ids) < k:
                    if left:
                        idx = i + offset
                    else:
                        idx = i - offset
                    if idx < 0 or idx > len(b) - 1:
                        offset += 1
                        left = not left
                        continue
                    if b[idx]:
                        ids.append(idx)
                    offset += 1
                    left = not left
            knn_m[i, ids] = 1

        knn_m = (knn_m + knn_m.T) / 2
        knn_m[np.nonzero(knn_m)] = 1
    else:
        knn_m = X
    return knn_m


def write_pdb(show_X, genechr, geneLst, write_path, sp, seed, prefix="chain"):
    if sp == "Axolotls":
        uniquechains = [
            chr for chr in sorted(set(genechr.keys())) if chr.startswith("chr")
        ]
    else:
        uniquechains = sorted(set(genechr.keys()))

    # i is global gene index in genome include terminals
    # j is global gene index in genome
    i = 1
    j = 1
    chain_idx = 0
    for chain in uniquechains:
        k = 0
        show_shape = show_X.T
        rows = []
        rows.append(
            "HEADER".ljust(10, " ")
            + "CHROMOSOMES".ljust(40, " ")
            + get_date_today()
            + "   "
            + str(seed).ljust(4, " ")
            + "\n"
        )
        rows.append(
            "TITLE".ljust(10, " ") + "CHROMOSOME CONFORMATION INFERED FROM sST DATA\n"
        )
        Is = []
        for g in genechr[chain]:
            if g in geneLst:
                idx = geneLst.index(g)
                # normal process
                rows.append(
                    "ATOM".ljust(6, " ")
                    + str(i).rjust(5, " ")
                    + "  "
                    + "O".ljust(3, " ")
                    + " "
                    + "GLY".ljust(3, " ")
                    + " "
                    + chr(65 + chain_idx)
                    + str(j).rjust(4, " ")
                    + "    "
                    + str("%.3f" % (show_shape[0, idx])).rjust(8, " ")
                    + str("%.3f" % (show_shape[1, idx])).rjust(8, " ")
                    + str("%.3f" % (show_shape[2, idx])).rjust(8, " ")
                    + "2.00".rjust(6, " ")
                    + "10.00".rjust(6, " ")
                    + "          "
                    + "O".rjust(2, " ")
                    + "\n"
                )
                Is.append(i)
                i += 1
                j += 1
                k += 1
        if k == 0:
            continue
        i += 1
        rows.append(
            "TER".ljust(6, " ")
            + str(i - 1).rjust(5, " ")
            + "  "
            + "".ljust(3, " ")
            + " "
            + "GLY".ljust(3, " ")
            + " "
            + chr(65 + chain_idx)
            + str(j - 1).rjust(4, " ")
            + "\n"
        )
        for l in Is:
            if l - 1 in Is and l + 1 in Is:
                rows.append(
                    "CONECT".ljust(6, " ")
                    + str(l).rjust(5, " ")
                    + str(l - 1).rjust(5, " ")
                    + str(l + 1).rjust(5, " ")
                    + "\n"
                )
            elif not l - 1 in Is:
                rows.append(
                    "CONECT".ljust(6, " ")
                    + str(l).rjust(5, " ")
                    + str(l + 1).rjust(5, " ")
                    + "\n"
                )
            elif not l + 1 in Is:
                rows.append(
                    "CONECT".ljust(6, " ")
                    + str(l).rjust(5, " ")
                    + str(l - 1).rjust(5, " ")
                    + "\n"
                )
        rows.append("END\n")
        out = open(write_path + "/" + prefix + chain + ".pdb", "wb", 100 * (2**20))
        data = "".join(rows).encode()
        out.write(data)
        out.close()
        chain_idx += 1


def read_gem_as_csv(path, sep="\t"):
    gem = pd.read_csv(
        path,
        sep=sep,
        comment="#",
        dtype=str,
        converters={
            "x": float,
            "y": float,
            "cell_id": str,
            "MIDCount": float,
            "MIDCounts": float,
            "expr": float,
        },
    )
    gem["x"] = gem["x"].astype(int)
    gem["y"] = gem["y"].astype(int)
    # cell column
    if "cell" in gem.columns:
        gem.rename(columns={"cell": "CellID"}, inplace=True)
    elif "cell_id" in gem.columns:
        gem.rename(columns={"cell_id": "CellID"}, inplace=True)
    # count column
    if "MIDCounts" in gem.columns:
        gem.rename(columns={"MIDCounts": "MIDCount"}, inplace=True)
    elif "expr" in gem.columns:
        gem.rename(columns={"expr": "MIDCount"}, inplace=True)
    # gene column
    if "gene" in gem.columns:
        gem.rename(columns={"gene": "geneID"}, inplace=True)

    return gem


def read_gem_as_adata(gem_path, sep="\t", SN="adata"):
    data = read_gem(file_path=gem_path, bin_type="cell_bins", sep=sep)
    data.tl.raw_checkpoint()
    adata = stereo_to_anndata(
        data, flavor="scanpy", sample_id=SN, reindex=False, output=None
    )
    return adata


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


def generate_random_rotation_matrices(n):
    # use the Rotation.random method to generate n random rotations
    rotations = R.random(n)
    # convert the rotations to matrices
    matrices = rotations.as_matrix()
    # return the matrices
    return matrices


def get_date_today():
    from datetime import datetime

    # get the current date as a datetime object
    today = datetime.today()
    # format the date as DD-MM-YY
    date_string = today.strftime("%d-%b-%y")
    # print the date string
    return date_string


def load_data(file):
    with open(file, "rb") as f:
        x = pickle.load(f)
    return x


def save_data(data, file):
    with open(file, "wb") as f:
        pickle.dump(data, f)


def generate_id():
    return "".join(random.choices(string.ascii_uppercase + string.digits, k=4))


def legalname(string):
    import re

    # Replace spaces with underscores
    string = re.sub(r"\s+", "_", string)

    # Remove non-alphanumeric characters
    string = re.sub(r"[^a-zA-Z0-9_\-\.]", "", string)

    # Remove leading/trailing underscores
    string = re.sub(r"^_+|_+$", "", string)

    # Remove leading/trailing hyphens
    string = re.sub(r"^-+|-+$", "", string)

    # Remove leading/trailing periods
    string = re.sub(r"^\.+|\.+$", "", string)

    return string


def mean_radius(x):
    center = np.mean(x, axis=0)
    dist = np.linalg.norm(x - center, axis=1)
    return np.mean(dist)  # Return the average distance, i.e. the average radius


def normalizeX(x, method="mean"):
    if method == "L2norm":
        return x / (np.linalg.norm(x) ** 2)
    elif method == "mean":
        r = mean_radius(x)
        return x / r


def get_GeneUID(gem):
    return list(
        gem.groupby(["geneID"])["MIDCount"]
        .count()
        .reset_index(name="Count")
        .sort_values(["Count"], ascending=False)
        .geneID
    )


def get_CellUID(gem):
    return list(gem.CellID.drop_duplicates())


def run_craft(
    gem_path,
    species,
    seed,
    out_path,
    sep,
    threshold_for_gene_filter=0.9,
    threshold_for_rmsd=0.25,
    Ngene_for_rotation_derivation=None,
    percent_of_gene_for_rotation_derivation=0.001,
):
    # set seed
    random.seed(seed)
    np.random.seed(seed)
    #################### INIT ####################
    models = []
    SN = os.path.basename(os.path.splitext(gem_path)[0])
    TID = generate_id()
    outpath = out_path + "/" + SN + "_" + TID + "/"

    # make dir
    Path(outpath).mkdir(parents=True, exist_ok=True)

    # start logging
    log_file = open(outpath + "/" + SN + "_" + TID + ".log", "w")
    sys.stdout = log_file

    # read input gem
    gem = read_gem_as_csv(gem_path, sep=sep)
    adata = read_gem_as_adata(gem_path, sep=sep, SN=SN)
    GeneUIDs = get_GeneUID(gem)
    if Ngene_for_rotation_derivation is None:
        Ngene_for_rotation_derivation = int(
            float(percent_of_gene_for_rotation_derivation) * len(GeneUIDs)
        )
    try:
        adata = craft(
            gem=gem,
            adata=adata,
            species=species,
            nderive=Ngene_for_rotation_derivation,
            thresh=threshold_for_gene_filter,
            thresh_rmsd=threshold_for_rmsd,
            seed=seed,
            samplename=SN,
            outpath=outpath,
        )
    except Exception as e:
        print(
            f"{type(e).__name__} at line {e.__traceback__.tb_lineno} of {__file__}: {e}",
            file=sys.stderr,
        )
        print("conformation reconstruction failed.", file=sys.stderr)
        sys.stdout = sys.__stdout__
        log_file.close()  # stop logging
    else:
        print("conformation reconstruction finished.")
        adata.write_h5ad(filename=outpath + "adata.h5ad")
        sys.stdout = sys.__stdout__
        log_file.close()  # stop logging


def craft(
    gem,
    adata,
    species,
    nderive=10,
    thresh=0.9,
    thresh_rmsd=0.25,
    seed=999,
    samplename=None,
    outpath=False,
):
    # set seed
    random.seed(seed)
    np.random.seed(seed)
    TID = generate_id()
    # run cytocraft
    GeneUIDs = get_GeneUID(gem)
    CellUIDs = get_CellUID(gem)

    print("Speceis: " + species)
    if samplename:
        print("Sample Name: " + samplename)

    print(
        "Seed: "
        + str(seed)
        + "\n"
        + "Cell Number: "
        + str(len(CellUIDs))
        + "\n"
        + "Gene Number: "
        + str(len(GeneUIDs))
        + "\n"
        + "Threshold for gene filter is: "
        + str(thresh)
        + "\n"
        + "Number of genes used for Rotation Derivation is: "
        + str(nderive)
        + "\n"
        + "Task ID: "
        + TID
        + "\n"
    )

    # reading corresponding gene annotation file as dictionary
    gene_chr = get_gene_chr(species)

    # generate and normalize observation TC matrix 'Z'
    W = genedistribution(gem, CellUIDs, GeneUIDs)
    W, GeneUIDs = filterGenes(W, GeneUIDs, threshold=thresh)
    W = normalizeW(W)
    # generate random Rotation Matrix 'R'
    RM = generate_random_rotation_matrices(int(W.shape[0] / 2))
    X, W, GeneUIDs = UpdateX(RM, W, GeneUIDs)
    if outpath is not False:
        write_pdb(
            X,
            gene_chr,
            GeneUIDs,
            write_path=outpath,
            sp=species,
            seed=seed,
            prefix=samplename + "_initial_chr",
        )

    for loop in range(30):  # update conformation F
        ## step1: derive Rotation Matrix R
        Mask = MASK(
            gem,
            GeneIDs=GeneUIDs,
            CellIDs=CellUIDs,
            Ngene=nderive,
        )
        RM, W, CellUIDs, adata = DeriveRotation(W, X, Mask, CellUIDs, adata)
        ## step2: update conformation F with R and Z
        try:
            newX, W, GeneUIDs, X = UpdateX(RM, W, GeneUIDs, X)
        except np.linalg.LinAlgError as e:
            print(
                f"{type(e).__name__} at line {e.__traceback__.tb_lineno} of {__file__}: {e}"
            )
            return "numpy.linalg.LinAlgError"
        rmsd, _, _ = numpy_svd_rmsd_rot(
            normalizeX(X, method="mean"), normalizeX(newX, method="mean")
        )
        print(
            "RMSD between New Configuration and Old Configuration for loop "
            + str(loop + 1)
            + " is: "
            + str(rmsd)
        )
        if outpath is not False:
            write_pdb(
                newX,
                gene_chr,
                GeneUIDs,
                write_path=outpath,
                sp=species,
                seed=seed,
                prefix=samplename + "_updated" + str(loop + 1) + "times_chr",
            )
        # renew conformation F
        X = newX

        ### test: save processing adata ###
        # adata.obsm["Rotation"] = RM
        # adata.uns["F"] = pd.DataFrame(X, index=GeneUIDs)
        # adata.uns["F"].columns = adata.uns["F"].columns.astype(str)
        # adata.uns["Z"] = W
        # adata.uns["reconstruction_celllist"] = CellUIDs
        # adata.write_h5ad(filename=outpath + "loop_" + str(loop) + "_adata.h5ad")

        ## step3: check if conformation converges
        if rmsd < thresh_rmsd:
            break
    print("Number of total Transcription centers is: " + str(X.shape[0]))

    adata.obsm["Rotation"] = RM
    adata.uns["F"] = pd.DataFrame(X, index=GeneUIDs)
    adata.uns["F"].columns = adata.uns["F"].columns.astype(str)
    adata.uns["Z"] = W
    adata.uns["reconstruction_celllist"] = CellUIDs
    return adata


def parse_args():
    parser = argparse.ArgumentParser(description="Cytocraft Main Function")

    parser.add_argument(
        "gem_path",
        type=str,
        help="Path to gem file",
    )
    parser.add_argument(
        "out_path",
        type=str,
        help="Output path to save results",
    )
    parser.add_argument(
        "species",
        type=str,
        help="Species of the input data, e.g. Human, Mouse",
        choices=["Human", "Mouse", "Mice", "Axolotls", "Axolotl", "Monkey"],
    )
    parser.add_argument(
        "-p",
        "--percent",
        type=float,
        help="percent of gene for rotation derivation, default: 0.001",
        default=0.001,
    )
    parser.add_argument(
        "-n",
        "--number",
        type=int,
        help="number of gene for rotation derivation, recommend: 10",
        default=None,
    )
    parser.add_argument(
        "-t",
        "--gene_filter_thresh",
        type=float,
        help="The maximum proportion of np.nans allowed in a column(gene) in W, default: 0.90",
        default=0.90,
    )
    parser.add_argument(
        "-r",
        "--rmsd_thresh",
        type=float,
        help="The maximum value of cvRMSD allowed in conformation convergence, default: 0.25",
        default=0.25,
    )
    parser.add_argument(
        "--sep", type=str, help="input gem file separator", default="\t"
    )
    parser.add_argument(
        "-c",
        "--celltype",
        type=str,
        help="Path of obs file containing cell types",
    )
    parser.add_argument(
        "--ctkey",
        type=str,
        help="Key of celltype column in the annotation file",
    )
    parser.add_argument(
        "--cikey",
        type=str,
        help="Key of cell id column in the annotation file",
    )
    parser.add_argument(
        "--csep", type=str, help="Annotation file separator", default="\t"
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed",
    )

    args = parser.parse_args()

    return args


def split_gem(gem_path, celltype, ctkey, cikey, gsep):
    split_gem_paths = {}
    gem = read_gem_as_csv(gem_path, sep=gsep)
    for ct in celltype[ctkey].dropna().drop_duplicates():
        ct_legal = legalname(ct)
        gem_ct_path = gem_path + "." + ctkey + "." + ct_legal + ".tsv"
        split_gem_paths[ct] = gem_ct_path
        print("split gem path of " + ct + ": " + gem_ct_path)
        if cikey is not None:
            cellids = celltype[celltype[ctkey] == ct][cikey].values.astype(str)
        else:
            cellids = celltype[celltype[ctkey] == ct].index.values.astype(str)
        gem[gem.CellID.isin(cellids)].to_csv(gem_ct_path, sep="\t", index=False)
    return split_gem_paths


def main():
    #################### SETTINGS ####################
    args = parse_args()
    if args.seed is not None:
        seed = args.seed
    else:
        seed = random.randint(0, 1000)

    #################### RUNNING #####################
    if args.celltype is not None:  # run multiple cell types
        print("Cell types have been set (path is %s)" % args.celltype + "\n")
        # read gem and obs
        obs = pd.read_csv(args.celltype, sep=args.csep, dtype=str)
        # split gem
        split_paths = split_gem(
            args.gem_path,
            celltype=obs,
            ctkey=args.ctkey,
            cikey=args.cikey,
            gsep=args.sep,
        )
        # run cytocraft by order
        for ct, ct_path in split_paths.items():
            run_craft(
                gem_path=ct_path,
                species=args.species,
                seed=seed,
                out_path=args.out_path,
                sep="\t",
                threshold_for_gene_filter=args.gene_filter_thresh,
                threshold_for_rmsd=args.rmsd_thresh,
                Ngene_for_rotation_derivation=args.number,
                percent_of_gene_for_rotation_derivation=args.percent,
            )
            os.remove(ct_path)
    else:  # run single cell type
        try:
            run_craft(
                gem_path=args.gem_path,
                species=args.species,
                seed=seed,
                out_path=args.out_path,
                sep=args.sep,
                threshold_for_gene_filter=args.gene_filter_thresh,
                threshold_for_rmsd=args.rmsd_thresh,
                Ngene_for_rotation_derivation=args.number,
                percent_of_gene_for_rotation_derivation=args.percent,
            )
        except Exception as e:
            print(
                f"{type(e).__name__} at line {e.__traceback__.tb_lineno} of {__file__}: {e}"
            )
            print("conformation reconstruction failed.")
        else:
            print("conformation reconstruction finished.")


if __name__ == "__main__":
    main()
