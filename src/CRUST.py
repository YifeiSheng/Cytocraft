import sys, os, argparse, pickle, copy, random, string, csv, warnings
import numpy as np
import pandas as pd
import scanpy as sc
import model, util
from tqdm import tqdm
from pathlib import Path
from scipy import linalg as LA
from scipy.spatial.transform import Rotation as R
from stereopy import *
from rigid import *
import util


"""
This module implements main function.
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
    mask = np.zeros((CellIDs.size, len(GeneIDs)))
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


def DeriveRotation(W, X, Mask):
    F = int(W.shape[0] / 2)
    Rotation = np.zeros((F, 3, 3))
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
        try:
            model = factor(Wi_filter)
            _, R, _ = numpy_svd_rmsd_rot(np.dot(model.Rs[idx], model.Ss[0]).T, Xi)
            Rotation[i] = R
        except ValueError:
            pass
    return Rotation


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
        except np.linalg.LinAlgError or LinAlgWarning:
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


def normalizeX(X):
    """
    X = (nGene,3)
    """
    if len(X.shape) == 2 and X.shape[1] == 3:
        result = np.empty_like(X)
        tl = -np.nanmean(X, axis=0)
        for i in range(X.shape[1]):
            result[:, i] = X[:, i] - np.nanmean(X[:, i])
        return result, tl
    else:
        print("X shape error!")


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
        show_shape = show_X.T
        out = open(write_path + "/" + prefix + chain + ".pdb", "w")
        out.writelines(
            "HEADER".ljust(10, " ")
            + "CHROMOSOMES".ljust(40, " ")
            + get_date_today()
            + "   "
            + str(seed).ljust(4, " ")
            + "\n"
        )
        out.writelines(
            "TITLE".ljust(10, " ") + "CHROMOSOME CONFORMATION INFERED FROM sST DATA\n"
        )
        Is = []
        for g in genechr[chain]:
            if g in geneLst:
                idx = geneLst.index(g)
                # normal process
                out.writelines(
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
        i += 1
        out.writelines(
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
                out.writelines(
                    "CONECT".ljust(6, " ")
                    + str(l).rjust(5, " ")
                    + str(l - 1).rjust(5, " ")
                    + str(l + 1).rjust(5, " ")
                    + "\n"
                )
            elif not l - 1 in Is:
                out.writelines(
                    "CONECT".ljust(6, " ")
                    + str(l).rjust(5, " ")
                    + str(l + 1).rjust(5, " ")
                    + "\n"
                )
            elif not l + 1 in Is:
                out.writelines(
                    "CONECT".ljust(6, " ")
                    + str(l).rjust(5, " ")
                    + str(l - 1).rjust(5, " ")
                    + "\n"
                )
        out.writelines("END\n")
        out.close()
        chain_idx += 1


def read_gem_as_csv(path):
    # from collections import defaultdict

    gem = pd.read_csv(
        path,
        sep="\t",
        comment="#",
        converters={
            "x": int,
            "y": int,
            "MIDCount": float,
            "MIDCounts": float,
            "expr": float,
        },
    )

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


def ReST3D(
    gem_path,
    species,
    seed,
    percent_of_gene_for_rotation_derivation,
    out_path,
):
    #################### INIT ####################
    models = []
    filename, _ = os.path.splitext(gem_path)
    SN = os.path.basename(filename)
    TID = generate_id()
    outpath = out_path + "/" + SN + "/" + TID + "/"

    # make dir
    Path(outpath).mkdir(parents=True, exist_ok=True)
    # start logging
    stdout = sys.stdout
    log_file = open(outpath + "/" + SN + "_" + TID + ".log", "w")
    sys.stdout = log_file

    #################### PROCESSING #####################
    # read input gem
    gem = read_gem_as_csv(gem_path)

    GeneUIDs = list(
        gem.groupby(["geneID"])["MIDCount"]
        .count()
        .reset_index(name="Count")
        .sort_values(["Count"], ascending=False)
        .geneID
    )
    CellUIDs = gem.CellID.drop_duplicates()

    print(
        "Speceis: "
        + species
        + "\n"
        + "File Name:"
        + filename
        + "\n"
        + "Seed: "
        + str(seed)
        + "\n"
        + "Cell Number: "
        + str(len(CellUIDs))
        + "\n"
        + "Gene Number: "
        + str(len(GeneUIDs))
        + "\n"
        + "Sample Name: "
        + SN
        + "\n"
        + "Proportion of genes used for Rotation Derivation: "
        + str(percent_of_gene_for_rotation_derivation)
        + "\n"
        + "Task ID: "
        + TID
        + "\n"
    )
    Ngene_for_rotation_derivation = int(
        percent_of_gene_for_rotation_derivation * len(GeneUIDs)
    )

    ### processing gtf
    if species == "Mouse" or species == "Mice":
        gtf_file = "/data1/shengyifei/ReST3D/gtf/mice_genes.gtf"
    elif species == "Axolotls":
        gtf_file = "/data1/shengyifei/ReST3D/gtf/AmexT_v47-AmexG_v6.0-DD.gtf"
    elif species == "Human" or species == "Monkey":
        gtf_file = "/data1/shengyifei/ReST3D/gtf/gencode.v44.basic.annotation.gtf"
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
                        gene_chr[chrom].append(gene_symbol)
                    except IndexError:
                        print("IndexError")
                        pass

    ##### generate random RM and derive X
    W = genedistribution(gem, CellUIDs.values, GeneUIDs)
    W, GeneUIDs = filterGenes(W, GeneUIDs, threshold=0.9)
    W = normalizeW(W)
    RM = generate_random_rotation_matrices(int(W.shape[0] / 2))
    X, W, GeneUIDs = UpdateX(RM, W, GeneUIDs)
    write_pdb(
        X,
        gene_chr,
        GeneUIDs,
        write_path=outpath,
        sp=species,
        seed=seed,
        prefix=SN + "_initial_chr",
    )
    ##### update X
    for loop in range(30):
        # step1: derive Rotation Matrix
        Mask = MASK(
            gem,
            GeneIDs=GeneUIDs,
            CellIDs=CellUIDs,
            Ngene=Ngene_for_rotation_derivation,
        )
        RM = DeriveRotation(W, X, Mask)
        # step2: update conformation X with RM and W
        try:
            newX, W, GeneUIDs, X = UpdateX(RM, W, GeneUIDs, X)
        except np.linalg.LinAlgError:
            return "numpy.linalg.LinAlgError"
        rmsd, _, _ = numpy_svd_rmsd_rot(
            X / np.linalg.norm(X), newX / np.linalg.norm(newX)
        )
        print("Distance between X_new and X_old is: " + str(rmsd))
        write_pdb(
            newX,
            gene_chr,
            GeneUIDs,
            write_path=outpath,
            sp=species,
            seed=seed,
            prefix=SN + "_updated" + str(loop) + "times_chr",
        )
        ## keep looping until X converges
        X = newX
        if rmsd < 0.01:
            break

    #################### SAVING #####################
    print("Number of total Transcription centers is: " + str(X.shape[0]))
    data = read_gem(file_path=gem_path, bin_type="cell_bins")
    data.tl.raw_checkpoint()
    adata = stereo_to_anndata(
        data, flavor="scanpy", sample_id="sample", reindex=False, output=None
    )
    adata.obsm["Rotation"] = RM
    adata.uns["X"] = pd.DataFrame(X, index=GeneUIDs)
    adata.uns["X"].columns = adata.uns["X"].columns.astype(str)
    adata.uns["W"] = W
    adata.write_h5ad(filename=outpath + "adata.h5ad")

    # stop logging
    sys.stdout = stdout
    log_file.close()


def parse_args():
    parser = argparse.ArgumentParser(description="ReST3D Main Function")

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
        choices=["Human", "Mouse", "Axolotls", "Monkey"],
    )
    parser.add_argument(
        "-p",
        "--percent",
        type=float,
        help="percent of gene for rotation derivation, default: 0.001",
        default=0.001,
    )

    parser.add_argument(
        "-c",
        "--celltype",
        type=str,
        help="Path of file containing cell types",
    )
    parser.add_argument(
        "--ctkey",
        type=str,
        help="Key of celltype column in the cell type file",
    )
    parser.add_argument(
        "--cikey",
        type=str,
        help="Key of cell id column in the cell type file",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed",
    )

    args = parser.parse_args()

    return args


def main():
    #################### SETTINGS ####################
    args = parse_args()
    if args.seed is not None:
        seed = args.seed
    else:
        seed = random.randint(0, 1000)
    random.seed(seed)
    np.random.seed(seed)

    #################### RUNNING ####################
    if args.celltype is not None and args.ctkey is not None and args.cikey is not None:
        print("Cell types have been set (path is %s)" % args.celltype + "\n")
        gem = read_gem_as_csv(args.gem_path)
        celltype = pd.read_csv(args.celltype, sep="\t", dtype=str)
        for ct in celltype[args.ctkey].drop_duplicates():
            ct_legal = legalname(ct)
            gem_ct_path = args.gem_path + "." + args.ctkey + "." + ct_legal + ".tsv"
            cellids = celltype[celltype[args.ctkey] == ct][args.cikey].values
            gem[gem.CellID.isin(cellids)].to_csv(gem_ct_path, sep="\t", index=False)
            try:
                ReST3D(
                    gem_ct_path,
                    args.species,
                    seed,
                    args.percent,
                    args.out_path + "/" + ct_legal + "/",
                )
            except IndexError:
                print("cell number of " + ct + " is not enough")
                pass
    else:
        ReST3D(
            args.gem_path,
            args.species,
            seed,
            args.percent,
            args.out_path,
        )


if __name__ == "__main__":
    main()
