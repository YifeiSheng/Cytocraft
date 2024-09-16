import sys, os, argparse, pickle, copy, random, string, csv, warnings
import numpy as np
import pandas as pd
import scanpy as sc
from tqdm import tqdm
from pathlib import Path
from scipy import linalg as LA
from scipy.linalg import LinAlgWarning
from scipy.spatial.transform import Rotation as R
import scipy.cluster.hierarchy as sch
from cytocraft import model
from cytocraft import util
from cytocraft.stereopy import *
from cytocraft.rigid import *
import warnings
import traceback

warnings.filterwarnings("ignore")

"""
This module implements main functions of Cytocraft.
Authors: Yifei Sheng, Shiying Li, Shuai Cheng Li
"""


def get_centers(gem, CellIDs, GeneIDs):
    """
    Calculate the transcription centers of genes for a given set of cells.

    Args:
        gem (DataFrame): The gene expression matrix.
        CellIDs (list): The list of cell IDs to consider.
        GeneIDs (list): The list of gene IDs to calculate the centers for.

    Returns:
        numpy.ndarray: The observed matrix of transcription centers.

    """
    Z = np.zeros((len(CellIDs) * 2, len(GeneIDs)))
    i = 0
    for c in CellIDs:
        j = 0
        # subset the gem in terms of individual cells
        gem_cell = gem[gem.CellID == c]
        for n in GeneIDs:
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
            Z[i * 2, j] = x_median
            Z[i * 2 + 1, j] = y_median
            j += 1
        i += 1
    return Z


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


def DeriveRotation(Z, F, Mask, CellUIDs, adata):
    """
    R-step: Derives rotation matrices for each cell based on the given inputs.

    Args:
        Z (numpy.ndarray): The input array of shape (N, M) representing gene expression levels.
        F (numpy.ndarray): The input array of shape (N, 3) representing cell positions.
        Mask (numpy.ndarray): The input array of shape (Ncell, M) representing the gene mask for each cell.
        CellUIDs (list): The list of unique identifiers for each cell.
        adata (pandas.DataFrame): The input dataframe containing additional cell data.

    Returns:
        Rotation (numpy.ndarray): The array of shape (Ncell, 3, 3) representing the rotation matrices for each cell.
        Z (numpy.ndarray): The updated array of shape (N', M) after removing cells.
        CellUIDs (list): The updated list of unique identifiers after removing cells.
        adata (pandas.DataFrame): The updated dataframe after removing cells.
    """
    Ncell = int(Z.shape[0] / 2)
    Rotation = np.zeros((Ncell, 3, 3))
    pop_indices = []
    for i in range(Ncell):
        try:
            # print("derive rotation for cell " + str(i))
            # select genes
            Zi = Z[:, Mask[i, :]]
            # filter cells
            Zi_filter = Zi[~np.isnan(Zi).any(axis=1), :]
            while Zi_filter.shape[0] < 6:
                # reduce the numebr of gene in Mask if cell number less than 3
                Mask[i, :] = change_last_true(Mask[i, :])
                # print("reduce one gene for cell " + str(i))
                # filter genes
                Zi = Z[:, Mask[i, :]]
                # filter cells
                Zi_filter = Zi[~np.isnan(Zi).any(axis=1), :]
            idx = int(find_subarray(Zi_filter, Zi[i * 2]) / 2)
            Fi = F[Mask[i, :], :]
            model = factor(Zi_filter)
            R = kabsch_numpy(np.dot(model.Rs[idx], model.Ss[0]).T, Fi)[0]
            Rotation[i] = R
        except Exception as err:
            print("Cell ID [" + str(CellUIDs[i]) + "] is removed due to: ", err)
            pop_indices.append(i)
    for i in sorted(pop_indices, reverse=True):
        Rotation = np.delete(Rotation, obj=i, axis=0)
        Z = np.delete(Z, obj=i * 2 + 1, axis=0)
        Z = np.delete(Z, obj=i * 2, axis=0)
        adata = adata[~(adata.obs.index.values == str(CellUIDs[i])), :]
        del CellUIDs[i]
    return Rotation, Z, CellUIDs, adata


def UpdateF(RM, Z, GeneUID, *F_old):
    """
    F-step: Update the coordinates of configuration based on the given rotation matrices RM and observation matrix of transcription centers Z.

    Parameters:
    - RM (numpy.ndarray): A 3-dimensional matrix representing the rotation matrices.
    - Z (numpy.ndarray): A 2-dimensional matrix representing the observation matrix of transcription centers.
    - GeneUID (list): A list of gene unique identifiers for transcription centers.
    - F_old (numpy.ndarray): The old coordinates of configuration.

    Returns:
    - newF (numpy.ndarray): A 2-dimensional matrix representing the updated coordinates of configuration.
    - Z (numpy.ndarray): A 2-dimensional matrix representing the filtered observation matrix of transcription centers Z after removing some genes.
    - GeneUID (list): A list of gene unique identifiers after removing some genes.
    - F_old (numpy.ndarray): A 2-dimensional matrix representing the filtered old coordinates of configuration after removing some genes.

    Note:
    - If F_old is not provided, the function will only update newF, Z, and GeneUID.
    - If F_old is provided, the function will update F_old as well.
    """
    Ncell = int(Z.shape[0] / 2)
    pop_indices = []
    for j in range(Z.shape[1]):
        a1 = b1 = c1 = d1 = a2 = b2 = c2 = d2 = a3 = b3 = c3 = d3 = 0
        for i in range(Ncell):
            if not (
                np.isnan(Z[i * 2 : i * 2 + 2, j]).any() or np.isnan(RM[i, :, :]).any()
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

                d1 += RM[i, 0, 0] * Z[i * 2, j] + RM[i, 0, 1] * Z[i * 2 + 1, j]
                d2 += RM[i, 1, 0] * Z[i * 2, j] + RM[i, 1, 1] * Z[i * 2 + 1, j]
                d3 += RM[i, 2, 0] * Z[i * 2, j] + RM[i, 2, 1] * Z[i * 2 + 1, j]
            else:
                # print("skip cell" + str(i) + "for gene" + str(j))
                pass

        args = np.array([[a1, b1, c1], [a2, b2, c2], [a3, b3, c3]])
        results = np.array([d1, d2, d3])
        try:
            newFi = LA.solve(args, results)
            try:
                newF = np.append(
                    newF,
                    [newFi],
                    axis=0,
                )
            except NameError:
                newF = np.array([newFi])
        except (np.linalg.LinAlgError, LinAlgWarning) as err:
            print("Gene No.[" + str(j + 1) + "] is removed due to: ", err)
            pop_indices.append(j)

    # Drop genes
    ## F_old also need to drop genes, so that F can be compared
    if F_old:
        F_old = F_old[0]
        for i in sorted(pop_indices, reverse=True):
            del GeneUID[i]
            Z = np.delete(Z, obj=i, axis=1)
            F_old = np.delete(F_old, obj=i, axis=0)
        return newF, Z, GeneUID, F_old
    else:
        for i in sorted(pop_indices, reverse=True):
            del GeneUID[i]
            Z = np.delete(Z, obj=i, axis=1)
        return newF, Z, GeneUID


def kabsch_numpy(in_crds1, in_crds2, center=False):
    """
    Returns rmsd and optional rotation between 2 sets of [nx3] arrays.

    This requires numpy for svd decomposition.
    The transform direction: transform(m, ref_crd) => target_crd.
    """
    crds1 = np.array(in_crds1)
    crds2 = np.array(in_crds2)
    assert crds1.shape[1] == 3
    assert crds1.shape == crds2.shape

    # Compute centroids
    centroid_P = np.mean(crds1, axis=0)
    centroid_Q = np.mean(crds2, axis=0)

    # Optimal translation
    t = centroid_Q - centroid_P

    # Center the points
    if center:
        crds1 = crds1 - centroid_P
        crds2 = crds2 - centroid_Q

    n_vec = np.shape(crds1)[0]
    # compute the covariance matrix
    correlation_matrix = np.dot(np.transpose(crds1), crds2)
    # SVD decomposition
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

    # optimal rotation
    rot33 = np.dot(v, w).transpose()

    return rot33, t, rmsd, is_reflection


# def kabsch_numpy(P, Q, center=False):
#    """
#    Computes the optimal rotation and translation to align two sets of points (P -> Q),
#    and their RMSD.
#
#    :param P: A Nx3 matrix of points
#    :param Q: A Nx3 matrix of points
#    :param center: Whether to center the points
#    :return: A tuple containing the optimal rotation matrix, the optimal
#             translation vector, and the RMSD.
#    """
#    P = np.array(P)
#    Q = np.array(Q)
#    assert P.shape == Q.shape, "Matrix dimensions must match"
#
#    # Compute centroids
#    centroid_P = np.mean(P, axis=0)
#    centroid_Q = np.mean(Q, axis=0)
#
#    # Optimal translation
#    t = centroid_Q - centroid_P
#
#    # Center the points
#    if center:
#        P = P - centroid_P
#        Q = Q - centroid_Q
#
#    # Compute the covariance matrix
#    H = np.dot(P.T, Q)
#
#    # SVD
#    U, S, Vt = np.linalg.svd(H)
#
#    # Validate right-handed coordinate system
#    is_reflection = np.linalg.det(np.dot(Vt.T, U.T)) < 0.0
#    if is_reflection:
#        Vt[-1, :] *= -1.0
#
#    # Optimal rotation
#    R = np.dot(Vt.T, U.T)
#
#    # RMSD
#    rmsd = np.sqrt(np.sum(np.square(np.dot(P, R.T) - Q)) / P.shape[0])
#
#    return R, t, rmsd, is_reflection


def normalizeZ(Z):
    normZ = np.empty_like(Z)
    for i in range(int(Z.shape[0])):
        normZ[i] = Z[i] - np.nanmean(Z[i])
    return normZ


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
                        elif species == "Axolotls" or species == "Axolotl":
                            gene_symbol = gene_names[1].split()[1].strip('"')
                        gene_chr[gene_symbol] = chrom
                    except IndexError as e:
                        print(
                            f"{type(e).__name__} at line {e.__traceback__.tb_lineno} of {__file__}: {e}"
                        )
                        pass
    return gene_chr


def gene_gene_distance_matrix(F):
    GeneList = list(F.index)
    N = len(GeneList)
    DM = np.zeros((N, N))
    for n, _ in enumerate(GeneList):
        for m, _ in enumerate(GeneList[: n + 1]):
            d = np.linalg.norm(F.iloc[n] - F.iloc[m])
            DM[n, m] = DM[m, n] = d
    return DM


def RMSD_distance_matrix(
    adatas, order=None, ngene=100, compare_method="pair", norm_method=None
):
    """
    Calculate the RMSD distance matrix between configurations of different cell types
    Parameters
    ----------
    Confs : dict
        The dictionary of the configurations of different cell types
    GeneLists : dict
        The dictionary of the gene lists of different cell types
    order : list, optional
        The order of the cell types, by default None
    ngene : int, optional
        The number of genes to be compared, by default 100
    compare_method : str, optional
        The method to compare the gene lists, by default "pair"
    norm_method : str, optional
        The method to normalize the configurations, by default None
    Returns
    -------
    np.ndarray
        The RMSD distance matrix
    """
    import numpy as np

    if order == None:
        keys = list(adatas.keys())
    else:
        keys = order

    GeneLists = {}
    for ct, adata in adatas.items():
        if "F" not in adata.uns.keys():
            print(f"Please run Cytocraft for {ct} first")
            return
        GeneLists[ct] = adata.uns["F"].index

    # calculate the distance matrix
    N = len(keys)
    DM = np.zeros((N, N))
    if compare_method == "complete":
        from functools import reduce

        intersected_values = reduce(np.intersect1d, GeneLists.values())
        intersected_values = intersected_values[:ngene]
        if len(intersected_values) < ngene:
            print(
                f"Warning: {len(intersected_values)} common genes between samples are less than {ngene}"
            )
    from tqdm import tqdm

    for n, key_n in enumerate(tqdm(keys)):
        for m, key_m in enumerate(keys[: n + 1]):
            if compare_method == "pair":
                intersected_values = np.intersect1d(GeneLists[key_n], GeneLists[key_m])
                intersected_values = intersected_values[:ngene]
                if len(intersected_values) < ngene:
                    print(
                        f"Warning: {len(intersected_values)} common genes between {key_n} and {key_m} are less than {ngene}"
                    )
            boolean_arrays_n = np.in1d(GeneLists[key_n], intersected_values)
            boolean_arrays_m = np.in1d(GeneLists[key_m], intersected_values)
            Conf_n = adatas[key_n].uns["F"][boolean_arrays_n]
            Conf_m = adatas[key_m].uns["F"][boolean_arrays_m]
            if norm_method:
                Conf_n = normalizeF(Conf_n, method=norm_method)
                Conf_m = normalizeF(Conf_m, method=norm_method)
            rmsd1 = kabsch_numpy(Conf_n, Conf_m)[2]
            rmsd2 = kabsch_numpy(mirror(Conf_n), Conf_m)[2]
            DM[n, m] = DM[m, n] = min(rmsd1, rmsd2)
    return DM


def expression_similarity(adatas, ngene, method="pearson", compare_method="pair"):
    if method == "pearson":
        similarity_matrix = np.zeros((len(adatas), len(adatas)))
        for i, sample1 in enumerate(adatas.keys()):
            for j, sample2 in enumerate(adatas.keys()):
                if compare_method == "pair":
                    intersected_values = np.intersect1d(
                        GeneLists[key_n], GeneLists[key_m]
                    )
                    # print("number of common gene: " + str(len(intersected_values)))
                    intersected_values = intersected_values[:ngene]
                elif compare_method == "complete":
                    intersected_values = reduce(np.intersect1d, GeneLists.values())
                if len(intersected_values) < ngene:
                    print(
                        f"Warning: {len(intersected_values)} common genes between {sample1} and {sample2} are less than {ngene}"
                    )
                boolean_arrays_n = np.in1d(GeneLists[sample1], intersected_values)
                boolean_arrays_m = np.in1d(GeneLists[sample2], intersected_values)
                average_expression1 = np.mean(
                    adatas[sample1].X[:, boolean_arrays_n], axis=0
                )
                average_expression2 = np.mean(
                    adatas[sample2].X[:, boolean_arrays_m], axis=0
                )
                similarity = np.corrcoef(average_expression1, average_expression2)[0, 1]
                similarity_matrix[i, j] = similarity
        return similarity_matrix
    # elif method == "dendrogram":
    #    Exp = np.zeros((len(adatas), ngene))
    #    for i, sample in enumerate(adatas.keys()):
    #        if not compare_method == "complate":
    #            print(
    #                "only 'complete' compare method is supported if using 'dendrogram' similarity"
    #            )
    #        intersected_values = reduce(np.intersect1d, GeneLists.values())[:ngene]
    #        if len(intersected_values) < ngene:
    #            print(
    #                f"Warning: {len(intersected_values)} common genes between samples are less than {ngene}"
    #            )
    #        boolean_arrays = np.in1d(GeneLists[sample], intersected_values)
    #        Exp[i] = np.mean(adatas[sample].X[:, boolean_arrays], axis=0)
    #    similarity_matrix = dendrogram_similarity(
    #        Exp, method="average", metric="euclidean"
    #    )
    #    return similarity_matrix
    else:
        print("method not supported")
        return


def mirror(F):
    mirrorF = np.copy(F)
    mirrorF[:, 2] = -mirrorF[:, 2]
    return mirrorF


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


def write_pdb(show_F, genechr, geneLst, write_path, sp, seed, prefix="chain"):
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
        show_shape = show_F.T
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
            "cell": str,
            "CellID": str,
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
    elif "GeneID" in gem.columns:
        gem.rename(columns={"GeneID": "geneID"}, inplace=True)

    return gem


def read_gem_as_adata(gem_path, sep="\t", SN="adata"):
    data = read_gem(file_path=gem_path, bin_type="cell_bins", sep=sep)
    data.tl.raw_checkpoint()
    adata = stereo_to_anndata(
        data, flavor="scanpy", sample_id=SN, reindex=False, output=None
    )
    return adata


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


def read_gem_header(gem_path, sep="\t"):
    with open(gem_path, "r") as f:
        header = f.readline().strip().split(sep)
        return header


def check_gem_header(header):
    if (
        ("geneID" in header or "gene" in header or "GeneID" in header)
        and ("cell" in header or "CellID" in header or "cell_id" in header)
        and "x" in header
        and "y" in header
        and ("MIDCount" in header or "MIDCounts" in header or "expr" in header)
    ):
        return True
    else:
        return False


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


def normalizeF(x, method="mean"):
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
    percent_of_gene_for_rotation_derivation=None,
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
    thresh_rmsd=0.01,
    seed=999,
    samplename=None,
    outpath=False,
):
    """
    Perform the cytocraft algorithm to generate a 3D reconstruction of transcription centers.

    Parameters:
    - gem (numpy.ndarray): The gene expression matrix.
    - adata (AnnData): The annotated data object containing cell and gene information.
    - species (str): The species of the data.
    - nderive (int, optional): The number of genes used for rotation derivation. Default is 10.
    - thresh (float, optional): The threshold for gene filtering. Default is 0.9.
    - thresh_rmsd (float, optional): The threshold for convergence of the conformation. Default is 0.25.
    - seed (int, optional): The random seed for reproducibility. Default is 999.
    - samplename (str, optional): The name of the sample. Default is None.
    - outpath (bool or str, optional): The output path for writing PDB files. Default is False.

    Returns:
    - adata (AnnData): The annotated data object with the 3D reconstruction information.

    """
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
        + "Cutoff for gene filter is: "
        + str(thresh)
        + "\n"
        + "Anchor Gene Number is: "
        + str(nderive)
        + "\n"
        + "Task ID: "
        + TID
        + "\n"
    )

    # reading corresponding gene annotation file as dictionary
    gene_chr = get_gene_chr(species)

    # generate and normalize observation TC matrix 'Z'
    Z = get_centers(gem, CellUIDs, GeneUIDs)
    Z, GeneUIDs = filterGenes(Z, GeneUIDs, threshold=thresh)
    Z = normalizeZ(Z)
    # generate random Rotation Matrix 'R'
    RM = generate_random_rotation_matrices(int(Z.shape[0] / 2))
    F, Z, GeneUIDs = UpdateF(RM, Z, GeneUIDs)
    if outpath is not False:
        write_pdb(
            F,
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
        RM, Z, CellUIDs, adata = DeriveRotation(Z, F, Mask, CellUIDs, adata)
        ## step2: update conformation F with R and Z
        try:
            newF, Z, GeneUIDs, F = UpdateF(RM, Z, GeneUIDs, F)
        except np.linalg.LinAlgError as e:
            print(
                f"{type(e).__name__} at line {e.__traceback__.tb_lineno} of {__file__}: {e}"
            )
            return "numpy.linalg.LinAlgError"
        rmsd = kabsch_numpy(
            normalizeF(F, method="mean"), normalizeF(newF, method="mean")
        )[2]
        print(
            "RMSD between New Configuration and Old Configuration for loop "
            + str(loop + 1)
            + " is: "
            + str(rmsd)
        )
        if outpath is not False:
            write_pdb(
                newF,
                gene_chr,
                GeneUIDs,
                write_path=outpath,
                sp=species,
                seed=seed,
                prefix=samplename + "_updated" + str(loop + 1) + "times_chr",
            )
        # renew conformation F
        F = newF

        ### test: save processing adata ###
        # adata.obsm["Rotation"] = RM
        # adata.uns["F"] = pd.DataFrame(F, index=GeneUIDs)
        # adata.uns["F"].columns = adata.uns["F"].columns.astype(str)
        # adata.uns["Z"] = Z
        # adata.uns["reconstruction_celllist"] = CellUIDs
        # adata.write_h5ad(filename=outpath + "loop_" + str(loop) + "_adata.h5ad")

        ## step3: check if conformation converges
        if rmsd < thresh_rmsd:
            break
    print("Number of total transcription centers is: " + str(F.shape[0]))

    adata.obsm["Rotation"] = RM
    adata.uns["F"] = pd.DataFrame(F, index=GeneUIDs)
    adata.uns["F"].columns = adata.uns["F"].columns.astype(str)
    adata.uns["Z"] = Z
    adata.uns["reconstruction_celllist"] = CellUIDs
    return adata


def parse_args():
    parser = argparse.ArgumentParser(description="Cytocraft Main Function")

    parser.add_argument(
        "-i",
        "--gem_path",
        type=str,
        help="Path of input gene expression matrix file",
        required=True,
    )
    parser.add_argument(
        "-o",
        "--out_path",
        type=str,
        help="Directory to save results",
        required=True,
    )
    parser.add_argument(
        "--species",
        type=str,
        help="Species of the input data",
        choices=["Human", "Mouse", "Mice", "Axolotls", "Axolotl", "Monkey"],
        required=True,
    )
    parser.add_argument(
        "-p",
        "--percent",
        type=float,
        help="Percent of anchor gene for rotation derivation",
        default=None,
    )
    parser.add_argument(
        "-n",
        "--number",
        type=int,
        help="Number of anchor gene for rotation derivation",
        default=10,
    )
    parser.add_argument(
        "-t",
        "--gene_filter_thresh",
        type=float,
        help="The maximum allowable proportion of np.nan values in a column (representing a gene) of the observed transcription centers (Z).",
        default=0.90,
    )
    parser.add_argument(
        "-r",
        "--rmsd_thresh",
        type=float,
        help="RMSD threshold. If the computed RMSD value is less than or equal to this threshold, it means the process has reached an acceptable level of similarity or convergence, and the loop is exited.",
        default=0.01,
    )
    parser.add_argument(
        "--sep",
        type=str,
        help="Separator of the input gene expression matrix file",
        default="\t",
    )
    parser.add_argument(
        "-c",
        "--celltype",
        type=str,
        help="Path of the annotation file containing cell id and cell type, multi-celltype mode only",
    )
    parser.add_argument(
        "--ctkey",
        type=str,
        help="Key of celltype column in the annotation file, multi-celltype mode only",
    )
    parser.add_argument(
        "--cikey",
        type=str,
        help="Key of cell id column in the annotation file, multi-celltype mode only",
    )
    parser.add_argument(
        "--csep",
        type=str,
        help="Separator of the annotation file, multi-celltype mode only",
        default="\t",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed, default: random int between 0 to 1000",
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

    ################ Check Gem Header ################
    gem_header = read_gem_header(args.gem_path, sep=args.sep)
    if not check_gem_header(gem_header):
        print(
            "Invalid gem header. Please make sure the gem file includes 'geneID', 'cell', 'x', 'y', 'MIDCount' columns."
        )
        return

    #################### RUNNING #####################
    if args.celltype is not None:  # run multiple cell types
        print("Running Multi-Celltype Mode")
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
        print("Running Single-Celltype Mode")
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
