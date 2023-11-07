import sys, os, argparse, pickle, copy, random, string, csv, warnings
import numpy as np
import pandas as pd
import scanpy as sc
import model, util

# from scipy.sparse import spmatrix, issparse, csr_matrix
# from anndata import AnnData
# from typing import Optional, Union
# from shapely.geometry import Point, MultiPoint
from tqdm import tqdm
from pathlib import Path
from scipy import linalg as LA

# from numpy.linalg import svd, solve, lstsq
# from model import BasisShapeModel
from stereopy import *
from rigid import *


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
    mask = np.zeros((CellIDs.size, GeneIDs.size))
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


def transformer(array, vector, scale=1, axis=0):
    result = np.empty_like(array)
    for i in range(array.shape[axis]):
        if axis == 0:
            result[i, :] = (array[i, :] + vector) * scale
        elif axis == 1:
            result[:, i] = (array[:, i] + vector) * scale
    return result


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


def get_date_today():
    from datetime import datetime

    # get the current date as a datetime object
    today = datetime.today()
    # format the date as DD-MM-YY
    date_string = today.strftime("%d-%b-%y")
    # print the date string
    return date_string


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
            + seed.ljust(4, " ")
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


def read_gem_as_csv(path):
    # from collections import defaultdict

    gem = pd.read_csv(
        path,
        sep="\t",
        comment="#",
        dtype=str,
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


def cluster(adata, clusterloops, maxdist=0.1):
    cluster = {}
    key = 0

    for _ in range(clusterloops):
        embedding = pd.DataFrame(adata.obsm["X_umap"], index=adata.obs_names)
        while embedding.shape[0]:
            center = embedding.sample()
            cluster[key] = np.array([])
            for idx, row in embedding.iterrows():
                vec_center = center.values
                vec_target = row.values
                d = np.linalg.norm(vec_center - vec_target)
                if d < maxdist:
                    cluster[key] = np.append(cluster[key], idx)
                    embedding = embedding.drop(idx)
            # print("Number of cells remaining:" + str(embedding.shape[0]))
            key += 1

    cluster_f = {key: cellids for key, cellids in cluster.items() if len(cellids) >= 3}

    print("Created " + str(len(cluster_f.keys())) + " near-neighbor cell groups")

    if len(cluster_f.keys()) < 100:
        warnings.warn(
            "Not enough near-neighbor cell groups. Suggest increasing the number of clustering loops. ",
            FutureWarning,
        )

    return cluster_f


def creat_models(cluster, gem, GeneUIDs):
    # init
    models = []

    ### creating models
    GeneDic = {}
    k = 0
    for i in cluster.keys():
        CellIDs = cluster[i]
        if CellIDs.size < 4:
            # print("Insufficient number of neighbors in the cluster")
            continue

        # compute gene distribution for every cell (W[i][n]); i for cell and n for gene
        W = genedistribution(gem, CellIDs, GeneUIDs)
        W = normalizeW(W)
        W_new = W[:, ~np.isnan(W).any(axis=0)]
        GeneUIDs_new = GeneUIDs[~np.isnan(W).any(axis=0)]

        if GeneUIDs_new.size > 4:
            try:
                inferred_model = factor(W_new)
                models.append(inferred_model)
                GeneDic[k] = GeneUIDs_new
                k += 1
                # print("model appended")
            except (np.linalg.LinAlgError, ValueError):
                pass
        else:
            # print("Insufficient number of common gene shared by neighbors")
            pass

    print("Created " + str(len(models)) + " models")

    return models, GeneDic


def assemble_models(models, GeneDic, Conf_sorted, init_index=0):
    Conf_sorted_index = Conf_sorted.index
    iidx = Conf_sorted_index[init_index]
    Conf_sorted_index = Conf_sorted_index.delete(init_index)
    igidx = GeneDic[iidx].index
    shape_iidx = models[iidx].Ss[0]
    signal = 0
    while signal < Conf_sorted_index.size:
        # print("Gene appended: " + str(shape_iidx.shape[1]))
        targetidx = Conf_sorted_index[0]
        Conf_sorted_index = Conf_sorted_index.delete(0)
        Geneset0 = igidx
        Geneset1 = GeneDic[targetidx].index
        Commonset = Geneset0.intersection(Geneset1)
        Common_idx0 = Geneset0.isin(Commonset)
        Common_idx1 = Geneset1.isin(Commonset)
        if len(Commonset) > 3:
            # print("appending model " + str(Conf_sorted_index[i]))
            # Alignment of common structure
            normed_shape0, T0 = normalizeX(shape_iidx[:, Common_idx0].T)
            normed_shape1, T1 = normalizeX(models[targetidx].Ss[0][:, Common_idx1].T)
            _, RM, is_reflection = numpy_svd_rmsd_rot(normed_shape0, normed_shape1)
            if is_reflection:
                models[targetidx].Ss[0][2, :] = np.flip(
                    models[targetidx].Ss[0][2, :], axis=0
                )
                normed_shape1[:, 2] = np.flip(normed_shape1[:, 2], axis=0)
                _, RM, _ = numpy_svd_rmsd_rot(normed_shape0, normed_shape1)
            rotated_normed_shape1 = np.dot(normed_shape1, RM)
            conf0 = models[iidx].W.shape[0] / models[iidx].W.shape[1]
            conf1 = models[targetidx].W.shape[0] / models[targetidx].W.shape[1]
            sumconf = conf0 + conf1
            shared_shape = (
                normed_shape0 * conf0 / sumconf
                + rotated_normed_shape1 * conf1 / sumconf
            )
            shape_iidx = np.concatenate(
                [
                    shared_shape.T,
                    transformer(shape_iidx[:, ~Common_idx0].T, T0, axis=0).T,
                    np.dot(
                        transformer(
                            models[targetidx].Ss[0][:, ~Common_idx1].T, T1, axis=0
                        ),
                        RM,
                    ).T,
                ],
                axis=1,
            )
            # Recording of normalized shape
            shape_iidx = normalizeX(shape_iidx.T)[0].T
            # Recording of gene indexes
            igidx = Commonset.append(Geneset0[~Common_idx0]).append(
                Geneset1[~Common_idx1]
            )
            signal = 0
        else:
            Conf_sorted_index = Conf_sorted_index.append(
                pd.Index(np.array([targetidx]))
            )
            signal += 1

    return shape_iidx, igidx


def ReST3D(
    gem_path,
    species,
    seed,
    percent_of_gene_for_rotation_derivation,
    clusterloops,
    maxdist,
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

    GeneUIDs = (
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
        + "Number of repetitions of clustering: "
        + str(clusterloops)
        + "\n"
        + "Upper limit of distance to neighbors in UMAP: "
        + str(maxdist)
        + "\n"
        + "Task ID: "
        + TID
        + "\n"
    )
    Ngene_for_rotation_derivation = int(
        percent_of_gene_for_rotation_derivation * len(GeneUIDs)
    )

    ### umap
    data = read_gem(file_path=gem_path, bin_type="cell_bins")
    data.tl.raw_checkpoint()
    adata = stereo_to_anndata(
        data, flavor="scanpy", sample_id="sample", reindex=False, output=None
    )
    sc.pp.neighbors(
        adata,
        method="umap",
        metric="euclidean",
        n_neighbors=10,
        n_pcs=None,
        use_rep="X",
    )
    sc.tl.umap(adata)

    ### clustering
    clusters = cluster(adata, clusterloops, maxdist)
    ### creating models
    models, genes_in_models = creat_models(clusters, gem, GeneUIDs)
    if len(models) == 0:
        return "No Model Created"
    elif len(models) < 20:
        warnings.warn(
            "Not enough models. Suggest reducing the maximum distance or increasing the number of clustering loops.",
            FutureWarning,
        )

    ### sorting models
    Conf = pd.Series(dtype="float64")
    for i in range(len(models)):
        score = pd.Series([models[i].W.shape[0] / models[i].W.shape[1]])
        Conf = pd.concat([Conf, score], ignore_index=True)
    Conf_sorted = Conf.sort_values(ascending=False)
    # Conf_sorted = Conf_sorted[Conf_sorted >= 1]

    ### assemble models
    init_idx = 0
    igidx = []
    while init_idx < len(Conf_sorted.index):
        shape_iidx_tmp, igidx_tmp = assemble_models(
            models, genes_in_models, Conf_sorted, init_index=init_idx
        )
        init_idx += 1
        print(len(igidx_tmp))
        if len(igidx_tmp) > len(igidx):
            shape_iidx = shape_iidx_tmp
            igidx = igidx_tmp

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

    ### write .pdb file
    write_pdb(
        shape_iidx.T,
        gene_chr,
        list(GeneUIDs[igidx]),
        write_path=outpath,
        sp=species,
        seed=seed,
        prefix=SN + "_updated_asm_chr",
    )

    # ##### update X
    W = genedistribution(gem, CellUIDs.values, GeneUIDs[igidx].values)
    W = normalizeW(W)

    rawX = shape_iidx.T
    X, _ = normalizeX(rawX)
    Mask = MASK(
        gem,
        GeneIDs=GeneUIDs[igidx].values,
        CellIDs=CellUIDs,
        Ngene=Ngene_for_rotation_derivation,
    )

    loop = 1
    while loop <= 30:
        # get rotation R through shared X and input W
        RM = DeriveRotation(W, X, Mask)

        # update X with R and W
        try:
            newX = UpdateX(RM, W)
        except np.linalg.LinAlgError:
            return "numpy.linalg.LinAlgError"
        rmsd, _, _ = numpy_svd_rmsd_rot(
            X / np.linalg.norm(X), newX / np.linalg.norm(newX)
        )
        print("Distance between X and newX is: " + str(rmsd))
        write_pdb(
            newX,
            gene_chr,
            list(GeneUIDs[igidx]),
            write_path=outpath,
            sp=species,
            seed=seed,
            prefix=SN + "_updated" + str(loop) + "times_chr",
        )
        ## keep looping until X converges
        # count loops
        loop += 1
        X = newX
        if rmsd < 0.02:
            break

    #################### SAVING #####################
    adata.obsm["Rotation"] = RM
    adata.uns["X"] = pd.DataFrame(X, index=GeneUIDs[igidx].values)
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
        "-l", "--loop", type=int, help="Cluster loops, default: 6", default=6
    )
    parser.add_argument(
        "-m",
        "--maxdist",
        type=float,
        help="max dist of clustering for umap embedding, default: 0.15",
        default=0.15,
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
                    args.loop,
                    args.maxdist,
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
            args.loop,
            args.maxdist,
            args.out_path,
        )


if __name__ == "__main__":
    main()
