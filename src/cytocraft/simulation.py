from matplotlib import pyplot as plt
import scipy.stats
from scipy.sparse import spmatrix, issparse, csr_matrix
from scipy.stats import multivariate_normal
from anndata import AnnData
from typing import Optional, Union
from shapely.geometry import Point, MultiPoint
from numpy.linalg import svd, solve, lstsq
from numpy.random import randn
from cytocraft.craft import *
from cytocraft.model import BasisShapeModel


def DeriveRotation(W, X, Mask):
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
        R = kabsch_numpy(np.dot(model.Rs[idx], model.Ss[0]).T, Xi)[0]
        Rotation[i] = R
    return Rotation


# def UpdateF(RM, W):
#    F = int(W.shape[0] / 2)
#    for j in range(W.shape[1]):
#        a1 = b1 = c1 = d1 = a2 = b2 = c2 = d2 = a3 = b3 = c3 = d3 = 0
#        for i in range(F):
#            if not (
#                np.isnan(W[i * 2 : i * 2 + 2, j]).any() or np.isnan(RM[i, :, :]).any()
#            ):
#                a1 += RM[i, 0, 0] * RM[i, 0, 0] + RM[i, 0, 1] * RM[i, 0, 1]
#                b1 += RM[i, 0, 0] * RM[i, 1, 0] + RM[i, 0, 1] * RM[i, 1, 1]
#                c1 += RM[i, 0, 0] * RM[i, 2, 0] + RM[i, 0, 1] * RM[i, 2, 1]
#
#                a2 += RM[i, 1, 0] * RM[i, 0, 0] + RM[i, 1, 1] * RM[i, 0, 1]
#                b2 += RM[i, 1, 0] * RM[i, 1, 0] + RM[i, 1, 1] * RM[i, 1, 1]
#                c2 += RM[i, 1, 0] * RM[i, 2, 0] + RM[i, 1, 1] * RM[i, 2, 1]
#
#                a3 += RM[i, 2, 0] * RM[i, 0, 0] + RM[i, 2, 1] * RM[i, 0, 1]
#                b3 += RM[i, 2, 0] * RM[i, 1, 0] + RM[i, 2, 1] * RM[i, 1, 1]
#                c3 += RM[i, 2, 0] * RM[i, 2, 0] + RM[i, 2, 1] * RM[i, 2, 1]
#
#                d1 += RM[i, 0, 0] * W[i * 2, j] + RM[i, 0, 1] * W[i * 2 + 1, j]
#                d2 += RM[i, 1, 0] * W[i * 2, j] + RM[i, 1, 1] * W[i * 2 + 1, j]
#                d3 += RM[i, 2, 0] * W[i * 2, j] + RM[i, 2, 1] * W[i * 2 + 1, j]
#            else:
#                # print("skip cell" + str(i) + "for gene" + str(j))
#                pass
#
#        args = np.array([[a1, b1, c1], [a2, b2, c2], [a3, b3, c3]])
#        results = np.array([d1, d2, d3])
#        newXi = LA.solve(args, results)
#        try:
#            newX = np.append(
#                newX,
#                [newXi],
#                axis=0,
#            )
#        except NameError:
#            newX = np.array([newXi])
#
#    return newX


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


def centerX(X):
    """
    X = (nGene,3)
    """
    if len(X.shape) == 2 and X.shape[1] == 3:
        mean_x, mean_y, mean_z = X.mean(axis=0)
        centered_X = X - [mean_x, mean_y, mean_z]
        return centered_X, [mean_x, mean_y, mean_z]
    else:
        print("X shape error!")


def main():
    # Read auguments
    GenomeStructure = pd.read_csv(sys.argv[1], delimiter="\t")
    sample_name = os.path.basename(sys.argv[1]).split(".")[0]
    Ngene = int(sys.argv[2])  ### simulated gene number
    Ncell = int(sys.argv[3])  ### simulated cell number
    rateCap = float(sys.argv[4])  ### set capture rate
    rateDrop = float(sys.argv[5])  ### set gene loss rate
    resolution = int(sys.argv[6])  ### difine the resolution
    Ngene_for_rotation_derivation = int(sys.argv[7])
    noise = int(sys.argv[8])
    mode = sys.argv[9].strip()
    outpath = sys.argv[10]
    csv = sys.argv[11]

    # Define outpath
    outpath = (
        outpath
        + "/"
        + sample_name
        + "_NG"
        + str(Ngene)
        + "_NC"
        + str(Ncell)
        + "_RC"
        + str(rateCap)
        + "_RD"
        + str(rateDrop)
        + "_RS"
        + str(resolution)
        + "_NGF"
        + str(Ngene_for_rotation_derivation)
        + "_NS"
        + str(noise)
        + "_"
        + mode
    )
    # Exit if the experiment has been done
    # result = pd.read_csv(csv, sep=",")
    # repeats = result.query(
    #     f"GeneNumber == {Ngene} and \\"
    #     f"CellNumber == {Ncell} and \\"
    #     f"CaptureRate == {rateCap} and \\"
    #     f"DropRate == {rateDrop} and \\"
    #     f"Resolution== {resolution} and \\"
    #     f"NumberofGeneforFactorization == {Ngene_for_rotation_derivation} and \\"
    #     f"Noise == {noise} and \\"
    #     f"Mode == '%s'" % mode
    # )
    # if len(repeats) >= 10:
    #     sys.exit(1)
    Path(outpath).mkdir(parents=True, exist_ok=True)
    # Check the number of arguments and set random seed
    if len(sys.argv) == 13:
        # Try to convert the 12th argument to an integer
        try:
            seed = int(sys.argv[12])
        except ValueError:
            # If the argument is not a valid integer, print an error message and exit
            print("Invalid seed argument. Please provide an integer.")
            sys.exit(1)
    elif len(sys.argv) == 12:
        # If the argument is not given, generate a random seed
        seed = random.randint(0, 1000000)
    else:
        print(
            "Argument length is not correct, please check each parameter for correctness."
        )
        sys.exit(1)
    random.seed(seed)
    np.random.seed(seed)

    if mode == "continous":
        Conformation = np.array(GenomeStructure[["x", "y", "z"]].head(Ngene))
    elif mode == "random":
        sampled_indices = sorted(
            np.random.choice(len(GenomeStructure), size=Ngene, replace=False)
        )
        Conformation = np.array(GenomeStructure[["x", "y", "z"]].iloc[sampled_indices])
    else:
        print("input mode is not correct")
    TID = generate_id()

    # start logging
    # stdout = sys.stdout
    # log_file = open(outpath + "/" + TID + ".log", "w")
    # sys.stdout = log_file
    print(f"The seed is {seed}.")

    ##### GENERATE SIMULATION DATA
    ScaledCoords, _ = scale_X(Conformation, resolution)
    simX, _ = centerX(ScaledCoords)
    # save_data(simX, outpath + "/"+TID+"_simX_resolution" + str(resolution) + ".pkl")

    #### generate random RM
    randRM = generate_random_rotation_matrices(Ncell)
    save_data(
        randRM,
        outpath
        + "/"
        + TID
        + "_randRM_nCell"
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

    # save_data(simW, outpath + "/HMEC_simW_" + str(Ncell) + ".pkl")

    # add Noise1: random noise on locs
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

    # Creat fixed Gaussian noise 3 matrix
    # noise_scale = int(noise / 2)
    # noise_matrix = np.random.normal(0, noise_scale, (UBx - LBx, UBy - LBy))
    if noise == 0:
        sigma = np.eye(2) * 8
    else:
        # add Noise2: random sigma
        A = np.random.rand(2, 2)  # generate a 2x2 random matrix
        sigma = (
            np.dot(A, A.transpose()) * 8
        )  # create a symmetric, positive semi-definite matrix for the covariance
    for c in range(Ncell):
        for g in range(Ngene):
            loc = simW[c * 2 : (c + 1) * 2, g]
            # add random noise 2 on exp matrix
            exp = (
                multivariate_normal.pdf(xy, mean=loc, cov=sigma, allow_singular=True)
                * 128
            )  # compute the multivariate normal pdf on the grid
            # add fixed noise 3 to exp matrix
            Matrix[c, g] = (
                exp.T.round().clip(min=0, max=127).astype("i1")
            )  # round and cast to integer

    #### drop counts based on capture rate
    CapMatrix = np.random.binomial(
        Matrix, rateCap
    )  # generate random numbers from a binomial distribution with Matrix as the number of trials and rateCap as the probability

    #### drop centers based on drop rate
    mask = (
        np.random.random((Ncell, Ngene)) < rateDrop
    )  # generate a boolean mask with rateDrop as the probability
    CapMatrix[mask] = (
        0  # set the elements of CapMatrix that correspond to True in the mask to zero
    )

    #### write gem as data frame
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

    ##### RUN Cytocraft
    sample_name = "HMEC"

    # read input gem
    models = []
    X = pd.Series(
        sorted(gem.geneID.drop_duplicates(), key=lambda fname: int(fname.strip("gene")))
    )
    Ngene = len(X)
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

    ##### update X
    W = get_centers(gem, gem.CellID.drop_duplicates().values, X)
    W = normalizeW(W)
    # get rotation R through shared X and input W
    RM = generate_random_rotation_matrices(int(W.shape[0] / 2))
    CellUIDs = list(gem.CellID.drop_duplicates())
    Mask = MASK(gem, GeneIDs=X, CellIDs=CellUIDs, Ngene=Ngene_for_rotation_derivation)
    X, _, _ = UpdateF(RM, W, X)
    write_sim_pdb(
        scale_X(X, 0.5)[0],
        prefix=TID
        + "_top100_nCell"
        + str(Ncell)
        + "_rateCap"
        + str(rateCap)
        + "_rateDrop"
        + str(rateDrop)
        + "_resolution"
        + str(resolution)
        + "_initial",
        outpath=outpath,
    )
    try:
        for loop in range(30):
            RM = DeriveRotation(W, X, Mask)
            try:
                X, _, _ = UpdateF(RM, W, X)
            except np.linalg.LinAlgError:
                return "numpy.linalg.LinAlgError"

            # test
            # save_data(W, outpath + "/HMEC_W_" + "loop_" + str(loop) + ".pkl")
            # save_data(X, outpath + "/HMEC_X_" + "loop_" + str(loop) + ".pkl")
            # save_data(RM, outpath + "/HMEC_RM_" + "loop_" + str(loop) + ".pkl")

            # evaluate
            rmsd1 = kabsch_numpy(
                normalizeF(simX, method="mean"), normalizeF(X, method="mean")
            )[2]
            X_mirror = np.copy(X)
            X_mirror[:, 2] = -X_mirror[:, 2]
            rmsd2 = kabsch_numpy(
                normalizeF(simX, method="mean"), normalizeF(X_mirror, method="mean")
            )[2]
            minrmsd = min(rmsd1, rmsd2)
            print(
                "Distance between ground truth and reconstructed structure for loop "
                + str(loop + 1)
                + " is: "
                + str(rmsd1)
                + " and "
                + str(rmsd2)
            )
            write_sim_pdb(
                scale_X(X, 0.5)[0],
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
                + str(loop + 1)
                + "times",
                outpath=outpath,
            )
            if minrmsd < 0.005:
                break
    except Exception as error:
        print(error)
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
            "NA",
            "NA",
            "NA",
        )
        if not csv == None:
            write_row_to_csv(csv, row)
        sys.exit(1)

    ####### evaluation
    if minrmsd == rmsd1:
        mirror = 0
    else:
        mirror = 1
    D = euclidean_distance_3d_matrix(X)
    S = distance_to_similarity(D)
    D_ = euclidean_distance_3d_matrix(simX)
    S_ = distance_to_similarity(D_)
    print(
        "RMSD Distance".ljust(18, " ")
        + "\t"
        + "Mirror"
        + "\t"
        + "Spearman Correlation Coefficient"
    )
    print(
        str(minrmsd)
        + "\t"
        + str(mirror)
        + "\t"
        + str(scipy.stats.spearmanr(S_.flatten(), S.flatten())[0])
    )

    # stop logging
    # sys.stdout = stdout
    # log_file.close()

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
        str(mirror),
        str(scipy.stats.spearmanr(S_.flatten(), S.flatten())[0]),
    )

    if not csv == None:
        write_row_to_csv(csv, row)


if __name__ == "__main__":
    main()
