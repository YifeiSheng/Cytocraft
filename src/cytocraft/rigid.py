from math import cos, sin
import numpy as np
from numpy.linalg import svd, solve, lstsq
from cytocraft import model
from cytocraft import util
from cytocraft.model import BasisShapeModel

"""
This module implements rigid factorization.
"""


def is_rotation_matrix(matrix):
    # Check if the matrix is square and has shape (3, 3)
    if not isinstance(matrix, np.ndarray) or matrix.shape != (3, 3):
        print("error1")
        return False

    # Check if the matrix is orthogonal
    if not np.allclose(matrix.T @ matrix, np.eye(3), atol=0.0001):
        print("error2")
        return False

    # Check if the matrix has determinant 1
    if not np.isclose(np.linalg.det(matrix), 1, atol=0.0001):
        print("error3")
        return False

    # Check if the matrix preserves the length of a random vector
    v = np.random.rand(3)
    if not np.isclose(np.linalg.norm(matrix @ v), np.linalg.norm(v), atol=0.0001):
        print("error4")
        return False

    # If all checks pass, return True
    return True


def factor_matrix(W, J=3):
    """Obtain the best rank J factorization MS to W

    Computes a factorization of W into M and B such that ||W-MB||_F
    is minimized.

    Input:
    W is a MxN matrix.

    Returns:
    M -- A MxR matrix
    B -- A RxN matrix
    """

    # Decompose using SVD.
    U, s, Vt = svd(W, full_matrices=False)

    # Compute the factorization.
    sqrt_sigma = np.diag(np.sqrt(s[:J]))
    M = np.dot(U[:, :J], sqrt_sigma)
    B = np.dot(sqrt_sigma, Vt[:J])

    return M, B


def factor(W):
    """
    This implements rigid factorization as described in

    Tomasi, C. & Kanade, T. "Shape and motion from image streams under
    orthography: a factorization method International Journal of Computer
    Vision, 1992
    """

    F = int(W.shape[0] / 2)
    N = W.shape[1]

    # Center W
    T = W.mean(axis=1)
    W = W - T[:, np.newaxis]

    # Factor W
    M_hat, B_hat = factor_matrix(W, J=3)

    # Where we will build the linear system.
    A = np.zeros((3 * F, 6))
    b = np.zeros((3 * F,))

    for f in range(F):
        # Extract the two rows.
        x_f, y_f = M_hat[f * 2 : f * 2 + 2]

        # Both rows must have unit length.
        A[f * 3] = util.vc(x_f, x_f)
        b[f * 3] = 1.0
        A[f * 3 + 1] = util.vc(y_f, y_f)
        b[f * 3 + 1] = 1.0

        # And be orthogonal.
        A[f * 3 + 2] = util.vc(x_f - y_f, x_f + y_f)

    # Recovec vech(Q) and Q
    vech_Q = lstsq(A, b, rcond=None)[0]
    Q = util.from_vech(vech_Q, 3, 3, sym=True)

    # Factor out G recovery matrix
    G, Gt = factor_matrix(Q)

    # Upgrade M and B.
    M = np.dot(M_hat, G)
    B = solve(G, B_hat)

    # Find actual rotations matrices.
    Rs = np.zeros((F, 3, 3))
    Rs[:, :2] = M.reshape(F, 2, 3)
    Rs[:, 2] = np.cross(Rs[:, 0], Rs[:, 1], axis=-1)
    # Rs[:, 2] = util.normed(np.cross(Rs[:, 0], Rs[:, 1], axis=-1))

    # check reflection
    # M = np.einsum("ijk, kl->ijl", Rs, np.linalg.inv(G))
    # is_reflection = (np.linalg.det(M) * np.linalg.det(G)) < 0.0
    # print(is_reflection)
    # I = np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]])
    # if is_reflection.any():
    #     print("reflect")
    #     Rs = np.einsum("ijk, kl->ijl", np.einsum("jk, ikl->ijl", I, Rs), I)

    # And 3D translations.
    Ts = np.zeros((F, 3))
    Ts[:, :2] = T.reshape(F, 2)

    model = BasisShapeModel(Rs, Bs=B[np.newaxis, :, :], Ts=Ts)

    return model
